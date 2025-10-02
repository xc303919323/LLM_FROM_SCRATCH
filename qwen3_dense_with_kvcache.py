import torch
import json
import math
from safetensors import safe_open
from transformers import AutoTokenizer

def custom_gemm(a, b):
    """自定义GEMM实现"""
    original_shape = a.shape
    a_flat = a.view(-1, a.size(-1))
    result = torch.mm(a_flat, b)
    return result.view(*original_shape[:-1], b.size(-1))

class WeightManager:
    def __init__(self, safetensors_path):
        self.weights = {}
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                self.weights[key] = f.get_tensor(key).clone().float()
        print(f"加载了 {len(self.weights)} 个权重")

    def get_weight(self, key):
        return self.weights.get(key)

class CustomRMSNorm:
    def __init__(self, weight, eps=1e-6):
        self.weight = weight
        self.eps = eps

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight

class CustomLinear:
    def __init__(self, weight, bias=None):
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        output = custom_gemm(x, self.weight.t())
        if self.bias is not None:
            output = output + self.bias
        return output

class CustomEmbedding:
    def __init__(self, weight):
        self.weight = weight

    def forward(self, input_ids):
        return self.weight[input_ids]

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """应用旋转位置编码"""
    if position_ids is None:
        seq_len = q.shape[-2]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=q.device)
        position_ids = position_ids.unsqueeze(0)

    cos = cos[position_ids]
    sin = sin[position_ids]

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class QwenRotaryEmbedding:
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        max_seq_len = max_position_embeddings
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)

    def forward(self, x, seq_len=None):
        if seq_len > self.cos_cached.shape[0]:
            t = torch.arange(seq_len, dtype=torch.float)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()

        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )

class KVCache:
    """简化的KV Cache实现"""
    def __init__(self):
        self.cache_k = None
        self.cache_v = None

    def update(self, k, v):
        """更新KV缓存"""
        if self.cache_k is None:
            # 首次缓存
            self.cache_k = k
            self.cache_v = v
        else:
            # 拼接新的k,v
            self.cache_k = torch.cat([self.cache_k, k], dim=-2)
            self.cache_v = torch.cat([self.cache_v, v], dim=-2)

        return self.cache_k, self.cache_v

    def reset(self):
        """重置缓存"""
        self.cache_k = None
        self.cache_v = None

class Qwen3AttentionWithKVCache:
    """带KV Cache的Qwen3注意力实现"""
    def __init__(self, layer_idx, weight_manager, config):
        self.layer_idx = layer_idx
        self.hidden_size = config['hidden_size']
        self.num_heads = config['num_attention_heads']
        self.num_key_value_heads = config['num_key_value_heads']
        self.head_dim = config['head_dim']
        self.max_position_embeddings = config['max_position_embeddings']

        assert self.num_heads % self.num_key_value_heads == 0

        # 投影层
        prefix = f"model.layers.{layer_idx}.self_attn"
        self.q_proj = CustomLinear(weight_manager.get_weight(f"{prefix}.q_proj.weight"))
        self.k_proj = CustomLinear(weight_manager.get_weight(f"{prefix}.k_proj.weight"))
        self.v_proj = CustomLinear(weight_manager.get_weight(f"{prefix}.v_proj.weight"))
        self.o_proj = CustomLinear(weight_manager.get_weight(f"{prefix}.o_proj.weight"))

        # RMSNorm for q和k
        self.q_norm = CustomRMSNorm(weight_manager.get_weight(f"{prefix}.q_norm.weight"))
        self.k_norm = CustomRMSNorm(weight_manager.get_weight(f"{prefix}.k_norm.weight"))

        # RoPE
        self.rotary_emb = QwenRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings
        )

        # KV Cache
        self.kv_cache = KVCache()

    def forward(self, hidden_states, attention_mask=None, position_ids=None, use_cache=False, cache_position=None):
        batch_size, seq_len, _ = hidden_states.shape
        # 投影
        q = self.q_proj.forward(hidden_states)
        k = self.k_proj.forward(hidden_states)
        v = self.v_proj.forward(hidden_states)

        # 重塑
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        # 应用RMSNorm
        q = self.q_norm.forward(q)
        k = self.k_norm.forward(k)

        # 转置为 (batch, heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 应用RoPE - 需要根据实际的序列位置计算
        if position_ids is None:
            if use_cache and self.kv_cache.cache_k is not None:
                # 增量推理：当前位置 = 已缓存的长度
                current_pos = self.kv_cache.cache_k.shape[-2]
                position_ids = torch.arange(current_pos, current_pos + seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
            else:
                # 首次推理：从0开始
                position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)

        cos, sin = self.rotary_emb.forward(q, seq_len=position_ids.max().item() + 1)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # 使用KV Cache
        if use_cache:
            k, v = self.kv_cache.update(k, v)
        #     print("k shape : ", k.shape)
        #     print("v shape : ", v.shape)
        # else:
        #     print("k shape : ", k.shape)
        #     print("v shape : ", v.shape)
        # GQA: 扩展k和v
        if self.num_key_value_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)

        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 处理attention mask
        if use_cache and self.kv_cache.cache_k is not None and seq_len == 1:
            # 增量推理时，query只有1个token，key有多个token，不需要掩码
            pass
        elif attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # 转回原始格式
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)
        
        return self.o_proj.forward(attn_output)

class CustomQwen3MLP:
    def __init__(self, layer_idx, weight_manager):
        prefix = f"model.layers.{layer_idx}.mlp"
        self.gate_proj = CustomLinear(weight_manager.get_weight(f"{prefix}.gate_proj.weight"))
        self.up_proj = CustomLinear(weight_manager.get_weight(f"{prefix}.up_proj.weight"))
        self.down_proj = CustomLinear(weight_manager.get_weight(f"{prefix}.down_proj.weight"))

    def forward(self, x):
        gate = torch.nn.functional.silu(self.gate_proj.forward(x))
        up = self.up_proj.forward(x)
        
        return self.down_proj.forward(gate * up)

class Qwen3DecoderLayerWithKVCache:
    def __init__(self, layer_idx, weight_manager, config):
        self.self_attn = Qwen3AttentionWithKVCache(layer_idx, weight_manager, config)
        self.mlp = CustomQwen3MLP(layer_idx, weight_manager)

        prefix = f"model.layers.{layer_idx}"
        self.input_layernorm = CustomRMSNorm(
            weight_manager.get_weight(f"{prefix}.input_layernorm.weight"),
            eps=config['rms_norm_eps']
        )
        self.post_attention_layernorm = CustomRMSNorm(
            weight_manager.get_weight(f"{prefix}.post_attention_layernorm.weight"),
            eps=config['rms_norm_eps']
        )

    def forward(self, hidden_states, attention_mask=None, position_ids=None, use_cache=False, cache_position=None):
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm.forward(hidden_states)
        hidden_states = self.self_attn.forward(
            hidden_states, attention_mask, position_ids, use_cache, cache_position
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm.forward(hidden_states)
        hidden_states = self.mlp.forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class Qwen3ModelWithKVCache:
    """带KV Cache的Qwen3模型实现"""
    def __init__(self, weight_manager, config):
        print("构建带KV Cache的Qwen3模型...")
        self.config = config

        # 嵌入层
        self.embed_tokens = CustomEmbedding(
            weight_manager.get_weight("model.embed_tokens.weight")
        )

        # Transformer层
        self.layers = []
        for i in range(config['num_hidden_layers']):
            self.layers.append(Qwen3DecoderLayerWithKVCache(i, weight_manager, config))

        # 最终norm
        self.norm = CustomRMSNorm(
            weight_manager.get_weight("model.norm.weight"),
            eps=config['rms_norm_eps']
        )

        # 输出层
        self.lm_head = CustomLinear(weight_manager.get_weight("lm_head.weight"))

        print(f"✅ 带KV Cache的Qwen3模型构建完成: {len(self.layers)}层")

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False, cache_position=None):
        batch_size, seq_len = input_ids.shape

        # 嵌入
        hidden_states = self.embed_tokens.forward(input_ids)

        # 因果掩码
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf')),
                diagonal=1
            )
            attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # 通过所有层
        for layer in self.layers:
            hidden_states = layer.forward(
                hidden_states, attention_mask, position_ids, use_cache, cache_position
            )

        # 最终norm
        hidden_states = self.norm.forward(hidden_states)

        # 输出
        logits = self.lm_head.forward(hidden_states)
        return logits

    def reset_kv_cache(self):
        """重置所有层的KV缓存"""
        for layer in self.layers:
            layer.self_attn.kv_cache.reset()

    def generate_with_kvcache(self, input_ids, tokenizer, max_new_tokens=50):
        """使用KV Cache的生成函数"""
        self.reset_kv_cache()

        generated = input_ids.clone()

        # 首次前向传播（处理完整的prompt）
        with torch.no_grad():
            logits = self.forward(input_ids, use_cache=True)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            print(tokenizer.decode(int(next_token), skip_special_tokens=True).strip())

        # 增量生成
        for i in range(max_new_tokens - 1):
            with torch.no_grad():
                # 只传入新的token
                logits = self.forward(next_token, use_cache=True)
                next_token_logits = logits[:, -1, :]

                if torch.isnan(next_token_logits).any():
                    print(f"在步骤{i+1}检测到NaN")
                    break

                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

                print(tokenizer.decode(int(next_token), skip_special_tokens=True).strip(), end="", flush=True)

                if next_token.item() in [151645, 151643]:
                    print()
                    break

        return generated

    def generate(self, input_ids, tokenizer, max_new_tokens=50):
        """标准生成函数（不使用KV Cache）"""
        generated = input_ids.clone()

        for i in range(max_new_tokens):
            logits = self.forward(generated)
            next_token_logits = logits[:, -1, :]

            if torch.isnan(next_token_logits).any():
                print(f"在步骤{i}检测到NaN")
                break

            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            print(tokenizer.decode(int(next_token), skip_special_tokens=True).strip(), end="", flush=True)

            if next_token.item() in [151645, 151643]:
                break

        return generated

def test_comparison():
    """对比测试标准版本和KV Cache版本"""
    print("🔧 Qwen3标准版本 vs KV Cache版本对比测试")
    print("=" * 60)

    model_path = "models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

    # 加载配置
    with open(f"{model_path}/config.json", 'r') as f:
        config = json.load(f)

    print(f"模型配置:")
    print(f"- 注意力头数: {config['num_attention_heads']}")
    print(f"- KV头数: {config['num_key_value_heads']}")
    print(f"- 头维度: {config['head_dim']}")
    print(f"- 隐藏维度: {config['hidden_size']}")

    # 创建模型
    weight_manager = WeightManager(f"{model_path}/model.safetensors")
    model = Qwen3ModelWithKVCache(weight_manager, config)

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 准备输入
    prompt = "介绍一下Python编程语言"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    input_ids = tokenizer.encode(text, return_tensors="pt")

    print(f"\n输入prompt: {prompt}")
    print(f"输入长度: {input_ids.shape[1]} tokens")

    # 测试标准版本
    print("\n" + "="*30)
    print("🔄 标准版本生成 (不使用KV Cache):")
    print("="*30)

    import time
    start_time = time.time()
    with torch.no_grad():
        standard_outputs = model.generate(input_ids, tokenizer, max_new_tokens=30)
    standard_time = time.time() - start_time

    # 测试KV Cache版本
    print("\n" + "="*30)
    print("🚀 KV Cache版本生成:")
    print("="*30)

    start_time = time.time()
    with torch.no_grad():
        kv_cache_outputs = model.generate_with_kvcache(input_ids, tokenizer, max_new_tokens=30)
    kv_cache_time = time.time() - start_time

    # 比较结果
    print(f"\n📊 性能对比:")
    print(f"标准版本耗时: {standard_time:.2f}秒")
    print(f"KV Cache耗时: {kv_cache_time:.2f}秒")
    print(f"加速比: {standard_time/kv_cache_time:.2f}x")

    # 检查输出是否一致
    standard_tokens = standard_outputs[0][len(input_ids[0]):].tolist()[:30]
    kv_cache_tokens = kv_cache_outputs[0][len(input_ids[0]):].tolist()[:30]

    print(f"\n🔍 输出一致性检查:")
    print(f"生成token数量 - 标准版本: {len(standard_tokens)}, KV Cache: {len(kv_cache_tokens)}")

    min_len = min(len(standard_tokens), len(kv_cache_tokens))
    matches = sum(1 for i in range(min_len) if standard_tokens[i] == kv_cache_tokens[i])

    print(f"前{min_len}个token匹配: {matches}/{min_len} ({matches/min_len*100:.1f}%)")

    if matches == min_len:
        print("✅ 输出完全一致!")
    else:
        print("⚠️ 输出存在差异")
        print(f"标准版本: {tokenizer.decode(standard_tokens[:10])}")
        print(f"KV Cache: {tokenizer.decode(kv_cache_tokens[:10])}")

def main():
    test_comparison()

if __name__ == "__main__":
    main()