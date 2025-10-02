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
    def __init__(self, dim, max_position_embeddings=131072, base=10000):
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
            self.cache_k = k
            self.cache_v = v
        else:
            self.cache_k = torch.cat([self.cache_k, k], dim=-2)
            self.cache_v = torch.cat([self.cache_v, v], dim=-2)

        return self.cache_k, self.cache_v

    def reset(self):
        """重置缓存"""
        self.cache_k = None
        self.cache_v = None

class DeepSeekAttentionWithKVCache:
    """DeepSeek-R1-Distill-Qwen注意力实现 (GQA: 12 heads, 2 KV heads)"""
    def __init__(self, layer_idx, weight_manager, config):
        self.layer_idx = layer_idx
        self.hidden_size = config['hidden_size']
        self.num_heads = config['num_attention_heads']
        self.num_key_value_heads = config['num_key_value_heads']
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config['max_position_embeddings']

        assert self.num_heads % self.num_key_value_heads == 0

        # 投影层
        prefix = f"model.layers.{layer_idx}.self_attn"
        q_bias = weight_manager.get_weight(f"{prefix}.q_proj.bias")
        k_bias = weight_manager.get_weight(f"{prefix}.k_proj.bias")
        v_bias = weight_manager.get_weight(f"{prefix}.v_proj.bias")

        self.q_proj = CustomLinear(weight_manager.get_weight(f"{prefix}.q_proj.weight"), q_bias)
        self.k_proj = CustomLinear(weight_manager.get_weight(f"{prefix}.k_proj.weight"), k_bias)
        self.v_proj = CustomLinear(weight_manager.get_weight(f"{prefix}.v_proj.weight"), v_bias)
        self.o_proj = CustomLinear(weight_manager.get_weight(f"{prefix}.o_proj.weight"))

        # 检查是否有q_norm和k_norm (某些版本没有)
        q_norm_weight = weight_manager.get_weight(f"{prefix}.q_norm.weight")
        k_norm_weight = weight_manager.get_weight(f"{prefix}.k_norm.weight")

        self.has_qk_norm = q_norm_weight is not None and k_norm_weight is not None
        if self.has_qk_norm:
            self.q_norm = CustomRMSNorm(q_norm_weight)
            self.k_norm = CustomRMSNorm(k_norm_weight)

        # RoPE
        self.rotary_emb = QwenRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=config.get('rope_theta', 10000)
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

        # 应用RMSNorm (如果有)
        if self.has_qk_norm:
            q = self.q_norm.forward(q)
            k = self.k_norm.forward(k)

        # 转置为 (batch, heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 应用RoPE
        if position_ids is None:
            if use_cache and self.kv_cache.cache_k is not None:
                current_pos = self.kv_cache.cache_k.shape[-2]
                position_ids = torch.arange(current_pos, current_pos + seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
            else:
                position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)

        cos, sin = self.rotary_emb.forward(q, seq_len=position_ids.max().item() + 1)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # 使用KV Cache
        if use_cache:
            k, v = self.kv_cache.update(k, v)

        # GQA: 扩展k和v
        if self.num_key_value_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)

        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 处理attention mask
        if use_cache and self.kv_cache.cache_k is not None and seq_len == 1:
            pass
        elif attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # 转回原始格式
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)

        return self.o_proj.forward(attn_output)

class CustomDeepSeekMLP:
    def __init__(self, layer_idx, weight_manager):
        prefix = f"model.layers.{layer_idx}.mlp"
        self.gate_proj = CustomLinear(weight_manager.get_weight(f"{prefix}.gate_proj.weight"))
        self.up_proj = CustomLinear(weight_manager.get_weight(f"{prefix}.up_proj.weight"))
        self.down_proj = CustomLinear(weight_manager.get_weight(f"{prefix}.down_proj.weight"))

    def forward(self, x):
        gate = torch.nn.functional.silu(self.gate_proj.forward(x))
        up = self.up_proj.forward(x)
        return self.down_proj.forward(gate * up)

class DeepSeekDecoderLayerWithKVCache:
    def __init__(self, layer_idx, weight_manager, config):
        self.self_attn = DeepSeekAttentionWithKVCache(layer_idx, weight_manager, config)
        self.mlp = CustomDeepSeekMLP(layer_idx, weight_manager)

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

class DeepSeekR1DistillQwenModel:
    """DeepSeek-R1-Distill-Qwen-1.5B模型实现"""
    def __init__(self, weight_manager, config):
        print("构建 DeepSeek-R1-Distill-Qwen-1.5B 模型...")
        self.config = config

        # 嵌入层
        self.embed_tokens = CustomEmbedding(
            weight_manager.get_weight("model.embed_tokens.weight")
        )

        # Transformer层
        self.layers = []
        for i in range(config['num_hidden_layers']):
            self.layers.append(DeepSeekDecoderLayerWithKVCache(i, weight_manager, config))

        # 最终norm
        self.norm = CustomRMSNorm(
            weight_manager.get_weight("model.norm.weight"),
            eps=config['rms_norm_eps']
        )

        # 输出层
        self.lm_head = CustomLinear(weight_manager.get_weight("lm_head.weight"))

        print(f"✅ DeepSeek-R1-Distill-Qwen-1.5B 模型构建完成: {len(self.layers)}层")

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

    def generate_with_kvcache(self, input_ids, tokenizer, max_new_tokens=100):
        """使用KV Cache的生成函数"""
        self.reset_kv_cache()

        generated = input_ids.clone()

        # 首次前向传播（处理完整的prompt）
        with torch.no_grad():
            logits = self.forward(input_ids, use_cache=True)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            token_text = tokenizer.decode(int(next_token), skip_special_tokens=True)
            if token_text.strip():
                print(token_text, end='', flush=True)

        # 增量生成
        for i in range(max_new_tokens - 1):
            with torch.no_grad():
                # 只传入新的token
                logits = self.forward(next_token, use_cache=True)
                next_token_logits = logits[:, -1, :]

                if torch.isnan(next_token_logits).any():
                    print(f"\n在步骤{i+1}检测到NaN")
                    break

                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

                token_text = tokenizer.decode(int(next_token), skip_special_tokens=True)
                if token_text.strip():
                    print(token_text, end='', flush=True)

                if next_token.item() in [151645, 151643]:
                    break

        print()  # 换行
        return generated

def main():
    """测试 DeepSeek-R1-Distill-Qwen-1.5B 模型"""
    print("🔧 DeepSeek-R1-Distill-Qwen-1.5B 推理测试")
    print("=" * 60)

    model_path = "DeepSeek-R1-Distill-Qwen-1___5B"

    # 加载配置
    with open(f"{model_path}/config.json", 'r') as f:
        config = json.load(f)

    print(f"模型配置:")
    print(f"- 模型类型: {config['model_type']}")
    print(f"- 注意力头数: {config['num_attention_heads']}")
    print(f"- KV头数: {config['num_key_value_heads']}")
    print(f"- 隐藏维度: {config['hidden_size']}")
    print(f"- 中间维度: {config['intermediate_size']}")
    print(f"- 层数: {config['num_hidden_layers']}")
    print(f"- 词汇表大小: {config['vocab_size']}")
    print(f"- 最大位置: {config['max_position_embeddings']}")

    # 创建模型
    weight_manager = WeightManager(f"{model_path}/model.safetensors")
    model = DeepSeekR1DistillQwenModel(weight_manager, config)

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 测试用例
    test_prompts = [
        "什么是人工智能?",
        "用Python写一个快速排序算法",
        "解释一下什么是深度学习"
    ]

    for prompt in test_prompts:
        print("\n" + "="*60)
        print(f"输入: {prompt}")
        print("="*60)

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        input_ids = tokenizer.encode(text, return_tensors="pt")

        print(f"输入token数: {input_ids.shape[1]}")
        print(f"输出: ", end='')

        import time
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate_with_kvcache(input_ids, tokenizer, max_new_tokens=100)
        elapsed_time = time.time() - start_time

        print(f"\n生成耗时: {elapsed_time:.2f}秒")
        print(f"生成token数: {outputs.shape[1] - input_ids.shape[1]}")

if __name__ == "__main__":
    main()
