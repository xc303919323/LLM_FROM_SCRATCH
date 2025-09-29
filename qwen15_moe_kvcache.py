import torch
import json
import math
from safetensors import safe_open
from transformers import AutoTokenizer
import os
from glob import glob

# 重用基础组件
def custom_gemm(a, b):
    """自定义GEMM实现"""
    original_shape = a.shape
    a_flat = a.view(-1, a.size(-1))
    result = torch.mm(a_flat, b)
    return result.view(*original_shape[:-1], b.size(-1))

class WeightManager:
    def __init__(self, model_path):
        self.weights = {}
        safetensors_files = glob(os.path.join(model_path, "model-*.safetensors"))
        print(f"加载权重文件: {len(safetensors_files)}个")

        for file_path in sorted(safetensors_files):
            print(f"加载 {os.path.basename(file_path)}")
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    self.weights[key] = f.get_tensor(key).clone().float()

        print(f"总共加载了 {len(self.weights)} 个权重")

    def get_weight(self, key):
        if key not in self.weights:
            return None
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
    def __init__(self, dim, max_position_embeddings=8192, base=1000000.0):
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

# KV Cache数据结构
class KVCache:
    """KV Cache管理类"""
    def __init__(self, max_batch_size=1, max_seq_len=8192, num_heads=16, head_dim=128):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim

        # 为每一层初始化KV cache
        self.k_cache = torch.zeros(max_batch_size, num_heads, max_seq_len, head_dim, dtype=torch.float32)
        self.v_cache = torch.zeros(max_batch_size, num_heads, max_seq_len, head_dim, dtype=torch.float32)
        self.seq_len = 0  # 当前缓存的序列长度

    def update(self, k, v, start_pos):
        """更新KV cache"""
        batch_size, num_heads, seq_len, head_dim = k.shape

        # 更新缓存
        self.k_cache[:batch_size, :num_heads, start_pos:start_pos + seq_len] = k
        self.v_cache[:batch_size, :num_heads, start_pos:start_pos + seq_len] = v

        # 更新序列长度
        self.seq_len = start_pos + seq_len

        # 返回完整的key和value
        return (
            self.k_cache[:batch_size, :num_heads, :self.seq_len],
            self.v_cache[:batch_size, :num_heads, :self.seq_len]
        )

    def get_cache(self, batch_size):
        """获取当前缓存"""
        return (
            self.k_cache[:batch_size, :, :self.seq_len],
            self.v_cache[:batch_size, :, :self.seq_len]
        )

# 支持KV Cache的注意力层
class Qwen2AttentionWithKVCache:
    """Qwen2注意力层（支持KV Cache）"""
    def __init__(self, layer_idx, weight_manager, config):
        self.layer_idx = layer_idx
        self.hidden_size = config['hidden_size']
        self.num_heads = config['num_attention_heads']
        self.num_key_value_heads = config['num_key_value_heads']
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config['max_position_embeddings']

        # 投影层（包含bias）
        prefix = f"model.layers.{layer_idx}.self_attn"
        self.q_proj = CustomLinear(
            weight_manager.get_weight(f"{prefix}.q_proj.weight"),
            weight_manager.get_weight(f"{prefix}.q_proj.bias")
        )
        self.k_proj = CustomLinear(
            weight_manager.get_weight(f"{prefix}.k_proj.weight"),
            weight_manager.get_weight(f"{prefix}.k_proj.bias")
        )
        self.v_proj = CustomLinear(
            weight_manager.get_weight(f"{prefix}.v_proj.weight"),
            weight_manager.get_weight(f"{prefix}.v_proj.bias")
        )
        self.o_proj = CustomLinear(weight_manager.get_weight(f"{prefix}.o_proj.weight"))

        # RoPE
        self.rotary_emb = QwenRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=config.get('rope_theta', 1000000.0)
        )

        # KV Cache
        self.kv_cache = KVCache(
            max_batch_size=8,
            max_seq_len=self.max_position_embeddings,
            num_heads=self.num_key_value_heads,
            head_dim=self.head_dim
        )

    def forward(self, hidden_states, attention_mask=None, position_ids=None, use_cache=False, start_pos=0):
        batch_size, seq_len, _ = hidden_states.shape

        # 投影
        q = self.q_proj.forward(hidden_states)
        k = self.k_proj.forward(hidden_states)
        v = self.v_proj.forward(hidden_states)

        # 重塑
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        # 转置为 (batch, heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 应用RoPE
        if position_ids is None:
            position_ids = torch.arange(start_pos, start_pos + seq_len, dtype=torch.long, device=q.device)
            position_ids = position_ids.unsqueeze(0)

        cos, sin = self.rotary_emb.forward(q, seq_len=start_pos + seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # 如果使用cache，更新并获取完整的k,v
        if use_cache:
            k, v = self.kv_cache.update(k, v, start_pos)

        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # 转回原始格式
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        return self.o_proj.forward(attn_output)

    def reset_cache(self):
        """重置KV cache"""
        self.kv_cache.seq_len = 0
        self.kv_cache.k_cache.zero_()
        self.kv_cache.v_cache.zero_()

class CorrectMoELayer:
    """正确的MoE层实现 - 实现真正的路由逻辑"""
    def __init__(self, layer_idx, weight_manager, config):
        self.num_experts = config['num_experts']
        self.num_experts_per_tok = config['num_experts_per_tok']
        self.norm_topk_prob = config.get('norm_topk_prob', False)

        # 创建所有专家
        self.experts = []
        for i in range(self.num_experts):
            expert_prefix = f"model.layers.{layer_idx}.mlp.experts.{i}"
            expert = {
                'gate_proj': CustomLinear(weight_manager.get_weight(f"{expert_prefix}.gate_proj.weight")),
                'up_proj': CustomLinear(weight_manager.get_weight(f"{expert_prefix}.up_proj.weight")),
                'down_proj': CustomLinear(weight_manager.get_weight(f"{expert_prefix}.down_proj.weight"))
            }
            self.experts.append(expert)

        # 共享专家
        shared_prefix = f"model.layers.{layer_idx}.mlp.shared_expert"
        self.shared_expert = {
            'gate_proj': CustomLinear(weight_manager.get_weight(f"{shared_prefix}.gate_proj.weight")),
            'up_proj': CustomLinear(weight_manager.get_weight(f"{shared_prefix}.up_proj.weight")),
            'down_proj': CustomLinear(weight_manager.get_weight(f"{shared_prefix}.down_proj.weight"))
        }

        # 门控网络（路由器）
        gate_weight = weight_manager.get_weight(f"model.layers.{layer_idx}.mlp.gate.weight")
        self.gate = CustomLinear(gate_weight)

        # 共享专家门控
        shared_gate_weight = weight_manager.get_weight(f"model.layers.{layer_idx}.mlp.shared_expert_gate.weight")
        self.shared_expert_gate = CustomLinear(shared_gate_weight)

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # 路由逻辑
        router_logits = self.gate.forward(hidden_states_flat)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1)

        # 选择top-k专家
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
        )

        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        # 初始化最终输出
        final_hidden_states = torch.zeros_like(hidden_states_flat)

        # 对每个选中的专家计算输出
        for i in range(self.num_experts_per_tok):
            expert_idx = selected_experts[:, i]
            expert_weights = routing_weights[:, i].unsqueeze(-1)

            for token_idx in range(hidden_states_flat.shape[0]):
                exp_id = expert_idx[token_idx].item()
                if exp_id < len(self.experts):
                    expert = self.experts[exp_id]
                    token_input = hidden_states_flat[token_idx:token_idx+1]

                    # 专家计算
                    gate_out = torch.nn.functional.silu(expert['gate_proj'].forward(token_input))
                    up_out = expert['up_proj'].forward(token_input)
                    expert_output = expert['down_proj'].forward(gate_out * up_out)

                    # 加权累加
                    final_hidden_states[token_idx] += expert_weights[token_idx] * expert_output.squeeze(0)

        # 共享专家
        shared_gate_logits = self.shared_expert_gate.forward(hidden_states_flat)
        shared_weight = torch.nn.functional.sigmoid(shared_gate_logits)

        shared_gate_out = torch.nn.functional.silu(self.shared_expert['gate_proj'].forward(hidden_states_flat))
        shared_up_out = self.shared_expert['up_proj'].forward(hidden_states_flat)
        shared_output = self.shared_expert['down_proj'].forward(shared_gate_out * shared_up_out)

        # 结合共享专家输出
        final_hidden_states += shared_weight * shared_output

        return final_hidden_states.view(batch_size, seq_len, hidden_dim)

class CorrectQwen2MoEDecoderLayerWithKVCache:
    def __init__(self, layer_idx, weight_manager, config):
        self.self_attn = Qwen2AttentionWithKVCache(layer_idx, weight_manager, config)
        self.mlp = CorrectMoELayer(layer_idx, weight_manager, config)

        prefix = f"model.layers.{layer_idx}"
        self.input_layernorm = CustomRMSNorm(
            weight_manager.get_weight(f"{prefix}.input_layernorm.weight"),
            eps=config['rms_norm_eps']
        )
        self.post_attention_layernorm = CustomRMSNorm(
            weight_manager.get_weight(f"{prefix}.post_attention_layernorm.weight"),
            eps=config['rms_norm_eps']
        )

    def forward(self, hidden_states, attention_mask=None, position_ids=None, use_cache=False, start_pos=0):
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm.forward(hidden_states)
        hidden_states = self.self_attn.forward(
            hidden_states, attention_mask, position_ids, use_cache=use_cache, start_pos=start_pos
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm.forward(hidden_states)
        hidden_states = self.mlp.forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def reset_cache(self):
        """重置KV cache"""
        self.self_attn.reset_cache()

class CorrectQwen2MoEModelWithKVCache:
    """支持KV Cache的Qwen1.5-MoE模型"""
    def __init__(self, weight_manager, config):
        print("构建支持KV Cache的Qwen1.5-MoE模型...")
        self.config = config

        # 嵌入层
        self.embed_tokens = CustomEmbedding(
            weight_manager.get_weight("model.embed_tokens.weight")
        )

        # Transformer层
        self.layers = []
        for i in range(config['num_hidden_layers']):
            self.layers.append(CorrectQwen2MoEDecoderLayerWithKVCache(i, weight_manager, config))

        # 最终norm
        self.norm = CustomRMSNorm(
            weight_manager.get_weight("model.norm.weight"),
            eps=config['rms_norm_eps']
        )

        # 输出层
        self.lm_head = CustomLinear(weight_manager.get_weight("lm_head.weight"))

        print(f"✅ 支持KV Cache的Qwen1.5-MoE模型构建完成: {len(self.layers)}层")

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False, start_pos=0):
        batch_size, seq_len = input_ids.shape

        # 嵌入
        hidden_states = self.embed_tokens.forward(input_ids)

        # 位置ids
        if position_ids is None:
            position_ids = torch.arange(start_pos, start_pos + seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # 因果掩码
        if attention_mask is None:
            if use_cache and start_pos > 0:
                # 增量生成时：新token可以看到所有之前的token
                total_len = start_pos + seq_len
                causal_mask = torch.zeros((seq_len, total_len), dtype=torch.float32)
                attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            else:
                # 正常情况：标准因果掩码
                causal_mask = torch.triu(
                    torch.full((seq_len, seq_len), float('-inf')),
                    diagonal=1
                )
                attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # 通过所有层
        for layer in self.layers:
            hidden_states = layer.forward(
                hidden_states, attention_mask, position_ids, use_cache=use_cache, start_pos=start_pos
            )

        # 最终norm
        hidden_states = self.norm.forward(hidden_states)

        # 输出
        logits = self.lm_head.forward(hidden_states)
        return logits

    def generate_with_kvcache(self, input_ids, tokenizer, max_new_tokens=50):
        """使用KV Cache的生成方法"""
        print("使用KV Cache进行生成...")

        # 重置所有层的cache
        for layer in self.layers:
            layer.reset_cache()

        generated = input_ids.clone()

        # 第一步：处理完整的prompt
        with torch.no_grad():
            logits = self.forward(input_ids, use_cache=True, start_pos=0)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            decoded = tokenizer.decode(int(next_token), skip_special_tokens=True).strip()
            print(f"{decoded}", end="")

        # 后续步骤：使用cache进行增量生成
        for i in range(1, max_new_tokens):
            with torch.no_grad():
                # 只处理新token
                logits = self.forward(next_token, use_cache=True, start_pos=generated.shape[1] - 1)
                next_token_logits = logits[:, -1, :]

                if torch.isnan(next_token_logits).any():
                    print(f"在步骤{i}检测到NaN")
                    break

                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

                decoded = tokenizer.decode(int(next_token), skip_special_tokens=True).strip()
                print(f"{decoded}", end="")

                if next_token.item() in [151645, 151643]:
                    break

        return generated

    def generate(self, input_ids, tokenizer, max_new_tokens=50):
        """标准生成方法（不使用cache，用于对比）"""
        print("不使用KV Cache进行生成...")
        generated = input_ids.clone()

        for i in range(max_new_tokens):
            logits = self.forward(generated)
            next_token_logits = logits[:, -1, :]

            if torch.isnan(next_token_logits).any():
                print(f"在步骤{i}检测到NaN")
                break

            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            decoded = tokenizer.decode(int(next_token), skip_special_tokens=True).strip()
            print(f"{decoded}", end="")

            if next_token.item() in [151645, 151643]:
                break

        return generated

def main():
    print("🔧 带KV Cache的Qwen1.5-MoE推理系统")
    print("=" * 50)

    model_path = "models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9"

    # 加载配置
    with open(f"{model_path}/config.json", 'r') as f:
        config = json.load(f)

    print(f"模型配置:")
    print(f"- 专家数量: {config['num_experts']}")
    print(f"- 每token专家数: {config['num_experts_per_tok']}")

    # 创建模型
    weight_manager = WeightManager(model_path)
    model = CorrectQwen2MoEModelWithKVCache(weight_manager, config)

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 准备输入
    prompt = "你好，请介绍一下乔布斯"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    print(f"\n开始推理，输入: {prompt}")

    # KV Cache生成
    print(f"\n=== 使用KV Cache生成 ===")
    with torch.no_grad():
        outputs_cache = model.generate_with_kvcache(input_ids, tokenizer, max_new_tokens=500)

    # print(f"\n\n=== 不使用KV Cache生成（对比） ===")
    # with torch.no_grad():
    #     outputs_normal = model.generate(input_ids, tokenizer, max_new_tokens=30)

    # print(f"\n\n✅ 生成完成")
    # print(f"KV Cache结果长度: {outputs_cache.shape[1]}")
    # print(f"普通生成结果长度: {outputs_normal.shape[1]}")

if __name__ == "__main__":
    main()