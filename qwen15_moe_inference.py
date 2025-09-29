import torch
import json
import math
from safetensors import safe_open
from transformers import AutoTokenizer
import os
from glob import glob

def custom_gemm(a, b):
    """自定义GEMM实现"""
    original_shape = a.shape
    a_flat = a.view(-1, a.size(-1))
    result = torch.mm(a_flat, b)
    return result.view(*original_shape[:-1], b.size(-1))

class WeightManager:
    def __init__(self, model_path):
        self.weights = {}

        # 找到所有safetensors文件
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
            print(f"警告: 权重 {key} 不存在")
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

class Qwen2Attention:
    """Qwen2注意力层（与Qwen3类似但使用不同的rope_theta）"""
    def __init__(self, layer_idx, weight_manager, config):
        self.layer_idx = layer_idx
        self.hidden_size = config['hidden_size']
        self.num_heads = config['num_attention_heads']
        self.num_key_value_heads = config['num_key_value_heads']
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config['max_position_embeddings']

        # 投影层
        prefix = f"model.layers.{layer_idx}.self_attn"
        self.q_proj = CustomLinear(weight_manager.get_weight(f"{prefix}.q_proj.weight"))
        self.k_proj = CustomLinear(weight_manager.get_weight(f"{prefix}.k_proj.weight"))
        self.v_proj = CustomLinear(weight_manager.get_weight(f"{prefix}.v_proj.weight"))
        self.o_proj = CustomLinear(weight_manager.get_weight(f"{prefix}.o_proj.weight"))

        # RoPE (注意Qwen1.5使用不同的base)
        self.rotary_emb = QwenRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=config.get('rope_theta', 1000000.0)
        )

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
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
        cos, sin = self.rotary_emb.forward(q, seq_len=seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

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

class MoEExpert:
    """单个MoE专家"""
    def __init__(self, expert_idx, layer_idx, weight_manager, intermediate_size):
        prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
        self.gate_proj = CustomLinear(weight_manager.get_weight(f"{prefix}.gate_proj.weight"))
        self.up_proj = CustomLinear(weight_manager.get_weight(f"{prefix}.up_proj.weight"))
        self.down_proj = CustomLinear(weight_manager.get_weight(f"{prefix}.down_proj.weight"))

    def forward(self, x):
        gate = torch.nn.functional.silu(self.gate_proj.forward(x))
        up = self.up_proj.forward(x)
        return self.down_proj.forward(gate * up)

class SharedExpert:
    """共享专家"""
    def __init__(self, layer_idx, weight_manager):
        prefix = f"model.layers.{layer_idx}.mlp.shared_expert"
        self.gate_proj = CustomLinear(weight_manager.get_weight(f"{prefix}.gate_proj.weight"))
        self.up_proj = CustomLinear(weight_manager.get_weight(f"{prefix}.up_proj.weight"))
        self.down_proj = CustomLinear(weight_manager.get_weight(f"{prefix}.down_proj.weight"))

    def forward(self, x):
        gate = torch.nn.functional.silu(self.gate_proj.forward(x))
        up = self.up_proj.forward(x)
        return self.down_proj.forward(gate * up)

class MoELayer:
    """MoE层实现"""
    def __init__(self, layer_idx, weight_manager, config):
        self.num_experts = config['num_experts']
        self.num_experts_per_tok = config['num_experts_per_tok']
        self.moe_intermediate_size = config['moe_intermediate_size']

        # 专家网络
        self.experts = []
        for i in range(self.num_experts):
            expert = MoEExpert(i, layer_idx, weight_manager, self.moe_intermediate_size)
            self.experts.append(expert)

        # 共享专家
        self.shared_expert = SharedExpert(layer_idx, weight_manager)

        # 路由器 (gate)
        gate_weight = weight_manager.get_weight(f"model.layers.{layer_idx}.mlp.gate.weight")
        if gate_weight is not None:
            # gate权重形状是 [num_experts, hidden_size]，但CustomLinear期望 [out_features, in_features]
            self.gate = CustomLinear(gate_weight)
        else:
            print(f"警告: 第{layer_idx}层没有找到gate权重")
            self.gate = None

        # 共享专家门控
        shared_gate_weight = weight_manager.get_weight(f"model.layers.{layer_idx}.mlp.shared_expert_gate.weight")
        if shared_gate_weight is not None:
            # shared_expert_gate权重形状是 [1, hidden_size]
            self.shared_expert_gate = CustomLinear(shared_gate_weight)
        else:
            print(f"警告: 第{layer_idx}层没有找到shared_expert_gate权重")
            self.shared_expert_gate = None

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 重塑为 (batch_size * seq_len, hidden_dim)
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # 路由逻辑
        if self.gate is not None:
            router_logits = self.gate.forward(hidden_states_flat)  # (batch_size * seq_len, num_experts)

            # 选择top-k专家
            top_k_logits, top_k_indices = torch.topk(router_logits, self.num_experts_per_tok, dim=-1)
            top_k_probs = torch.softmax(top_k_logits, dim=-1)

            # 初始化输出
            final_hidden_states = torch.zeros_like(hidden_states_flat)

            # 更高效的专家处理方式
            for token_idx in range(hidden_states_flat.shape[0]):
                token_input = hidden_states_flat[token_idx:token_idx+1]
                token_output = torch.zeros_like(token_input)

                for k in range(self.num_experts_per_tok):
                    expert_idx = top_k_indices[token_idx, k].item()
                    expert_weight = top_k_probs[token_idx, k].item()

                    expert_output = self.experts[expert_idx].forward(token_input)
                    token_output += expert_weight * expert_output

                final_hidden_states[token_idx:token_idx+1] = token_output
        else:
            # 如果没有gate，使用第一个专家
            final_hidden_states = self.experts[0].forward(hidden_states_flat)

        # 共享专家
        if self.shared_expert_gate is not None:
            shared_gate_output = self.shared_expert_gate.forward(hidden_states_flat)
            shared_gate_probs = torch.sigmoid(shared_gate_output)
            shared_output = self.shared_expert.forward(hidden_states_flat)
            final_hidden_states += shared_gate_probs * shared_output
        else:
            # 如果没有门控，直接加上共享专家输出
            shared_output = self.shared_expert.forward(hidden_states_flat)
            final_hidden_states += shared_output

        # 重塑回原始形状
        return final_hidden_states.view(batch_size, seq_len, hidden_dim)

class Qwen2MoEDecoderLayer:
    def __init__(self, layer_idx, weight_manager, config):
        self.self_attn = Qwen2Attention(layer_idx, weight_manager, config)
        self.mlp = MoELayer(layer_idx, weight_manager, config)

        prefix = f"model.layers.{layer_idx}"
        self.input_layernorm = CustomRMSNorm(
            weight_manager.get_weight(f"{prefix}.input_layernorm.weight"),
            eps=config['rms_norm_eps']
        )
        self.post_attention_layernorm = CustomRMSNorm(
            weight_manager.get_weight(f"{prefix}.post_attention_layernorm.weight"),
            eps=config['rms_norm_eps']
        )

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm.forward(hidden_states)
        hidden_states = self.self_attn.forward(hidden_states, attention_mask, position_ids)
        hidden_states = residual + hidden_states

        # MLP (MoE)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm.forward(hidden_states)
        hidden_states = self.mlp.forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class Qwen2MoEModel:
    """Qwen1.5-MoE模型实现"""
    def __init__(self, weight_manager, config):
        print("构建Qwen1.5-MoE模型...")
        self.config = config

        # 嵌入层
        self.embed_tokens = CustomEmbedding(
            weight_manager.get_weight("model.embed_tokens.weight")
        )

        # Transformer层
        self.layers = []
        for i in range(config['num_hidden_layers']):
            print(f"构建第{i}层...")
            self.layers.append(Qwen2MoEDecoderLayer(i, weight_manager, config))

        # 最终norm
        self.norm = CustomRMSNorm(
            weight_manager.get_weight("model.norm.weight"),
            eps=config['rms_norm_eps']
        )

        # 输出层
        self.lm_head = CustomLinear(weight_manager.get_weight("lm_head.weight"))

        print(f"✅ Qwen1.5-MoE模型构建完成: {len(self.layers)}层")

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        batch_size, seq_len = input_ids.shape

        # 嵌入
        hidden_states = self.embed_tokens.forward(input_ids)

        # 位置ids
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # 因果掩码
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf')),
                diagonal=1
            )
            attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # 通过所有层
        for layer in self.layers:
            hidden_states = layer.forward(hidden_states, attention_mask, position_ids)

        # 最终norm
        hidden_states = self.norm.forward(hidden_states)

        # 输出
        logits = self.lm_head.forward(hidden_states)
        return logits

    def generate(self, input_ids, tokenizer, max_new_tokens=50):
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
            print(f"步骤{i}: {decoded}")

            if next_token.item() in [151645, 151643]:  # EOS tokens
                break

        return generated

def main():
    print("🔧 Qwen1.5-MoE推理系统")
    print("=" * 50)

    model_path = "models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9"

    # 加载配置
    with open(f"{model_path}/config.json", 'r') as f:
        config = json.load(f)

    print(f"模型配置:")
    print(f"- 注意力头数: {config['num_attention_heads']}")
    print(f"- 隐藏维度: {config['hidden_size']}")
    print(f"- 专家数量: {config['num_experts']}")
    print(f"- 每token专家数: {config['num_experts_per_tok']}")
    print(f"- MoE中间维度: {config['moe_intermediate_size']}")
    print(f"- 共享专家中间维度: {config['shared_expert_intermediate_size']}")

    # 创建模型
    weight_manager = WeightManager(model_path)
    model = Qwen2MoEModel(weight_manager, config)

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 准备输入
    prompt = "人工智能是"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    print(f"\n开始推理，输入: {prompt}")
    print(f"输入长度: {input_ids.shape[1]} tokens")

    # 推理
    with torch.no_grad():
        outputs = model.generate(input_ids, tokenizer, max_new_tokens=20)

    # 输出结果
    output_text = tokenizer.decode(outputs[0].tolist(), skip_special_tokens=True)
    print(f"\n🎯 完整输出:")
    print(output_text)
    print(f"\n✅ 生成了 {outputs.shape[1] - input_ids.shape[1]} 个新token")

if __name__ == "__main__":
    main()