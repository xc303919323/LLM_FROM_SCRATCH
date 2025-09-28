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

    # 应用旋转
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

        # 预计算频率
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        # 预计算cos和sin
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
            # 动态扩展
            t = torch.arange(seq_len, dtype=torch.float)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()

        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )

class CustomQwen3Attention:
    """正确的Qwen3注意力实现"""
    def __init__(self, layer_idx, weight_manager, config):
        self.layer_idx = layer_idx
        self.hidden_size = config['hidden_size']
        self.num_heads = config['num_attention_heads']
        self.num_key_value_heads = config['num_key_value_heads']
        self.head_dim = config['head_dim']
        self.max_position_embeddings = config['max_position_embeddings']

        # 验证维度 - Qwen3的特殊结构
        # q的输出维度是 num_heads * head_dim
        # k,v的输出维度是 num_key_value_heads * head_dim
        # 最终的输出维度回到 hidden_size
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

        # 应用RMSNorm
        q = self.q_norm.forward(q)
        k = self.k_norm.forward(k)

        # 转置为 (batch, heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 应用RoPE
        cos, sin = self.rotary_emb.forward(q, seq_len=seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # GQA: 扩展k和v
        if self.num_key_value_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)

        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
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

class CustomQwen3DecoderLayer:
    def __init__(self, layer_idx, weight_manager, config):
        self.self_attn = CustomQwen3Attention(layer_idx, weight_manager, config)
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

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm.forward(hidden_states)
        hidden_states = self.self_attn.forward(hidden_states, attention_mask, position_ids)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm.forward(hidden_states)
        hidden_states = self.mlp.forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class CorrectQwen3Model:
    """正确的Qwen3模型实现"""
    def __init__(self, weight_manager, config):
        print("构建正确的Qwen3模型...")
        self.config = config

        # 嵌入层
        self.embed_tokens = CustomEmbedding(
            weight_manager.get_weight("model.embed_tokens.weight")
        )

        # Transformer层
        self.layers = []
        for i in range(config['num_hidden_layers']):
            self.layers.append(CustomQwen3DecoderLayer(i, weight_manager, config))

        # 最终norm
        self.norm = CustomRMSNorm(
            weight_manager.get_weight("model.norm.weight"),
            eps=config['rms_norm_eps']
        )

        # 输出层
        self.lm_head = CustomLinear(weight_manager.get_weight("lm_head.weight"))

        print(f"✅ Qwen3模型构建完成: {len(self.layers)}层")

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
            # print(next_token.shape)
            print(tokenizer.decode(int(next_token), skip_special_tokens=True).strip())
            # print("generated : ", generated.shape)
            if next_token.item() in [151645, 151643]:
                break

        return generated

def main():
    print("🔧 正确的Qwen3自定义GEMM推理系统")
    print("=" * 50)

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
    model = CorrectQwen3Model(weight_manager, config)

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 准备输入
    prompt = "你好，请介绍一下小米公司"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    input_ids = tokenizer.encode(text, return_tensors="pt")

    print(f"\n开始推理，输入长度: {input_ids.shape[1]}")

    # 推理
    with torch.no_grad():
        outputs = model.generate(input_ids, tokenizer, max_new_tokens=500)

    # 解析结果
    output_ids = outputs[0][len(input_ids[0]):].tolist()

    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()

    print(f"\n🎯 结果:")
    print(f"输入: {prompt}")
    if thinking_content:
        print(f"思考: {thinking_content[:100]}...")
    if content:
        print(f"回复: {content}")
    else:
        print("⚠️ 回复为空")

    print(f"\n✅ 生成了 {len(output_ids)} 个token")

if __name__ == "__main__":
    main()