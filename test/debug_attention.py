import torch
import json
from safetensors import safe_open
from transformers import AutoTokenizer
from qwen3_with_kvcache import *
from correct_qwen3_inference import CustomQwen3Attention

def debug_attention_layer():
    print("🔍 调试单个注意力层")
    print("=" * 40)

    model_path = "models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

    # 加载配置
    with open(f"{model_path}/config.json", 'r') as f:
        config = json.load(f)

    # 创建权重管理器
    weight_manager = WeightManager(f"{model_path}/model.safetensors")

    # 创建原始注意力层和KV Cache注意力层
    original_attn = CustomQwen3Attention(0, weight_manager, config)
    kvcache_attn = Qwen3AttentionWithKVCache(0, weight_manager, config)

    # 准备测试输入
    batch_size, seq_len, hidden_size = 1, 12, config['hidden_size']
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    print(f"输入形状: {hidden_states.shape}")

    # 创建因果掩码
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float('-inf')),
        diagonal=1
    )
    attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    # 测试原始注意力层
    with torch.no_grad():
        output1 = original_attn.forward(hidden_states, attention_mask)
        print(f"原始注意力输出形状: {output1.shape}")

    # 测试KV Cache注意力层（不使用缓存）
    kvcache_attn.kv_cache.reset()
    with torch.no_grad():
        output2 = kvcache_attn.forward(hidden_states, attention_mask, use_cache=False)
        print(f"KV Cache注意力输出形状（无缓存）: {output2.shape}")

    # 比较输出
    diff = torch.abs(output1 - output2).max()
    print(f"输出最大差异（无缓存）: {diff.item()}")

    if diff < 1e-5:
        print("✅ 无缓存模式一致!")
    else:
        print("❌ 无缓存模式不一致!")
        return

    # 测试KV Cache注意力层（使用缓存）
    kvcache_attn.kv_cache.reset()
    with torch.no_grad():
        output3 = kvcache_attn.forward(hidden_states, attention_mask, use_cache=True)
        print(f"KV Cache注意力输出形状（有缓存）: {output3.shape}")

    # 比较缓存和无缓存的输出
    diff2 = torch.abs(output1 - output3).max()
    print(f"输出最大差异（有缓存 vs 原始）: {diff2.item()}")

    if diff2 < 1e-5:
        print("✅ 有缓存模式一致!")
    else:
        print("❌ 有缓存模式不一致!")

    # 测试增量推理
    print("\n=== 增量推理测试 ===")

    # 单个token输入
    single_token = hidden_states[:, :1, :]  # 只取第一个token

    with torch.no_grad():
        output4 = kvcache_attn.forward(single_token, use_cache=True)
        print(f"增量推理输出形状: {output4.shape}")

    print("增量推理测试完成")

if __name__ == "__main__":
    debug_attention_layer()