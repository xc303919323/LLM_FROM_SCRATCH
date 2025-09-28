import torch
import json
from safetensors import safe_open
from transformers import AutoTokenizer
from qwen3_with_kvcache import *

def debug_kv_cache():
    print("🔍 调试KV Cache实现")
    print("=" * 40)

    model_path = "models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

    # 加载配置
    with open(f"{model_path}/config.json", 'r') as f:
        config = json.load(f)

    # 创建模型
    weight_manager = WeightManager(f"{model_path}/model.safetensors")
    model = Qwen3ModelWithKVCache(weight_manager, config)

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 简单测试输入
    test_text = "Hello"
    input_ids = tokenizer.encode(test_text, return_tensors="pt")
    print(f"输入: {test_text}")
    print(f"Token IDs: {input_ids}")

    # 测试标准前向传播
    print("\n标准前向传播:")
    with torch.no_grad():
        logits1 = model.forward(input_ids, use_cache=False)
        next_token1 = torch.argmax(logits1[:, -1, :], dim=-1, keepdim=True)
        print(f"下一个token: {next_token1.item()}")
        print(f"解码: {tokenizer.decode(next_token1.item())}")

    # 测试KV Cache前向传播
    print("\nKV Cache前向传播:")
    model.reset_kv_cache()
    with torch.no_grad():
        logits2 = model.forward(input_ids, use_cache=True)
        next_token2 = torch.argmax(logits2[:, -1, :], dim=-1, keepdim=True)
        print(f"下一个token: {next_token2.item()}")
        print(f"解码: {tokenizer.decode(next_token2.item())}")

    # 检查logits是否一致
    print(f"\nLogits一致性:")
    print(f"Logits形状 - 标准: {logits1.shape}, KV Cache: {logits2.shape}")

    # 比较最后一个位置的logits
    diff = torch.abs(logits1[:, -1, :] - logits2[:, -1, :]).max()
    print(f"最大差异: {diff.item()}")

    if diff < 1e-5:
        print("✅ Logits一致!")
    else:
        print("❌ Logits不一致!")
        # 显示前10个logits值
        print("标准版本前10个logits:", logits1[0, -1, :10])
        print("KV Cache前10个logits:", logits2[0, -1, :10])

    # 测试增量推理
    print(f"\n增量推理测试:")
    with torch.no_grad():
        # 使用KV Cache的下一个token
        logits3 = model.forward(next_token2, use_cache=True)
        next_token3 = torch.argmax(logits3[:, -1, :], dim=-1, keepdim=True)
        print(f"增量推理下一个token: {next_token3.item()}")
        print(f"解码: {tokenizer.decode(next_token3.item())}")

    # 比较与完整序列的结果
    full_sequence = torch.cat([input_ids, next_token2], dim=1)
    print(f"\n完整序列对比:")
    with torch.no_grad():
        logits_full = model.forward(full_sequence, use_cache=False)
        next_token_full = torch.argmax(logits_full[:, -1, :], dim=-1, keepdim=True)
        print(f"完整序列下一个token: {next_token_full.item()}")

    if next_token3.item() == next_token_full.item():
        print("✅ 增量推理正确!")
    else:
        print("❌ 增量推理错误!")

    # 测试完整的生成过程对比
    print(f"\n完整生成过程对比:")

    # 标准生成
    model.reset_kv_cache()  # 重置缓存
    with torch.no_grad():
        standard_result = model.generate(input_ids, tokenizer, max_new_tokens=10)
    print("标准生成结果:", tokenizer.decode(standard_result[0].tolist()))

    # KV Cache生成
    model.reset_kv_cache()  # 重置缓存
    with torch.no_grad():
        kvcache_result = model.generate_with_kvcache(input_ids, tokenizer, max_new_tokens=10)
    print("KV Cache生成结果:", tokenizer.decode(kvcache_result[0].tolist()))

if __name__ == "__main__":
    debug_kv_cache()