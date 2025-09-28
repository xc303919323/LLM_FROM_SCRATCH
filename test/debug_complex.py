import torch
import json
from safetensors import safe_open
from transformers import AutoTokenizer
from qwen3_with_kvcache import *

def debug_complex_input():
    print("🔍 调试复杂输入的KV Cache实现")
    print("=" * 50)

    model_path = "models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

    # 加载配置
    with open(f"{model_path}/config.json", 'r') as f:
        config = json.load(f)

    # 创建模型
    weight_manager = WeightManager(f"{model_path}/model.safetensors")
    model = Qwen3ModelWithKVCache(weight_manager, config)

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 准备复杂输入
    prompt = "介绍一下Python编程语言"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    input_ids = tokenizer.encode(text, return_tensors="pt")

    print(f"Chat template文本:\n{text}")
    print(f"\nToken IDs: {input_ids}")
    print(f"输入长度: {input_ids.shape[1]} tokens")
    print(f"解码回来: {tokenizer.decode(input_ids[0].tolist())}")

    # 测试第一步
    print("\n=== 第一步对比 ===")

    # 标准前向传播
    model.reset_kv_cache()
    with torch.no_grad():
        logits1 = model.forward(input_ids, use_cache=False)
        next_token1 = torch.argmax(logits1[:, -1, :], dim=-1, keepdim=True)
        print(f"标准版本下一个token: {next_token1.item()}")
        print(f"解码: '{tokenizer.decode(next_token1.item())}'")

    # KV Cache前向传播
    model.reset_kv_cache()
    with torch.no_grad():
        logits2 = model.forward(input_ids, use_cache=True)
        next_token2 = torch.argmax(logits2[:, -1, :], dim=-1, keepdim=True)
        print(f"KV Cache版本下一个token: {next_token2.item()}")
        print(f"解码: '{tokenizer.decode(next_token2.item())}'")

    # 检查logits差异
    diff = torch.abs(logits1[:, -1, :] - logits2[:, -1, :]).max()
    print(f"Logits最大差异: {diff.item()}")

    if next_token1.item() == next_token2.item():
        print("✅ 第一步一致!")
    else:
        print("❌ 第一步不一致!")
        return

    # 测试第二步
    print("\n=== 第二步对比 ===")

    # 标准版本: 完整序列
    full_seq1 = torch.cat([input_ids, next_token1], dim=1)
    with torch.no_grad():
        logits1_step2 = model.forward(full_seq1, use_cache=False)
        next_token1_step2 = torch.argmax(logits1_step2[:, -1, :], dim=-1, keepdim=True)
        print(f"标准版本第二步token: {next_token1_step2.item()}")
        print(f"解码: '{tokenizer.decode(next_token1_step2.item())}'")

    # KV Cache版本: 增量推理
    with torch.no_grad():
        logits2_step2 = model.forward(next_token2, use_cache=True)
        next_token2_step2 = torch.argmax(logits2_step2[:, -1, :], dim=-1, keepdim=True)
        print(f"KV Cache版本第二步token: {next_token2_step2.item()}")
        print(f"解码: '{tokenizer.decode(next_token2_step2.item())}'")

    # 检查第二步logits差异
    diff2 = torch.abs(logits1_step2[:, -1, :] - logits2_step2[:, -1, :]).max()
    print(f"第二步Logits最大差异: {diff2.item()}")

    if next_token1_step2.item() == next_token2_step2.item():
        print("✅ 第二步一致!")
    else:
        print("❌ 第二步不一致!")
        print("可能是位置编码或缓存逻辑问题")

if __name__ == "__main__":
    debug_complex_input()