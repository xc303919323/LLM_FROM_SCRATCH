import torch
import json
from safetensors import safe_open
from transformers import AutoTokenizer
from qwen3_with_kvcache import *
from correct_qwen3_inference import CorrectQwen3Model

def debug_embedding_and_early_layers():
    print("🔍 调试嵌入层和早期层")
    print("=" * 40)

    model_path = "models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

    # 加载配置
    with open(f"{model_path}/config.json", 'r') as f:
        config = json.load(f)

    # 创建权重管理器
    weight_manager = WeightManager(f"{model_path}/model.safetensors")

    # 创建原始模型和KV Cache模型
    original_model = CorrectQwen3Model(weight_manager, config)
    kvcache_model = Qwen3ModelWithKVCache(weight_manager, config)

    # 测试输入
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompt = "介绍一下Python编程语言"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    input_ids = tokenizer.encode(text, return_tensors="pt")

    print(f"输入token ids: {input_ids}")

    # 比较嵌入层输出
    embed1 = original_model.embed_tokens.forward(input_ids)
    embed2 = kvcache_model.embed_tokens.forward(input_ids)

    embed_diff = torch.abs(embed1 - embed2).max()
    print(f"嵌入层差异: {embed_diff.item()}")

    if embed_diff < 1e-10:
        print("✅ 嵌入层一致!")
    else:
        print("❌ 嵌入层不一致!")
        return

    # 测试第一层的输出
    print("\n=== 第一层对比 ===")

    # 因果掩码
    seq_len = input_ids.shape[1]
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float('-inf')),
        diagonal=1
    )
    attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    # 位置ids
    position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)

    # 原始模型第一层
    layer1_input = embed1
    layer1_output_orig = original_model.layers[0].forward(layer1_input, attention_mask, position_ids)

    # KV Cache模型第一层（不使用缓存）
    kvcache_model.reset_kv_cache()
    layer1_output_kv_nocache = kvcache_model.layers[0].forward(layer1_input, attention_mask, position_ids, use_cache=False)

    layer1_diff = torch.abs(layer1_output_orig - layer1_output_kv_nocache).max()
    print(f"第一层输出差异（无缓存）: {layer1_diff.item()}")

    if layer1_diff < 1e-5:
        print("✅ 第一层无缓存一致!")
    else:
        print("❌ 第一层无缓存不一致!")

    # KV Cache模型第一层（使用缓存）
    kvcache_model.reset_kv_cache()
    layer1_output_kv_cache = kvcache_model.layers[0].forward(layer1_input, attention_mask, position_ids, use_cache=True)

    layer1_diff_cache = torch.abs(layer1_output_orig - layer1_output_kv_cache).max()
    print(f"第一层输出差异（有缓存）: {layer1_diff_cache.item()}")

    if layer1_diff_cache < 1e-5:
        print("✅ 第一层有缓存一致!")
    else:
        print("❌ 第一层有缓存不一致!")

    # 测试完整模型前向传播
    print("\n=== 完整模型前向传播对比 ===")

    # 原始模型
    with torch.no_grad():
        logits_orig = original_model.forward(input_ids)
        next_token_orig = torch.argmax(logits_orig[:, -1, :], dim=-1, keepdim=True)

    # KV Cache模型（不使用缓存）
    kvcache_model.reset_kv_cache()
    with torch.no_grad():
        logits_kv_nocache = kvcache_model.forward(input_ids, use_cache=False)
        next_token_kv_nocache = torch.argmax(logits_kv_nocache[:, -1, :], dim=-1, keepdim=True)

    # KV Cache模型（使用缓存）
    kvcache_model.reset_kv_cache()
    with torch.no_grad():
        logits_kv_cache = kvcache_model.forward(input_ids, use_cache=True)
        next_token_kv_cache = torch.argmax(logits_kv_cache[:, -1, :], dim=-1, keepdim=True)

    print(f"原始模型下一个token: {next_token_orig.item()} -> '{tokenizer.decode(next_token_orig.item())}'")
    print(f"KV模型无缓存下一个token: {next_token_kv_nocache.item()} -> '{tokenizer.decode(next_token_kv_nocache.item())}'")
    print(f"KV模型有缓存下一个token: {next_token_kv_cache.item()} -> '{tokenizer.decode(next_token_kv_cache.item())}'")

    # 比较logits
    logits_diff_nocache = torch.abs(logits_orig[:, -1, :] - logits_kv_nocache[:, -1, :]).max()
    logits_diff_cache = torch.abs(logits_orig[:, -1, :] - logits_kv_cache[:, -1, :]).max()

    print(f"Logits差异（无缓存）: {logits_diff_nocache.item()}")
    print(f"Logits差异（有缓存）: {logits_diff_cache.item()}")

if __name__ == "__main__":
    debug_embedding_and_early_layers()