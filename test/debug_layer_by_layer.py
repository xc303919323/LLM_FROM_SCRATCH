import torch
import json
from safetensors import safe_open
from transformers import AutoTokenizer
from qwen3_with_kvcache import *
from correct_qwen3_inference import CorrectQwen3Model

def debug_layer_by_layer():
    print("🔍 逐层调试KV Cache实现")
    print("=" * 50)

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

    # 初始化
    batch_size, seq_len = input_ids.shape

    # 嵌入
    hidden_states_orig = original_model.embed_tokens.forward(input_ids)
    hidden_states_kv = kvcache_model.embed_tokens.forward(input_ids)

    # 位置ids
    position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)

    # 因果掩码
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float('-inf')),
        diagonal=1
    )
    attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    # 重置KV缓存
    kvcache_model.reset_kv_cache()

    # 逐层比较
    for layer_idx in range(min(3, len(original_model.layers))):  # 只测试前3层
        print(f"\n=== 第{layer_idx}层 ===")

        # 原始模型
        hidden_states_orig = original_model.layers[layer_idx].forward(
            hidden_states_orig, attention_mask, position_ids
        )

        # KV Cache模型
        hidden_states_kv = kvcache_model.layers[layer_idx].forward(
            hidden_states_kv, attention_mask, position_ids, use_cache=True
        )

        # 比较
        diff = torch.abs(hidden_states_orig - hidden_states_kv).max()
        print(f"第{layer_idx}层输出差异: {diff.item()}")

        if diff > 1e-5:
            print(f"❌ 第{layer_idx}层出现差异!")
            break
        else:
            print(f"✅ 第{layer_idx}层一致")

    # 如果前面的层都一致，继续测试剩余层
    if diff <= 1e-5:
        print(f"\n继续测试剩余层...")
        for layer_idx in range(3, len(original_model.layers)):
            # 原始模型
            hidden_states_orig = original_model.layers[layer_idx].forward(
                hidden_states_orig, attention_mask, position_ids
            )

            # KV Cache模型
            hidden_states_kv = kvcache_model.layers[layer_idx].forward(
                hidden_states_kv, attention_mask, position_ids, use_cache=True
            )

            # 比较
            diff = torch.abs(hidden_states_orig - hidden_states_kv).max()

            if diff > 1e-5:
                print(f"❌ 第{layer_idx}层出现差异: {diff.item()}")
                break
            elif layer_idx % 5 == 0:  # 每5层报告一次
                print(f"✅ 第{layer_idx}层一致: {diff.item()}")

    print(f"\n问题出现在第{layer_idx}层")

if __name__ == "__main__":
    debug_layer_by_layer()