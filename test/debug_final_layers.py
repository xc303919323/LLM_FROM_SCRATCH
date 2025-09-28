import torch
import json
from safetensors import safe_open
from transformers import AutoTokenizer
from qwen3_with_kvcache import *
from correct_qwen3_inference import CorrectQwen3Model

def debug_final_layers():
    print("🔍 调试最终层（norm和lm_head）")
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

    # 完整前向传播到最后一层之前
    with torch.no_grad():
        # 原始模型
        batch_size, seq_len = input_ids.shape

        # 嵌入
        hidden_states_orig = original_model.embed_tokens.forward(input_ids)

        # 位置ids
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)

        # 因果掩码
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf')),
            diagonal=1
        )
        attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # 通过所有层
        for layer in original_model.layers:
            hidden_states_orig = layer.forward(hidden_states_orig, attention_mask, position_ids)

        # KV Cache模型
        kvcache_model.reset_kv_cache()
        hidden_states_kv = kvcache_model.embed_tokens.forward(input_ids)

        # 通过所有层
        for layer in kvcache_model.layers:
            hidden_states_kv = layer.forward(
                hidden_states_kv, attention_mask, position_ids, use_cache=True
            )

        # 比较所有层后的hidden states
        diff_before_norm = torch.abs(hidden_states_orig - hidden_states_kv).max()
        print(f"所有层后的hidden states差异: {diff_before_norm.item()}")

        if diff_before_norm > 1e-5:
            print("❌ 所有层后就已经不一致!")
            return

        # 应用最终norm
        normed_orig = original_model.norm.forward(hidden_states_orig)
        normed_kv = kvcache_model.norm.forward(hidden_states_kv)

        diff_after_norm = torch.abs(normed_orig - normed_kv).max()
        print(f"Norm后的差异: {diff_after_norm.item()}")

        # 比较norm权重
        norm_weight_diff = torch.abs(
            original_model.norm.weight - kvcache_model.norm.weight
        ).max()
        print(f"Norm权重差异: {norm_weight_diff.item()}")

        # 应用lm_head
        logits_orig = original_model.lm_head.forward(normed_orig)
        logits_kv = kvcache_model.lm_head.forward(normed_kv)

        diff_logits = torch.abs(logits_orig - logits_kv).max()
        print(f"Logits差异: {diff_logits.item()}")

        # 比较lm_head权重
        lm_head_weight_diff = torch.abs(
            original_model.lm_head.weight - kvcache_model.lm_head.weight
        ).max()
        print(f"LM head权重差异: {lm_head_weight_diff.item()}")

        # 检查具体的logits值
        print(f"\n原始模型最后位置前10个logits: {logits_orig[0, -1, :10]}")
        print(f"KV Cache模型最后位置前10个logits: {logits_kv[0, -1, :10]}")

        # 分别用原始模型进行正常推理和使用缓存推理
        print(f"\n=== 对比原始模型的缓存和非缓存版本 ===")

        # 原始模型正常推理
        logits_orig_normal = original_model.forward(input_ids)

        # 手动调用KV Cache模型的正常推理
        logits_kv_normal = kvcache_model.forward(input_ids, use_cache=False)

        # KV Cache模型缓存推理
        kvcache_model.reset_kv_cache()
        logits_kv_cached = kvcache_model.forward(input_ids, use_cache=True)

        print(f"原始模型正常推理: {torch.argmax(logits_orig_normal[0, -1, :]).item()}")
        print(f"KV模型正常推理: {torch.argmax(logits_kv_normal[0, -1, :]).item()}")
        print(f"KV模型缓存推理: {torch.argmax(logits_kv_cached[0, -1, :]).item()}")

        diff_normal = torch.abs(logits_orig_normal - logits_kv_normal).max()
        diff_cached = torch.abs(logits_orig_normal - logits_kv_cached).max()

        print(f"正常推理差异: {diff_normal.item()}")
        print(f"缓存推理差异: {diff_cached.item()}")

if __name__ == "__main__":
    debug_final_layers()