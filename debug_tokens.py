import torch
import json
from transformers import AutoTokenizer
from correct_qwen3_inference import WeightManager, CorrectQwen3Model

def debug_tokens():
    model_path = "models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

    # 加载配置和模型
    with open(f"{model_path}/config.json", 'r') as f:
        config = json.load(f)

    weight_manager = WeightManager(f"{model_path}/model.safetensors")
    model = CorrectQwen3Model(weight_manager, config)
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

    print(f"开始生成...")

    # 生成更多token
    with torch.no_grad():
        outputs = model.generate(input_ids, max_new_tokens=300)

    output_ids = outputs[0][len(input_ids[0]):].tolist()

    print(f"生成了 {len(output_ids)} 个token")
    print(f"前20个token: {output_ids[:20]}")
    print(f"后20个token: {output_ids[-20:]}")

    # 检查特殊token
    think_start = 151667  # <think>
    think_end = 151668    # </think>

    think_start_positions = [i for i, token in enumerate(output_ids) if token == think_start]
    think_end_positions = [i for i, token in enumerate(output_ids) if token == think_end]

    print(f"\n特殊token分析:")
    print(f"<think> (151667) 出现位置: {think_start_positions}")
    print(f"</think> (151668) 出现位置: {think_end_positions}")

    # 解码部分内容看看
    print(f"\n前50个token解码:")
    print(tokenizer.decode(output_ids[:50], skip_special_tokens=False))

    print(f"\n后50个token解码:")
    print(tokenizer.decode(output_ids[-50:], skip_special_tokens=False))

    # 如果找到</think>，解析thinking和content
    if think_end_positions:
        end_pos = think_end_positions[0]
        thinking_ids = output_ids[:end_pos]
        content_ids = output_ids[end_pos+1:]  # 跳过</think>

        thinking_content = tokenizer.decode(thinking_ids, skip_special_tokens=True).strip()
        content = tokenizer.decode(content_ids, skip_special_tokens=True).strip()

        print(f"\n✅ 找到thinking结束标记!")
        print(f"Thinking内容: {thinking_content[:100]}...")
        print(f"Content内容: {content}")
    else:
        print(f"\n⚠️ 没有找到thinking结束标记，可能需要生成更多token")

if __name__ == "__main__":
    debug_tokens()