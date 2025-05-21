import torch # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import json
from rouge_score import rouge_scorer  # type: ignore
from peft import PeftModel # type: ignore

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

with open("../dataset/syn/news_unbias.json", 'r', encoding='utf-8') as file:
    news_data = json.load(file)

model_name = "/data/opensource_model/Qwen2-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

lora_path = '/data/liqiuyu/LLaMA-Factory-news/saves/qwen2-lora-news-checkpoint-200/checkpoint-200'

model = PeftModel.from_pretrained(model, model_id=lora_path)

def generate_prompt(instruction):
    return (
        # f"This is a summarization task. When summarizing, due to positional bias, please do not summarize using the information at the beginning of the above content, but use all the information to fully consider the meaning expressed in the above content. Please summarize the following content: {instruction}"
        f"This is a summarization task. Please summarize the following content: {instruction}"
    )

def evaluate_sample(instruction):
    instruction = instruction[:2000]
    prompt = generate_prompt(instruction)
    messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = generated_text.replace(prompt, "")
    return response

sum_s = 0
total = len(news_data)
count = 0
for row in news_data:
    predicted = evaluate_sample(row["instruction"])
    s = scorer.score(predicted, row["output"])["rougeL"].fmeasure
    print(s)
    sum_s = sum_s + s
    count = count + 1
print(f"Gold-3 lora on NEWS: {sum_s / total:.2%}")
