import pandas as pd # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline # type: ignore
import logging
from peft import PeftModel # type: ignore

sampled_data = pd.read_csv("../dataset/syn/sampled_hans.txt", sep="\t")

model_name = "/data/opensource_model/Qwen2-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

lora_path = '/data/liqiuyu/LLaMA-Factory-prompt-enhance/saves/qwen2-lora-updated-mnli-lap-neg/checkpoint-1800'

model = PeftModel.from_pretrained(model, model_id=lora_path)

logging.basicConfig(
    filename="log_post/hans.txt", 
    level=logging.INFO,           
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def log_message(message):
    logging.info(message)

def generate_prompt(premise, conclusion):
    return (
       f"Given the premise '{premise}' and the hypothesis '{conclusion}', please classify the relationship between the premise and hypothesis, and answer only: neutral, entailment, or contradiction. Note that there is a lexical overlap bias between the premise and hypothesis. Do not judge their relationship based on the degree of overlap of words in the premise and hypothesis, but rather judge their relationship based on semantics and logic. For example, the premise 'The artist said that the athletes saw the president .' and the hypothesis 'The athletes saw the president .', their relationship is not 'entailment'."
    )

"""
def generate_prompt(premise, conclusion):
    return (
        f"Given the premise '{premise}' and the hypothesis '{conclusion}', please classify the relationship between the premise and hypothesis, and answer only: neutral, entailment, or contradiction."
    )
"""

def evaluate_sample(premise, hypothesis, gold_label):
    prompt = generate_prompt(premise, hypothesis)
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
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
    log_message(f"generated_text: {generated_text}\n")
    response = generated_text.replace(prompt, "")
    log_message(f"response: {response.lower()}\n")
    if "entailment" == response.lower():
        predicted_label = "entailment"
    else:
        predicted_label = "non-entailment"
    return predicted_label == gold_label, predicted_label


subset_1 = sampled_data[sampled_data["heuristic"] == "lexical_overlap"]
entail_count = 0
non_entail_count = 0
entail_correct = 0
non_entail_correct = 0
print("lexical_overlap")
for _, row in subset_1.iterrows():
    if row["gold_label"] == "entailment":
        is_correct, predicted = evaluate_sample(row["sentence1"], row["sentence2"], row["gold_label"])
        log_message(f"entail_count: {entail_count}, gold_label: {row['gold_label']}, predicted: {predicted}, is_correct: {is_correct}\n")
        entail_count = entail_count + 1
        if is_correct:
            entail_correct += 1
    if row["gold_label"] == "non-entailment":
        is_correct, predicted = evaluate_sample(row["sentence1"], row["sentence2"], row["gold_label"])
        log_message(f"non_entail_count: {non_entail_count}, gold_label: {row['gold_label']}, predicted: {predicted}, is_correct: {is_correct}\n")
        non_entail_count = non_entail_count + 1
        if is_correct:
            non_entail_correct += 1
print(f"Accuracy for lexical_overlap entailment: {entail_correct / entail_count:.2%}")
log_message(f"Accuracy for lexical_overlap entailment: {entail_correct / entail_count:.2%}")
print(f"Accuracy for lexical_overlap non-entailment: {non_entail_correct / non_entail_count:.2%}")
log_message(f"Accuracy for lexical_overlap non-entailment: {non_entail_correct / non_entail_count:.2%}")


subset_2 = sampled_data[sampled_data["heuristic"] == "subsequence"]
entail_count = 0
non_entail_count = 0
entail_correct = 0
non_entail_correct = 0
print("subsequence")
for _, row in subset_2.iterrows():
    if row["gold_label"] == "entailment":
        is_correct, predicted = evaluate_sample(row["sentence1"], row["sentence2"], row["gold_label"])
        log_message(f"entail_count: {entail_count}, gold_label: {row['gold_label']}, predicted: {predicted}, is_correct: {is_correct}\n")
        entail_count = entail_count + 1
        if is_correct:
            entail_correct += 1
    if row["gold_label"] == "non-entailment":
        is_correct, predicted = evaluate_sample(row["sentence1"], row["sentence2"], row["gold_label"])
        log_message(f"non_entail_count: {non_entail_count}, gold_label: {row['gold_label']}, predicted: {predicted}, is_correct: {is_correct}\n")
        non_entail_count = non_entail_count + 1
        if is_correct:
            non_entail_correct += 1

print(f"Accuracy for subsequence entailment: {entail_correct / entail_count:.2%}")
log_message(f"Accuracy for subsequence entailment: {entail_correct / entail_count:.2%}")
print(f"Accuracy for subsequence non-entailment: {non_entail_correct / non_entail_count:.2%}")
log_message(f"Accuracy for subsequence non-entailment: {non_entail_correct / non_entail_count:.2%}")


subset_3 = sampled_data[sampled_data["heuristic"] == "constituent"]
entail_count = 0
non_entail_count = 0
entail_correct = 0
non_entail_correct = 0
print("constituent")
for _, row in subset_3.iterrows():
    if row["gold_label"] == "entailment":
        is_correct, predicted = evaluate_sample(row["sentence1"], row["sentence2"], row["gold_label"])
        log_message(f"entail_count: {entail_count}, gold_label: {row['gold_label']}, predicted: {predicted}, is_correct: {is_correct}\n")
        entail_count = entail_count + 1
        if is_correct:
            entail_correct += 1
    if row["gold_label"] == "non-entailment":
        is_correct, predicted = evaluate_sample(row["sentence1"], row["sentence2"], row["gold_label"])
        log_message(f"non_entail_count: {non_entail_count}, gold_label: {row['gold_label']}, predicted: {predicted}, is_correct: {is_correct}\n")
        non_entail_count = non_entail_count + 1
        if is_correct:
            non_entail_correct += 1
print(f"Accuracy for constituent entailment: {entail_correct / entail_count:.2%}")
log_message(f"Accuracy for constituent entailment: {entail_correct / entail_count:.2%}")
print(f"Accuracy for constituent non-entailment: {non_entail_correct / non_entail_count:.2%}")
log_message(f"Accuracy for constituent non-entailment: {non_entail_correct / non_entail_count:.2%}")
