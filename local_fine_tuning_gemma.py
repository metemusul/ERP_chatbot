import os
import json
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# ✅ 1. Cihaz kontrolü (GPU varsa kullan, yoksa CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Kullanılan cihaz: {device}")

# ✅ 2. Model ve tokenizer (lokal klasörden yükle)
model_path = "./local_models/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)

# Pad token ayarı
tokenizer.pad_token = tokenizer.eos_token

# ✅ 3. PEFT (LoRA) yapılandırması
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# ✅ 4. JSONL dosyasını yükle ve formatla
def load_jsonl_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data

def format_example(example):
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output_text = example.get("output", "")
    
    if input_text:
        prompt = f"Soru: {instruction}\nGirdi: {input_text}\nYanıt: {output_text}"
    else:
        prompt = f"Soru: {instruction}\nYanıt: {output_text}"

    return {"text": prompt}

raw_data = load_jsonl_dataset("erp_finetune_dataset.jsonl")
formatted_data = [format_example(example) for example in raw_data]
dataset = Dataset.from_list(formatted_data)

# ✅ 5. Tokenizasyon
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True)

# ✅ 6. Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./gemma-2b-it-lora-finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=False  # ✅ Değişiklik burada
)


# ✅ 8. Trainer başlat
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ✅ 9. Eğitimi başlat
trainer.train()

# ✅ 10. Fine-tune edilmiş modeli kaydet
model.save_pretrained("./gemma-2b-it-lora-finetuned")
tokenizer.save_pretrained("./gemma-2b-it-lora-finetuned")

print("✅ Fine-tuning tamamlandı ve model kaydedildi.")
