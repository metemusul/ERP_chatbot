from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset
import torch
import json

# 💾 1. Model ve Tokenizer Yolu
model_id = "Qwen/Qwen1.5-1.8B-Chat"

# ✅ 2. Quantization ayarı (GPU destekli 4-bit)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token

# ✅ 3. LoRA ile modeli fine-tune'a hazırla
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# 📄 4. JSONL veri setini yükle
def load_jsonl_dataset(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            instruction = item['instruction']
            input_text = item.get('input', '')
            output = item['output']
            prompt = f"Soru: {instruction}\nGirdi: {input_text}\nYanıt: {output}"
            data.append({"text": prompt})
    return Dataset.from_list(data)

dataset = load_jsonl_dataset("erp_finetune_dataset.jsonl")

# ✂️ 5. Tokenize et
def tokenize(example):
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True)

# ⚙️ 6. Eğitim ayarları
training_args = TrainingArguments(
    output_dir="./qwen-erp-finetuned",
    per_device_train_batch_size=1,
    num_train_epochs=5,
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    fp16=True,
    report_to="none"
)

# 🔁 7. Collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# 🚀 8. Trainer başlat
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 🔥 9. Eğitimi başlat
trainer.train()

# 💾 10. Kaydet
model.save_pretrained("./qwen-erp-finetuned")
tokenizer.save_pretrained("./qwen-erp-finetuned")
