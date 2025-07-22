from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

model_path = "./qwen-erp-finetuned"
device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float32,
    device_map=None
)
model = base_model.to(device)

# Eğer adapter varsa ve ayrı bir dosyada ise şunu kullanabilirsin:
# model = PeftModel.from_pretrained(base_model, "./adapter_path").to(device)

instruction = "Yeni bir müşteri eklemek için hangi bilgileri girmem gerekir?"
input_text = "ERP sisteminde müşteri tanımlama ekranına girdim."
prompt = f"<|user|>\n{instruction}\n{input_text}\n<|assistant|>"

inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id,
        early_stopping=False
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n💬 Yanıt:", response.split("<|assistant|>")[-1].strip())