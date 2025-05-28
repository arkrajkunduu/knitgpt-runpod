import base64
import io
from flask import Flask, request, jsonify
from PIL import Image
import torch
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel
from huggingface_hub import login

# 🔐 Hugging Face login
login(token="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")  # 🔁 Replace this with your real Hugging Face token

app = Flask(__name__)

# Model config
base_model_id = "google/paligemma2-3b-pt-224"
adapter_repo_id = "arkrajkundu/knitGPT_v1.0"
image_resize = (128, 128)
max_new_tokens = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model + processor
processor = PaliGemmaProcessor.from_pretrained(base_model_id)
model = PaliGemmaForConditionalGeneration.from_pretrained(base_model_id)
model = PeftModel.from_pretrained(model, adapter_repo_id)
model.to(device)
model.eval()

@app.route("/run", methods=["POST"])
def run():
    data = request.get_json()
    prompt = data["input"].get("prompt", "How to knit this pattern?")
    image_base64 = data["input"].get("image_base64")

    if not image_base64:
        return jsonify({"error": "Missing image_base64"}), 400

    try:
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(image_resize)
    except Exception as e:
        return jsonify({"error": f"Invalid image data: {str(e)}"}), 400

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return jsonify({"output": output})
