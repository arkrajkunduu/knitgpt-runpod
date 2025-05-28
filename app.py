import os
from flask import Flask, request, jsonify
import torch
from PIL import Image
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel

app = Flask(__name__)

base_model_id = "google/paligemma2-3b-pt-224"
adapter_repo_id = "arkrajkundu/knitGPT_v1.0"
image_resize = (128, 128)
max_new_tokens = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = PaliGemmaProcessor.from_pretrained(base_model_id)
model = PaliGemmaForConditionalGeneration.from_pretrained(base_model_id)
model = PeftModel.from_pretrained(model, adapter_repo_id)
model.to(device)
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Image is required"}), 400
    image_file = request.files["image"]

    try:
        image = Image.open(image_file).convert("RGB").resize(image_resize)
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

    prompt = request.form.get("prompt", "How to knit this pattern?")
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    )
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return jsonify({"output": output})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)