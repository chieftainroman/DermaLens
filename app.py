import os
import io
import base64
import threading

from dotenv import load_dotenv
from openai import OpenAI

from flask import Flask, render_template, request, flash

import json
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# =========================
# Инициализация Flask
# =========================
app = Flask(__name__)
app.secret_key = "super-secret-key-change-me"  # нужен для flash-сообщений

# =========================
# OpenAI + .env
# =========================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# Настройки устройства
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Лок для потокобезопасного доступа к модели
model_lock = threading.Lock()

# =========================
# Загружаем классы болезней
# =========================
with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

num_classes = len(class_names)
print("Num classes:", num_classes)

# =========================
# Трансформации (как при обучении)
# =========================
img_size = 224
preprocess = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =========================
# Модель ResNet50 + веса
# =========================
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

state_dict = torch.load("models/best_resnet50_dermnet.pth", map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

print("Model loaded (ResNet50).")


# =========================
# Grad-CAM
# =========================
class GradCAM:
    """
    Простая реализация Grad-CAM для последнего сверточного блока ResNet50.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        """
        input_tensor: (1, 3, H, W)
        return: cam (H,W in [0,1]), class_idx, probs (np.array num_classes)
        """
        self.model.zero_grad()

        output = self.model(input_tensor)  # (1, num_classes)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        score = output[0, class_idx]
        score.backward()

        gradients = self.gradients[0]      # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # Усредняем градиенты по spatial-осям -> веса
        weights = gradients.mean(dim=(1, 2))  # (C,)

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32).to(device)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = torch.relu(cam)

        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cam.detach().cpu().numpy()

        probs = torch.softmax(output.detach(), dim=1).cpu().numpy()[0]

        return cam, class_idx, probs


# Глобальный Grad-CAM для последнего блока
target_layer = model.layer4
grad_cam = GradCAM(model, target_layer)


# =========================
# OpenAI advice (ENGLISH)
# =========================
def get_ai_advice(class_name, prob, probs, class_names):
    """
    Ask OpenAI for a textual explanation of the results.
    IMPORTANT: This is NOT a diagnosis. It is general information only.
    """

    # Build top-3 classes for context
    top_indices = np.argsort(probs)[::-1][:3]
    top_info = []
    for idx in top_indices:
        top_info.append(f"{class_names[idx]}: {probs[idx]*100:.1f}%")
    top_text = ", ".join(top_info)

    prompt = f"""
You are a cautious medical AI assistant helping a user understand the result
of a NON-SPECIALIZED computer program that analyzes skin images.
This program CAN BE WRONG and IS NOT a medical device.

GIVEN:
- Predicted class: "{class_name}"
- Probability of this class: {prob*100:.1f}%
- Top-3 classes and probabilities: {top_text}

TASK:
1. In clear, simple English, explain what this MIGHT mean in very general terms
   (no confident diagnosis, no statements that this is definitely the condition).
2. Very clearly state that:
   - this is NOT a medical diagnosis,
   - the program can be inaccurate or completely wrong,
   - the user MUST see a doctor (preferably a dermatologist or primary care doctor)
     for an in-person examination.
3. Give 2–4 safe, general precautions, for example:
   - do not try to cut, burn, or pop anything yourself,
   - protect the area from sun exposure,
   - monitor for changes (size, color, shape, bleeding, pain),
   - if there is strong pain, bleeding, or rapid change, seek urgent medical care.
4. Do NOT mention any specific medicines, creams, pills, dosages, or treatment plans.
5. Do NOT say that “everything is fine” or that the user “doesn’t need to see a doctor”.
   ALWAYS recommend an in-person medical evaluation.
6. Style: short paragraphs, friendly and calm tone, easy to understand
   for a non-medical person.
"""

    response = client.responses.create(
        model="gpt-5.1",   # or "gpt-5.1-mini" if хочешь дешевле/быстрее
        input=prompt,
    )

    advice_text = response.output_text
    return advice_text


# =========================
# Обработка загруженного изображения
# =========================
def analyze_uploaded_image(file_storage):
    """
    Принимает Flask FileStorage, возвращает:
    - class_name
    - prob
    - advice (string)
    - overlay_png_base64 (string "data:image/png;base64,...")
    """
    # Читаем файл в память
    file_bytes = file_storage.read()
    if not file_bytes:
        raise ValueError("Empty file")

    # Декодируем в OpenCV BGR
    np_arr = np.frombuffer(file_bytes, np.uint8)
    orig_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if orig_bgr is None:
        raise ValueError("Unable to decode image")

    # Масштабируем до 224x224
    resized_bgr = cv2.resize(orig_bgr, (img_size, img_size))
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(resized_rgb)
    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

    # Защита модели локом (для нескольких одновременных запросов)
    with model_lock:
        cam, class_idx, probs = grad_cam.generate(input_tensor)

    class_name = class_names[class_idx]
    prob = probs[class_idx]

    # Heatmap
    cam_uint8 = np.uint8(cam * 255.0)
    cam_uint8 = cv2.resize(cam_uint8, (img_size, img_size))
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

    # Накладываем heatmap на оригинал
    overlay = cv2.addWeighted(resized_bgr, 0.5, heatmap, 0.5, 0)

    # Подпись на картинке
    text = f"{class_name}: {prob*100:.1f}%"
    cv2.putText(
        overlay,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Конвертируем overlay в PNG base64
    _, buffer = cv2.imencode(".png", overlay)
    png_bytes = buffer.tobytes()
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    overlay_data_uri = f"data:image/png;base64,{b64}"

    advice = get_ai_advice(class_name, prob, probs, class_names)

    return class_name, float(prob), advice, overlay_data_uri



@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            flash("Please upload an image file.")
        else:
            try:
                class_name, prob, advice, overlay_data_uri = analyze_uploaded_image(file)
                result = {
                    "class_name": class_name,
                    "prob": f"{prob*100:.1f}",
                    "advice": advice,
                    "overlay_data_uri": overlay_data_uri,
                }
            except Exception as e:
                print("Error while processing image:", e)
                flash(f"Error while processing image: {e}")

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
