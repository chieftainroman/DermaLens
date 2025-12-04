import json
import sys
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# =========================
# 1. Настройки устройства
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# 2. Загружаем классы болезней
# =========================
with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

num_classes = len(class_names)
print("Num classes:", num_classes)


# =========================
# 3. Трансформации (как при обучении)
# =========================
img_size = 224
preprocess = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# =========================
# 4. Модель ResNet18 + веса
# =========================
# 4. Модель ResNet18 + веса
# 4. Модель ResNet50 + веса
model = models.resnet50(weights=None)  # <-- ВАЖНО: здесь resnet50
num_ftrs = model.fc.in_features        # для ResNet50 это 2048
model.fc = nn.Linear(num_ftrs, num_classes)  # num_classes = 23

state_dict = torch.load("models/best_resnet50_dermnet.pth", map_location=device)
model.load_state_dict(state_dict)  # веса полностью совпадут по форме
model = model.to(device)
model.eval()

print("Model loaded (ResNet50).")



print("Model loaded.")


# =========================
# 5. Реализация Grad-CAM
# =========================
class GradCAM:
    """
    Простейшая реализация Grad-CAM для последнего сверточного блока ResNet18.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Регистрируем хуки
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            # out: (B, C, H, W)
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out[0]: (B, C, H, W)
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        """
        input_tensor: (1, 3, H, W)
        class_idx: индекс класса, для которого считаем Grad-CAM
        return:
            cam: np.array (H, W) в диапазоне [0, 1]
            class_idx: индекс класса, который реально использовали
            probs: массив вероятностей по всем классам
        """
        self.model.zero_grad()

        output = self.model(input_tensor)  # (1, num_classes)

        # Если класс не задан — берём предсказанный (argmax)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Берём логит нужного класса и делаем backward
        score = output[0, class_idx]
        score.backward()

        # Достаём активации и градиенты
        gradients = self.gradients[0]   # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # Усредняем градиенты по spatial-осям -> веса
        weights = gradients.mean(dim=(1, 2))  # (C,)

        # Линейная комбинация карт признаков с весами
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32).to(device)  # (H, W)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        # ReLU
        cam = torch.relu(cam)

        # Нормализация [0,1]
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cam.detach().cpu().numpy()

        # Вероятности (softmax)
        probs = torch.softmax(output.detach(), dim=1).cpu().numpy()[0]

        return cam, class_idx, probs


# Target layer для Grad-CAM — последний сверточный блок ResNet18
target_layer = model.layer4
grad_cam = GradCAM(model, target_layer)

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
        model="gpt-5.1",   # you can switch models if needed
        input=prompt,
    )

    advice_text = response.output_text
    return advice_text



# =========================
# 6. Предобработка картинки и вывод
# =========================
def run_gradcam_on_image(image_path):
    # Читаем исходное изображение (BGR)
    orig_bgr = cv2.imread(image_path)
    if orig_bgr is None:
        raise ValueError(f"Не удалось прочитать изображение: {image_path}")

    # Масштабируем до 224x224 (как на вход модели)
    resized_bgr = cv2.resize(orig_bgr, (img_size, img_size))
    # Для модели нужен RGB
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(resized_rgb)
    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)  # (1,3,224,224)

    # Считаем Grad-CAM
    cam, class_idx, probs = grad_cam.generate(input_tensor)
    class_name = class_names[class_idx]
    prob = probs[class_idx]

    print(f"Predicted class: {class_name} ({prob*100:.2f}%)")
    
    try:
        advice = get_ai_advice(class_name, prob, probs, class_names)
        print("\n=== AI-совет (НЕ диагноз!) ===")
        print(advice)
    except Exception as e:
        print("Не удалось получить совет от OpenAI:", e)
        advice = None

    # CAM сейчас (H,W) в [0,1] -> делаем 0-255
    cam_uint8 = np.uint8(cam * 255.0)
    cam_uint8 = cv2.resize(cam_uint8, (img_size, img_size))

    # Цветная карта (тепловая)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

    # Накладываем heatmap на исходный resized_bgr
    overlay = cv2.addWeighted(resized_bgr, 0.5, heatmap, 0.5, 0)

    # Подпись сверху
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

    return overlay


def main():
    if len(sys.argv) < 2:
        print("Использование: python cam_viewer.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    overlay = run_gradcam_on_image(image_path)

    cv2.imshow("DermNet Grad-CAM", overlay)
    print("Нажми любую клавишу в окне, чтобы закрыть.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
