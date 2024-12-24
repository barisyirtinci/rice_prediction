# app.py
# ------------------------------------------------------
# Flask uygulaması: Rice Image Classification (CPU)
# ------------------------------------------------------

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template
import werkzeug.utils

# =========================================
# 1. Model Mimarisi (CNN)
# =========================================
class CNN(nn.Module):
    def __init__(self, unique_classes=5):
        super(CNN, self).__init__()
        # Bu yapı eğitimde kullandığınız CNN ile aynı olmalı
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(32),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(64),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(128),
        )
        self.dense_layers = nn.Sequential(
            nn.Linear(107648, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, unique_classes)  # Kaç sınıfınız varsa
        )

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(out.size(0), -1)
        out = self.dense_layers(out)
        return out

# =========================================
# 2. Label -> Class Mapping
# =========================================
rice_variety_labels = {
    0: "Arborio",
    1: "Basmati",
    2: "Ipsala",
    3: "Jasmine",
    4: "Karacadag"
}

# =========================================
# 3. Modeli CPU'da Yükleme
# =========================================
def load_model(model_path="rice_classification_model.pth"):
    # Model nesnesi
    model = CNN(unique_classes=len(rice_variety_labels))
    # Ağırlıkları yükle
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()  # Inference modu
    return model

# Global alan: model ve transform bir kez yüklenir
model = load_model("rice_classification_model.pth")

transform_ops = transforms.Compose([
    transforms.Resize((250, 250)),  # Eğitimde kullandığınız boyut
    transforms.ToTensor(),
    transforms.Normalize([0.0], [1.0])  # Eğitimde kullandığınız normalizasyon
])

# =========================================
# 4. Flask Uygulaması
# =========================================
app = Flask(__name__)

@app.route('/')
def index_page():
    """
    templates/index.html dosyasını render eder.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Dosya yükleme (drag&drop veya file seç) -> Model tahmini -> JSON yanıt
    """
    if 'file' not in request.files:
        return jsonify({"error": "Dosya yüklenmedi."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Dosya adı boş!"}), 400

    # Dosya güvenli ismi
    filename = werkzeug.utils.secure_filename(file.filename)
    # Geçici kaydedilen dizin (isterseniz iptal edebilirsiniz)
    upload_dir = 'uploads'
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, filename)
    file.save(file_path)

    # Resmi aç, modele hazırla
    image = Image.open(file_path).convert("RGB")
    image_tensor = transform_ops(image).unsqueeze(0)  # [1, 3, 250, 250]

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, predicted_idx = torch.max(probs, 1)

    conf = conf.item()  # 0-1 arası
    predicted_idx = predicted_idx.item()
    predicted_label = rice_variety_labels[predicted_idx]

    # Tahmin sonrası dosyayı silelim (isteğe bağlı)
    os.remove(file_path)

    # JSON formatında sonuç döndür
    return jsonify({
        "label": predicted_label,
        "confidence": f"{conf*100:.2f} %"
    }), 200

# Flask'i çalıştır
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
