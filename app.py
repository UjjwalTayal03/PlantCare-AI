from flask import Flask, request, jsonify
import torch
from torchvision import models, transforms
from PIL import Image
from flask_cors import CORS   # add this import

app = Flask(__name__)
CORS(app)   # add this line here

# load pretrained model
model = models.resnet18(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

labels = [
    "Healthy Plant",
    "Leaf Disease",
    "Bacterial Infection",
    "Fungal Infection"
]

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]

    img = Image.open(file).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)

    pred = output.argmax(1).item()

    result = labels[pred % len(labels)]

    return jsonify({"prediction": result})


@app.route("/")
def home():
    return "PlantCare AI Backend Running"


if __name__ == "__main__":
    app.run()