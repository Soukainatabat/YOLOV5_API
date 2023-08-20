import sys
sys.path.insert(0, './yolov5')

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO

app = FastAPI()

# Charger le modèle YOLOv5 avec PyTorch
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp24/weights/last.pt', force_reload=True)

# Définir la route pour la prédiction de détection d'objets
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Lire l'image à partir du fichier téléchargé
    image = Image.open(BytesIO(file.file.read()))
    
    # Effectuer la prédiction de détection d'objets
    results = model(image)
    
    # Extraire les informations de prédiction
    detections = results.pred[0]
    labels = detections[:, -1].cpu().numpy()
    scores = detections[:, 4].cpu().numpy()
    
    # Construire les résultats de prédiction
    prediction_results = []
    for label, score in zip(labels, scores):
        prediction_results.append({"label": int(label), "score": float(score)})
    
    return {"predictions": prediction_results}
