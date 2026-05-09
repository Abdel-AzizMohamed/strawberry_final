import io
import joblib
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
from preprocesser import process_image, extract_features

model = joblib.load("model.pkl")
label_map = joblib.load("label_map.pkl")
scaler = joblib.load("scaler.pkl")
inv_label_map = {v: k for k, v in label_map.items()}

app = FastAPI()

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image_np = np.array(image)

    processed_img = process_image(image_np)
    features = extract_features(processed_img).reshape(1, -1)
    scaled_data = scaler.transform(features)

    prediction = model.predict(scaled_data)[0]
    disease_name = inv_label_map.get(prediction, "Unknown")

    return {
        "filename": file.filename,
        "prediction": inv_label_map.get(prediction)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)