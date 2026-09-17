from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# Tải mô hình và bộ chuẩn hóa
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI(
    title="Iris Classification API",
    description="SVM model for the Iris dataset",
    version="1.0.0",
)

app.mount("/images", StaticFiles(directory="images"), name="images")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


species = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica",
}


@app.get("/")
def home():
    return FileResponse("api.html")


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: IrisInput):

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]]

    # Chuẩn hóa dữ liệu đầu vào
    features_scaled = scaler.transform(features)

    # Dự đoán loài hoa
    prediction = int(model.predict(features_scaled)[0])

    return {
        "prediction": species[prediction]
    }
