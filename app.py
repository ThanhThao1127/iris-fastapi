from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timezone, timedelta
import joblib
import time
import json
import os


# 1. CẤU HÌNH FILE JSON

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

USERS_FILE = os.path.join(BASE_DIR, "users.json")
HISTORY_FILE = os.path.join(BASE_DIR, "history.json")

# Tạo file nếu chưa có
if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=4)

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=4)


# 2. ĐỌC / LƯU JSON

def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# 3. TẢI MODEL

model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")


# 4. KHỞI TẠO API

app = FastAPI(
    title="Iris Classification API",
    description="SVM model for the Iris dataset",
    version="1.0.0"
)

app.mount("/images", StaticFiles(directory="images"), name="images")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# 5. DỮ LIỆU INPUT

class RegisterInput(BaseModel):
    username: str
    password: str


class LoginInput(BaseModel):
    username: str
    password: str


class ChangePasswordInput(BaseModel):
    user_id: int
    old_password: str
    new_password: str


class PredictInput(BaseModel):
    user_id: int
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


species = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}


# 6. TRANG CHỦ

@app.get("/")
def home():
    return FileResponse("api.html")


@app.get("/health")
def health():
    return {"status": "healthy"}


# 7. ĐĂNG KÝ

@app.post("/register")
def register(data: RegisterInput):

    users = load_json(USERS_FILE)

    # Kiểm tra tên đăng nhập
    for user in users:
        if user["username"] == data.username:
            return {
                "success": False,
                "message": "Tên đăng nhập đã tồn tại!"
            }

    # Tạo UserID mới
    if users:
        new_user_id = max(
            user["user_id"]
            for user in users
        ) + 1
    else:
        new_user_id = 1

    # Lưu tài khoản vào JSON
    users.append({
        "user_id": new_user_id,
        "username": data.username,
        "password": data.password
    })

    save_json(USERS_FILE, users)

    return {
        "success": True,
        "message": "Đăng ký thành công!",
        "user_id": new_user_id
    }


# 8. ĐĂNG NHẬP

@app.post("/login")
def login(data: LoginInput):

    users = load_json(USERS_FILE)

    for user in users:
        if (
            user["username"] == data.username
            and user["password"] == data.password
        ):
            return {
                "success": True,
                "message": "Đăng nhập thành công!",
                "user_id": user["user_id"],
                "username": user["username"]
            }

    return {
        "success": False,
        "message": "Tên đăng nhập hoặc mật khẩu không đúng!"
    }


# 9. ĐỔI MẬT KHẨU

@app.post("/change-password")
def change_password(data: ChangePasswordInput):

    users = load_json(USERS_FILE)

    for user in users:

        if user["user_id"] == data.user_id:

            # Kiểm tra mật khẩu cũ
            if user["password"] != data.old_password:
                return {
                    "success": False,
                    "message": "Mật khẩu cũ không đúng!"
                }

            # Cập nhật mật khẩu trong JSON
            user["password"] = data.new_password

            save_json(USERS_FILE, users)

            return {
                "success": True,
                "message": "Đổi mật khẩu thành công!"
            }

    return {
        "success": False,
        "message": "Không tìm thấy tài khoản!"
    }


# 10. DỰ ĐOÁN + LƯU JSON

@app.post("/predict")
def predict(data: PredictInput):

    # Bắt đầu tính thời gian
    start_time = time.time()

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    # Chuẩn hóa dữ liệu
    scaled_features = scaler.transform(features)

    # Dự đoán
    prediction = int(
        model.predict(scaled_features)[0]
    )

    prediction_name = species[prediction]

    # Thời gian xử lý
    processing_time = time.time() - start_time

    # Thời gian Việt Nam UTC+7
    vietnam_time = (
        datetime.now(timezone.utc)
        + timedelta(hours=7)
    ).replace(tzinfo=None)

    # Lưu vào JSON
    history_data = load_json(HISTORY_FILE)

    # Tạo ID cho lần dự đoán
    if history_data:
        new_prediction_id = max(
            item["prediction_id"]
            for item in history_data
        ) + 1
    else:
        new_prediction_id = 1

    history_data.append({
        "prediction_id": new_prediction_id,
        "user_id": data.user_id,
        "sepal_length": data.sepal_length,
        "sepal_width": data.sepal_width,
        "petal_length": data.petal_length,
        "petal_width": data.petal_width,
        "model_name": "SVM",
        "prediction": prediction_name,
        "processing_time": round(processing_time, 4),
        "created_at": vietnam_time.strftime(
            "%d/%m/%Y %H:%M:%S"
        )
    })

    save_json(HISTORY_FILE, history_data)

    # Trả kết quả về web
    return {
        "success": True,
        "prediction": prediction_name,
        "prediction_id": new_prediction_id,
        "user_id": data.user_id
    }


# 11. LỊCH SỬ

@app.get("/history/{user_id}")
def history(user_id: int):

    history_data = load_json(HISTORY_FILE)

    # Chỉ lấy lịch sử của tài khoản đang đăng nhập
    user_history = [
        item
        for item in history_data
        if item["user_id"] == user_id
    ]

    # Mới nhất lên trước
    user_history.reverse()

    return {
        "success": True,
        "user_id": user_id,
        "history": user_history
    }