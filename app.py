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


# =========================================================
# ĐƯỜNG DẪN FILE
# =========================================================

# Lấy thư mục chứa file api.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File lưu dữ liệu
USERS_FILE = os.path.join(BASE_DIR, "users.json")
HISTORY_FILE = os.path.join(BASE_DIR, "history.json")


# =========================================================
# TẠO FILE JSON NẾU CHƯA CÓ
# =========================================================

if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=4)

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=4)


# =========================================================
# ĐỌC JSON
# =========================================================

def load_json(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


# =========================================================
# LƯU JSON
# =========================================================

def save_json(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=4
        )


# =========================================================
# TẢI MODEL
# =========================================================

model = joblib.load(
    os.path.join(BASE_DIR, "svm_model.pkl")
)

scaler = joblib.load(
    os.path.join(BASE_DIR, "scaler.pkl")
)


# =========================================================
# KHỞI TẠO FASTAPI
# =========================================================

app = FastAPI(
    title="Iris Classification API",
    description="SVM model for the Iris dataset",
    version="1.0.0"
)


# =========================================================
# HÌNH ẢNH
# =========================================================

app.mount(
    "/images",
    StaticFiles(
        directory=os.path.join(BASE_DIR, "images")
    ),
    name="images"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# =========================================================
# MODEL INPUT
# =========================================================

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


# =========================================================
# LOÀI HOA IRIS
# =========================================================

species = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}


# =========================================================
# TRANG CHỦ
# =========================================================

@app.get("/")
def home():
    return FileResponse(
        os.path.join(BASE_DIR, "api.html")
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# =========================================================
# ĐĂNG KÝ
# =========================================================

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

    # Tạo User ID mới
    if users:

        new_user_id = max(
            user["user_id"]
            for user in users
        ) + 1

    else:

        new_user_id = 1

    # Tạo tài khoản mới
    new_user = {
        "user_id": new_user_id,
        "username": data.username,
        "password": data.password
    }

    users.append(new_user)

    # Lưu vào users.json
    save_json(
        USERS_FILE,
        users
    )

    return {
        "success": True,
        "message": "Đăng ký thành công!",
        "user_id": new_user_id
    }


# =========================================================
# ĐĂNG NHẬP
# =========================================================

@app.post("/login")
def login(data: LoginInput):

    users = load_json(USERS_FILE)

    for user in users:

        if (
            user["username"] == data.username
            and
            user["password"] == data.password
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


# =========================================================
# ĐỔI MẬT KHẨU
# =========================================================

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

            # Cập nhật mật khẩu
            user["password"] = data.new_password

            # Lưu lại users.json
            save_json(
                USERS_FILE,
                users
            )

            return {
                "success": True,
                "message": "Đổi mật khẩu thành công!"
            }

    return {
        "success": False,
        "message": "Không tìm thấy tài khoản!"
    }


# =========================================================
# DỰ ĐOÁN + LƯU LỊCH SỬ
# =========================================================

@app.post("/predict")
def predict(data: PredictInput):

    start_time = time.time()

    # Dữ liệu đầu vào
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

    # Tên loài
    prediction_name = species[prediction]

    # Thời gian xử lý
    processing_time = time.time() - start_time

    # Thời gian Việt Nam
    vietnam_time = (
        datetime.now(timezone.utc)
        + timedelta(hours=7)
    ).replace(tzinfo=None)

    # =====================================================
    # ĐỌC HISTORY.JSON
    # =====================================================

    history_data = load_json(HISTORY_FILE)

    # =====================================================
    # TẠO ID DỰ ĐOÁN
    # =====================================================

    if history_data:

        new_prediction_id = max(
            item["prediction_id"]
            for item in history_data
        ) + 1

    else:

        new_prediction_id = 1

    # =====================================================
    # THÊM LỊCH SỬ
    # =====================================================

    new_history = {

        "prediction_id": new_prediction_id,

        "user_id": data.user_id,

        "sepal_length": data.sepal_length,

        "sepal_width": data.sepal_width,

        "petal_length": data.petal_length,

        "petal_width": data.petal_width,

        "model_name": "SVM",

        "prediction": prediction_name,

        "processing_time": round(
            processing_time,
            4
        ),

        "created_at": vietnam_time.strftime(
            "%d/%m/%Y %H:%M:%S"
        )
    }

    history_data.append(
        new_history
    )

    # =====================================================
    # LƯU HISTORY.JSON
    # =====================================================

    save_json(
        HISTORY_FILE,
        history_data
    )

    return {
        "success": True,
        "prediction": prediction_name
    }


# =========================================================
# LỊCH SỬ DỰ ĐOÁN
# =========================================================

@app.get("/history/{user_id}")
def history(user_id: int):

    history_data = load_json(
        HISTORY_FILE
    )

    # Chỉ lấy lịch sử của user hiện tại
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

