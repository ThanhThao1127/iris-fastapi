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
import pyodbc


# 1. CẤU HÌNH FILE JSON

USERS_FILE = "users.json"
HISTORY_FILE = "history.json"

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


# 3. KẾT NỐI SQL SERVER

SERVER = r"DESKTOP-9RB4JUU"
DATABASE = "IrisDB"
DRIVER = "ODBC Driver 17 for SQL Server"


def get_connection():
    connection_string = (
        f"DRIVER={{{DRIVER}}};"
        f"SERVER={SERVER};"
        f"DATABASE={DATABASE};"
        f"Trusted_Connection=yes;"
    )

    return pyodbc.connect(connection_string)


# 4. TẢI MODEL

model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")


# 5. KHỞI TẠO API

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


# 6. DỮ LIỆU INPUT

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


# 7. TRANG CHỦ

@app.get("/")
def home():
    return FileResponse("api.html")


@app.get("/health")
def health():
    return {"status": "healthy"}


# 8. ĐĂNG KÝ

@app.post("/register")
def register(data: RegisterInput):

    users = load_json(USERS_FILE)

    # Kiểm tra tên đăng nhập trong JSON
    for user in users:
        if user["username"] == data.username:
            return {
                "success": False,
                "message": "Tên đăng nhập đã tồn tại!"
            }

    # Thêm tài khoản vào SQL
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Kiểm tra username trong SQL
        cursor.execute(
            "SELECT UserID FROM Users WHERE Username = ?",
            data.username
        )

        if cursor.fetchone():
            conn.close()

            return {
                "success": False,
                "message": "Tên đăng nhập đã tồn tại trong SQL!"
            }

        # Thêm user và lấy UserID
        cursor.execute(
            """
            INSERT INTO Users (Username, Password)
            OUTPUT INSERTED.UserID
            VALUES (?, ?)
            """,
            data.username,
            data.password
        )

        new_user_id = cursor.fetchone()[0]

        conn.commit()
        conn.close()

    except Exception as e:
        return {
            "success": False,
            "message": f"Lỗi kết nối SQL: {str(e)}"
        }

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


# 9. ĐĂNG NHẬP

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


# 10. ĐỔI MẬT KHẨU

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

            # Cập nhật SQL
            try:
                conn = get_connection()
                cursor = conn.cursor()

                cursor.execute(
                    """
                    UPDATE Users
                    SET Password = ?
                    WHERE UserID = ?
                    """,
                    data.new_password,
                    data.user_id
                )

                conn.commit()
                conn.close()

            except Exception as e:
                return {
                    "success": False,
                    "message": f"Lỗi SQL: {str(e)}"
                }

            # Cập nhật JSON
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


# 11. DỰ ĐOÁN + LƯU JSON + LƯU SQL

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

    # 12. LƯU VÀO JSON

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

    # 13. LƯU CÙNG DỮ LIỆU VÀO SQL

    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO Predictions (
                UserID,
                SepalLength,
                SepalWidth,
                PetalLength,
                PetalWidth,
                ModelName,
                Prediction,
                ProcessingTime,
                CreatedAt
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            data.user_id,
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width,
            "SVM",
            prediction_name,
            round(processing_time, 4),
            vietnam_time
        )

        conn.commit()
        conn.close()

    except Exception as e:
        print("Lỗi lưu SQL:", e)

    # 14. TRẢ KẾT QUẢ VỀ WEB

    return {
        "prediction": prediction_name
    }


# 15. LỊCH SỬ

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