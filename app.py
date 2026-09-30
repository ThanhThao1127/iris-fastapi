from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timezone, timedelta

import joblib
import time
import os
import psycopg2
from psycopg2.extras import RealDictCursor


# ĐƯỜNG DẪN FILE
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# DATABASE
DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection():
    if not DATABASE_URL:
        raise Exception("Chưa cấu hình DATABASE_URL trên Render!")

    return psycopg2.connect(
        DATABASE_URL,
        connect_timeout=5
    )


# TẠO DATABASE TABLE
def create_tables():

    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Bảng Users
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Users (
                UserID SERIAL PRIMARY KEY,
                Username VARCHAR(50) UNIQUE NOT NULL,
                Password VARCHAR(255) NOT NULL
            )
        """)

        # Bảng Predictions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Predictions (
                PredictionID SERIAL PRIMARY KEY,

                UserID INTEGER NOT NULL,

                SepalLength DOUBLE PRECISION,
                SepalWidth DOUBLE PRECISION,
                PetalLength DOUBLE PRECISION,
                PetalWidth DOUBLE PRECISION,

                ModelName VARCHAR(50),
                Prediction VARCHAR(50),

                ProcessingTime DOUBLE PRECISION,

                CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                CONSTRAINT fk_predictions_users
                    FOREIGN KEY (UserID)
                    REFERENCES Users(UserID)
                    ON DELETE CASCADE
            )
        """)

        conn.commit()

        cursor.close()
        conn.close()

        print("Kết nối PostgreSQL thành công!")
        print("Đã kiểm tra / tạo bảng Users và Predictions.")

    except Exception as e:
        print("Không thể kết nối PostgreSQL:", e)


# KHỞI TẠO FASTAPI
app = FastAPI(
    title="Iris Classification API",
    description="SVM model for the Iris dataset",
    version="1.0.0"
)


# TẠO DATABASE KHI SERVER KHỞI ĐỘNG
@app.on_event("startup")
def startup_event():
    create_tables()


# HÌNH ẢNH
app.mount(
    "/images",
    StaticFiles(
        directory=os.path.join(BASE_DIR, "images")
    ),
    name="images"
)


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# MODEL INPUT
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


# LOÀI HOA IRIS
species = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}


# TRANG CHỦ
@app.get("/")
def home():
    return FileResponse(
        os.path.join(BASE_DIR, "api.html")
    )


# HEALTH CHECK
@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ĐĂNG KÝ
@app.post("/register")
def register(data: RegisterInput):

    conn = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Kiểm tra username
        cursor.execute(
            """
            SELECT UserID
            FROM Users
            WHERE Username = %s
            """,
            (data.username,)
        )

        existing_user = cursor.fetchone()

        if existing_user:
            cursor.close()
            conn.close()

            return {
                "success": False,
                "message": "Tên đăng nhập đã tồn tại!"
            }

        # Lưu mật khẩu trực tiếp
        password = data.password

        cursor.execute(
            """
            INSERT INTO Users
            (Username, Password)
            VALUES (%s, %s)
            RETURNING UserID
            """,
            (
                data.username,
                password
            )
        )

        user_id = cursor.fetchone()[0]

        conn.commit()

        cursor.close()
        conn.close()

        return {
            "success": True,
            "message": "Đăng ký thành công!",
            "user_id": user_id
        }

    except Exception as e:

        if conn:
            conn.rollback()
            conn.close()

        print("Lỗi đăng ký:", e)

        return {
            "success": False,
            "message": "Không thể đăng ký tài khoản!"
        }


# ĐĂNG NHẬP
@app.post("/login")
def login(data: LoginInput):

    conn = None

    try:
        conn = get_connection()
        cursor = conn.cursor(
            cursor_factory=RealDictCursor
        )

        # Kiểm tra trực tiếp
        cursor.execute(
            """
            SELECT UserID, Username
            FROM Users
            WHERE Username = %s
            AND Password = %s
            """,
            (
                data.username,
                data.password
            )
        )

        user = cursor.fetchone()

        cursor.close()
        conn.close()

        if user:
            return {
                "success": True,
                "message": "Đăng nhập thành công!",
                "user_id": user["userid"],
                "username": user["username"]
            }

        return {
            "success": False,
            "message": "Tên đăng nhập hoặc mật khẩu không đúng!"
        }

    except Exception as e:

        if conn:
            conn.close()

        print("Lỗi đăng nhập:", e)

        return {
            "success": False,
            "message": "Không thể kết nối cơ sở dữ liệu!"
        }


# ĐỔI MẬT KHẨU
@app.post("/change-password")
def change_password(data: ChangePasswordInput):

    conn = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Kiểm tra mật khẩu cũ
        cursor.execute(
            """
            SELECT UserID
            FROM Users
            WHERE UserID = %s
            AND Password = %s
            """,
            (
                data.user_id,
                data.old_password
            )
        )

        user = cursor.fetchone()

        if not user:
            cursor.close()
            conn.close()

            return {
                "success": False,
                "message": "Mật khẩu cũ không đúng!"
            }

        # Lưu mật khẩu mới
        new_password = data.new_password

        cursor.execute(
            """
            UPDATE Users
            SET Password = %s
            WHERE UserID = %s
            """,
            (
                new_password,
                data.user_id
            )
        )

        conn.commit()

        cursor.close()
        conn.close()

        return {
            "success": True,
            "message": "Đổi mật khẩu thành công!"
        }

    except Exception as e:

        if conn:
            conn.rollback()
            conn.close()

        print("Lỗi đổi mật khẩu:", e)

        return {
            "success": False,
            "message": "Không thể đổi mật khẩu!"
        }


# DỰ ĐOÁN + LƯU LỊCH SỬ
@app.post("/predict")
def predict(data: PredictInput):

    start_time = time.time()

    # Kiểm tra user
    conn = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT UserID
            FROM Users
            WHERE UserID = %s
            """,
            (data.user_id,)
        )

        user = cursor.fetchone()

        if not user:
            cursor.close()
            conn.close()

            return {
                "success": False,
                "message": "Không tìm thấy tài khoản!"
            }

        # Dữ liệu đầu vào
        features = [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]]

        # Chuẩn hóa
        scaled_features = scaler.transform(
            features
        )

        # Dự đoán
        prediction = int(
            model.predict(
                scaled_features
            )[0]
        )

        prediction_name = species[
            prediction
        ]

        # Thời gian xử lý
        processing_time = (
            time.time() - start_time
        )

        # Thời gian Việt Nam
        vietnam_time = (
            datetime.now(timezone.utc)
            + timedelta(hours=7)
        ).replace(tzinfo=None)

        # Lưu vào database
        cursor.execute(
            """
            INSERT INTO Predictions
            (
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
            VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                data.user_id,
                data.sepal_length,
                data.sepal_width,
                data.petal_length,
                data.petal_width,
                "SVM",
                prediction_name,
                round(
                    processing_time,
                    4
                ),
                vietnam_time
            )
        )

        conn.commit()

        cursor.close()
        conn.close()

        return {
            "success": True,
            "prediction": prediction_name
        }

    except Exception as e:

        if conn:
            conn.rollback()
            conn.close()

        print("Lỗi dự đoán:", e)

        return {
            "success": False,
            "message": "Không thể lưu kết quả dự đoán!"
        }


# LỊCH SỬ DỰ ĐOÁN
@app.get("/history/{user_id}")
def history(user_id: int):

    conn = None

    try:
        conn = get_connection()

        cursor = conn.cursor(
            cursor_factory=RealDictCursor
        )

        cursor.execute(
            """
            SELECT
                PredictionID,
                UserID,
                SepalLength,
                SepalWidth,
                PetalLength,
                PetalWidth,
                ModelName,
                Prediction,
                ProcessingTime,
                CreatedAt
            FROM Predictions
            WHERE UserID = %s
            ORDER BY PredictionID DESC
            """,
            (user_id,)
        )

        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        history_data = []

        for item in rows:

            history_data.append({

                "prediction_id": item["predictionid"],

                "user_id": item["userid"],

                "sepal_length": item["sepallength"],

                "sepal_width": item["sepalwidth"],

                "petal_length": item["petallength"],

                "petal_width": item["petalwidth"],

                "model_name": item["modelname"],

                "prediction": item["prediction"],

                "processing_time": item["processingtime"],

                "created_at": item["createdat"].strftime(
                    "%d/%m/%Y %H:%M:%S"
                )
            })

        return {

            "success": True,

            "user_id": user_id,

            "history": history_data

        }

    except Exception as e:

        if conn:
            conn.close()

        print("Lỗi lấy lịch sử:", e)

        return {

            "success": False,

            "user_id": user_id,

            "history": []

        }


# TẢI MODEL
model = joblib.load(
    os.path.join(
        BASE_DIR,
        "svm_model.pkl"
    )
)

scaler = joblib.load(
    os.path.join(
        BASE_DIR,
        "scaler.pkl"
    )
)

