from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

iris = datasets.load_iris()

X = iris.data
y = iris.target

# 1. Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 2. Chuẩn hóa dữ liệu
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Khởi tạo mô hình SVM
model = SVC(kernel="linear")

# 4. Huấn luyện mô hình
model.fit(X_train_scaled, y_train)

# 5. Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test_scaled)

# 6. Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))

# 7. Chuẩn hóa toàn bộ dữ liệu
X_scaled = scaler.fit_transform(X)

# 8. Huấn luyện lại mô hình trên toàn bộ dữ liệu
model.fit(X_scaled, y)

# 9. Lưu mô hình và bộ chuẩn hóa
joblib.dump(model, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved!")