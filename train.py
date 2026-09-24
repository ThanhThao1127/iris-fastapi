from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import joblib

# 1. Tải dữ liệu
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 2. Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Huấn luyện SVM
model = SVC(kernel="linear")
model.fit(X_train_scaled, y_train)

# 5. Dự đoán và đánh giá
y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 6. Confusion Matrix(ma trận nhầm lẫn)
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot()

plt.xlabel("Loài hoa dự đoán")
plt.ylabel("Loài hoa thực tế")
plt.title("Ma trận nhầm lẫn - SVM")

plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# 7. Phân bố dữ liệu
plt.figure(figsize=(8, 6))

for i, name in enumerate(iris.target_names):
    plt.scatter(X[y == i, 2], X[y == i, 3], label=name.capitalize())

plt.xlabel("Petal Length (Chiều dài cánh hoa)")
plt.ylabel("Petal Width (Chiều rộng cánh hoa)")
plt.title("Phân bố dữ liệu Iris")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("iris_visualization.png", dpi=300)
plt.show()

# 8. Đường phân chia tuyến tính
X_2d = X[:, [2, 3]]
scaler_2d = StandardScaler()
X_2d_scaled = scaler_2d.fit_transform(X_2d)

model_2d = SVC(kernel="linear")
model_2d.fit(X_2d_scaled, y)

x_min, x_max = X_2d_scaled[:, 0].min() - 1, X_2d_scaled[:, 0].max() + 1
y_min, y_max = X_2d_scaled[:, 1].min() - 1, X_2d_scaled[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.2)

for i, name in enumerate(iris.target_names):
    plt.scatter(
        X_2d_scaled[y == i, 0],
        X_2d_scaled[y == i, 1],
        label=name.capitalize()
    )

plt.contour(xx, yy, Z, levels=[0.5, 1.5], linewidths=1)
plt.xlabel("Petal Length (Chiều dài cánh hoa)")
plt.ylabel("Petal Width (Chiều rộng cánh hoa)")
plt.title("Đường phân chia tuyến tính")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("svm_decision_boundary.png", dpi=300)
plt.show()

# 9. Huấn luyện lại trên toàn bộ dữ liệu
X_scaled = scaler.fit_transform(X)
model.fit(X_scaled, y)

# 10. Lưu mô hình
joblib.dump(model, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")
