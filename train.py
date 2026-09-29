from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import joblib

# 1. TẢI DỮ LIỆU
iris = datasets.load_iris()
X, y = iris.data, iris.target

pastel_colors = ["#F7B7C8", "#A8D8B9", "#C9B6E4"]


# 2. CHIA DỮ LIỆU
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 3. CHUẨN HÓA DỮ LIỆU
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 4. HUẤN LUYỆN SVM
model = SVC(kernel="linear", probability=True)
model.fit(X_train_scaled, y_train)


# 5. ĐÁNH GIÁ MÔ HÌNH
y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, target_names=iris.target_names
))


# 6. CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

fig, ax = plt.subplots(figsize=(7, 6))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
).plot(
    ax=ax,
    cmap="Pastel2",
    colorbar=False,
    values_format="d"
)

for text in ax.texts:
    text.set_color("#444444")
    text.set_fontsize(16)
    text.set_fontweight("bold")

ax.set_xlabel("Loài hoa dự đoán", fontsize=11)
ax.set_ylabel("Loài hoa thực tế", fontsize=11)
ax.set_title("Ma trận nhầm lẫn - SVM", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()


# 7. PHÂN BỐ DỮ LIỆU
plt.figure(figsize=(8, 6))

for i, name in enumerate(iris.target_names):
    plt.scatter(
        X[y == i, 2], X[y == i, 3],
        label=name.capitalize(),
        color=pastel_colors[i],
        s=55, alpha=0.85,
        edgecolors="white", linewidths=0.8
    )

plt.xlabel("Petal Length (Chiều dài cánh hoa)", fontsize=11)
plt.ylabel("Petal Width (Chiều rộng cánh hoa)", fontsize=11)
plt.title("Phân bố dữ liệu Iris", fontsize=14, fontweight="bold")
plt.legend()
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("iris_visualization.png", dpi=300, bbox_inches="tight")
plt.show()


# 8. ĐƯỜNG PHÂN CHIA TUYẾN TÍNH
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

Z = model_2d.predict(
    np.c_[xx.ravel(), yy.ravel()]
).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.18, cmap="Pastel2")

for i, name in enumerate(iris.target_names):
    plt.scatter(
        X_2d_scaled[y == i, 0],
        X_2d_scaled[y == i, 1],
        label=name.capitalize(),
        color=pastel_colors[i],
        s=55, alpha=0.9,
        edgecolors="white", linewidths=0.8
    )

plt.contour(
    xx, yy, Z,
    levels=[0.5, 1.5],
    colors="#8C7AA9",
    linewidths=1.2
)

plt.xlabel("Petal Length (Chiều dài cánh hoa)", fontsize=11)
plt.ylabel("Petal Width (Chiều rộng cánh hoa)", fontsize=11)
plt.title("Đường phân chia tuyến tính của SVM", fontsize=14, fontweight="bold")
plt.legend()
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("svm_decision_boundary.png", dpi=300, bbox_inches="tight")
plt.show()


# 9. SCATTER MATRIX
feature_names = [
    "Sepal Length",
    "Sepal Width",
    "Petal Length",
    "Petal Width"
]

fig, axes = plt.subplots(4, 4, figsize=(12, 12))

for row in range(4):
    for col in range(4):
        ax = axes[row, col]

        for i, name in enumerate(iris.target_names):
            if row == col:
                ax.hist(
                    X[y == i, col],
                    bins=10,
                    alpha=0.6,
                    color=pastel_colors[i],
                    label=name.capitalize()
                )
            else:
                ax.scatter(
                    X[y == i, col],
                    X[y == i, row],
                    color=pastel_colors[i],
                    s=20,
                    alpha=0.7,
                    edgecolors="white",
                    linewidths=0.4
                )

        ax.grid(True, alpha=0.15)

        if row == 3:
            ax.set_xlabel(feature_names[col], fontsize=9)

        if col == 0:
            ax.set_ylabel(feature_names[row], fontsize=9)

plt.suptitle(
    "Scatter Matrix - Mối quan hệ giữa các đặc trưng Iris",
    fontsize=16,
    fontweight="bold",
    y=0.995
)

plt.tight_layout()
plt.savefig("scatter_matrix.png", dpi=300, bbox_inches="tight")
plt.show()


# 10. HUẤN LUYỆN LẠI TOÀN BỘ DỮ LIỆU
X_scaled = scaler.fit_transform(X)
model.fit(X_scaled, y)


# 11. LƯU MÔ HÌNH
joblib.dump(model, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nĐã lưu mô hình và scaler thành công!")