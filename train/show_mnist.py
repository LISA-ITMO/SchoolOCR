import numpy as np
import keras
import matplotlib.pyplot as plt

# Загрузка модели и данных
model = keras.models.load_model("mnist_with_x_model.keras")
(x_test, y_test) = np.load("x_test_with_x.npy"), np.load("y_test_with_x.npy")
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'X']

# Выбираем 10 случайных изображений 'X'
x_indices = np.where(np.argmax(y_test, axis=1) == 10)[0]
selected_indices = np.random.choice(x_indices, size=10, replace=False)

# Создаем фигуру
plt.figure(figsize=(15, 8))

for i, idx in enumerate(selected_indices):
    # Получаем изображение и предсказание
    image = x_test[idx]
    true_label = class_names[np.argmax(y_test[idx])]
    pred = model.predict(np.expand_dims(image, axis=0), verbose=0)
    pred_class = class_names[np.argmax(pred)]
    confidence = np.max(pred)

    # Отображаем изображение
    plt.subplot(3, 10, i + 1)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"True: {true_label}")
    plt.axis('off')

    # Отображаем предсказание
    plt.subplot(3, 10, i + 11)
    bars = plt.bar(class_names, pred[0], color=['r' if c == pred_class else 'b' for c in class_names])
    plt.xticks(rotation=90)
    plt.title(f"Pred: {pred_class}\n({confidence:.1%})")
    plt.ylim(0, 1)
    if i == 0:
        plt.ylabel("Probability")

plt.tight_layout()
plt.show()