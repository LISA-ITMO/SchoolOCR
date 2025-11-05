import os
import numpy as np
import keras
from keras import layers
from keras import regularizers
import ssl
import requests
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

requests.packages.urllib3.disable_warnings()

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

num_classes = 12
input_shape = (28, 28, 1)
batch_size = 128
epochs = 150

def analyze_class_distribution(directory):
    """Анализирует распределение классов в директории"""
    class_counts = {}
    total_files = 0
    
    for class_name in sorted(os.listdir(directory)):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
            class_counts[class_name] = count
            total_files += count
    
    return class_counts, total_files

print("="*60)
print("АНАЛИЗ РАСПРЕДЕЛЕНИЯ КЛАССОВ")
print("="*60)

train_class_counts, nb_train_samples = analyze_class_distribution('dataset/train')
val_class_counts, nb_val_samples = analyze_class_distribution('dataset/val')
test_class_counts, nb_test_samples = analyze_class_distribution('dataset/test')

print("\nTrain set:")
for class_name, count in train_class_counts.items():
    print(f"  Класс {class_name}: {count} образцов ({count/nb_train_samples*100:.2f}%)")
print(f"  Всего: {nb_train_samples}")

print("\nValidation set:")
for class_name, count in val_class_counts.items():
    print(f"  Класс {class_name}: {count} образцов ({count/nb_val_samples*100:.2f}%)")
print(f"  Всего: {nb_val_samples}")

print("\nTest set:")
for class_name, count in test_class_counts.items():
    print(f"  Класс {class_name}: {count} образцов ({count/nb_test_samples*100:.2f}%)")
print(f"  Всего: {nb_test_samples}")

class_names = sorted(train_class_counts.keys())
class_counts_array = np.array([train_class_counts[c] for c in class_names])

class_weights_inverse = {}
total_samples = sum(class_counts_array)
for i, class_name in enumerate(class_names):
    class_weights_inverse[i] = total_samples / (num_classes * class_counts_array[i])

class_labels = []
for i, class_name in enumerate(class_names):
    class_labels.extend([i] * train_class_counts[class_name])
class_weights_sklearn = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(class_labels),
    y=class_labels
)
class_weights_balanced = {i: class_weights_sklearn[i] for i in range(num_classes)}

print("\n" + "="*60)
print("ВЕСА КЛАССОВ (для компенсации дисбаланса)")
print("="*60)
for i, class_name in enumerate(class_names):
    print(f"  Класс {class_name}: {class_weights_balanced[i]:.4f}")

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    shear_range=0.15,
    fill_mode='nearest'
)

val_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(28, 28),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=True
)

val_generator = val_test_datagen.flow_from_directory(
    'dataset/val',
    target_size=(28, 28),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(28, 28),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=False
)

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * y_true * tf.pow((1 - y_pred), self.gamma)
        focal_loss = weight * cross_entropy
        
        return tf.reduce_sum(focal_loss, axis=1)

def create_improved_model():
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        
        layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation="relu", 
                     kernel_regularizer=regularizers.L2(0.0001)),
        layers.BatchNormalization(),
        layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation="relu",
                     kernel_regularizer=regularizers.L2(0.0001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation="relu",
                     kernel_regularizer=regularizers.L2(0.0001)),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation="relu",
                     kernel_regularizer=regularizers.L2(0.0001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation="relu",
                     kernel_regularizer=regularizers.L2(0.0001)),
        layers.BatchNormalization(),
        layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation="relu",
                     kernel_regularizer=regularizers.L2(0.0001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(256, activation="relu", kernel_regularizer=regularizers.L2(0.0001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.L2(0.0001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    
    return model

model = create_improved_model()

metrics = [
    "accuracy",
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
]

USE_FOCAL_LOSS = False

if USE_FOCAL_LOSS:
    print("\nИспользуется Focal Loss для борьбы с дисбалансом классов")
    model.compile(
        loss=FocalLoss(gamma=2.0, alpha=0.25),
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001),
        metrics=metrics
    )
    class_weight_to_use = None 
else:
    print("\nИспользуется Categorical Crossentropy + Class Weights")
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001),
        metrics=metrics
    )
    class_weight_to_use = class_weights_balanced

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=25, 
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='mnist/best_model_balanced.h5', 
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=1
    ),
]

print("\n" + "="*60)
print("НАЧАЛО ОБУЧЕНИЯ")
print("="*60)

history = model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=nb_val_samples // batch_size,
    callbacks=callbacks,
    class_weight=class_weight_to_use,
    verbose=1
)

model.load_weights('mnist/best_model_balanced.h5')

print("\n" + "="*60)
print("ОЦЕНКА НА ТЕСТОВОМ НАБОРЕ")
print("="*60)
scores = model.evaluate(test_generator, steps=nb_test_samples // batch_size, verbose=1)

print("\n" + "="*60)
print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
print("="*60)
print(f"Test Loss: {scores[0]:.4f}")
print(f"Test Accuracy: {scores[1]:.4f} ({scores[1]*100:.2f}%)")
print(f"Test Precision: {scores[2]:.4f}")
print(f"Test Recall: {scores[3]:.4f}")
print(f"Test AUC: {scores[4]:.4f}")
print("="*60)

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

test_generator.reset()
y_pred_probs = model.predict(test_generator, steps=nb_test_samples // batch_size + 1, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

y_true = test_generator.classes[:len(y_pred)]

print("\n" + "="*60)
print("ДЕТАЛЬНЫЙ ОТЧЕТ ПО КЛАССАМ")
print("="*60)
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('mnist/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\nMatрица ошибок сохранена в 'mnist/confusion_matrix.png'")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(history.history['precision'], label='Train Precision', linewidth=2)
axes[1, 0].plot(history.history['val_precision'], label='Val Precision', linewidth=2)
axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(history.history['recall'], label='Train Recall', linewidth=2)
axes[1, 1].plot(history.history['val_recall'], label='Val Recall', linewidth=2)
axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mnist/training_history_balanced.png', dpi=300, bbox_inches='tight')
print("Графики обучения сохранены в 'mnist/training_history_balanced.png'")

print("\n" + "="*60)
print("АНАЛИЗ ТОЧНОСТИ ПО КЛАССАМ")
print("="*60)
for i, class_name in enumerate(class_names):
    class_indices = np.where(y_true == i)[0]
    if len(class_indices) > 0:
        class_accuracy = np.mean(y_pred[class_indices] == y_true[class_indices])
        print(f"Класс {class_name}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%) - {len(class_indices)} образцов")