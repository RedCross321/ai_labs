import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Константы
IMG_SIZE = (128, 128)  # Размер, к которому будем приводить все изображения
BATCH_SIZE = 32
EPOCHS = 15
TARGET_CLASS = "cubic_negative_a"  # Наш целевой класс (y=ax^3+bx^2+cx+d, a < 0)

def load_images_from_folder(folder_path, label):
    """Загрузка изображений из папки и присвоение им метки класса"""
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            try:
                # Загружаем изображение и преобразуем его в RGB (если оно в другом формате)
                img = Image.open(img_path).convert('RGB')
                # Изменяем размер изображения
                img = img.resize(IMG_SIZE)
                # Конвертируем в массив numpy
                img_array = np.array(img) / 255.0  # Нормализация
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Ошибка при загрузке {img_path}: {e}")
    
    return np.array(images), np.array(labels)

def create_model():
    """Создание модели нейронной сети"""
    # Используем предобученную модель MobileNetV2 без верхних слоев
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base_model.trainable = False  # Замораживаем веса базовой модели
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Бинарная классификация
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def visualize_predictions(model, test_images, test_labels, class_names, num_samples=10):
    """Визуализация предсказаний модели"""
    # Получаем предсказания
    predictions = model.predict(test_images)
    predictions = (predictions > 0.5).astype(int)
    
    # Выбираем случайные образцы для визуализации
    indices = np.random.randint(0, len(test_images), num_samples)
    
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i+1)
        plt.imshow(test_images[idx])
        true_label = class_names[test_labels[idx]]
        pred_label = class_names[predictions[idx][0]]
        title_color = 'green' if true_label == pred_label else 'red'
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=title_color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_results.png')
    plt.close()

def main():
    # Пути к папкам с данными
    data_folders = [
        os.path.join('d:/Учеба/тек семак/ai', str(i)) for i in range(1, 7)
    ]
    
    # Предполагаем, что одна из папок содержит наш целевой класс (y=ax^3+bx^2+cx+d, a < 0)
    # Например, папка №3 содержит кубические функции с отрицательным a
    target_folder = os.path.join('d:/Учеба/тек семак/ai', '3')  # Предполагаемая папка с целевым классом
    
    # Загрузка изображений целевого класса
    print(f"Загрузка изображений целевого класса из {target_folder}")
    target_images, target_labels = load_images_from_folder(target_folder, 1)  # 1 - принадлежит целевому классу
    
    # Загрузка изображений других классов
    other_images = []
    other_labels = []
    for folder in data_folders:
        if folder != target_folder:  # Пропускаем целевую папку
            print(f"Загрузка изображений из {folder}")
            imgs, lbls = load_images_from_folder(folder, 0)  # 0 - не принадлежит целевому классу
            other_images.append(imgs)
            other_labels.append(lbls)
    
    # Объединяем все изображения не целевого класса
    other_images = np.vstack(other_images)
    other_labels = np.hstack(other_labels)
    
    # Объединяем все данные
    all_images = np.vstack([target_images, other_images])
    all_labels = np.hstack([target_labels, other_labels])
    
    print(f"Всего изображений: {len(all_images)}")
    print(f"Изображений целевого класса: {np.sum(all_labels == 1)}")
    print(f"Изображений не целевого класса: {np.sum(all_labels == 0)}")
    
    # Разделение на обучающую, валидационную и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Создание и обучение модели
    model = create_model()
    
    # Аугментация данных для обучающей выборки
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    
    # Обучение модели
    print("Обучение модели...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val)
    )
    
    # Оценка модели
    print("Оценка модели...")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Точность на тестовой выборке: {test_acc:.4f}")
    
    # Предсказания на тестовой выборке
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int)
    
    # Метрики качества
    print("\nОтчет о классификации:")
    print(classification_report(y_test, y_pred_classes))
    
    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred_classes)
    print("Матрица ошибок:")
    print(cm)
    
    # Визуализация результатов обучения
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Точность на обучении')
    plt.plot(history.history['val_accuracy'], label='Точность на валидации')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Потери на обучении')
    plt.plot(history.history['val_loss'], label='Потери на валидации')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    # Визуализация предсказаний
    class_names = {0: "Другая функция", 1: TARGET_CLASS}
    visualize_predictions(model, X_test, y_test, class_names)
    
    # Сохранение модели
    model.save('function_classifier.h5')
    print("Модель сохранена как 'function_classifier.h5'")
    
    # Создание изображений с подписями для тестовой выборки
    test_output_dir = 'test_results'
    os.makedirs(test_output_dir, exist_ok=True)
    
    for i in range(len(X_test)):
        img = X_test[i]
        true_label = class_names[y_test[i]]
        pred_label = class_names[y_pred_classes[i][0]]
        
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f"True: {true_label}\nPredicted: {pred_label}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(test_output_dir, f'test_img_{i}.png'))
        plt.close()

if __name__ == "__main__":
    main()
