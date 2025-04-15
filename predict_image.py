import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import sys
import os

def predict_image(model_path, image_path):
    """
    Загружает отдельное изображение и выполняет предсказание с помощью модели.
    
    Аргументы:
        model_path: Путь к файлу модели (.h5)
        image_path: Путь к изображению, которое нужно классифицировать
    
    Возвращает:
        Предсказание (0 или 1) и вероятность
    """
    # Загрузка обученной модели
    model = load_model(model_path)
    
    # Загрузка изображения и изменение размера до того, который ожидает модель (150x150)
    img = load_img(image_path, target_size=(150, 150))
    
    # Преобразование изображения в массив и нормализация
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Нормализация до [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Добавление размерности батча
    
    # Выполнение предсказания
    prediction = model.predict(img_array)
    probability = prediction[0][0]
    
    # Интерпретация предсказания (настройте согласно классам вашей модели)
    class_prediction = 1 if probability >= 0.5 else 0
    class_name = "Моя функция" if class_prediction == 0 else "Не моя функция"
    
    # Отображение изображения с предсказанием
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f'Предсказание: {class_name}\nВероятность: {probability:.4f}')
    plt.axis('off')
    plt.show()
    
    return class_prediction, probability

if __name__ == "__main__":
    # Если скрипт запускается напрямую, можно передать путь к изображению как аргумент
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        model_path = "model.h5"  # Настройте этот путь, если необходимо
        
        if not os.path.exists(image_path):
            print(f"Ошибка: Изображение не найдено по пути {image_path}")
            sys.exit(1)
            
        if not os.path.exists(model_path):
            print(f"Ошибка: Модель не найдена по пути {model_path}")
            sys.exit(1)
            
        prediction, probability = predict_image(model_path, image_path)
        print(f"Предсказание: {'Моя функция' if prediction == 0 else 'Не моя функция'}")
        print(f"Вероятность: {probability:.4f}")
    else:
        print("Использование: python predict_image.py путь/к/изображению.jpg")
