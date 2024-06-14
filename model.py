# import torch
# from transformers import ViTForImageClassification, ViTFeatureExtractor, ViTImageProcessor
# from PIL import Image
# import os

# # Загрузка модели и feature extractor
# model_name = 'th041/vit-weld-classify'
# # Загрузка конфигурации ViTImageProcessor, если она была изменена и сохранена
# # feature_extractor = ViTImageProcessor.from_pretrained('model/')
# feature_extractor = ViTImageProcessor.from_pretrained(model_name)
# # Загрузка модели
# # model = ViTForImageClassification.from_pretrained('model/')
# # Загрузите модель без классификатора
# # model = ViTForImageClassification.from_pretrained('model/', ignore_mismatched_sizes=True)
# model = ViTForImageClassification.from_pretrained(model_name)


# # Замените классификатор на новый с правильным количеством классов
# num_classes = 2  # Замените на актуальное количество классов
# model.classifier = torch.nn.Linear(model.config.hidden_size, num_classes)


# # Подготовка изображения
# def prepare_image(image_path):
#     image = Image.open(image_path).convert("RGB")
#     inputs = feature_extractor(images=image, return_tensors="pt")
#     return inputs

# # Классификация изображения
# def classify_image(image_path):
#     inputs = prepare_image(image_path)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     logits = outputs.logits
#     predicted_class_idx = logits.argmax(-1).item()
    
#     # Словарь с именами классов
#     class_names = {0: "Дефект отсутствует", 1: "Дефект есть"}
#     print(class_names[predicted_class_idx])
#     return class_names[predicted_class_idx]
#     # return predicted_class_idx

# # Пример использования
# image_path = 'uploads/Welding.jpg'  # Замените на путь к вашему изображению
# result = classify_image(image_path)
# print(f'Оценка: {result}')


import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import os
import numpy as np
import random

# Функция для установки всех seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Устанавливаем фиксированный seed
set_seed(42)

# Загрузка модели и feature extractor
model_name = 'th041/vit-weld-classify'
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# Замените классификатор на новый с правильным количеством классов
num_classes = 2  # Замените на актуальное количество классов
model.classifier = torch.nn.Linear(model.config.hidden_size, num_classes)

# Переключаем модель в режим оценки
model.eval()

# Подготовка изображения
def prepare_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs

# Классификация изображения
def classify_image(image_path):
    if not os.path.exists(image_path):
        print(f"Файл не найден: {image_path}")
        return "Файл не найден"
    
    inputs = prepare_image(image_path)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    
    # Словарь с именами классов
    class_names = {1: "Дефект отсутствует", 0: "Дефект есть"}
    result = class_names.get(predicted_class_idx, "Неизвестно")
    print(result)
    return result

# Пример использования
image_path = 'uploads/Welding3.jpg'  # Замените на путь к вашему изображению
result = classify_image(image_path)
print(f'Оценка: {result}')
