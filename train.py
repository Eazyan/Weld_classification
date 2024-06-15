# import torch
# from transformers import ViTForImageClassification, ViTFeatureExtractor
# from data import train_dataloader

# # Загрузка предобученной модели
# model_name = 'th041/vit-weld-classify'
# feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
# model = ViTForImageClassification.from_pretrained(model_name)

# # Изменение выходного слоя для многоклассовой классификации
# num_classes = 5  # Количество классов дефектов
# model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)

# # Определение гиперпараметров и подготовка к обучению
# learning_rate = 5e-5
# num_epochs = 10
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# criterion = torch.nn.CrossEntropyLoss()

# # Обучение модели
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for images, labels in train_dataloader:
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(images).logits
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()

#     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_dataloader)}")

# # Сохранение дообученной модели
# model.save_pretrained("/model")
# feature_extractor.save_pretrained("/model")



# import torch
# from transformers import ViTForImageClassification, ViTFeatureExtractor
# from data import datas, train_dataloader
# import os

# # Загрузка предобученной модели
# model_name = 'th041/vit-weld-classify'
# feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
# model = ViTForImageClassification.from_pretrained(model_name)

# # Изменение выходного слоя для многоклассовой классификации
# num_classes = len(datas['Defect'].unique())  # Количество классов дефектов
# model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)

# # Определение гиперпараметров и подготовка к обучению
# learning_rate = 5e-5
# num_epochs = 5
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# criterion = torch.nn.CrossEntropyLoss()

# # Обучение модели
# device = torch.device('cuda' if torch.cuda.is_available() else 'mps') #Использование Nvidia CUDA или Apple Metal
# model.to(device)

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for images, labels in train_dataloader:
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(images).logits
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()

#     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_dataloader)}")

# # Сохранение дообученной модели
# save_path = "model"
# os.makedirs(save_path, exist_ok=True)
# model.save_pretrained(save_path)
# feature_extractor.save_pretrained(save_path)

# print(f"Model and feature extractor saved to {save_path}")


# # ПРОВЕРЬ, ЕСТЬ ЛИ НОВЫЙ СКРЫТЫЙ СЛОЙ!

import os
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import shutil
import torch

# Путь к вашему датасету
dataset_path = 'dataset/'

# Путь к тренировочным и валидационным наборам данных
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')

# Разделение данных на тренировочные и валидационные
images = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
annotations = [f for f in os.listdir(dataset_path) if f.endswith('.txt') and f != 'classes.txt']

from sklearn.model_selection import train_test_split

# Предположим, что images - это список путей к изображениям
train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

print(f"Обучающая выборка: {len(train_images)} изображений")
print(f"Валидационная выборка: {len(val_images)} изображений")

train_images, val_images = train_test_split(images, test_size=0.2, train_size=0.8, random_state=42)

# Создание папок для тренировочных и валидационных данных
# os.makedirs(train_path, exist_ok=True)
# os.makedirs(val_path, exist_ok=True)

# Копирование файлов в соответствующие папки
# for image in train_images:
#     shutil.copy(os.path.join(dataset_path, image), os.path.join(train_path, image))
#     annotation = image.replace('.jpg', '.txt')
#     if annotation in annotations:
#         shutil.copy(os.path.join(dataset_path, annotation), os.path.join(train_path, annotation))

# for image in val_images:
#     shutil.copy(os.path.join(dataset_path, image), os.path.join(val_path, image))
#     annotation = image.replace('.jpg', '.txt')
#     if annotation in annotations:
#         shutil.copy(os.path.join(dataset_path, annotation), os.path.join(val_path, annotation))

# Создание YAML файла для конфигурации данных
data_yaml = """
train: {}/train
val: {}/val
nc: 5
names: ['adj', 'int', 'geo', 'pro', 'non']
""".format(dataset_path, dataset_path)

with open(os.path.join(dataset_path, 'data.yaml'), 'w') as f:
    f.write(data_yaml)

# Определение устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

# # Загрузка и тренировка модели
model = YOLO('yolov8n.pt')  # Путь к предобученной модели YOLOv8
model.to(device)  # Перенос модели на устройство
model.train(data=os.path.join(dataset_path, 'data.yaml'), epochs=1, imgsz=640, device=device)

# Сохранение модели
model_path = 'modelTrained/best.pt'
os.makedirs(os.path.dirname(model_path), exist_ok=True)
model.save(model_path)