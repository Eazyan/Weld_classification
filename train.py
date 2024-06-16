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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Загрузка и тренировка модели
model = YOLO('yolov10n.pt')
model.to(device)
model.train(data=os.path.join(dataset_path, 'data.yaml'), epochs=100, imgsz=640, device=device)

# Сохранение модели
model_path = 'modelTrained/best.pt'
os.makedirs(os.path.dirname(model_path), exist_ok=True)
model.save(model_path)