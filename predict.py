import torch
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Загрузить модель YOLOv10
model_path = 'runs/detect/train12/weights/best.pt'
model = YOLO(model_path)

# Словарь замены имен классов
class_name_mapping = {
    'adj': 'Прилегающие дефекты', 
    'int': 'Дефекты целостности', 
    'geo': 'Дефекты геометрии',
    'pro': 'Дефекты постобработки', 
    'non': 'Дефекты невыполнения'
}

# Обработать изображение
image_path = 'dataset/dataset/1 (1).jpg'
results = model(image_path)

# Функция для рисования меток на изображении
def plot_boxes(results, class_name_mapping):
    img = results.orig_img
    boxes = results.boxes.xyxy
    scores = results.boxes.conf
    classes = results.boxes.cls

    for box, score, cls in zip(boxes, scores, classes):
        label = class_name_mapping.get(model.names[int(cls)], 'unknown')
        color = (0, 255, 0)  # Зеленый цвет для меток
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f'{label} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Вызвать функцию для рисования меток
plot_boxes(results[0], class_name_mapping)
