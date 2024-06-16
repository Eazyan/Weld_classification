# Первый с прямогульниками___________________________________________________

# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# import torch
# from transformers import ViTForImageClassification, ViTImageProcessor
# from PIL import Image
# import io
# import cv2
# import numpy as np
# import base64
# import uvicorn
# import random

# # Инициализация FastAPI
# app = FastAPI()

# # Настройка CORS (если необходимо)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Функция для установки всех seeds
# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# # Устанавливаем фиксированный seed
# set_seed(42)

# # Загрузка модели и feature extractor
# model_name = 'th041/vit-weld-classify'
# feature_extractor = ViTImageProcessor.from_pretrained(model_name)
# model = ViTForImageClassification.from_pretrained(model_name)

# # Замените классификатор на новый с правильным количеством классов
# num_classes = 2  # Замените на актуальное количество классов
# model.classifier = torch.nn.Linear(model.config.hidden_size, num_classes)

# # Переключаем модель в режим оценки
# model.eval()

# # Подготовка изображения
# def prepare_image(image):
#     inputs = feature_extractor(images=image, return_tensors="pt")
#     return inputs

# # Функция для классификации изображений
# def classify_image(image):
#     inputs = prepare_image(image)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     logits = outputs.logits
#     predicted_class_idx = logits.argmax(-1).item()
#     class_names = {1: "Дефект отсутствует", 0: "Дефект есть"}
#     return class_names.get(predicted_class_idx, "Неизвестно")

# # Функция для выделения шва с помощью OpenCV
# def detect_weld(image: Image.Image):
#     open_cv_image = np.array(image)
#     gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
#     _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     if contours:
#         contour = max(contours, key=cv2.contourArea)
#         x, y, w, h = cv2.boundingRect(contour)
#         cv2.rectangle(open_cv_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         cropped_image = open_cv_image[y:y + h, x:x + w]
#         return cropped_image, open_cv_image
#     else:
#         return open_cv_image, open_cv_image

# @app.post("/")
# async def upload_file(file: UploadFile = File(...)):
#     contents = await file.read()
#     image = Image.open(io.BytesIO(contents)).convert("RGB")

#     cropped_image, annotated_image = detect_weld(image)
#     result = classify_image(cropped_image)

#     _, buffer = cv2.imencode('.jpg', annotated_image)
#     annotated_image_encoded = base64.b64encode(buffer).decode('utf-8')

#     return JSONResponse(content={"result": result, "annotated_image": annotated_image_encoded})

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)







#Слишком точно, в обмен на красоту________________________________________


# import torch
# from transformers import ViTForImageClassification, ViTImageProcessor
# from PIL import Image
# import io
# import cv2
# import numpy as np
# import random
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import base64

# app = FastAPI()

# # Настройка CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Разрешить все домены
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Установка фиксированного seed для воспроизводимости
# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# set_seed(42)

# model_name = 'th041/vit-weld-classify'
# feature_extractor = ViTImageProcessor.from_pretrained(model_name)
# model = ViTForImageClassification.from_pretrained(model_name)

# num_classes = 2  # Замените на актуальное количество классов
# model.classifier = torch.nn.Linear(model.config.hidden_size, num_classes)

# model.eval()

# def classify_image(image_bytes):
#     image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#     inputs = feature_extractor(images=image, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model(**inputs)
#     logits = outputs.logits
#     predicted_class_idx = logits.argmax(-1).item()
    
#     class_names = {1: "Дефект отсутствует", 0: "Дефект есть"}
#     result = class_names.get(predicted_class_idx, "Неизвестно")
#     return result, image

# def detect_and_draw_weld(image):
#     image_cv = np.array(image)
#     image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    
#     # Преобразуем в серый и улучшаем контрастность
#     gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     gray = clahe.apply(gray)
    
#     # Применяем гауссово размытие для уменьшения шума
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
#     # Применяем адаптивную бинаризацию
#     _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     # Применяем морфологические операции для улучшения выделения контуров
#     kernel = np.ones((5, 5), np.uint8)
#     morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
#     # Применяем фильтр Канни для детектирования краев
#     edges = cv2.Canny(morph, 50, 150)
    
#     # Найти контуры на изображении
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Отфильтровать контуры по площади
#     min_contour_area = 100  # минимальная площадь контура, которую будем учитывать
#     contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
#     # Найти самый большой контур
#     if contours:
#         largest_contour = max(contours, key=cv2.contourArea)
#         cv2.drawContours(image_cv, [largest_contour], -1, (0, 255, 0), 2)
    
#     return image_cv

# @app.post("/")
# async def classify_weld(file: UploadFile = File(...)):
#     try:
#         image_bytes = await file.read()
#         result, image = classify_image(image_bytes)
        
#         # Обработка изображения для обводки шва
#         annotated_image_cv = detect_and_draw_weld(image)
        
#         _, buffer = cv2.imencode('.jpg', annotated_image_cv)
#         annotated_image = base64.b64encode(buffer).decode('utf-8')
        
#         return JSONResponse(content={"result": result, "annotated_image": annotated_image})
#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)

# if __name__ == "__main__":
#     uvicorn.run(
#         app, 
#         host="0.0.0.0", 
#         port=8000, 
#         # ssl_certfile="/path/to/your/fullchain.pem", 
#         # ssl_keyfile="/path/to/your/privkey.pem"
#     )




# Улучшенные прямоугольники_________________________________________


# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# import torch
# from transformers import ViTForImageClassification, ViTImageProcessor
# from PIL import Image
# import io
# import cv2
# import numpy as np
# import base64
# import uvicorn
# import random

# # Инициализация FastAPI
# app = FastAPI()

# # Настройка CORS (если необходимо)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Функция для установки всех seeds
# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# # Устанавливаем фиксированный seed
# set_seed(42)

# # Загрузка модели и feature extractor
# model_name = 'th041/vit-weld-classify'
# feature_extractor = ViTImageProcessor.from_pretrained(model_name)
# model = ViTForImageClassification.from_pretrained(model_name)

# # Замените классификатор на новый с правильным количеством классов
# num_classes = 2  # Замените на актуальное количество классов
# model.classifier = torch.nn.Linear(model.config.hidden_size, num_classes)

# # Переключаем модель в режим оценки
# model.eval()

# # Подготовка изображения
# def prepare_image(image):
#     inputs = feature_extractor(images=image, return_tensors="pt")
#     return inputs

# # Функция для классификации изображений
# def classify_image(image):
#     inputs = prepare_image(image)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     logits = outputs.logits
#     predicted_class_idx = logits.argmax(-1).item()
#     class_names = {1: "Дефект отсутствует", 0: "Дефект есть"}
#     return class_names.get(predicted_class_idx, "Неизвестно")

# # Функция для выделения шва с помощью OpenCV
# def detect_weld(image: Image.Image):
#     open_cv_image = np.array(image)
#     gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)

#     # Увеличим контрастность с помощью CLAHE
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     gray = clahe.apply(gray)

#     # Применим размытие для удаления шумов
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#     # Используем метод Canny для выделения краев
#     edges = cv2.Canny(blurred, 50, 150)

#     # Применим морфологические операции для удаления мелких шумов
#     kernel = np.ones((5, 5), np.uint8)
#     closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

#     # Найдем контуры
#     contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     if contours:
#         contour = max(contours, key=cv2.contourArea)
#         x, y, w, h = cv2.boundingRect(contour)
#         cv2.rectangle(open_cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cropped_image = open_cv_image[y:y + h, x:x + w]
#         return cropped_image, open_cv_image
#     else:
#         return open_cv_image, open_cv_image

# @app.post("/")
# async def upload_file(file: UploadFile = File(...)):
#     contents = await file.read()
#     image = Image.open(io.BytesIO(contents)).convert("RGB")

#     cropped_image, annotated_image = detect_weld(image)
#     result = classify_image(cropped_image)

#     _, buffer = cv2.imencode('.jpg', annotated_image)
#     annotated_image_encoded = base64.b64encode(buffer).decode('utf-8')

#     return JSONResponse(content={"result": result, "annotated_image": annotated_image_encoded})

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)




















# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# import torch
# from transformers import ViTForImageClassification, ViTImageProcessor
# from PIL import Image
# import io
# import cv2
# import numpy as np
# import base64
# import uvicorn
# import random
# from typing import List

# # Инициализация FastAPI
# app = FastAPI()

# # Настройка CORS (если необходимо)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Функция для установки всех seeds
# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# # Устанавливаем фиксированный seed
# set_seed(42)

# # Загрузка модели и feature extractor
# model_name = 'th041/vit-weld-classify'
# feature_extractor = ViTImageProcessor.from_pretrained(model_name)
# model = ViTForImageClassification.from_pretrained(model_name)

# # Замените классификатор на новый с правильным количеством классов
# num_classes = 2  # Замените на актуальное количество классов
# model.classifier = torch.nn.Linear(model.config.hidden_size, num_classes)

# # Переключаем модель в режим оценки
# model.eval()

# # Подготовка изображения
# def prepare_image(image):
#     inputs = feature_extractor(images=image, return_tensors="pt")
#     return inputs

# # Функция для классификации изображений
# def classify_image(image):
#     inputs = prepare_image(image)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     logits = outputs.logits
#     predicted_class_idx = logits.argmax(-1).item()
#     class_names = {1: "Дефект отсутствует", 0: "Дефект есть"}
#     return class_names.get(predicted_class_idx, "Неизвестно")

# # Функция для выделения шва с помощью OpenCV
# def detect_weld(image: Image.Image):
#     open_cv_image = np.array(image)
#     gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
#     _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     if contours:
#         contour = max(contours, key=cv2.contourArea)
#         x, y, w, h = cv2.boundingRect(contour)
#         cv2.rectangle(open_cv_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         cropped_image = open_cv_image[y:y + h, x:x + w]
#         return cropped_image, open_cv_image
#     else:
#         return open_cv_image, open_cv_image

# @app.post("/")
# async def upload_files(files: List[UploadFile] = File(...)):
#     results = []

#     for file in files:
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents)).convert("RGB")
        
#         cropped_image, annotated_image = detect_weld(image)
#         result = classify_image(cropped_image)
        
#         _, buffer = cv2.imencode('.jpg', annotated_image)
#         annotated_image_encoded = base64.b64encode(buffer).decode('utf-8')
        
#         results.append({"result": result, "annotated_image": annotated_image_encoded})

#     return JSONResponse(content=results)

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)














# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# import torch
# from PIL import Image
# import io
# import cv2
# import numpy as np
# import base64
# import uvicorn
# import random
# from typing import List
# from ultralytics import YOLO

# # Инициализация FastAPI
# app = FastAPI()

# # Настройка CORS (если необходимо)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Функция для установки всех seeds
# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# # Устанавливаем фиксированный seed
# set_seed(42)

# # Определение устройства
# device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

# # Загрузка обученной модели
# model = YOLO('modelTrained/best.pt')
# model.to(device)  # Перенос модели на устройство

# # Функция для подготовки изображения
# def prepare_image(image):
#     return np.array(image)

# # Функция для классификации изображений
# def classify_image(image):
#     image = image.to(device)  # Перенос изображения на устройство
#     results = model(image)
#     return results

# # Функция для выделения шва с помощью OpenCV
# def detect_weld(image: Image.Image):
#     open_cv_image = np.array(image)
#     gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
#     _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     if contours:
#         contour = max(contours, key=cv2.contourArea)
#         x, y, w, h = cv2.boundingRect(contour)
#         cv2.rectangle(open_cv_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         cropped_image = open_cv_image[y:y + h, x:x + w]
#         return cropped_image, open_cv_image
#     else:
#         return open_cv_image, open_cv_image

# @app.post("/")
# async def upload_files(files: List[UploadFile] = File(...)):
#     results = []

#     for file in files:
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents)).convert("RGB")
        
#         cropped_image, annotated_image = detect_weld(image)
#         result = classify_image(cropped_image)
        
#         _, buffer = cv2.imencode('.jpg', annotated_image)
#         annotated_image_encoded = base64.b64encode(buffer).decode('utf-8')
        
#         results.append({"result": result, "annotated_image": annotated_image_encoded})

#     return JSONResponse(content=results)

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)










# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# import torch
# from PIL import Image
# import io
# import cv2
# import numpy as np
# import base64
# from typing import List
# from ultralytics import YOLO

# # Инициализация FastAPI
# app = FastAPI()

# # Настройка CORS (если необходимо)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Определение устройства
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Загрузка обученной модели
# model = YOLO('modelTrained/best.pt')
# model.to(device)  # Перенос модели на устройство

# # Функция для подготовки изображения
# def prepare_image(image):
#     return np.array(image)

# # Функция для визуализации предсказаний
# def visualize_predictions(image, results):
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             label = result.names[int(box.cls)]
#             confidence = box.conf[0]
#             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(image, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#     return image

# @app.post("/")
# async def upload_files(files: List[UploadFile] = File(...)):
#     results = []

#     for file in files:
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents)).convert("RGB")
        
#         image_np = prepare_image(image)
#         prediction_results = model.predict(image_np, save=False)

#         annotated_image = visualize_predictions(image_np.copy(), prediction_results)

#         _, buffer = cv2.imencode('.jpg', annotated_image)
#         annotated_image_encoded = base64.b64encode(buffer).decode('utf-8')
        
#         # Формирование списка меток и координат для JSON ответа
#         labels_and_boxes = []
#         for result in prediction_results:
#             for box in result.boxes:
#                 labels_and_boxes.append({
#                     "label": result.names[int(box.cls)],
#                     "confidence": box.conf[0].item(),
#                     "coordinates": [int(coord) for coord in box.xyxy[0]]
#                 })

#         results.append({
#             "result": labels_and_boxes,
#             "annotated_image": annotated_image_encoded
#         })

#     return JSONResponse(content=results)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



#_____________________________________________________НОРМАЛЬНАЯ ВЕРСИЯ
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# import torch
# import cv2
# import numpy as np
# import base64
# import uvicorn
# from ultralytics import YOLO
# from typing import List
# import io
# from PIL import Image
# import matplotlib.pyplot as plt


# # Инициализация FastAPI
# app = FastAPI()

# # Настройка CORS (если необходимо)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Загрузить модель YOLOv8
# model_path = 'runs/detect/train12/weights/best.pt'  # Укажите путь к вашей обученной модели
# model = YOLO(model_path)

# # Словарь замены имен классов
# class_name_mapping = {
#     'adj': 'adj',  # пример замены, измените в соответствии с вашими классами
#     'int': 'Дефекты целостности',  # пример замены, измените в соответствии с вашими классами
#     'geo': 'Дефекты геометрии',      # пример замены, измените в соответствии с вашими классами
#     'pro': 'Дефекты постобработки',  # пример замены, измените в соответствии с вашими классами
#     'non': 'Дефекты невыполнения'  # пример замены, измените в соответствии с вашими классами
# }

# # Функция для рисования меток на изображении
# def plot_boxes(results, class_name_mapping):
#     img = results.orig_img
#     boxes = results.boxes.xyxy
#     scores = results.boxes.conf
#     classes = results.boxes.cls

#     for box, score, cls in zip(boxes, scores, classes):
#         label = class_name_mapping.get(model.names[int(cls)], 'unknown')
#         color = (0, 255, 0)  # Зеленый цвет для меток
#         x1, y1, x2, y2 = map(int, box)
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(img, f'{label} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#     return img

# @app.post("/")
# async def upload_files(files: List[UploadFile] = File(...)):
#     results = []

#     for file in files:
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents)).convert("RGB")
#         image_np = np.array(image)

#         # Обработать изображение
#         results_yolo = model(image_np)

#         # Визуализация меток на изображении
#         annotated_image = plot_boxes(results_yolo[0], class_name_mapping)

#         # Кодирование изображения в base64 для ответа
#         _, buffer = cv2.imencode('.jpg', annotated_image)
#         annotated_image_encoded = base64.b64encode(buffer).decode('utf-8')

#         # Словарь с результатами
#         result = {
#             "boxes": results_yolo[0].boxes.xyxy.tolist(),
#             "scores": results_yolo[0].boxes.conf.tolist(),
#             "classes": [class_name_mapping.get(model.names[int(cls)], 'unknown') for cls in results_yolo[0].boxes.cls],
#             "annotated_image": annotated_image_encoded
#         }

#         results.append(result)

#     return JSONResponse(content=results)

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)




#________________________________________________РАБОЧАЯ
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import cv2
import numpy as np
import base64
import uvicorn
from ultralytics import YOLO
from typing import List
import io
from PIL import Image, ImageDraw, ImageFont

# Инициализация FastAPI
app = FastAPI()

# Настройка CORS (если необходимо)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузить модель YOLOv8
model_path = 'runs/detect/train12/weights/best.pt'  # Укажите путь к вашей обученной модели
model = YOLO(model_path)

# Словарь замены имен классов
class_name_mapping = {
    'adj': 'Прилегающие дефекты',
    'int': 'Дефекты целостности',
    'geo': 'Дефекты геометрии',
    'pro': 'Дефекты постобработки',
    'non': 'Дефекты невыполнения'
}

# Функция для рисования меток на изображении
def plot_boxes(results, class_name_mapping):
    img = results.orig_img
    boxes = results.boxes.xyxy
    scores = results.boxes.conf
    classes = results.boxes.cls

    # Конвертируем изображение в PIL формат для поддержки кириллицы
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Загрузка шрифта, поддерживающего кириллицу
    font_path = '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'  # Укажите путь к шрифту, поддерживающему кириллицу
    font = ImageFont.truetype(font_path, 20)

    for box, score, cls in zip(boxes, scores, classes):
        label = class_name_mapping.get(model.names[int(cls)], 'unknown')
        color = (0, 255, 0)  # Зеленый цвет для меток
        x1, y1, x2, y2 = map(int, box)

        # Рисуем прямоугольник и текст на изображении
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, y1 - 25), f'{label} {score:.2f}', font=font, fill=color)

    # Конвертируем изображение обратно в формат OpenCV
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img

@app.post("/")
async def upload_files(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)

        # Обработать изображение
        results_yolo = model(image_np)

        # Визуализация меток на изображении
        annotated_image = plot_boxes(results_yolo[0], class_name_mapping)

        # Кодирование изображения в base64 для ответа
        _, buffer = cv2.imencode('.jpg', annotated_image)
        annotated_image_encoded = base64.b64encode(buffer).decode('utf-8')

        # Словарь с результатами
        result = {
            "boxes": results_yolo[0].boxes.xyxy.tolist(),
            "scores": results_yolo[0].boxes.conf.tolist(),
            "classes": [class_name_mapping.get(model.names[int(cls)], 'unknown') for cls in results_yolo[0].boxes.cls],
            "annotated_image": annotated_image_encoded
        }

        results.append(result)

    return JSONResponse(content=results)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


