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


from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import io
import cv2
import numpy as np
import base64
import uvicorn
import random

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
def prepare_image(image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs

# Функция для классификации изображений
def classify_image(image):
    inputs = prepare_image(image)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    class_names = {1: "Дефект отсутствует", 0: "Дефект есть"}
    return class_names.get(predicted_class_idx, "Неизвестно")

# Функция для выделения шва с помощью OpenCV
def detect_weld(image: Image.Image):
    open_cv_image = np.array(image)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)

    # Увеличим контрастность с помощью CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Применим размытие для удаления шумов
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Используем метод Canny для выделения краев
    edges = cv2.Canny(blurred, 50, 150)

    # Применим морфологические операции для удаления мелких шумов
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Найдем контуры
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(open_cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped_image = open_cv_image[y:y + h, x:x + w]
        return cropped_image, open_cv_image
    else:
        return open_cv_image, open_cv_image

@app.post("/")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    cropped_image, annotated_image = detect_weld(image)
    result = classify_image(cropped_image)

    _, buffer = cv2.imencode('.jpg', annotated_image)
    annotated_image_encoded = base64.b64encode(buffer).decode('utf-8')

    return JSONResponse(content={"result": result, "annotated_image": annotated_image_encoded})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
