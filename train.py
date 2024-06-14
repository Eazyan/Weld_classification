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



import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from data import datas, train_dataloader
import os

# Загрузка предобученной модели
model_name = 'th041/vit-weld-classify'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# Изменение выходного слоя для многоклассовой классификации
num_classes = len(datas['Defect'].unique())  # Количество классов дефектов
model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)

# Определение гиперпараметров и подготовка к обучению
learning_rate = 5e-5
num_epochs = 5
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Обучение модели
device = torch.device('cuda' if torch.cuda.is_available() else 'mps') #Использование Nvidia CUDA или Apple Metal
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_dataloader)}")

# Сохранение дообученной модели
save_path = "model"
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
feature_extractor.save_pretrained(save_path)

print(f"Model and feature extractor saved to {save_path}")


# ПРОВЕРЬ, ЕСТЬ ЛИ НОВЫЙ СКРЫТЫЙ СЛОЙ!