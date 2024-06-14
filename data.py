# import pandas as pd
# from torchvision import transforms
# from torch.utils.data import DataLoader, Dataset
# from PIL import Image
# import os

# # Чтение CSV-файла
# csv_file = '/Users/eazyan/Documents/AtomHack/FirstTest/dataset/train/_classes.csv'
# data = pd.read_csv(csv_file)

# # Кастомный Dataset
# class WeldDefectsDataset(Dataset):
#     def __init__(self, csv_data, root_dir, transform=None):
#         self.csv_data = csv_data
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.csv_data)

#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir, self.csv_data.iloc[idx, 0])
#         image = Image.open(img_name).convert("RGB")
#         label = int(self.csv_data.iloc[idx, 1])
#         if self.transform:
#             image = self.transform(image)
#         return image, label

# # Трансформации
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

# # Создание Dataset и DataLoader
# root_dir = 'dataset'  # Корневая папка с подкаталогами train, test и valid
# train_data = data[data['filename'].str.contains('train')]
# valid_data = data[data['filename'].str.contains('valid')]
# test_data = data[data['filename'].str.contains('test')]

# train_dataset = WeldDefectsDataset(csv_data=train_data, root_dir=root_dir, transform=transform)
# valid_dataset = WeldDefectsDataset(csv_data=valid_data, root_dir=root_dir, transform=transform)
# test_dataset = WeldDefectsDataset(csv_data=test_data, root_dir=root_dir, transform=transform)

# train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
# test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)




import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# Чтение CSV-файла
csv_file = '/Users/eazyan/Documents/AtomHack/FirstTest/dataset/train/_classes.csv'
datas = pd.read_csv(csv_file)
# Удаление пробелов из названий столбцов
datas.columns = datas.columns.str.strip()

# Проверка содержимого CSV-файла
print(datas.head())

# Кастомный Dataset
class WeldDefectsDataset(Dataset):
    def __init__(self, csv_data, root_dir, transform=None):
        self.csv_data = csv_data
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.csv_data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = int(self.csv_data.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        return image, label

# Трансформации
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Создание Dataset и DataLoader
root_dir = 'dataset/train'  # Корневая папка с изображениями для обучения
train_dataset = WeldDefectsDataset(csv_data=datas, root_dir=root_dir, transform=transform)

# Проверка длины датасета
print(f"Train dataset length: {len(train_dataset)}")

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)



