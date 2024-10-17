import os
import numpy as np
import torch
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2

class ImageLabelDataset(Dataset):
    def __init__(self, image_folder, label_folder,transform=None):
        self.image_folder = image_folder

        self.label_folder = label_folder
        self.transform = transform

        self.image_filenames = os.listdir(image_folder)

        self.label_filenames = os.listdir(label_folder)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        label_name = self.label_filenames[idx]

        image_path = os.path.join(self.image_folder, image_name)
        label_path = os.path.join(self.label_folder, label_name)

        image = cv2.imread(image_path)
        label = cv2.imread(label_path, 0)

        if label is None:
            # 可以在这里进行异常处理，比如跳过当前样本
            print(f"Failed to read label image: {label_path}")
            return None

        label = label / 255.0

        image = cv2.resize(image, (256, 256))
        label = cv2.resize(label, (256, 256))

        if self.transform:
            image = self.transform(image)

        label = torch.from_numpy(label).long()

        return image, label


def prepare_dataloader(image_folder, label_folder, batch_size, shuffle, num_workers=4):
    transform = ToTensor()  # 可以根据需要添加其他数据增强操作

    dataset = ImageLabelDataset(image_folder, label_folder, transform=transform)

    valid_indices = []
    for idx, filename in enumerate(dataset.image_filenames):
        result = dataset.__getitem__(idx)
        if result is not None:
            valid_indices.append(idx)

    filtered_image_filenames = [dataset.image_filenames[i] for i in valid_indices]
    dataset.image_filenames = filtered_image_filenames

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loader