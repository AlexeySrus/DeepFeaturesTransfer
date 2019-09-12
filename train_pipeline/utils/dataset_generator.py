import cv2
import os
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from train_pipeline.utils.edge_detector.edge_detector import EdgeDetector


def load_image(path):
    img = cv2.imread(path, 1)
    if img is None:
        print('IMAGGGGGGG', path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class ImagenetLoader(Dataset):
    def __init__(self, folder_path, size=(224, 224)):
        self.images_pathes = [
            os.path.join(
                os.path.join(folder_path, subfolder_name),
                img_name
            )
            for subfolder_name in os.listdir(
                folder_path
            )
            for img_name in os.listdir(
                os.path.join(folder_path, subfolder_name)
            )
        ]

        self.size = size

        self.edge_detector = EdgeDetector()

    def __len__(self):
        return len(self.images_pathes)

    def __getitem__(self, idx):
        image = load_image(self.images_pathes[idx])
        image = cv2.resize(image, self.size)
        edge = self.edge_detector.detect(image)

        image = image.transpose(2, 0, 1)

        return torch.FloatTensor(image) / 255.0 - 0.5, \
               torch.FloatTensor(edge).unsqueeze(0) / 255.0 - 0.5
