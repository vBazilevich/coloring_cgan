import torch
import torchvision
import os
import cv2

class ImageColorizationDataset(torch.utils.data.Dataset):
    def __init__(self, path, transforms, label_transforms):
        super().__init__()
        self.transforms = transforms
        self.label_transforms = label_transforms
        self.labels = []
        for root, _, files in os.walk(path):
            for image in files:
                if image == '.gitignore':
                    continue
                img = cv2.imread(root + '/' + image)
                img = cv2.resize(img, (64, 64))
                self.labels.append(img)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img_lab = cv2.cvtColor(label, cv2.COLOR_RGB2Lab)
        inp = img_lab[:, :, 0] # Extract luma
        return self.transforms(inp), self.label_transforms(label)

    def __len__(self):
        return len(self.labels)
