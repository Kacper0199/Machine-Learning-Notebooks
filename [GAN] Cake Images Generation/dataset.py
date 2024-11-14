import os
from torchvision import transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image


class CakeDataset(Dataset):
    def __init__(self, data_path, image_size=64, augmentation=False, augmentation_crop_scale_low=0.8):
        self.data_path = data_path
        self.image_files = [f for f in os.listdir(
            data_path) if f.endswith('.jpg')]

        if augmentation:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(
                    (image_size, image_size), scale=(augmentation_crop_scale_low, 1)),
                transforms.ToTensor(),
                # Normalization to values [-1, 1]
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                # Normalization to values [-1, 1]
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load and transform image
        image_path = os.path.join(self.data_path, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        return self.transform(image)

    def show_samples(self, num_samples=6):
        plt.figure(figsize=(10, 5))
        for i in range(num_samples):
            img = self[i]  # Get image using __getitem__ method
            # Denormalization and transforming into format [H, W, C]
            img = (img * 0.5 + 0.5).permute(1, 2, 0)
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(img)
            plt.axis("off")
        plt.show()
