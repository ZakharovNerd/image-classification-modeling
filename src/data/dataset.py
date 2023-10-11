import os

from PIL import Image
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, image_names, path, ohe_encoder, transform=None):
        super().__init__()
        self.image_names = image_names
        self.path = path
        self.ohe_encoder = ohe_encoder
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # join image path with image name to get image path
        filename = self.image_names[idx] + ".jpg"
        img_path = os.path.join(self.path, filename)
        # open image using image path
        image = Image.open(img_path).convert("RGB")
        # encode labels to be able to process it
        label_encoded = self.ohe_encoder[idx]
        # move to device
        if self.transform:
            image = self.transform(image)
        return image, label_encoded
