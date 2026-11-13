
import cv2
import numpy as np
from torch.utils.data import Dataset


IMAGE_SIZE = 256


def read_xray(path, image_size=IMAGE_SIZE):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Cannot read X-ray: {path}")

    image = cv2.resize(
        image, (image_size, image_size),
        interpolation=cv2.INTER_AREA
    )

    image = image.astype(np.float32) / 255.0
    return image[np.newaxis, :, :]


def read_mask(path, image_size=IMAGE_SIZE):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")

    mask = cv2.resize(
        mask, (image_size, image_size),
        interpolation=cv2.INTER_NEAREST
    )

    mask = (mask > 0).astype(np.float32)
    return mask[np.newaxis, :, :]


class KneeDataset(Dataset):

    def __init__(self, df, augment=False, image_size=IMAGE_SIZE):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image = read_xray(row["xrays"], self.image_size)
        mask = read_mask(row["masks"], self.image_size)

        if self.augment:
            # Apply the same flip to image and mask.
            if np.random.rand() < 0.5:
                image = np.flip(image, axis=2).copy()
                mask = np.flip(mask, axis=2).copy()

            # Mild brightness and contrast variation.
            if np.random.rand() < 0.5:
                contrast = np.random.uniform(0.9, 1.1)
                brightness = np.random.uniform(-0.05, 0.05)

                image = np.clip(
                    image * contrast + brightness, 0, 1
                )

        return {
            "xray": np.ascontiguousarray(image, dtype=np.float32),
            "mask": np.ascontiguousarray(mask, dtype=np.float32)
        }
