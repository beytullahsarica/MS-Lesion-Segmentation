# -*- coding: utf-8 -*-
import numpy as np


class MSDataset:
    def __init__(self, image_paths=None, mask_paths=None, train_images=None, train_labels=None, augmentation=None, preprocessing=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.train_images = train_images
        self.train_labels = train_labels
        self.load_data()

    def load_data(self):
        if self.train_images is None and self.train_labels is None:
            self.train_images = np.load(self.image_paths).astype(np.float32)
            self.train_labels = np.load(self.mask_paths).astype(np.float32)

    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, idx):
        image = self.train_images[idx]
        mask = self.train_labels[idx]

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def get_images(self):
        return self.train_images

    def get_labels(self):
        return self.train_labels


def main():
    pass


if __name__ == "__main__":
    main()
