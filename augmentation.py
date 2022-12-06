# -*- coding: utf-8 -*-
import albumentations as A


# define heavy augmentations
def get_training_augmentation(width=224, height=224):
    train_transform = [
        A.RandomCrop(width=width, height=height, p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.25),
        A.IAAEmboss(p=0.25),
        A.Blur(p=0.01, blur_limit=3),
        A.OneOf([
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
        ], p=0.8)]
    return A.Compose(train_transform)


def get_validation_augmentation(width=224, height=224):
    return A.Compose([A.PadIfNeeded(width, height)])


def get_preprocessing(preprocessing_fn):
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)


def main():
    pass


if __name__ == "__main__":
    main()
