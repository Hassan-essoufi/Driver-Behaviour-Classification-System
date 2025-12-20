import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
#Image loading
def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path)

    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Convert image to RGB
    image = image.convert("RGB")

    return image

def get_train_transforms(input_size=(224,224)):
    """
    Data augmentation pipeline for training data.
    """
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])

def get_val_transforms(input_size=(224, 224)):
    """
    Validation transforms (no augmentation).
    """
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])

import numpy as np 
import matplotlib.pyplot as plt
train_transform = transforms.Compose([
    transforms.Resize(224),            # 1. Redimensionner
    transforms.RandomHorizontalFlip(p=0.5),   # 2. Retournement horizontal
    transforms.RandomRotation(degrees=10),    # 3. Rotation
    transforms.ColorJitter(                   # 4. Variations de couleur
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),
    transforms.ToTensor(),                    # 5. Convertir PIL → Tensor (C, H, W)
    transforms.Normalize(                     # 6. Normaliser (nécessite Tensor!)
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])

def show_original_and_transformed(image_path):
    """
    Affiche l'image originale et transformée côte à côte
    """
    # Charger l'image
    img = Image.open(image_path).convert('RGB')
    
    # Appliquer la transformation
    transformed_tensor = train_transform(img)
    
    # Pour afficher avec matplotlib, on doit:
    # 1. Dénormaliser le tensor
    # 2. Convertir de (C, H, W) à (H, W, C)
    # 3. Convertir en numpy
    
    # Fonction de dénormalisation
    def denormalize(tensor, mean, std):
        """Inverse la normalisation"""
        tensor = tensor.clone()  # Ne pas modifier l'original
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)  # t = t * s + m
        return tensor
    
    # Dénormaliser
    denormalized = denormalize(transformed_tensor, IMAGENET_MEAN, IMAGENET_STD)
    
    # Convertir pour affichage: (C, H, W) → (H, W, C)
    img_for_display = denormalized.permute(1, 2, 0).numpy()
    
    # Clip les valeurs entre 0 et 1 (après dénormalisation)
    img_for_display = np.clip(img_for_display, 0, 1)
    
    # Créer la figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Image originale
    axes[0].imshow(img)
    axes[0].set_title('Image originale\nTaille: {}'.format(img.size))
    axes[0].axis('off')
    
    # Image transformée
    axes[1].imshow(img_for_display)
    axes[1].set_title('Après transformations\nShape: {}'.format(transformed_tensor.shape))
    axes[1].axis('off')
    
    plt.suptitle('Comparaison: Original vs Transformée', fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Afficher des informations détaillées
    print("="*60)
    print("INFORMATIONS DE TRANSFORMATION")
    print("="*60)
    print(f"Original - Type: {type(img)}, Taille: {img.size}, Mode: {img.mode}")
    print(f"Transformé - Type: {type(transformed_tensor)}, Shape: {transformed_tensor.shape}")
    print(f"Transformé - Min: {transformed_tensor.min():.3f}, Max: {transformed_tensor.max():.3f}")
    print(f"Transformé - Mean: [{transformed_tensor.mean(dim=(1,2))[0]:.3f}, "
          f"{transformed_tensor.mean(dim=(1,2))[1]:.3f}, "
          f"{transformed_tensor.mean(dim=(1,2))[2]:.3f}]")
    print("="*60)

# Utilisation
show_original_and_transformed("C:/Users/hp/Desktop/Projects/Driver_Gesture_Detection_System/img_186.jpg")