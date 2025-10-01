"""
MIMIC-IV Image Transforms Module

This module provides image transformation utilities optimized for MIMIC-IV chest X-ray data,
particularly for use with EfficientNet models.
"""

from typing import Tuple
import torchvision.transforms as transforms


def get_efficientnet_transforms(
    target_size: Tuple[int, int] = (224, 224),
    is_training: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> transforms.Compose:
    """
    Get EfficientNet-optimized transforms for medical imaging.
    
    These transforms are specifically designed for chest X-ray images and follow
    best practices for medical image preprocessing with EfficientNet models.
    
    Args:
        target_size: Target image size (height, width)
        is_training: Whether to apply training augmentations
        mean: Normalization mean values (ImageNet default)
        std: Normalization std values (ImageNet default)
        
    Returns:
        Composed transforms
    """
    # Base transforms for medical imaging
    transform_list = [
        transforms.Resize((int(target_size[0] * 1.1), int(target_size[1] * 1.1))),
        transforms.CenterCrop(target_size),
    ]
    
    # Add training augmentations
    if is_training:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ])
    
    # Final transforms
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return transforms.Compose(transform_list)


def get_chest_xray_transforms(
    target_size: Tuple[int, int] = (224, 224),
    is_training: bool = True,
    use_chest_specific_augmentation: bool = True
) -> transforms.Compose:
    """
    Get chest X-ray specific transforms with medical imaging considerations.
    
    These transforms are specifically designed for chest X-ray images and include
    augmentations that are appropriate for medical imaging tasks.
    
    Args:
        target_size: Target image size (height, width)
        is_training: Whether to apply training augmentations
        use_chest_specific_augmentation: Whether to use chest-specific augmentations
        
    Returns:
        Composed transforms
    """
    # Base transforms
    transform_list = [
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ]
    
    # Add training augmentations if specified
    if is_training and use_chest_specific_augmentation:
        # Chest X-ray specific augmentations
        transform_list.insert(-1, transforms.RandomHorizontalFlip(p=0.5))
        transform_list.insert(-1, transforms.RandomRotation(degrees=5))  # Smaller rotation for medical images
        transform_list.insert(-1, transforms.ColorJitter(brightness=0.05, contrast=0.05))  # Subtle changes
    
    # Normalization (ImageNet standard)
    transform_list.append(transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ))
    
    return transforms.Compose(transform_list)


def get_minimal_transforms(
    target_size: Tuple[int, int] = (224, 224)
) -> transforms.Compose:
    """
    Get minimal transforms for inference or validation.
    
    Args:
        target_size: Target image size (height, width)
        
    Returns:
        Composed transforms with only resizing and normalization
    """
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
