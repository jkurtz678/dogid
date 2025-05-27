from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from config import *  # Using absolute import path

def load_dog_data():
    data_set_means = [0.4762, 0.4519, 0.3910]
    data_set_stds = [0.2580, 0.2524, 0.2570]

    # Training transform with augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(CROP_SIZE, scale=(0.8, 1.0)),  # More varied crops
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=COLOR_JITTER['brightness'],
            contrast=COLOR_JITTER['contrast'],
            saturation=COLOR_JITTER['saturation'],
            hue=COLOR_JITTER['hue']
        ),
        transforms.RandomAffine(
            degrees=ROTATION_DEGREES,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_set_means, 
                           std=data_set_stds)
    ])

    # Validation/Test transform without augmentation
    val_transform = transforms.Compose([
        transforms.Resize(256),  # Resize to slightly larger than crop size
        transforms.CenterCrop(CROP_SIZE),  # Center crop for consistency
        transforms.ToTensor(),
        transforms.Normalize(mean=data_set_means, 
                           std=data_set_stds)
    ])

    # Load full dataset with training transforms
    train_dataset = datasets.ImageFolder(root="images", transform=train_transform)

    # Calculate raw stats
    raw_transform = transforms.Compose([
        transforms.Resize((CROP_SIZE, CROP_SIZE)),
        transforms.ToTensor()
    ])
    raw_dataset = datasets.ImageFolder(root="images", transform=raw_transform)
    raw_img, _ = raw_dataset[0]
    
    print(f"\nRaw image stats (before normalization):")
    for i, channel in enumerate(['Red', 'Green', 'Blue']):
        print(f"{channel} channel - Mean: {raw_img[i].mean():.4f}, Std: {raw_img[i].std():.4f}")
    
    # Calculate splits
    train_size = int(TRAIN_SPLIT * len(train_dataset))
    validation_size = int(VAL_SPLIT * len(train_dataset))
    test_size = len(train_dataset) - train_size - validation_size

    # Load datasets with appropriate transforms
    val_dataset = datasets.ImageFolder(root="images", transform=val_transform)
    test_dataset = datasets.ImageFolder(root="images", transform=val_transform)
    
    # Create splits
    train_dataset, _, _ = random_split(
        train_dataset, 
        [train_size, validation_size, test_size]
    )
    _, val_dataset, _ = random_split(
        val_dataset, 
        [train_size, validation_size, test_size]
    )
    _, _, test_dataset = random_split(
        test_dataset, 
        [train_size, validation_size, test_size]
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader
