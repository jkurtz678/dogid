from timeit import default_timer as timer

from tqdm.auto import tqdm 
import torch
from torch import nn
from torchvision import transforms
import math
from util import get_device
#from convolutional import ImprovedConvolutionalModel
from resnet import create_resnet18 
from helper_functions import accuracy_fn
from training import train_step, test_step, eval_model
from pathlib import Path

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=120, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing  # confidence for the true class
        self.smoothing = smoothing
        self.classes = classes

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

def load_dog_data():
    from torchvision import datasets
    from torch.utils.data import random_split, DataLoader

    data_set_means = [0.4762, 0.4519, 0.3910]
    data_set_stds = [0.2580, 0.2524, 0.2570]

     # Training transform with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize larger then crop
        transforms.RandomCrop(224),     # Random crop for variation
        transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to flip left-right
        transforms.ColorJitter(
            brightness=0.2,   # Adjust brightness
            contrast=0.2,     # Adjust contrast
            saturation=0.2,   # Adjust saturation
            hue=0.1          # Slight hue adjustment
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_set_means, 
                           std=data_set_stds)
    ])

    # Validation/Test transform without augmentation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Just resize, no augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=data_set_means, 
                           std=data_set_stds)
    ])

    # Load full dataset with training transforms
    train_dataset = datasets.ImageFolder(root="images", transform=train_transform)

    # Let's also verify our normalization values are appropriate
    # Get raw stats before normalization
    raw_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    raw_dataset = datasets.ImageFolder(root="images", transform=raw_transform)
    raw_img, _ = raw_dataset[0]
    
    print(f"\nRaw image stats (before normalization):")
    for i, channel in enumerate(['Red', 'Green', 'Blue']):
        print(f"{channel} channel - Mean: {raw_img[i].mean():.4f}, Std: {raw_img[i].std():.4f}")

    # Print dataset info
    #print(f"Total number of classes: {len(train_dataset.classes)}")
    #print(f"Class mapping: {train_dataset.class_to_idx}")
    #print(f"Class distribution:")
    #class_counts = {}
    #for _, label in train_dataset.samples:
    #    class_counts[label] = class_counts.get(label, 0) + 1
    #for class_name, idx in train_dataset.class_to_idx.items():
    #    print(f"  {class_name}: {class_counts.get(idx, 0)} images")
    
    # Calculate splits
    train_size = int(0.8 * len(train_dataset))
    validation_size = int(0.1 * len(train_dataset))
    test_size = len(train_dataset) - train_size - validation_size

    # Create splits
    train_dataset, val_dataset, test_dataset = random_split(
        train_dataset, 
        [train_size, validation_size, test_size]
    )

    # Override transforms for validation and test datasets
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # No need to shuffle validation
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # No need to shuffle test

    return train_loader, val_loader, test_loader



def run():

    device = get_device()
    model = create_resnet18(num_classes=120)
    def initialize_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)
    
    model.apply(initialize_weights)

    #loss_fn = nn.CrossEntropyLoss()
    loss_fn = LabelSmoothingLoss(classes=120, smoothing=0.1)
    #optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
    optimizer = torch.optim.SGD(
                          model.parameters(), 
                          lr=0.005,
                          momentum=0.9, 
                          weight_decay=1e-3,
                          nesterov=True  # Add Nesterov momentum
                        )  

    def run_training():
        train_time_start = timer()
        epochs = 20

        total_steps = epochs * len(train_loader)
        num_warmup_steps = len(train_loader) // 8  # Only 1/8th of an epoch
    
        # More aggressive warmup scheduler
        def get_warmup_lr_lambda(current_step):
            if current_step < num_warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1.0, num_warmup_steps))
            # Cosine decay after warmup
            return 0.5 * (1.0 + math.cos(math.pi * (current_step - num_warmup_steps) / (total_steps - num_warmup_steps)))
        
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_warmup_lr_lambda)

        # Main scheduler for after warmup
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,  # Total number of epochs
            eta_min=1e-6   # Minimum learning rate
        )

        for epoch in tqdm(range(epochs)):
            print(f"Epoch: {epoch}\n-------------")

            for batch in train_loader:
                images, labels = batch
                print(f"Input stats - min: {images.min():.2f}, max: {images.max():.2f}, mean: {images.mean():.2f}")
                print(f"Label range: {labels.min()}-{labels.max()}, unique labels: {len(torch.unique(labels))}")
                break

            train_loss, train_acc = train_step(
                data_loader=train_loader,
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                accuracy_fn=accuracy_fn,
                device=device,
                # Pass warmup info to train_step
                warmup_scheduler=warmup_scheduler if epoch == 0 else None  # Only use warmup in first epoch
            )
            
            # test_loss, test_acc = test_step(data_loader=test_loader,
            #           model=model,
            #           loss_fn=loss_fn,
            #           accuracy_fn=accuracy_fn,
            #           device=device,
            # )

            val_loss, val_acc = test_step(data_loader=val_loader,  # Use validation loader here
                    model=model,
                    loss_fn=loss_fn,
                    accuracy_fn=accuracy_fn,
                    device=device,
            )

            lr_scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}")
            print(f"Current LR: {current_lr}")

            #print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}")
            #print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}")


        train_time_end = timer()
        total_time = train_time_end - train_time_start 
        print(f"Train time on {device}: {total_time:.3f} seconds")

    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(exist_ok=True)
    MODEL_NAME = "doggie_convolutional_model_improved.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    if not MODEL_SAVE_PATH.exists():
        print("No saved model found, starting training...")
        train_loader, val_loader, test_loader = load_dog_data()
        run_training()
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    else:
        print("Saved model found, skipping training, loading model...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        model.to(device)

        test_loader = load_dog_data()[-1]  # Only need the test_loader here


    print("Evaluating model...")
    model_results = eval_model(
        model=model, 
        data_loader=test_loader,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        device=device
    )
    print("model results: ", model_results)


# if this is the main file, run the run function
if __name__ == "__main__":
    run()

