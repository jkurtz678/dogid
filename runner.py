from timeit import default_timer as timer

from tqdm.auto import tqdm 
import torch
from torch import nn
from torchvision import transforms
from util import get_device
#from convolutional import ImprovedConvolutionalModel
from resnet import create_resnet18 
from helper_functions import accuracy_fn
from training import train_step, test_step, eval_model
from pathlib import Path


def load_dog_data():
    from torchvision import datasets
    from torch.utils.data import random_split, DataLoader

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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    # Validation/Test transform without augmentation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Just resize, no augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

   # Load full dataset with training transforms
    train_dataset = datasets.ImageFolder(root="images", transform=train_transform)
    
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
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

    def run_training():
        train_time_start = timer()
        epochs = 30

         # Warmup parameters
        warmup_factor = 1.0 / 1000  # Start with lr/1000 and gradually increase
        warmup_iters = min(1000, len(train_loader) - 1)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=warmup_factor, 
            end_factor=1.0, 
            total_iters=warmup_iters
        )

        # Main scheduler for after warmup
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,  # Total number of epochs
            eta_min=1e-6   # Minimum learning rate
        )

        for epoch in tqdm(range(epochs)):
            print(f"Epoch: {epoch}\n-------------")

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

