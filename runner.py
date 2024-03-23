from timeit import default_timer as timer

from tqdm.auto import tqdm 
import torch
from torch import nn
from torchvision import transforms
from util import get_device
from convolutional import ConvolutionalModel
from helper_functions import accuracy_fn
from training import train_step, test_step, eval_model
from pathlib import Path

device = get_device()

def load_dog_data():
    from torchvision import datasets
    from torch.utils.data import random_split, DataLoader

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root="images", transform=transform)

    train_size = int(0.8 * len(dataset))
    validation_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - validation_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    return train_loader, val_loader, test_loader

model = ConvolutionalModel(input_shape=3, hidden_units=256, output_shape=120)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

def run_training():
    train_time_start = timer()
    epochs = 1
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n-------------")

        train_loss, train_acc = train_step(data_loader=train_loader,
                   model=model,
                   loss_fn=loss_fn,
                   optimizer=optimizer,
                   accuracy_fn=accuracy_fn,
                   device=device
        )
        
        test_loss, test_acc = test_step(data_loader=test_loader,
                  model=model,
                  loss_fn=loss_fn,
                  accuracy_fn=accuracy_fn,
                  device=device,
        )
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}")


    train_time_end = timer()
    total_time = train_time_end - train_time_start 
    print(f"Train time on {device}: {total_time:.3f} seconds")

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(exist_ok=True)
MODEL_NAME = "doggie_convolutional_model_1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

if not MODEL_SAVE_PATH.exists():
    train_loader, val_loader, test_loader = load_dog_data()
    run_training()
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
else:
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.to(device)

    test_loader = load_dog_data()[-1]  # Only need the test_loader here


model_results = eval_model(
    model=model, 
    data_loader=test_loader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
    device=device
)
print("model results: ", model_results)

