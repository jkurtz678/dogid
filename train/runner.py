import torch
from torch import nn
from utils.device import get_device
from core.data_loading import load_dog_data
from core.loss import LabelSmoothingLoss
from engine.trainer import Trainer
from models.resnet import create_resnet18
from config import *

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)
        nn.init.constant_(m.bias, 0)

def run():
    device = get_device()
    model = create_resnet18(num_classes=NUM_CLASSES)
    # Don't apply initialize_weights - it overwrites pretrained weights!
    
    loss_fn = LabelSmoothingLoss(classes=NUM_CLASSES, smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True
    )

    # Warmup scheduler for first 5 epochs, then cosine annealing
    warmup_epochs = 5
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,  # Start at 10% of learning rate
        end_factor=1.0,    # Ramp up to full learning rate
        total_iters=warmup_epochs
    )
    
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS - warmup_epochs,
        eta_min=1e-6
    )
    
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        lr_scheduler=lr_scheduler
    )

    if not MODEL_SAVE_PATH.exists():
        print("No saved model found, starting training...")
        train_loader, val_loader, test_loader = load_dog_data()
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=EPOCHS
        )
        trainer.save_model(MODEL_SAVE_PATH)
    else:
        print("Saved model found, skipping training, loading model...")
        trainer.load_model(MODEL_SAVE_PATH)
        test_loader = load_dog_data()[-1]

    print("Evaluating model...")
    results = trainer.evaluate(test_loader)
    print("Model results:", results)

if __name__ == "__main__":
    run()
