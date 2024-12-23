import torch
from pathlib import Path
from .steps import train_step, test_step, eval_model

class Trainer:
    def __init__(self, model, loss_fn, optimizer, device):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

    def train(self, train_loader, val_loader, epochs):
        self.model.train()
        for epoch in range(epochs):
            print(f"Epoch: {epoch+1}")
            train_loss, train_acc = train_step(
                model=self.model,
                data_loader=train_loader,
                loss_fn=self.loss_fn,
                optimizer=self.optimizer,
                accuracy_fn=self.accuracy_fn,
                device=self.device
            )
            
            val_loss, val_acc = test_step(
                data_loader=val_loader,
                model=self.model,
                loss_fn=self.loss_fn,
                accuracy_fn=self.accuracy_fn,
                device=self.device
            )
            
            print(
                f"Train loss: {train_loss:.4f} | "
                f"Train acc: {train_acc:.4f} | "
                f"Val loss: {val_loss:.4f} | "
                f"Val acc: {val_acc:.4f}"
            )

    def evaluate(self, test_loader):
        return eval_model(
            model=self.model,
            data_loader=test_loader,
            loss_fn=self.loss_fn,
            accuracy_fn=self.accuracy_fn,
            device=self.device
        )

    def save_model(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: Path):
        self.model.load_state_dict(torch.load(path))

    @staticmethod
    def accuracy_fn(y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct / len(y_pred)) * 100
        return acc