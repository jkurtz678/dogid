import torch
from pathlib import Path
from .steps import train_step, test_step, eval_model
from utils.logging.tensorboard import TensorboardLogger

class Trainer:
    def __init__(self, model, loss_fn, optimizer, device, lr_scheduler):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.logger = TensorboardLogger()
        self.lr_scheduler = lr_scheduler

    def train(self, train_loader, val_loader, epochs):
        self.model.train()
        for epoch in range(epochs):
            print(f"Epoch: {epoch+1}")
            
            # Training phase with mixed precision
            train_loss, train_acc = train_step(
                model=self.model,
                data_loader=train_loader,
                loss_fn=self.loss_fn,
                optimizer=self.optimizer,
                accuracy_fn=self.accuracy_fn,
                device=self.device,
                logger=self.logger,
                epoch=epoch,
                use_amp=True  # Enable mixed precision
            )

            self.lr_scheduler.step()
            
            # Log training metrics
            self.logger.log_metrics(
                metrics={
                    "loss": train_loss,
                    "accuracy": train_acc,
                },
                step=epoch,
                prefix="train/"
            )
            
            # Log model weights and gradients
            for name, param in self.model.named_parameters():
                self.logger.log_histogram(f"weights/{name}", param.data, epoch)
                if param.grad is not None:
                    self.logger.log_histogram(f"gradients/{name}", param.grad, epoch)
            
            # Validation phase
            val_loss, val_acc = test_step(
                data_loader=val_loader,
                model=self.model,
                loss_fn=self.loss_fn,
                accuracy_fn=self.accuracy_fn,
                device=self.device
            )
            
            # Log validation metrics
            self.logger.log_metrics(
                metrics={
                    "loss": val_loss,
                    "accuracy": val_acc,
                },
                step=epoch,
                prefix="val/"
            )
            
            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.log_metrics(
                metrics={"learning_rate": current_lr},
                step=epoch
            )
            
            print(
                f"Train loss: {train_loss:.4f} | "
                f"Train acc: {train_acc:.4f} | "
                f"Val loss: {val_loss:.4f} | "
                f"Val acc: {val_acc:.4f}"
            )

    def evaluate(self, test_loader):
        results = eval_model(
            model=self.model,
            data_loader=test_loader,
            loss_fn=self.loss_fn,
            accuracy_fn=self.accuracy_fn,
            device=self.device
        )
        
        # Log test results
        self.logger.log_metrics(
            metrics={
                "loss": results["model_loss"],
                "accuracy": results["model_acc"]
            },
            step=0,
            prefix="test/"
        )
        
        return results

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

    def __del__(self):
        """Ensure TensorBoard writer is properly closed."""
        if hasattr(self, 'logger'):
            self.logger.close()
