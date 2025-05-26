import torch
from timeit import default_timer as timer

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device,
               warmup_scheduler=None,
               logger=None,
               epoch=None):
    train_loss, train_acc = 0,0
    model.to(device)
    
    for batch, (X, y) in enumerate(data_loader):
        global_step = None
        if logger is not None and epoch is not None:
            global_step = epoch * len(data_loader) + batch

        batch_time_start = timer() 
        X, y = X.to(device), y.to(device)

        # 1. forward pass
        y_pred = model(X)

        if batch == 0:  # First batch of each epoch
            print(f"Model output stats:")
            print(f"  Shape: {y_pred.shape}")
            print(f"  Min: {y_pred.min().item():.4f}")
            print(f"  Max: {y_pred.max().item():.4f}")
            print(f"  Mean: {y_pred.mean().item():.4f}")
            print(f"  Std: {y_pred.std().item():.4f}")

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1)) # Go from logits -> pred labels

        # 3. Backward pass
        optimizer.zero_grad()

        # 4. Backpropagation
        loss.backward()


        if batch % 100 == 0:
            total_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"Gradient norm: {total_norm:.4f}")

            # Print layer-wise gradients
            print("\nLayer-wise gradient norms:")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name}: {param.grad.norm().item():.4f}")

        # 5. Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=15.0)

        # 6. Optimizer step
        optimizer.step()

        # batch level loss logging for tensorboard
        if logger is not None and epoch is not None and global_step is not None:
            current_lr = optimizer.param_groups[0]['lr']
            logger.log_metrics(
                metrics={
                    "batch_loss": loss.item(),
                    "batch_accuracy": train_acc/(batch+1),
                    "learning_rate": current_lr  # Add this line

                },
                step=global_step,
                prefix="train/batch/"
            )
            
            # Optionally log gradient norms at batch level too
            if batch % 100 == 0:  # Keep the same frequency as your current gradient logging
                total_norm = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                logger.log_metrics(
                    metrics={"gradient_norm": total_norm},
                    step=global_step,
                    prefix="train/batch/"
                )

        # 7. Step warmup scheduler if it exists and we're in warmup phase
        if warmup_scheduler is not None:
            warmup_scheduler.step()
            if batch % 10 == 0:  # Print LR less frequently to reduce output
                print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        batch_time_end = timer()
        total_time = batch_time_end - batch_time_start 
        print(f"Batch {batch}/{len(data_loader)}: Loss: {loss}, Accuracy: {train_acc/(batch+1):.2f}, time: {total_time:.3f} seconds")
    
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    return train_loss, train_acc

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device):
    test_loss, test_acc = 0,0
    model.to(device)
    model.eval()

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            # 1. Forward pass 
            test_pred = model(X)

            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                                    y_pred=test_pred.argmax(dim=1)
            )

        test_loss /= len(data_loader) 
        test_acc /= len(data_loader)

        return test_loss, test_acc

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device: torch.device):
    """Evaluates a given model on a given dataset.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.
        device (str, optional): Target device to compute on. Defaults to device.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to the target device
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        
        # Scale loss and acc
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {"model_name": model.__class__.__name__,
            "model_loss": loss.item(),
            "model_acc": acc}

def make_predictions(model: torch.nn.Module, data: list, device: torch.device):
    pred_probs = []
    model.eval()
    with torch.inference_mode(): 
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device)
            pred_logit = model(sample)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
            pred_probs.append(pred_prob.cpu())
    return torch.stack(pred_probs)
