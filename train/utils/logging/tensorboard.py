from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import datetime

class TensorboardLogger:
    def __init__(self, log_dir: str = "logs"):
        """Initialize TensorBoard logger.
        
        Args:
            log_dir: Base directory for logs. A timestamp will be appended.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = Path(log_dir) / timestamp
        self.writer = SummaryWriter(log_path)
        print(f"TensorBoard logs will be saved to: {log_path}")

    def log_metrics(self, metrics: dict, step: int, prefix: str = ""):
        """Log metrics to TensorBoard.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Current step (epoch or batch number)
            prefix: Optional prefix for metric names (e.g., 'train/' or 'val/')
        """
        for name, value in metrics.items():
            self.writer.add_scalar(f"{prefix}{name}", value, step)

    def log_histogram(self, name: str, values, step: int):
        """Log histogram of values (useful for weights, gradients, etc.).
        
        Args:
            name: Name of the histogram
            values: Tensor of values
            step: Current step
        """
        self.writer.add_histogram(name, values, step)

    def log_images(self, name: str, images, step: int, normalize: bool = True):
        """Log images to TensorBoard.
        
        Args:
            name: Name for the image group
            images: Tensor of images (B, C, H, W)
            step: Current step
            normalize: Whether to normalize the images
        """
        self.writer.add_images(name, images, step, dataformats='NCHW')

    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()
