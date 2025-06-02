import re
import sys
from pathlib import Path
from datetime import datetime

class TrainingSummaryAnalyzer:
    def __init__(self):
        self.epochs = []
        self.gradient_norms = []
        self.batch_data = []
        
    def parse_training_output(self, output_text):
        """Parse training output and extract key metrics"""
        lines = output_text.strip().split('\n')
        
        current_epoch = None
        for line in lines:
            # Parse epoch summary lines
            if "Train loss:" in line:
                match = re.search(r'Train loss: ([\d.]+).*Train acc: ([\d.]+).*Val loss: ([\d.]+).*Val acc: ([\d.]+)', line)
                if match:
                    epoch_data = {
                        'epoch': current_epoch,
                        'train_loss': float(match.group(1)),
                        'train_acc': float(match.group(2)),
                        'val_loss': float(match.group(3)),
                        'val_acc': float(match.group(4))
                    }
                    self.epochs.append(epoch_data)
            
            # Parse epoch numbers
            elif "Epoch:" in line:
                match = re.search(r'Epoch: (\d+)', line)
                if match:
                    current_epoch = int(match.group(1))
            
            # Parse gradient norms
            elif "Gradient norm:" in line:
                match = re.search(r'Gradient norm: ([\d.]+)', line)
                if match:
                    self.gradient_norms.append(float(match.group(1)))
            
            # Parse batch accuracy to track early training
            elif "Batch" in line and "Accuracy:" in line:
                match = re.search(r'Batch (\d+)/(\d+).*Accuracy: ([\d.]+)', line)
                if match:
                    batch_num = int(match.group(1))
                    total_batches = int(match.group(2))
                    accuracy = float(match.group(3))
                    if batch_num <= 20:  # Only track first 20 batches
                        self.batch_data.append({'batch': batch_num, 'accuracy': accuracy})
    
    def analyze_performance(self):
        """Analyze training performance and generate insights"""
        if not self.epochs:
            return "❌ No epoch data found - training may have failed"
        
        analysis = []
        analysis.append("=== TRAINING PERFORMANCE ANALYSIS ===")
        analysis.append(f"Total epochs completed: {len(self.epochs)}")
        analysis.append("")
        
        # Initial performance
        first_epoch = self.epochs[0]
        analysis.append("📊 INITIAL PERFORMANCE:")
        analysis.append(f"  Epoch 1: Train {first_epoch['train_acc']:.1f}% | Val {first_epoch['val_acc']:.1f}%")
        
        # Check for pretrained model success
        if first_epoch['train_acc'] > 25:
            analysis.append("  ✅ Good initial accuracy - pretrained weights working")
        elif first_epoch['train_acc'] > 10:
            analysis.append("  ⚠️  Moderate initial accuracy - transfer learning partially working")
        else:
            analysis.append("  ❌ Poor initial accuracy - pretrained weights not working properly")
        
        # Final performance
        final_epoch = self.epochs[-1]
        analysis.append("")
        analysis.append("📊 FINAL PERFORMANCE:")
        analysis.append(f"  Final: Train {final_epoch['train_acc']:.1f}% | Val {final_epoch['val_acc']:.1f}%")
        
        # Best validation accuracy
        best_val_epoch = max(self.epochs, key=lambda x: x['val_acc'])
        analysis.append(f"  Best validation: {best_val_epoch['val_acc']:.1f}% (Epoch {best_val_epoch['epoch']})")
        
        # Convergence analysis
        analysis.append("")
        analysis.append("📈 TRAINING DYNAMICS:")
        
        if len(self.epochs) > 5:
            # Check if still improving
            recent_val_acc = [e['val_acc'] for e in self.epochs[-5:]]
            if max(recent_val_acc) == recent_val_acc[-1]:
                analysis.append("  📈 Still improving - consider training longer")
            elif recent_val_acc[-1] < max(recent_val_acc) - 2:
                analysis.append("  📉 Validation accuracy declining - possible overfitting")
            else:
                analysis.append("  ➡️  Converged - validation accuracy stable")
        
        # Overfitting check
        if len(self.epochs) > 10:
            train_val_gap = final_epoch['train_acc'] - final_epoch['val_acc']
            if train_val_gap > 10:
                analysis.append("  ⚠️  Large train/val gap - significant overfitting")
            elif train_val_gap > 5:
                analysis.append("  ⚠️  Moderate train/val gap - some overfitting")
            else:
                analysis.append("  ✅ Small train/val gap - good generalization")
        
        # Gradient analysis
        if self.gradient_norms:
            analysis.append("")
            analysis.append("🔢 GRADIENT ANALYSIS:")
            initial_grad = self.gradient_norms[0] if self.gradient_norms else 0
            final_grad = self.gradient_norms[-1] if len(self.gradient_norms) > 1 else initial_grad
            
            analysis.append(f"  Initial gradient norm: {initial_grad:,.0f}")
            if len(self.gradient_norms) > 1:
                analysis.append(f"  Final gradient norm: {final_grad:,.0f}")
                
            if initial_grad > 100000:
                analysis.append("  ❌ Very high initial gradients - may indicate setup issues")
            elif initial_grad > 10000:
                analysis.append("  ⚠️  High initial gradients - monitor for stability")
            else:
                analysis.append("  ✅ Reasonable gradient magnitudes")
        
        # Early batch analysis
        if self.batch_data:
            analysis.append("")
            analysis.append("🚀 EARLY TRAINING BEHAVIOR:")
            first_batch_acc = self.batch_data[0]['accuracy'] if self.batch_data else 0
            batch_10_acc = next((b['accuracy'] for b in self.batch_data if b['batch'] == 10), None)
            
            analysis.append(f"  First batch accuracy: {first_batch_acc:.1f}%")
            if batch_10_acc:
                analysis.append(f"  Batch 10 accuracy: {batch_10_acc:.1f}%")
                improvement = batch_10_acc - first_batch_acc
                if improvement > 5:
                    analysis.append("  ✅ Fast initial learning")
                elif improvement > 0:
                    analysis.append("  ➡️  Gradual initial learning")
                else:
                    analysis.append("  ❌ No improvement in first 10 batches")
        
        # Key epochs summary
        analysis.append("")
        analysis.append("📋 KEY EPOCHS SUMMARY:")
        key_epochs = [1, 5, 10, 25, len(self.epochs)]
        for epoch_num in key_epochs:
            if epoch_num <= len(self.epochs):
                epoch_data = self.epochs[epoch_num - 1]
                analysis.append(f"  Epoch {epoch_num:2d}: Train {epoch_data['train_acc']:5.1f}% | Val {epoch_data['val_acc']:5.1f}% | Loss {epoch_data['train_loss']:.3f}")
        
        return "\n".join(analysis)

def analyze_training_from_file(log_file_path):
    """Analyze training from a log file"""
    analyzer = TrainingSummaryAnalyzer()
    
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()
        analyzer.parse_training_output(content)
        return analyzer.analyze_performance()
    except Exception as e:
        return f"❌ Error analyzing log file: {e}"

def analyze_training_from_text(output_text):
    """Analyze training from raw text output"""
    analyzer = TrainingSummaryAnalyzer()
    analyzer.parse_training_output(output_text)
    return analyzer.analyze_performance()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Analyze from file
        log_file = sys.argv[1]
        print(analyze_training_from_file(log_file))
    else:
        # Read from stdin
        content = sys.stdin.read()
        print(analyze_training_from_text(content))