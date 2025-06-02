#!/bin/bash

# Enhanced training script that captures output and generates performance summary

echo "Starting training with automatic performance analysis..."
echo "=============================================="

# Create logs directory if it doesn't exist
mkdir -p training_logs

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="training_logs/training_${TIMESTAMP}.log"

# Run training and capture output to both console and file
echo "Training output will be saved to: $LOG_FILE"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the training script, capturing output with progress indicator
python3 runner.py 2>&1 | tee "$LOG_FILE" | while IFS= read -r line; do
    echo "$line"
    
    # Extract and display progress
    if [[ "$line" =~ Epoch:\ ([0-9]+) ]]; then
        current_epoch="${BASH_REMATCH[1]}"
        echo "🔄 Progress: Epoch $current_epoch/50 ($(( current_epoch * 100 / 50 ))%)"
    elif [[ "$line" =~ Train\ loss:.*Train\ acc:\ ([0-9.]+).*Val\ acc:\ ([0-9.]+) ]]; then
        train_acc="${BASH_REMATCH[1]}"
        val_acc="${BASH_REMATCH[2]}"
        echo "📊 Latest: Train ${train_acc}% | Val ${val_acc}%"
        echo "----------------------------------------"
    fi
done

# Check if training completed successfully
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "🎯 TRAINING COMPLETED - Generating Analysis..."
    echo "=============================================="
    echo ""
    
    # Generate performance summary
    python3 utils/training_summary.py "$LOG_FILE"
    
    echo ""
    echo "=============================================="
    echo "📁 Full training log saved to: $LOG_FILE"
    echo "🔍 For detailed TensorBoard analysis:"
    echo "   tensorboard --logdir=logs --port=6006"
    echo "=============================================="
else
    echo ""
    echo "❌ Training failed! Check the log file for details: $LOG_FILE"
    exit 1
fi