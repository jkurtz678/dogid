#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Add the current directory to PYTHONPATH and run the training script
PYTHONPATH=$PYTHONPATH:. python3 train/runner.py