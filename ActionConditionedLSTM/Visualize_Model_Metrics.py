import os
import json
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from losses import weighted_loss

from config import (
    MODEL_PATH,
    TRAINING_HISTORY_PATH,
    TRAINING_METRICS_PATH
)

def main():
    
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at: {MODEL_PATH}")
        print("Please train the model first using Train_Action_Model.py.")
    else:
        print(f"Loading model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH, custom_objects={'weighted_loss': weighted_loss})
        
        print("\n--- Summary ---")
        model.summary()
        
        print("\n--- Architecture Details ---")
        print(f"Total Parameters: {model.count_params():,}")
        print(f"Number of Layers: {len(model.layers)}")
        print(f"Input Shape (Pose): {model.input[0].shape}")
        print(f"Input Shape (Action): {model.input[1].shape}")
        print(f"Output Shape: {model.output.shape}")
    
    if not os.path.exists(TRAINING_HISTORY_PATH):
        print(f"Training history file not found at: {TRAINING_HISTORY_PATH}")
        print("To generate this file, please re-run Train_Action_Model.py (it will save history.json automatically).")
        return
        
    print(f"Loading training history from {TRAINING_HISTORY_PATH}...")
    with open(TRAINING_HISTORY_PATH, 'r') as f:
        history = json.load(f)
        
    if 'loss' not in history:
        print("Error: 'loss' key not found in training history.")
        return
        
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (weighted_loss)', color=color, fontweight='bold')
    ax1.plot(history['loss'], color=color, label='Training Loss', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)

    if 'lr' in history:
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Learning Rate', color=color, fontweight='bold')
        ax2.plot(history['lr'], color=color, linestyle='--', label='Learning Rate', linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_yscale('log') # often better for LR visualization

    plt.title('Training Metrics: Loss & Learning Rate', fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    plt.savefig(TRAINING_METRICS_PATH, dpi=300)
    print(f"\n[Visualized] Saved training metrics plot to '{TRAINING_METRICS_PATH}'")

if __name__ == "__main__":
    main()
