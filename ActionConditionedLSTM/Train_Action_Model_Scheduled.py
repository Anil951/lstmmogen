import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback
from losses import weighted_loss
from tensorflow.keras.optimizers import Adam
from config import (
    SEQUENCE_LENGTH,
    NUM_JOINTS,
    NUM_CLASSES,
    PROCESSED_DATASET_PATH,
    MODEL_PATH,
    TRAINING_HISTORY_PATH,
    TARGET_SAMPLING_PROB
)

class ScheduledSamplingCallback(Callback):
    def __init__(self, sampling_prob_var, target_prob=0.5, epochs=40):
        super().__init__()
        self.sampling_prob_var = sampling_prob_var
        self.target_prob = target_prob
        self.epochs = epochs

    def on_epoch_begin(self, epoch, logs=None):
        # Linear decay from 0.0 (100% teacher forcing) to target_prob
        new_prob = min(self.target_prob, (epoch / self.epochs) * self.target_prob)
        self.sampling_prob_var.assign(new_prob)
        print(f"\n[Scheduled Sampling] Epoch {epoch+1}/{self.epochs} - Prediction Sampling Prob: {new_prob:.4f}")

class ScheduledSamplingLSTM(tf.keras.Model):
    def __init__(self, sequence_length=SEQUENCE_LENGTH, num_joints=NUM_JOINTS, num_classes=NUM_CLASSES):
        super().__init__()
        self.sequence_length = sequence_length
        self.num_joints = num_joints
        self.num_classes = num_classes
        self.pose_dim = num_joints * 2
        
        # Action embedding
        self.action_dense = layers.Dense(16, activation='relu', name='action_embedding')
        
        # Recurrent Core
        self.lstm_cell_1 = layers.LSTMCell(256, dropout=0.2)
        self.lstm_cell_2 = layers.LSTMCell(128, dropout=0.2)
        
        # Feed-forward mapping
        self.dense_1 = layers.Dense(128, activation='relu')
        self.ln = layers.LayerNormalization()
        self.dropout = layers.Dropout(0.2)
        self.dense_2 = layers.Dense(64, activation='relu')
        self.dense_out = layers.Dense(self.pose_dim)
        
        # Sampling probability variable (not trainable)
        self.sampling_prob = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    def call(self, inputs, training=None):
        pose_inputs, action_inputs = inputs
        # pose_inputs: (batch, 20, 13, 2)
        # action_inputs: (batch, 20, 7)
        
        batch_size = tf.shape(pose_inputs)[0]
        
        # Flatten pose inputs: (batch, 20, 26)
        flat_pose_inputs = tf.reshape(pose_inputs, (batch_size, self.sequence_length, self.pose_dim))
        
        # Embed action inputs: (batch, 20, 16)
        flat_action_inputs = self.action_dense(action_inputs)
        
        # Initialize LSTM states
        state_h1 = tf.zeros((batch_size, 256))
        state_c1 = tf.zeros((batch_size, 256))
        state_h2 = tf.zeros((batch_size, 128))
        state_c2 = tf.zeros((batch_size, 128))
        
        outputs = []
        
        # Initial input at t=0 is always the ground truth
        prev_pred = flat_pose_inputs[:, 0]
        
        for t in range(self.sequence_length):
            # Select input for timestep t
            if t == 0:
                current_input = flat_pose_inputs[:, 0]
            else:
                if training:
                    # Random decision for each sample in the batch based on current sampling_prob
                    random_choices = tf.random.uniform((batch_size,), 0.0, 1.0)
                    use_pred = random_choices < self.sampling_prob
                    # If use_pred is True, use prev_pred; else use ground truth
                    current_input = tf.where(
                        tf.expand_dims(use_pred, -1),
                        prev_pred,
                        flat_pose_inputs[:, t]
                    )
                else:
                    # During generation/inference, we always feed the previous prediction back
                    current_input = prev_pred
            
            # Combine pose input (26) with action embedding (16) -> (batch, 42)
            combined = tf.concat([current_input, flat_action_inputs[:, t]], axis=-1)
            
            # Step through LSTM cells
            out1, [state_h1, state_c1] = self.lstm_cell_1(combined, [state_h1, state_c1], training=training)
            out2, [state_h2, state_c2] = self.lstm_cell_2(out1, [state_h2, state_c2], training=training)
            
            # Step through feed-forward layers
            dense1 = self.dense_1(out2)
            dense1 = self.ln(dense1)
            if training:
                dense1 = self.dropout(dense1, training=training)
            dense2 = self.dense_2(dense1)
            pred = self.dense_out(dense2) # (batch, 26)
            
            outputs.append(pred)
            prev_pred = pred
            
        # Stack outputs along time dimension: (batch, 20, 26)
        stacked_outputs = tf.stack(outputs, axis=1)
        # Reshape to final output shape: (batch, 20, 13, 2)
        final_output = tf.reshape(stacked_outputs, (batch_size, self.sequence_length, self.num_joints, 2))
        return final_output
    
    # We must provide get_config so we can save/load easily if needed,
    # or rely on save_weights since it's a custom Model loop.
    def get_config(self):
        return {
            "sequence_length": self.sequence_length,
            "num_joints": self.num_joints,
            "num_classes": self.num_classes
        }
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def main():
    if not os.path.exists(PROCESSED_DATASET_PATH):
        print(f"Error: Processed dataset '{PROCESSED_DATASET_PATH}' not found.")
        return

    print(f"Loading processed dataset from '{PROCESSED_DATASET_PATH}'...")
    data = np.load(PROCESSED_DATASET_PATH)
    X_pose = data['X']        # (N, 20, 13, 2)
    y_target = data['y']      # (N, 20, 13, 2)
    X_action = data['actions']# (N, 20, 7)

    np.random.seed(42)
    indices = np.arange(X_pose.shape[0])
    np.random.shuffle(indices)
    X_pose = X_pose[indices]
    y_target = y_target[indices]
    X_action = X_action[indices]

    print(f"Loaded and shuffled X_pose: {X_pose.shape}, X_action: {X_action.shape}, y_target: {y_target.shape}")

    # Build custom scheduled sampling model
    model = ScheduledSamplingLSTM()
    
    # Run a dummy batch through it to initialize shapes properly so we can print a summary
    _ = model([X_pose[:2], X_action[:2]])
    model.summary()

    # We use learning rate 0.001 and the custom scheduled sampling loss
    model.compile(optimizer=Adam(learning_rate=0.001), loss=weighted_loss)

    epochs = 40
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
    scheduled_sampling_cb = ScheduledSamplingCallback(model.sampling_prob, target_prob=TARGET_SAMPLING_PROB, epochs=epochs)

    # Train model
    print(f"\nStarting Scheduled Sampling model training with weighted_loss for {epochs} epochs...")
    history = model.fit(
        [X_pose, X_action], 
        y_target, 
        epochs=epochs, 
        batch_size=128, 
        validation_split=0.15,
        callbacks=[reduce_lr, scheduled_sampling_cb]
    )

    # We save as tf format (SavedModel) instead of H5 because custom step models serialize better this way.
    # We will save to a slightly different name so the original isn't destroyed immediately if this fails.
    scheduled_model_path = MODEL_PATH.replace('.h5', '_scheduled')
    model.save(scheduled_model_path, save_format='tf')
    print(f"\nScheduled Sampling Model saved to '{scheduled_model_path}'")

    # Override the history
    history_path_scheduled = TRAINING_HISTORY_PATH.replace('.json', '_scheduled.json')
    with open(history_path_scheduled, 'w') as f:
        json_history = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        json.dump(json_history, f)
    print(f"Saved training history to '{history_path_scheduled}'.")

if __name__ == "__main__":
    main()
