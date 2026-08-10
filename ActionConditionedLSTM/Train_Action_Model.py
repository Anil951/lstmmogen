import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau
from losses import weighted_loss
from tensorflow.keras.optimizers import Adam


class ActionRNNModel:
    def __init__(self, sequence_length=20, num_joints=13, num_classes=3):
        self.sequence_length = sequence_length
        self.num_joints = num_joints
        self.num_classes = num_classes
        self.pose_dim = num_joints * 2
        self.model = self.create_model()

    def create_model(self):
        pose_input = Input(shape=(self.sequence_length, self.num_joints, 2), name='pose_input')
        action_input = Input(shape=(self.sequence_length, self.num_classes), name='action_input')

        # Flatten 13x2 joints into 26 coordinates per frame
        pose_reshaped = layers.Reshape((self.sequence_length, self.pose_dim))(pose_input)

        # Action Embedding: Dense projection of one-hot action vector to 16 features
        action_emb = layers.TimeDistributed(layers.Dense(16, activation='relu'), name='action_embedding')(action_input)

        # Concatenate pose features (26) and action embedding (16) -> 42 features per timestep
        combined_input = layers.Concatenate(axis=-1)([pose_reshaped, action_emb])

        # Deep LSTM network
        x = layers.LSTM(256, return_sequences=True, dropout=0.2)(combined_input)
        x = layers.LSTM(128, return_sequences=True, dropout=0.2)(x)
        # Wider Dense layers with Batch Normalization
        x = layers.TimeDistributed(layers.Dense(128, activation='relu'))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.TimeDistributed(layers.Dense(64, activation='relu'))(x)
        # Output mapping
        x = layers.TimeDistributed(layers.Dense(self.pose_dim))(x)
        output = layers.Reshape((self.sequence_length, self.num_joints, 2), name='pose_output')(x)

        model = models.Model(inputs=[pose_input, action_input], outputs=output, name='Action_RNN_Model')
        return model

    def compile(self):
        self.model.compile(optimizer=Adam(learning_rate=0.01), loss=weighted_loss)

    def fit(self, X_pose, X_action, y_target, epochs=200, batch_size=64, reduce_lr_callback=None):
        callbacks = [reduce_lr_callback] if reduce_lr_callback else []
        history = self.model.fit(
            [X_pose, X_action],
            y_target,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks
        )
        return history

    def save(self, model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        print(f"Action RNN model successfully saved to '{model_path}'")

    def summary(self):
        self.model.summary()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(script_dir)
    iofiles_dir = os.path.join(workspace_dir, 'iofiles')
    os.makedirs(iofiles_dir, exist_ok=True)

    dataset_file = os.path.join(iofiles_dir, 'processed_dataset.npz')
    model_path = os.path.join(iofiles_dir, 'action_rnn_model.h5')

    if not os.path.exists(dataset_file):
        print(f"Error: Processed dataset '{dataset_file}' not found. Please run Dataset_Preprocessor.py first.")
        return

    print(f"Loading processed dataset from '{dataset_file}'...")
    data = np.load(dataset_file)
    X_pose = data['X']        # (N, 20, 13, 2)
    y_target = data['y']      # (N, 20, 13, 2)
    X_action = data['actions']# (N, 20, 3)

    print(f"Loaded X_pose: {X_pose.shape}, X_action: {X_action.shape}, y_target: {y_target.shape}")

    rnn_model = ActionRNNModel(sequence_length=20, num_joints=13, num_classes=3)
    rnn_model.summary()
    rnn_model.compile()

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-20)

    # Train model
    print("\nStarting model training with weighted_loss (1.0 * MSE + 2.0 * MPJPE)...")
    rnn_model.fit(X_pose, X_action, y_target, epochs=100, batch_size=64, reduce_lr_callback=reduce_lr)

    rnn_model.save(model_path)

if __name__ == "__main__":
    main()
