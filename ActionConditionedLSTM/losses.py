import tensorflow as tf

def weighted_loss(y_true, y_pred):
    """
    Combined Loss: 1.0 * MSE + 2.0 * MPJPE (Mean Per Joint Position Error).
    Computes Euclidean distance per joint for positional accuracy combined with MSE.
    """
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    per_joint_dist = tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1) + 1e-8)
    mpjpe = tf.reduce_mean(per_joint_dist)
    return 1.0 * mse + 2.0 * mpjpe
