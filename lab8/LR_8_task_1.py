import numpy as np
import tensorflow as tf


n_samples, batch_size, num_steps = 1000, 100, 20000
X_data = np.random.uniform(1, 10, (n_samples, 1)).astype(np.float32)
y_data = (2 * X_data + 1 + np.random.normal(0, 2, (n_samples, 1))).astype(np.float32)

k = tf.Variable(tf.random.normal((1, 1)), name='slope')
b = tf.Variable(tf.zeros((1,)), name='bias')
y_pred = tf.matmul(X_data, k) + b

optimizer = tf.optimizers.SGD(learning_rate=0.0001)
display_step = 100

for i in range(num_steps):
    indices = np.random.choice(n_samples, batch_size)
    X_batch, y_batch = X_data[indices], y_data[indices]

    with tf.GradientTape() as tape:
        y_pred = tf.matmul(X_batch, k) + b
        loss_val = tf.reduce_sum((y_batch - y_pred) ** 2)

    if np.isnan(loss_val.numpy()):
        print(f"NaN value was found on step: {i + 1}")
        break

    gradients = tape.gradient(loss_val, [k, b])
    optimizer.apply_gradients(zip(gradients, [k, b]))

    if (i + 1) % display_step == 0:
        print(f'Епоха {i + 1}: {loss_val.numpy():.8f}, k={k.numpy()[0][0]:.4f}, b={b.numpy()[0]:.4f}')
