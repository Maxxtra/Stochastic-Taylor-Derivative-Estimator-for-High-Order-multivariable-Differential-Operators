import jax
import jax.numpy as jnp
from jax.experimental.jet import jet
import time  # Import time for timing the computation

# Simple MLP
def net(params, x):
    for W, b in params:
        x = jnp.tanh(x @ W + b)
    return x

def sample_jets(key, batch_size, dim):
    return jax.random.normal(key, (batch_size, dim))

def estimate_laplacian(params, x, jets):
    def f(z):
        return jnp.sum(net(params, z)**2)

    def estimator(x_i, v_i):
        _, jvp = jet(f, (x_i,), ((v_i,),))
        return jvp

    return jnp.array([estimator(x[i], jets[i]) for i in range(x.shape[0])])

# Setup
key = jax.random.PRNGKey(0)
params = [(jax.random.normal(key, (128, 128)), jax.random.normal(key, (128,))) for _ in range(3)]
x = jax.random.normal(key, (16, 128))
jets = sample_jets(key, 16, 128)

# Timing the single GPU execution
start = time.time()
result = estimate_laplacian(params, x, jets)
end = time.time()

total_time = end - start
print("Laplacian estimates:", result.shape)
print(f"Total computation time on single GPU: {total_time:.4f} seconds")
