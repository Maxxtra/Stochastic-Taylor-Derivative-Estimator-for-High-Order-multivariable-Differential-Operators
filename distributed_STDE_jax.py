import jax
import jax.numpy as jnp
from jax.experimental.jet import jet
import numpy as np
import time

# ---------- MLP Definition ----------
def net(params, x):
    for W, b in params:
        x = jnp.tanh(x @ W + b)
    return x

# ---------- Jet Sampling ----------
def sample_jets(key, batch_size, dim):
    return jax.random.normal(key, (batch_size, dim))

# ---------- STDE Laplacian Estimation ----------
def estimate_laplacian(params, x, jets):
    def f(z):
        return jnp.sum(net(params, z)**2)

    def estimator(x_i, v_i):
        _, jvp = jet(f, (x_i,), ((v_i,),))
        return jvp

    return jnp.array([estimator(x[i], jets[i]) for i in range(x.shape[0])])

# ---------- Distributed Runner ----------
def run_distributed_stde(batch_size, dim=128, layers=3, hidden_dim=128, save_results=False):
    num_devices = len(jax.devices())
    assert batch_size % num_devices == 0
    chunk_size = batch_size // num_devices

    print(f"Running STDE with {num_devices} GPUs, batch_size={batch_size}, dim={dim}")

    key = jax.random.PRNGKey(0)

    # Random neural net params
    params = [(jax.random.normal(key, (hidden_dim, hidden_dim)),
               jax.random.normal(key, (hidden_dim,))) for _ in range(layers)]

    # Input and jets
    x = jax.random.normal(key, (batch_size, dim))
    jets = sample_jets(key, batch_size, dim)

    # Shard input to devices
    x_chunks = [x[i * chunk_size:(i + 1) * chunk_size] for i in range(num_devices)]
    jet_chunks = [jets[i * chunk_size:(i + 1) * chunk_size] for i in range(num_devices)]
    x_sharded = jax.device_put_sharded(x_chunks, jax.devices())
    jet_sharded = jax.device_put_sharded(jet_chunks, jax.devices())

    # Distributed computation
    @jax.pmap
    def parallel_estimate(x, jet):
        return estimate_laplacian(params, x, jet)

    # Timing the multi-GPU execution
    start = time.time()
    result = parallel_estimate(x_sharded, jet_sharded)
    end = time.time()

    total_time = end - start
    print(f"Estimated Laplacians shape: {result.shape}")
    print(f"Total computation time on {num_devices} GPUs: {total_time:.4f} seconds")

    if save_results:
        from pathlib import Path
        Path("results").mkdir(exist_ok=True)
        np.save("results/laplacian_estimates.npy", np.array(result))

    return result, total_time

if __name__ == "__main__":
    run_distributed_stde(batch_size=128, dim=128, save_results=True)
