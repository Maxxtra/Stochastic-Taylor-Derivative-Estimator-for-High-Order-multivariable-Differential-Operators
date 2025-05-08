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

# ---------- Potential Energy ----------
def potential_energy(x):
    # Harmonic oscillator potential: V(x) = 0.5 * ||x||^2
    return 0.5 * jnp.sum(x**2, axis=1)

# ---------- Laplacian Estimation ----------
def estimate_laplacian(params, x, jets):
    def f(z):
        return jnp.sum(net(params, z)**2)

    def estimator(x_i, v_i):
        _, jvp = jet(f, (x_i,), ((v_i,),))
        return jvp

    return jnp.array([estimator(x[i], jets[i]) for i in range(x.shape[0])])

# ---------- Energy Estimation ----------
def run_distributed_energy(batch_size, dim=128, layers=3, hidden_dim=128, save_results=False):
    num_devices = len(jax.devices())
    assert batch_size % num_devices == 0
    chunk_size = batch_size // num_devices

    print(f"Running Many-body Energy Estimator with {num_devices} GPUs, batch_size={batch_size}, dim={dim}")

    key = jax.random.PRNGKey(0)
    subkeys = jax.random.split(key, 4)

    # Random neural net params
    params = [(jax.random.normal(subkeys[0], (hidden_dim, hidden_dim)),
               jax.random.normal(subkeys[1], (hidden_dim,))) for _ in range(layers)]

    # Input and jets
    x = jax.random.normal(subkeys[2], (batch_size, dim))
    jets = sample_jets(subkeys[3], batch_size, dim)

    # Shard inputs
    x_chunks = [x[i * chunk_size:(i + 1) * chunk_size] for i in range(num_devices)]
    jet_chunks = [jets[i * chunk_size:(i + 1) * chunk_size] for i in range(num_devices)]
    x_sharded = jax.device_put_sharded(x_chunks, jax.devices())
    jet_sharded = jax.device_put_sharded(jet_chunks, jax.devices())

    # Replicate parameters across devices
    params_replicated = jax.device_put_replicated(params, jax.devices())

    # Distributed computation
    @jax.pmap
    def parallel_energy(x, jet, params):
        # For each device, compute psi, laplacian, and energy density
        psi = net(params, x)
        laplacian = estimate_laplacian(params, x, jet)
        V = potential_energy(x)
        kinetic = -0.5 * laplacian
        energy_density = kinetic + V
        return energy_density

    # Timing the multi-GPU execution
    start = time.time()
    local_energies = parallel_energy(x_sharded, jet_sharded, params_replicated)
    end = time.time()

    total_time = end - start
    total_energy = jnp.mean(local_energies)

    print(f"Mean Energy: {total_energy:.6f}")
    print(f"Total computation time: {total_time:.4f} seconds")

    if save_results:
        from pathlib import Path
        Path("results").mkdir(exist_ok=True)
        np.save("results/energy_density.npy", np.array(local_energies))

    return float(total_energy), total_time


# ---------- Single GPU Execution ----------
def run_single_gpu(batch_size, dim=128, layers=3, hidden_dim=128, save_results=False):
    num_devices = 1  # Single device mode
    print(f"Running Many-body Energy Estimator with {num_devices} GPU, batch_size={batch_size}, dim={dim}")

    key = jax.random.PRNGKey(0)
    subkeys = jax.random.split(key, 4)

    # Random neural net params
    params = [(jax.random.normal(subkeys[0], (hidden_dim, hidden_dim)),
               jax.random.normal(subkeys[1], (hidden_dim,))) for _ in range(layers)]

    # Input and jets
    x = jax.random.normal(subkeys[2], (batch_size, dim))
    jets = sample_jets(subkeys[3], batch_size, dim)

    # No sharding needed for single GPU
    x_sharded = jax.device_put(x, jax.devices()[0])
    jet_sharded = jax.device_put(jets, jax.devices()[0])

    # Replicate parameters across single GPU
    params_replicated = jax.device_put(params, jax.devices()[0])

    # Distributed computation (Single device)
    @jax.jit
    def single_gpu_energy(x, jet, params):
        # For single GPU, compute psi, laplacian, and energy density
        psi = net(params, x)
        laplacian = estimate_laplacian(params, x, jet)
        V = potential_energy(x)
        kinetic = -0.5 * laplacian
        energy_density = kinetic + V
        return energy_density

    # Timing the single-GPU execution
    start = time.time()
    local_energies = single_gpu_energy(x_sharded, jet_sharded, params_replicated)
    end = time.time()

    total_time = end - start
    total_energy = jnp.mean(local_energies)

    print(f"Mean Energy (Single GPU): {total_energy:.6f}")
    print(f"Total computation time (Single GPU): {total_time:.4f} seconds")

    if save_results:
        from pathlib import Path
        Path("results").mkdir(exist_ok=True)
        np.save("results/energy_density_single_gpu.npy", np.array(local_energies))

    return float(total_energy), total_time


# ---------- Main Function to Compare Results ----------
if __name__ == "__main__":
    batch_size = 256
    dim = 128

    # Run on multi-GPU
    multi_gpu_energy, multi_gpu_time = run_distributed_energy(batch_size=batch_size, dim=dim, save_results=True)

    # Run on single GPU for comparison
    single_gpu_energy, single_gpu_time = run_single_gpu(batch_size=batch_size, dim=dim, save_results=True)

    # Compare results
    print(f"\nComparison of Results:")
    print(f"Multi-GPU Mean Energy: {multi_gpu_energy:.6f} (Time: {multi_gpu_time:.4f}s)")
    print(f"Single-GPU Mean Energy: {single_gpu_energy:.6f} (Time: {single_gpu_time:.4f}s)")


#Running Many-body Energy Estimator with 4 GPUs, batch_size=128, dim=128
#Mean Energy: 60.312904
#Total computation time: 10.3192 seconds
#Running Many-body Energy Estimator with 1 GPU, batch_size=128, dim=128
#Mean Energy (Single GPU): 60.312904
#Total computation time (Single GPU): 43.5802 seconds

#Comparison of Results:
#Multi-GPU Mean Energy: 60.312904 (Time: 10.3192s)
#Single-GPU Mean Energy: 60.312904 (Time: 43.5802s)
