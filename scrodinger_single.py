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
        return jnp.sum(net(params, z) ** 2)

    def estimator(x_i, v_i):
        _, jvp = jet(f, (x_i,), ((v_i,),))
        return jvp

    return jnp.array([estimator(x[i], jets[i]) for i in range(x.shape[0])])

# ---------- Single-GPU Runner ----------
def run_single_gpu_stde(batch_size=128, dim=128, layers=3, hidden_dim=128, save_results=False):
    print(f"Running Many-body Energy Estimator on 1 GPU, batch_size={batch_size}, dim={dim}")

    key = jax.random.PRNGKey(0)

    # Random neural net params
    params = [(jax.random.normal(key, (hidden_dim, hidden_dim)),
               jax.random.normal(key, (hidden_dim,))) for _ in range(layers)]

    # Input and jets
    x = jax.random.normal(key, (batch_size, dim))
    jets = sample_jets(key, batch_size, dim)

    # Time the computation
    start = time.time()
    laplacians = estimate_laplacian(params, x, jets)
    energies = 0.5 * laplacians  # Harmonic potential: Kinetic only here
    end = time.time()

    mean_energy = jnp.mean(energies)
    print(f"Mean Energy: {mean_energy:.6f}")
    print(f"Total computation time: {end - start:.4f} seconds")

    if save_results:
        from pathlib import Path
        Path("results").mkdir(exist_ok=True)
        np.save("results/laplacian_estimates_single_gpu.npy", np.array(laplacians))

    return mean_energy, end - start

if __name__ == "__main__":
    run_single_gpu_stde(batch_size=128, dim=128, save_results=True)

# Running Many-body Energy Estimator on 1 GPU, batch_size=128, dim=128
# Mean Energy: -1.006690
# Total computation time: 2.6487 seconds
