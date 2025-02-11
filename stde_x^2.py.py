import torch
import random

class STDE:
    def __init__(
        self,
        input_dim,
        order,
        operator_coefficients,
        rand_batch_size=0,
        use_abs_for_sampling=True,
    ):
        """
        Initialize the STDE class.

        Args:
            input_dim (int): Dimensionality of the input space (e.g., 2 for 2D).
            order (int): Maximum order of the differential operator (optional usage).
            operator_coefficients (dict): Coefficients for the operator, specified as:
                {multi_index (tuple of ints): coefficient (float)},
                where each multi_index is a sequence of dimension indices.
                E.g., (0,0) means d^2/dx^2 for dimension 0 in 2D.
            rand_batch_size (int): Number of dimensions to sample for sparse computation. 
                                   If 0, use all dimensions.
            use_abs_for_sampling (bool): If True, use the absolute value of coefficients 
                                         to compute sampling probabilities. This helps
                                         avoid negative or zero probabilities.
        """
        self.input_dim = input_dim
        self.order = order  # You can use this to validate multi-index lengths if desired.
        self.coefficients = operator_coefficients

        # Validate that multi-indices don't exceed the declared order (optional check)
        for idx in self.coefficients.keys():
            if len(idx) > self.order:
                raise ValueError(f"Multi-index {idx} exceeds the specified order {self.order}")

        # Convert your coefficient dict into a list of (multi_index, coeff) so we can sample easily
        self.indices = list(self.coefficients.keys())  # Multi-index terms
        self.rand_batch_size = rand_batch_size or input_dim  # Default to all dimensions if 0

        # Compute sampling probabilities (handle negatives / zero sum)
        self.use_abs_for_sampling = use_abs_for_sampling
        self.probabilities = self._compute_sampling_probabilities()

    def _compute_sampling_probabilities(self):
        """
        Compute sampling probabilities for each multi-index based on coefficients.
        Returns:
            list: Sampling probabilities for each term in self.indices.
        """
        coeff_values = torch.tensor(
            [self.coefficients[idx] for idx in self.indices],
            dtype=torch.float32
        )
        if self.use_abs_for_sampling:
            # Use absolute values for probabilities to avoid negative or zero-sum
            prob_values = coeff_values.abs()
        else:
            # If you *know* all coefficients are non-negative or can handle sign logic differently
            prob_values = coeff_values.clone()

        total_sum = prob_values.sum().item()

        # If total_sum is 0, all coefficients might be zero or negative. 
        # Handle carefully (e.g., uniform distribution among all indices or raise error).
        if total_sum == 0.0:
            raise ValueError("Sum of absolute coefficients is zero. Cannot sample indices.")

        return (prob_values / total_sum).tolist()

    def _generate_random_jets(self, num_samples):
        """
        Generate random sparse k-jets for the input dimension and order.

        Args:
            num_samples (int): Number of jets to generate.

        Returns:
            list of tuples: List of sampled multi-indices.
        """
        sampled_indices = random.choices(self.indices, weights=self.probabilities, k=num_samples)
        return sampled_indices

    def _sample_dimensions(self):
        """
        Randomly sample a subset of dimensions for sparse computation.

        Returns:
            Tensor: Indices of sampled dimensions.
        """
        if self.rand_batch_size < self.input_dim:
            return torch.randperm(self.input_dim)[:self.rand_batch_size]
        return torch.arange(self.input_dim)

    def apply_operator(self, func, inputs, num_samples=10):
        """
        Apply the differential operator using sparse random jets.

        Args:
            func (callable): Function u(x, ...) to which the operator is applied. 
                             Should return a tensor (possibly with batch dimensions).
            inputs (list of torch.Tensor): List of input tensors, each requires_grad=True.
            num_samples (int): Number of random jets to sample.

        Returns:
            torch.Tensor: Approximated operator value at the given inputs (scalar).
        """
        # Start operator_value as a torch Tensor (for device consistency)
        operator_value = torch.zeros(1, dtype=inputs[0].dtype, device=inputs[0].device)

        # Gather random multi-indices (jets) and dimension subset
        sampled_jets = self._generate_random_jets(num_samples)
        sampled_dims = self._sample_dimensions()

        for jet in sampled_jets:
            # Perform a forward pass once for this jet
            # If your function is expensive, see notes on optimizing repeated calls below.
            output = func(*inputs)

            # If output has a batch dimension, reduce to scalar now so that
            # the derivative is well-defined. (We do this once at the start.)
            output = output.mean()  # ensures scalar

            # We'll apply partial derivatives in sequence
            for order_idx, dim_idx in enumerate(jet):
                # If dim_idx is not in sampled_dims, skip it
                if dim_idx not in sampled_dims:
                    # For this code path, we do nothing and break from partial derivatives
                    # because once we skip, we can't keep differentiating something we didn't compute.
                    break

                # Compute the gradient of `output` w.r.t. inputs[dim_idx].
                grad = torch.autograd.grad(
                    outputs=output,
                    inputs=inputs[dim_idx],
                    grad_outputs=None,  # = ones_like(output) for a scalar, if output is scalar
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0]

                # If grad is None, dimension not used, or something else. Just treat it as zero.
                if grad is None:
                    grad = torch.zeros_like(inputs[dim_idx])
                
                # If we're not at the last partial derivative, we need
                # to set `output` to something that we can differentiate again.
                # `grad` might be a non-scalar if inputs[dim_idx] has shape > 1.
                # We can reduce it to a scalar to differentiate again.
                if order_idx < len(jet) - 1:
                    output = grad.mean()
                else:
                    # This is the last partial derivative in `jet`, so we add it to the operator.
                    # The sign of the coefficient is the same as in the dictionary if we're not
                    # using absolute values for sampling. If using abs for sampling, multiply
                    # by sign.
                    coeff = self.coefficients.get(jet, 0.0)
                    if self.use_abs_for_sampling:
                        # Use the original sign (coeff could be negative)
                        # prob was computed with abs, but we still want the correct sign for PDE
                        # so we do:
                        # contribution = sign(coeff) * grad.mean()
                        # or just directly multiply by 'coeff', since we do not lose sign in the dict.
                        contribution = coeff * grad.mean()
                    else:
                        contribution = coeff * grad.mean()
                        
                    operator_value += contribution

        # -- Dimension sampling correction
        # We are skipping partial derivatives on unsampled dimensions.
        # This factor "scales" the result to approximate what we would get with all dimensions.
        # NOTE: This is a heuristic. In some PDEs, it can introduce bias or high variance.
        correction_factor = self.input_dim / float(len(sampled_dims))
        operator_value *= correction_factor

        # Normalize by number of sampled jets
        operator_value /= num_samples

        return operator_value
    

import torch
import random

def test_stde_1d(STDE_class):
    """
    Test the STDE class on a simple 1D function f(x) = x^2,
    whose second derivative is 2 everywhere.

    Args:
        STDE_class: The STDE class to be tested.
    """

    # ----- Reproducibility -----
    torch.manual_seed(0)
    random.seed(0)

    # ----- Define the test function f(x) -----
    # f(x) = x^2
    def f(x):
        return x**2

    # ----- Define operator coefficients for the 1D second derivative -----
    # In your format, (0, 0) means "take second derivative wrt dimension 0"
    operator_coeffs = {
        (0, 0): 1.0  # coefficient = 1 for second derivative
    }

    # Create an STDE instance
    stde = STDE_class(
        input_dim=1,           # only one dimension
        order=2,               # up to second derivative
        operator_coefficients=operator_coeffs,
        rand_batch_size=0,     # use all dimensions (just 1 anyway)
        use_abs_for_sampling=True
    )

    # ----- Define input x -----
    x = torch.tensor([3.0], requires_grad=True)  # choose any value, e.g. 3.0

    # ----- Apply the operator -----
    # Increase num_samples for better accuracy if needed
    approx_value = stde.apply_operator(f, [x], num_samples=1000)

    # ----- Print results -----
    print("Approximate second derivative of x^2 at x=3.0:", approx_value.item())
    print("Exact second derivative is 2.0")
    print("Error:", abs(approx_value.item() - 2.0))


test_stde_1d(STDE)