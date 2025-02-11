#load both the pytorch (for tensor handling) and time library (to record time for each function)
import torch
import time

# let's define the function
def f(x):
    return torch.exp(x) #fx = e^x

# create functions to evaluate using AD
def compute_with_typical_ad(inputs):
    start_time = time.time()
    for x in inputs:
        x = x.requires_grad_()
        y = f(x)
        y.backward()
    end_time = time.time()
    return end_time - start_time

# AD + Taylor: Compute derivatives once and reuse
def compute_with_taylor_expansion(inputs, expansion_point):
    start_time = time.time()
    # Compute derivatives at the expansion point
    a = torch.tensor([expansion_point], requires_grad=True)
    y = f(a)
    y.backward()
    f_prime = a.grad  # First derivative

    # Use Taylor expansion to approximate for other inputs
    for x in inputs:
        _ = y.item() + f_prime.item() * (x - expansion_point)
    end_time = time.time()
    return end_time - start_time

# lett's perform testing the input below
inputs = torch.tensor([1.1, 1.2, 1.3, 1.4, 1.5], dtype=torch.float32)
expansion_point = 1.0 #the value nearest to a

# Measure time for both methods
time_typical = compute_with_typical_ad(inputs)
time_taylor = compute_with_taylor_expansion(inputs, expansion_point)

print(f"Time for typical AD: {time_typical:.6f} seconds")
print(f"Time for AD with Taylor expansion: {time_taylor:.6f} seconds")
print(f"AD with taylor expansion is {time_typical/time_taylor:.0f} times faster than typical AD")