import torch
import triton
import triton.language as tl
import benchmark

# Default device for testing. We do not force the CPUDriver.
from triton.backends.triton_shared.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())

device=torch.device("cpu")


@triton.jit
def unsqueeze_2d_to_3d_kernel(
    # Pointers
    input_ptr, output_ptr,
    # Strides
    stride_im, stride_in,
    stride_od, stride_om, stride_on,
    # The dimension to unsqueeze
    DIM: tl.constexpr,
):
    """
    A "safe" Triton kernel to unsqueeze a 2D tensor into a 3D tensor.
    This kernel is launched on a 3D grid matching the output shape.
    Each program instance copies exactly one element.
    """
    # 1. Get the 3D coordinates (d, m, n) of the output element to compute
    d = tl.program_id(axis=0)
    m = tl.program_id(axis=1)
    n = tl.program_id(axis=2)

    # 2. Calculate the destination address in the 3D output tensor
    output_addr = output_ptr + d * stride_od + m * stride_om + n * stride_on

    # 3. The core logic: map the 3D output coordinate back to a 2D input coordinate
    #    and calculate the source address.
    if DIM == 0:
        # Output (0, m, n) maps to Input (m, n)
        input_addr = input_ptr + m * stride_im + n * stride_in
    elif DIM == 1:
        # Output (d, 0, n) maps to Input (d, n)
        # Note: In this case, program_id(0) is the first dim of the input,
        # and program_id(2) is the second dim.
        input_addr = input_ptr + d * stride_im + n * stride_in
    else: # DIM == 2
        # Output (d, m, 0) maps to Input (d, m)
        input_addr = input_ptr + d * stride_im + m * stride_in

    # 4. Load the value from the source and store it in the destination
    value = tl.load(input_addr)
    tl.store(output_addr, value)


def unsqueeze(a: torch.Tensor, dim: int):
    """
    A robust `unsqueeze` function that materializes a new tensor.
    Uses Triton on GPU and falls back to PyTorch on CPU.
    """
    # CPU Fallback: This is the safest and most efficient path for CPU.
    if a.device.type == 'cpu':
        return torch.unsqueeze(a, dim)

    # --- GPU Path ---
    assert a.dim() == 2, "This Triton implementation only supports unsqueezing a 2D tensor."
    assert 0 <= dim <= 2, "Dimension must be 0, 1, or 2 for unsqueezing a 2D tensor."

    # Calculate output shape by inserting a '1' at the specified dimension
    output_shape = list(a.shape)
    output_shape.insert(dim, 1)
    
    # Create the output tensor
    output = torch.empty(output_shape, device=a.device, dtype=a.dtype)

    # The grid is 3D, matching the output tensor's shape
    grid = (output.shape[0], output.shape[1], output.shape[2])

    # Launch the kernel
    unsqueeze_2d_to_3d_kernel[grid](
        a, output,
        # Input strides (2)
        a.stride(0), a.stride(1),
        # Output strides (3)
        output.stride(0), output.stride(1), output.stride(2),
        DIM=dim,
    )
    return output


def test_unsqueeze(device):
    """
    Tests the Triton `unsqueeze` implementation against `torch.unsqueeze`.
    """
    print(f"Running correctness test on device: {device}...")
    
    a = torch.randn((128, 256), device=device, dtype=torch.float32)

    for dim in range(3):
        torch_output = torch.unsqueeze(a, dim)
        triton_output = unsqueeze(a, dim)
        
        assert torch.equal(torch_output, triton_output), f"Unsqueeze on dim={dim} failed!"
        print(f"✅ Test passed for dim={dim}. Shape: {triton_output.shape}")


@benchmark.measure()
def bench_unsqueeze(M, N, dim, provider, dev_string):
    """
    Benchmarks the performance of `unsqueeze` operations.
    """
    a = torch.randn((M, N), device=dev_string, dtype=torch.float32)
    
    if provider == 'torch':
        # We call .contiguous() to make a fair comparison, as our Triton
        # kernel always materializes a new, contiguous tensor.
        torch.unsqueeze(a, dim).contiguous()
    elif provider == 'triton':
        unsqueeze(a, dim)
    else:
        raise ValueError(f"Unknown provider: {provider}")


if __name__ == "__main__":
    # --- Run tests and benchmarks on CPU to verify the fallback path ---
    print("=========================================")
    print("         RUNNING ON CPU                  ")
    print("=========================================")
    cpu_device = torch.device("cpu")
    
    # 1. Run correctness tests
    test_unsqueeze(device=cpu_device)

    # 2. Run benchmarks
    print("\nRunning benchmark on CPU...")
    M, N = 1024, 1024
    for dim in range(3):
        print(f"\n--- Benchmarking unsqueeze dim={dim} for shape=({M},{N}) on CPU ---")
        # On CPU, 'triton' provider will fall back to torch
        for provider in ['torch', 'triton']:
            bench_unsqueeze(M, N, dim, provider, 'cpu')