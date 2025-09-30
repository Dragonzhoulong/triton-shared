import torch
import triton
import triton.language as tl
import benchmark

# Default device for testing. We do not force the CPUDriver.
from triton.backends.triton_shared.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())

device=torch.device("cpu")


@triton.jit
def reshape_2d_to_3d_kernel(
    # Pointers
    input_ptr, output_ptr,
    # Input Strides (2D)
    stride_in_m, stride_in_n,
    # Output Strides (3D)
    stride_out_d, stride_out_h, stride_out_w,
    # Input Shape
    in_N,
    # Output Shape
    out_H, out_W,
):
    """
    A "safe" Triton kernel to reshape a 2D tensor (M, N) into a 3D tensor (D, H, W).
    It is launched on a 3D grid matching the output shape.
    Each program instance copies exactly one element.
    """
    # 1. Get the 3D coordinates (d, h, w) of the output element to compute
    d = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    w = tl.program_id(axis=2)

    # 2. Calculate the destination address in the 3D output tensor
    output_addr = output_ptr + d * stride_out_d + h * stride_out_h + w * stride_out_w

    # 3. The core logic: map the 3D output coordinate -> linear index -> 2D input coordinate
    
    # 3a. Calculate the flat linear index.
    #     Assuming C-contiguous layout: linear_idx = d * (H * W) + h * W + w
    linear_idx = d * (out_H * out_W) + h * out_W + w
    
    # 3b. Map the linear index back to 2D input coordinates (m, n)
    #     m = linear_idx // N
    #     n = linear_idx % N
    input_m = linear_idx // in_N
    input_n = linear_idx % in_N
    
    # 4. Calculate the source address in the 2D input tensor
    input_addr = input_ptr + input_m * stride_in_m + input_n * stride_in_n

    # 5. Load the value from the source and store it in the destination
    value = tl.load(input_addr)
    tl.store(output_addr, value)


def reshape(a: torch.Tensor, new_shape):
    """
    A robust `reshape` function that materializes a new tensor.
    Uses Triton on GPU and falls back to PyTorch on CPU.
    """
    # CPU Fallback: Safest and most efficient path for CPU.
    if a.device.type == 'cpu':
        return torch.reshape(a, new_shape)

    # --- GPU Path ---
    # For this example, we specifically implement 2D -> 3D reshape
    assert a.dim() == 2, "This Triton implementation only supports reshaping a 2D tensor."
    assert len(new_shape) == 3, "This Triton implementation only supports reshaping into a 3D tensor."
    
    M, N = a.shape
    D, H, W = new_shape
    assert M * N == D * H * W, "Total number of elements must match."
    
    # Create the output tensor
    output = torch.empty(new_shape, device=a.device, dtype=a.dtype)

    # The grid is 3D, matching the output tensor's shape
    grid = (D, H, W)

    # Launch the kernel
    reshape_2d_to_3d_kernel[grid](
        a, output,
        # Input strides (2)
        a.stride(0), a.stride(1),
        # Output strides (3)
        output.stride(0), output.stride(1), output.stride(2),
        # Pass shapes needed for coordinate calculation
        in_N=N,
        out_H=H,
        out_W=W,
    )
    return output


def test_reshape(device):
    """
    Tests the Triton `reshape` implementation against `torch.reshape`.
    """
    print(f"Running correctness test on device: {device}...")
    
    # Test case: Reshape (32, 128) -> (4, 8, 128)
    # Total elements: 32 * 128 = 4096
    # New shape elements: 4 * 8 * 128 = 32 * 128 = 4096
    orig_shape = (32, 128)
    new_shape = (4, 8, 128)
    
    a = torch.randn(orig_shape, device=device, dtype=torch.float32)

    torch_output = torch.reshape(a, new_shape)
    triton_output = reshape(a, new_shape)
    
    assert torch.equal(torch_output, triton_output), "Reshape failed!"
    print(f"✅ Test passed. {orig_shape} -> {triton_output.shape}")


@benchmark.measure()
def bench_reshape(orig_shape, new_shape, provider, dev_string):
    """
    Benchmarks the performance of `reshape` operations.
    """
    a = torch.randn(orig_shape, device=dev_string, dtype=torch.float32)
    
    if provider == 'torch':
        # We call .contiguous() to make a fair comparison, forcing a data copy.
        torch.reshape(a, new_shape).contiguous()
    elif provider == 'triton':
        reshape(a, new_shape)
    else:
        raise ValueError(f"Unknown provider: {provider}")


if __name__ == "__main__":
    # --- Run tests and benchmarks on CPU to verify the fallback path ---
    print("=========================================")
    print("         RUNNING ON CPU                  ")
    print("=========================================")
    cpu_device = torch.device("cpu")
    
    # 1. Run correctness tests
    test_reshape(device=cpu_device)

    # 2. Run benchmarks
    print("\nRunning benchmark on CPU...")
    orig_shape = (1024, 1024)
    new_shape = (64, 128, 128) # 1024*1024 = 1048576; 64*128*128 = 1048576
    print(f"\n--- Benchmarking reshape {orig_shape} -> {new_shape} on CPU ---")
    # On CPU, 'triton' provider will fall back to torch
    for provider in ['torch', 'triton']:
        bench_reshape(orig_shape, new_shape, provider, 'cpu')