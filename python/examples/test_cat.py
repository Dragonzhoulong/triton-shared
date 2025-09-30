import torch
import triton
import triton.language as tl
import benchmark

# Default device for testing.
# We will not force the CPUDriver anymore to allow for robust fallback.
from triton.backends.triton_shared.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())

device=torch.device("cpu")


@triton.jit
def cat_2d_kernel(
    # Pointers to the two input tensors and the output tensor
    input_ptr_1, input_ptr_2, output_ptr,
    # Shapes and strides for all tensors
    M1, N1,
    M2, N2,
    out_M, out_N,
    stride_in1_m, stride_in1_n,
    stride_in2_m, stride_in2_n,
    stride_out_m, stride_out_n,
    # The dimension to concatenate along (0 for rows, 1 for columns)
    DIM: tl.constexpr,
    # The size of the first tensor in the concatenation dimension, used as the split point
    split_point: tl.constexpr,
):
    """
    A "safe" Triton kernel to concatenate two 2D tensors.
    Each program instance handles one element of the output tensor.
    """
    # 1. Get the global row (m) and column (n) of the output element to compute
    m = tl.program_id(axis=0)
    n = tl.program_id(axis=1)

    # Pointer to the destination address in the output tensor
    output_addr = output_ptr + m * stride_out_m + n * stride_out_n

    # 2. The core logic: decide which input tensor to read from
    if DIM == 0:
        # Concatenating along rows
        if m < split_point:
            # This element comes from the first tensor
            # The local coordinates are the same as the global ones: (m, n)
            input_addr = input_ptr_1 + m * stride_in1_m + n * stride_in1_n
            value = tl.load(input_addr)
            tl.store(output_addr, value)
        else:
            # This element comes from the second tensor
            # We need to calculate the local row index: m - split_point
            local_m = m - split_point
            input_addr = input_ptr_2 + local_m * stride_in2_m + n * stride_in2_n
            value = tl.load(input_addr)
            tl.store(output_addr, value)
    else: # DIM == 1
        # Concatenating along columns
        if n < split_point:
            # This element comes from the first tensor
            # The local coordinates are (m, n)
            input_addr = input_ptr_1 + m * stride_in1_m + n * stride_in1_n
            value = tl.load(input_addr)
            tl.store(output_addr, value)
        else:
            # This element comes from the second tensor
            # We need to calculate the local column index: n - split_point
            local_n = n - split_point
            input_addr = input_ptr_2 + m * stride_in2_m + local_n * stride_in2_n
            value = tl.load(input_addr)
            tl.store(output_addr, value)


def cat(tensors, dim):
    """
    A robust `cat` function for two 2D tensors.
    Uses Triton on GPU and falls back to PyTorch on CPU.
    """
    # For now, this implementation is limited to two tensors
    assert len(tensors) == 2, "This Triton `cat` implementation only supports two tensors."
    a, b = tensors
    
    # Robust CPU Fallback
    if a.device.type == 'cpu':
        return torch.cat(tensors, dim=dim)

    # --- GPU Path ---
    assert a.dim() == 2 and b.dim() == 2, "Only 2D tensors are supported"

    # Validate shapes
    if dim == 0:
        assert a.shape[1] == b.shape[1], "Tensors must have same number of columns to cat on dim 0"
    else: # dim == 1
        assert a.shape[0] == b.shape[0], "Tensors must have same number of rows to cat on dim 1"

    # Calculate output shape and create output tensor
    output_shape = list(a.shape)
    output_shape[dim] += b.shape[dim]
    output = torch.empty(output_shape, device=a.device, dtype=a.dtype)

    # The grid size is the shape of the output tensor
    grid = (output.shape[0], output.shape[1])
    
    # The split point is the size of the first tensor along the cat dimension
    split_point = a.shape[dim]
    
    # Launch the kernel
    cat_2d_kernel[grid](
        a, b, output,
        a.shape[0], a.shape[1],
        b.shape[0], b.shape[1],
        output.shape[0], output.shape[1],
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        output.stride(0), output.stride(1),
        DIM=dim,
        split_point=split_point,
    )
    return output


def test_cat(device):
    """
    Tests the Triton `cat` implementation against `torch.cat`.
    """
    print(f"Running correctness test on device: {device}...")
    
    # Test case 1: Concatenate along dimension 0 (rows)
    a_dim0 = torch.randn((64, 128), device=device, dtype=torch.float32)
    b_dim0 = torch.randn((32, 128), device=device, dtype=torch.float32)
    
    torch_output_0 = torch.cat([a_dim0, b_dim0], dim=0)
    triton_output_0 = cat([a_dim0, b_dim0], dim=0)
    
    assert torch.equal(torch_output_0, triton_output_0), "Cat on dim=0 failed!"
    print("✅ Test passed for dim=0")

    # Test case 2: Concatenate along dimension 1 (columns)
    a_dim1 = torch.randn((256, 48), device=device, dtype=torch.float32)
    b_dim1 = torch.randn((256, 96), device=device, dtype=torch.float32)

    torch_output_1 = torch.cat([a_dim1, b_dim1], dim=1)
    triton_output_1 = cat([a_dim1, b_dim1], dim=1)

    assert torch.equal(torch_output_1, triton_output_1), "Cat on dim=1 failed!"
    print("✅ Test passed for dim=1")


@benchmark.measure()
def bench_cat(m1, n1, m2, n2, dim, provider, dev_string):
    """
    Benchmarks the performance of `cat` operations.
    """
    a = torch.randn((m1, n1), device=dev_string, dtype=torch.float32)
    b = torch.randn((m2, n2), device=dev_string, dtype=torch.float32)
    
    if provider == 'torch':
        torch.cat([a, b], dim=dim)
    elif provider == 'triton':
        cat([a, b], dim=dim)
    else:
        raise ValueError(f"Unknown provider: {provider}")


if __name__ == "__main__":
    # --- Run tests and benchmarks on CPU to verify the fallback path ---
    print("=========================================")
    print("         RUNNING ON CPU                  ")
    print("=========================================")
    cpu_device = torch.device("cpu")
    
    # 1. Run correctness tests
    test_cat(device=cpu_device)

    # 2. Run benchmarks
    print("\nRunning benchmark on CPU...")
    # Benchmark dim=0
    m1, m2, n = 1024, 1024, 512
    print(f"\n--- Benchmarking cat dim=0 for shapes=({m1},{n}), ({m2},{n}) on CPU ---")
    for provider in ['torch', 'triton']:
        bench_cat(m1, n, m2, n, 0, provider, 'cpu')
        
    # Benchmark dim=1
    m, n1, n2 = 512, 1024, 1024
    print(f"\n--- Benchmarking cat dim=1 for shapes=({m},{n1}), ({m},{n2}) on CPU ---")
    for provider in ['torch', 'triton']:
        bench_cat(m, n1, m, n2, 1, provider, 'cpu')