import torch
import triton
import triton.language as tl
import benchmark

# 设置 Triton 在 CPU 上运行 (用于开发和测试)
from triton.backends.triton_shared.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())
device=torch.device("cpu")


@triton.jit
def transpose_kernel(
    input_ptr,
    output_ptr,
    M, N,
    stride_im, stride_in,
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    A Triton kernel for 2D matrix transpose, inspired by the arange/broadcasting style.
    Each program in the grid transposes one block.
    """
    # 1. 确定当前程序实例负责处理哪个块
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # 2. 计算当前块的起始偏移量
    #    offs_m_base 是当前块所有行的起始索引
    #    offs_n_base 是当前块所有列的起始索引
    offs_m_base = pid_m * BLOCK_SIZE_M
    offs_n_base = pid_n * BLOCK_SIZE_N

    # 3. 使用 tl.arange 创建块内的相对偏移量
    #    offs_m_arange -> [0, 1, 2, ..., BLOCK_SIZE_M-1]
    #    offs_n_arange -> [0, 1, 2, ..., BLOCK_SIZE_N-1]
    offs_m_arange = tl.arange(0, BLOCK_SIZE_M)
    offs_n_arange = tl.arange(0, BLOCK_SIZE_N)

    # 4. 计算输入块中每个元素的绝对索引
    #    通过将基础偏移量与相对偏移量相加得到
    offs_m = offs_m_base + offs_m_arange
    offs_n = offs_n_base + offs_n_arange
    
    # 5. 计算输入和输出的指针
    #    这部分与您提供的 reshape_kernel 逻辑一致，但应用于块
    #    输入指针: 正常访问 (m, n)
    input_ptrs = input_ptr + (offs_m[:, None] * stride_im + offs_n[None, :] * stride_in)
    #    输出指针: 转置访问 (n, m)
    output_ptrs = output_ptr + (offs_n[:, None] * stride_om + offs_m[None, :] * stride_on)

    # 6. 创建掩码以处理边界情况（非方形或大小不完美的矩阵）
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # 7. 加载、存储
    block = tl.load(input_ptrs, mask=mask, other=0.0)
    tl.store(output_ptrs, block, mask=mask)


def transpose(a: torch.Tensor):
    """
    Triton implementation of torch.transpose for 2D tensors.
    """
    assert a.is_contiguous(), "Input tensor must be contiguous"
    assert a.dim() == 2, "Only 2D tensors are supported"
    
    M, N = a.shape
    output = torch.empty((N, M), device=a.device, dtype=a.dtype)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    transpose_kernel[grid](
        a, output,
        M, N,
        a.stride(0), a.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=16,
    )
    return output


def test_transpose(device):
    """
    Tests the Triton `transpose` implementation against `torch.transpose`.
    """
    shape = (179, 321)
    a = torch.randn(shape, device=device, dtype=torch.float32)
    
    torch_output = torch.transpose(a, 0, 1)
    triton_output = transpose(a)

    print("Running correctness test...")
    print(f"Input shape: {a.shape}")
    print("Triton output shape:", triton_output.shape)
    print("PyTorch output shape:", torch_output.shape)
    
    assert torch.equal(triton_output, torch_output), "Transpose implementation failed!"
    print("✅ Test passed!")


@benchmark.measure()
def bench_transpose(M, N, provider):
    """
    Benchmarks the performance of `transpose` operations.
    """
    a = torch.randn((M, N), device='cpu', dtype=torch.float32)
    if provider == 'torch':
        torch.transpose(a, 0, 1).contiguous()
    elif provider == 'triton':
        transpose(a)
    else:
        raise ValueError(f"Unknown provider: {provider}")


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    
    # 1. 运行正确性测试
    test_transpose(device=device)

    print("\nRunning benchmark...")
    # 2. 运行基准测试
    for M in [128, 512, 1024, 2048]:
        N = M
        print(f"\n--- Benchmarking for shape=({M}, {N}) ---")
        for provider in ['torch', 'triton']:
             bench_transpose(M, N, provider)
             
        N = M // 2
        print(f"\n--- Benchmarking for shape=({M}, {N}) ---")
        for provider in ['torch', 'triton']:
             bench_transpose(M, N, provider)