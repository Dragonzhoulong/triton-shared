import torch
import triton
import triton.language as tl
import benchmark

# 设置 Triton 在 CPU 上运行 (用于开发和测试)
from triton.backends.triton_shared.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())
device=torch.device("cpu")


@triton.jit
def ones_kernel(
    # 输出张量的指针
    output_ptr,
    # 输出张量的维度
    M, N,
    # 输出张量的步长 (strides)
    stride_om, stride_on,
    # Meta-parameters for launching the kernel
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Triton kernel to fill a 2D tensor with ones.
    """
    # -----------------------------------------------------------
    # 1. 映射程序 ID (pid) 到它应该计算的输出块
    #    每个程序实例处理输出张量的一个块 (block)
    # -----------------------------------------------------------
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # -----------------------------------------------------------
    # 2. 计算当前块内每个元素的行和列偏移量
    # -----------------------------------------------------------
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # -----------------------------------------------------------
    # 3. 创建指向输出块的指针
    #    这结合了基地址、偏移量和步长来计算每个元素的内存地址
    # -----------------------------------------------------------
    output_ptrs = output_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    
    # -----------------------------------------------------------
    # 4. 创建一个掩码 (mask) 以防止内存越界
    #    这确保我们只在张量的有效范围内写入数据
    # -----------------------------------------------------------
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # -----------------------------------------------------------
    # 5. 将数值 1.0 写入到指定的内存位置
    #    `tl.store` 会根据掩码来决定是否执行写入操作
    # -----------------------------------------------------------
    one = 1.0
    tl.store(output_ptrs, one, mask=mask)


def ones(shape, dtype=torch.float32, device='cpu'):
    """
    Triton implementation of torch.ones for 2D tensors.
    """
    # 检查约束
    assert len(shape) == 2, "Only 2D shapes are supported for this implementation"
    M, N = shape
    
    # 分配一个空的输出张量
    # 内核将在这个张量上进行写入操作
    output = torch.empty(shape, device=device, dtype=dtype)
    
    # 启动内核的网格 (Grid)
    # 网格的大小应该向上取整，以确保覆盖整个输出张量
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    # 启动内核
    ones_kernel[grid](
        output,
        M, N,
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=16,
    )
    return output


def test_ones(device):
    """
    Tests the Triton `ones` implementation against `torch.ones`.
    """
    # 定义测试用的张量形状
    shape = (179, 321)
    
    # 运行 Triton 和 PyTorch 的实现
    triton_output = ones(shape, device=device, dtype=torch.float32)
    torch_output = torch.ones(shape, device=device, dtype=torch.float32)

    # 比较结果
    print("Running correctness test...")
    print(f"Shape: {shape}")
    print("Triton output shape:", triton_output.shape)
    print("PyTorch output shape:", torch_output.shape)
    # 对于 `ones` 操作，结果应该是精确相等的，所以使用 `torch.equal`
    assert torch.equal(triton_output, torch_output), "Tensor values are not equal"
    print("✅ Test passed!")


@benchmark.measure()
def bench_ones(M, N, provider):
    """
    Benchmarks the performance of `ones` operations.
    """
    shape = (M, N)
    if provider == 'torch':
        torch.ones(shape, device='cpu', dtype=torch.float32)
    elif provider == 'triton':
        ones(shape, device='cpu', dtype=torch.float32)
    else:
        raise ValueError(f"Unknown provider: {provider}")


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    
    # 1. 首先运行正确性测试
    test_ones(device=device)

    print("\nRunning benchmark...")
    # 2. 然后运行基准测试
    # 针对不同大小的张量进行基准测试
    for M in [128, 512, 1024, 2048, 4096]:
        N = M
        
        print(f"\n--- Benchmarking for shape=({M}, {N}) ---")
        # 对比 torch 和 triton 的性能
        for provider in ['torch', 'triton']:
             bench_ones(M, N, provider)