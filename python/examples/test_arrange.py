import torch
import triton
import triton.language as tl
import benchmark

# 设置 Triton 在 CPU 上运行 (用于开发和测试)
from triton.backends.triton_shared.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())
device=torch.device("cpu")


@triton.jit
def arange_kernel(
    # 输出张量的指针
    output_ptr,
    # 张量中的元素总数
    num_elements,
    # 输出张量的步长 (对于 1D 张量，通常是 1)
    stride_o,
    # Meta-parameter: 每个程序实例处理的元素数量
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to create a 1D tensor with values equal to their indices.
    """
    # -----------------------------------------------------------
    # 1. 获取程序 ID (pid)。由于是 1D 网格，我们只关心 axis=0
    #    `pid` 告诉我们当前程序实例是第几个块 (block)
    # -----------------------------------------------------------
    pid = tl.program_id(axis=0)

    # -----------------------------------------------------------
    # 2. 计算当前块负责的全局索引 (offsets)
    #    例如，如果 BLOCK_SIZE=1024, 第 0 个程序处理 [0, 1023],
    #    第 1 个程序处理 [1024, 2047]，以此类推。
    # -----------------------------------------------------------
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # -----------------------------------------------------------
    # 3. 创建一个掩码 (mask) 以防止写入越界
    #    这对于最后一个块尤其重要，因为它可能不需要填满整个 BLOCK_SIZE
    # -----------------------------------------------------------
    mask = offsets < num_elements

    # -----------------------------------------------------------
    # 4. 计算要写入的值。对于 arange, 值就是索引本身。
    # -----------------------------------------------------------
    values = offsets.to(tl.float32) # 将索引转换为浮点数以进行存储

    # -----------------------------------------------------------
    # 5. 计算内存地址并将值写入
    # -----------------------------------------------------------
    output_ptrs = output_ptr + offsets * stride_o
    tl.store(output_ptrs, values, mask=mask)


def arange(end, dtype=torch.float32, device='cpu'):
    """
    Triton implementation of torch.arange for 1D tensors.
    """
    # 获取要创建的元素总数
    N = end
    
    # 分配一个空的输出张量
    output = torch.empty((N,), device=device, dtype=dtype)
    
    # 如果要创建的张量为空，直接返回
    if N == 0:
        return output

    # 启动内核的网格 (Grid)
    # 这是一个 1D 网格。网格大小是元素总数除以块大小，向上取整。
    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
    
    # 启动内核
    # 注意，我们选择了一个相对较大的 BLOCK_SIZE，因为这是一个内存密集型操作，
    # 较大的块可以减少内核启动的开销。
    arange_kernel[grid](
        output,
        N,
        output.stride(0),
        BLOCK_SIZE=1024,
    )
    return output


def test_arange(device):
    """
    Tests the Triton `arange` implementation against `torch.arange`.
    """
    # 定义测试用的张量大小
    N = 4096
    
    # 运行 Triton 和 PyTorch 的实现
    triton_output = arange(N, device=device, dtype=torch.float32)
    torch_output = torch.arange(N, device=device, dtype=torch.float32)

    # 比较结果
    print("Running correctness test...")
    print(f"Size: {N}")
    print("Triton output shape:", triton_output.shape)
    print("PyTorch output shape:", torch_output.shape)
    # arange 的结果应该是精确相等的
    assert torch.equal(triton_output, torch_output), "Tensor values are not equal"
    print("✅ Test passed!")


@benchmark.measure()
def bench_arange(N, provider):
    """
    Benchmarks the performance of `arange` operations.
    """
    if provider == 'torch':
        torch.arange(N, device='cpu', dtype=torch.float32)
    elif provider == 'triton':
        arange(N, device='cpu', dtype=torch.float32)
    else:
        raise ValueError(f"Unknown provider: {provider}")


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    
    # 1. 首先运行正确性测试
    test_arange(device=device)

    print("\nRunning benchmark...")
    # 2. 然后运行基准测试
    # 针对不同大小的张量进行基准测试
    for N in [512, 1024, 2048, 4096, 8192, 16384]:
        
        print(f"\n--- Benchmarking for size={N} ---")
        # 对比 torch 和 triton 的性能
        for provider in ['torch', 'triton']:
             bench_arange(N, provider)