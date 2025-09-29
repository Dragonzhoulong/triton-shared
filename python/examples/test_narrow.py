import torch
import triton
import triton.language as tl
import benchmark

# 设置 Triton 在 CPU 上运行 (用于开发和测试)
from triton.backends.triton_shared.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())
device=torch.device("cpu")


@triton.jit
def narrow_kernel(
    # 输入和输出张量的指针
    input_ptr,
    output_ptr,
    # 输入张量的维度
    M, N,
    # 输入和输出张量的步长 (strides)
    # 步长表示在某个维度上移动一个元素时，内存地址需要增加多少
    stride_im, stride_in,
    stride_om, stride_on,
    # Narrow 操作的参数
    DIM: tl.constexpr,    # 要进行 narrow 操作的维度
    START: tl.constexpr,  # narrow 的起始索引
    # Meta-parameters for launching the kernel
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Triton kernel for a 2D narrow operation.
    Output has shape (length, N) if DIM is 0, or (M, length) if DIM is 1.
    """
    # -----------------------------------------------------------
    # 映射程序 ID (pid) 到它应该计算的输出块
    # 每个程序实例处理输出张量的一个块 (block)
    # -----------------------------------------------------------
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # -----------------------------------------------------------
    # 计算输出张量中的偏移量
    # tl.arange 创建一个从 0 到 BLOCK_SIZE 的序列
    # offs_m 和 offs_n 代表当前块内每个元素的行和列索引
    # -----------------------------------------------------------
    offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # -----------------------------------------------------------
    # 计算输入张量中对应的偏移量
    # 这是 narrow 操作的核心逻辑
    # -----------------------------------------------------------
    if DIM == 0:
        # 如果在维度 0 (行) 上 narrow，输入行索引 = 输出行索引 + start
        offs_im = offs_om + START
        offs_in = offs_on
    else: # DIM == 1
        # 如果在维度 1 (列) 上 narrow，输入列索引 = 输出列索引 + start
        offs_im = offs_om
        offs_in = offs_on + START

    # -----------------------------------------------------------
    # 创建输入和输出的指针块
    # -----------------------------------------------------------
    input_ptrs = input_ptr + (offs_im[:, None] * stride_im + offs_in[None, :] * stride_in)
    output_ptrs = output_ptr + (offs_om[:, None] * stride_om + offs_on[None, :] * stride_on)

    # -----------------------------------------------------------
    # 创建掩码 (mask) 以防止内存越界
    # 这是至关重要的，因为块的大小可能不是维度的整数倍
    # -----------------------------------------------------------
    # 获取输出张量的形状
    if DIM == 0:
        # 如果在维度 0 上 narrow, 输出的行数是 length
        output_shape_m = tl.num_programs(0) * BLOCK_SIZE_M # Placeholder, actual length is passed in python
        output_shape_n = N
    else: # DIM == 1
        output_shape_m = M
        output_shape_n = tl.num_programs(1) * BLOCK_SIZE_N # Placeholder

    # The python wrapper will ensure the grid is sized for the output,
    # but we still need to get the actual output dimensions for the mask.
    # A cleaner way is to pass output_shape_m and output_shape_n as arguments.
    # For simplicity here, we assume grid covers output exactly.
    # Let's refine this by passing output shapes.
    # (Note: This part is conceptually tricky. The most robust way is to pass output dimensions to the kernel)
    # For this implementation, the python wrapper will set a grid that matches the output tensor's shape,
    # so we just need to use the output offsets for masking.
    
    # 获取输出张量的形状 (由 Python 代码传入更佳，但这里为了简化)
    # 在这个例子中，我们假设启动网格完美匹配输出形状
    # 所以，我们直接用输出偏移量来创建掩码
    output_rows = tl.num_programs(0) * BLOCK_SIZE_M
    output_cols = tl.num_programs(1) * BLOCK_SIZE_N
    
    # 修正: 直接传递输出形状到内核是更清晰的做法。
    # 为了遵循简单模式，我们假设 python 端的 grid 计算是精确的
    # `output_mask` 确保我们只在输出张量的有效范围内写入
    
    # Let's stick to the simplest approach: the grid is launched according to the output shape.
    # The mask is therefore against the offsets of the output.
    # The python wrapper calculates the output shape. Let's assume M and N in the kernel
    # refer to the *output* shape for masking purposes. This is a common pattern.
    # Let's adjust the signature to be clearer.
    # REVISED KERNEL SIGNATURE AND LOGIC
    # See below for the final, cleaner implementation.

    # -- Final Kernel Implementation --
    # Let's pass all shape info explicitly for clarity.
    
    # @triton.jit -> See final code block for the complete, clean version.
    # The logic remains the same, but the parameters will be more explicit.
    # The core logic is loading from `input_ptrs` and storing to `output_ptrs`.
    
    # 从输入指针加载数据块
    # `mask` 参数确保我们不会读取输入张量边界之外的数据
    input_mask = (offs_im[:, None] >= 0) & (offs_im[:, None] < M) & \
                 (offs_in[None, :] >= 0) & (offs_in[None, :] < N)

    data = tl.load(input_ptrs, mask=input_mask, other=0.0)

    # 将数据块写入输出指针
    # 这里的掩码应该基于输出的形状，但由于网格是为输出配置的，
    # 并且我们通常假设块内的偏移量不会超出输出的逻辑边界（通过python的cdiv保证），
    # 我们可以简化这个掩码。
    # To be perfectly safe, a mask on output is good.
    output_shape_m = tl.num_programs(0) * BLOCK_SIZE_M
    output_shape_n = tl.num_programs(1) * BLOCK_SIZE_N
    # (This is still confusing).
    
    # -- LET'S RESTART THE KERNEL TO BE SUPER CLEAR AND ROBUST --
    # This is a better, more readable and robust kernel.

@triton.jit
def clean_narrow_kernel(
    input_ptr, output_ptr,
    # Input shape
    input_shape_m, input_shape_n,
    # Output shape
    output_shape_m, output_shape_n,
    # Strides
    stride_im, stride_in,
    stride_om, stride_on,
    # Narrow params
    DIM: tl.constexpr,
    START: tl.constexpr,
    # Meta params
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # 1. Calculate offsets for the *output* block this program is responsible for
    offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # 2. Create pointers to the output block
    output_ptrs = output_ptr + (offs_om[:, None] * stride_om + offs_on[None, :] * stride_on)
    
    # 3. Create a mask to avoid writing out of bounds of the output tensor
    output_mask = (offs_om[:, None] < output_shape_m) & (offs_on[None, :] < output_shape_n)

    # 4. Calculate corresponding offsets for the *input* tensor
    if DIM == 0:
        offs_im = offs_om + START
        offs_in = offs_on
    else: # DIM == 1
        offs_im = offs_om
        offs_in = offs_on + START
        
    # 5. Create pointers to the input block
    input_ptrs = input_ptr + (offs_im[:, None] * stride_im + offs_in[None, :] * stride_in)
    
    # 6. Create a mask to avoid reading out of bounds of the input tensor
    # (This is implicitly handled by the logic, since valid output indices + start should map to valid input indices)
    # However, an explicit mask is safer if there are complex cases.
    input_mask = (offs_im[:, None] < input_shape_m) & (offs_in[None, :] < input_shape_n)
    
    # 7. Load from input and store to output
    # We only load/store where the output mask is valid.
    data = tl.load(input_ptrs, mask=output_mask & input_mask, other=0.0)
    tl.store(output_ptrs, data, mask=output_mask)


def narrow(a: torch.Tensor, dim, start, length):
    """
    Triton implementation of torch.narrow for 2D tensors.
    """
    # 检查约束
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert a.dim() == 2, "Only 2D tensors are supported"
    M, N = a.shape
    
    # 计算输出形状并分配输出张量
    output_shape = list(a.shape)
    output_shape[dim] = length
    c = torch.empty(output_shape, device=a.device, dtype=a.dtype)
    
    # 如果输出长度为0，直接返回空张量
    if length == 0:
        return c

    # 启动内核的网格 (Grid)
    # 网格的大小应该覆盖整个输出张量
    grid = lambda META: (
        triton.cdiv(c.shape[0], META['BLOCK_SIZE_M']),
        triton.cdiv(c.shape[1], META['BLOCK_SIZE_N']),
    )

    # 启动内核
    clean_narrow_kernel[grid](
        a, c,
        M, N,
        c.shape[0], c.shape[1],
        a.stride(0), a.stride(1),
        c.stride(0), c.stride(1),
        DIM=dim,
        START=start,
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=16,
    )
    return c


def test_narrow(device):
    """
    Tests the Triton narrow implementation against torch.narrow.
    """
    torch.manual_seed(0)
    # 定义测试用的张量形状和 narrow 参数
    rows = 179
    cols = 321
    dim = 1
    start = 100
    length = 150

    # 创建一个随机张量
    a = torch.randn((rows, cols), device=device, dtype=torch.float32)

    # 运行 Triton 和 PyTorch 的实现
    triton_output = narrow(a, dim, start, length)
    torch_output = torch.narrow(a, dim, start, length)

    # 比较结果
    print("Triton output shape:", triton_output.shape)
    print("PyTorch output shape:", torch_output.shape)
    torch.testing.assert_close(triton_output, torch_output, atol=1e-5, rtol=0)
    print("Test passed!")


@benchmark.measure()
def bench_narrow(M, N, dim, start, length, provider):
    """
    Benchmarks the performance of narrow operations.
    """
    a = torch.randn((M, N), device='cpu', dtype=torch.float32)
    if provider == 'torch':
        torch.narrow(a, dim, start, length)
    if provider == 'triton':
        narrow(a, dim, start, length)


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    
    print("Running correctness test...")
    test_narrow(device=device)

    print("\nRunning benchmark...")
    # 针对不同大小的张量进行基准测试
    for M in [128, 512, 1024, 2048]:
        N = M
        dim = 0
        start = M // 4
        length = M // 2
        print(f"\n--- Benchmarking for shape ({M}, {N}), dim={dim}, start={start}, length={length} ---")
        for provider in ['torch', 'triton']:
             bench_narrow(M, N, dim, start, length, provider)