# ==========================================
# Triton GEMM Alpha/Beta with MLIR Dump
# ==========================================
import os
import time
import torch
import triton
import triton.language as tl
from triton.backends.triton_shared.driver import CPUDriver
import triton.runtime.driver

# -----------------------------
# MLIR / LLVM Dump Setup
# -----------------------------
os.environ['MLIR_ENABLE_DUMP'] = '1'  # 或者 'kernelName' 指定 kernel
os.environ['MLIR_DUMP_PATH'] = './triton_mlir_dumps'
os.environ['LLVM_IR_ENABLE_DUMP'] = '1'
# 可选：解释器模式
# os.environ['TRITON_INTERPRET'] = '1'

# 切换到自定义 CPUDriver（新硬件 backend）
triton.runtime.driver.set_active(CPUDriver())
device = torch.device("cpu")  # 前端测试用，不依赖 CUDA

# -----------------------------
# Triton GEMM kernel
# -----------------------------
@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    alpha, beta,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 初始化 accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + offs_m[:, None]*stride_am + offs_k[None, :]*stride_ak,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))
        b = tl.load(B_ptr + offs_k[:, None]*stride_bk + offs_n[None, :]*stride_bn,
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N))
        acc += tl.dot(a, b)

    c = tl.load(C_ptr + offs_m[:, None]*stride_cm + offs_n[None, :]*stride_cn,
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    acc = alpha * acc + beta * c
    tl.store(C_ptr + offs_m[:, None]*stride_cm + offs_n[None, :]*stride_cn,
             acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# -----------------------------
# Python wrapper
# -----------------------------
def triton_gemm_alpha_beta(A, B, C=None, alpha=1.0, beta=0.0,
                            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match"
    if C is None:
        C = torch.zeros((M, N), device=A.device, dtype=A.dtype)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        alpha, beta,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return C

# -----------------------------
# Quick self-check
# -----------------------------
def quick_check_alpha_beta():
    print(">>> Running quick GEMM check...")
    M, K, N = 64, 64, 64
    alpha, beta = 1.2, 0.8
    A = torch.randn((M, K), device=device, dtype=torch.float32)
    B = torch.randn((K, N), device=device, dtype=torch.float32)
    C = torch.randn((M, N), device=device, dtype=torch.float32)
    C_out = triton_gemm_alpha_beta(A, B, C.clone(), alpha=alpha, beta=beta)
    C_ref = torch.addmm(beta*C, A, B, alpha=alpha)
    max_diff = (C_out - C_ref).abs().max()
    print(f"Quick check max difference: {max_diff.item():.3e}")
    assert max_diff < 1e-5

# -----------------------------
# CLI Quick benchmark
# -----------------------------
def run_cli_benchmark():
    print(">>> Running CLI benchmark (if CUDA available, will time)")
    quick_check_alpha_beta()
    if torch.cuda.is_available():
        M, K, N = 512, 512, 512
        A = torch.randn((M, K), device='cuda', dtype=torch.float32)
        B = torch.randn((K, N), device='cuda', dtype=torch.float32)
        C_old = torch.randn((M, N), device='cuda', dtype=torch.float32)
        alpha, beta = 1.0, 1.0

        # warmup
        _ = triton_gemm_alpha_beta(A, B, C=C_old.clone(),
                                   alpha=alpha, beta=beta,
                                   BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(3):
            _ = triton_gemm_alpha_beta(A, B, C=C_old.clone(),
                                       alpha=alpha, beta=beta,
                                       BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
        torch.cuda.synchronize()
        t_triton = (time.perf_counter() - t0)/3

        # torch timing
        t0 = time.perf_counter()
        for _ in range(3):
            _ = alpha*(A@B) + beta*C_old
        torch.cuda.synchronize()
        t_torch = (time.perf_counter() - t0)/3

        print(f"512^3 | triton: {t_triton*1000:.3f} ms | torch: {t_torch*1000:.3f} ms")
    else:
        print("CUDA not available — skipping high-res GPU benchmark.")

# -----------------------------
# pytest correctness test
# -----------------------------
def test_triton_gemm_correctness():
    M, K, N = 64, 64, 64
    alpha, beta = 1.1, 0.9
    A = torch.randn((M, K), device=device, dtype=torch.float32)
    B = torch.randn((K, N), device=device, dtype=torch.float32)
    C = torch.randn((M, N), device=device, dtype=torch.float32)
    C_out = triton_gemm_alpha_beta(A, B, C.clone(), alpha=alpha, beta=beta)
    C_ref = torch.addmm(beta*C, A, B, alpha=alpha)
    max_diff = (C_out - C_ref).abs().max()
    print(f"[pytest] max difference: {max_diff.item():.3e}")
    assert max_diff < 1e-5

# -----------------------------
# Main entry
# -----------------------------
if __name__ == "__main__":
    print("ENTRY __main__")
    print(f"MLIR dump enabled: {os.environ.get('MLIR_ENABLE_DUMP')}")
    print(f"MLIR dump path: {os.environ.get('MLIR_DUMP_PATH')}")
    run_cli_benchmark()
