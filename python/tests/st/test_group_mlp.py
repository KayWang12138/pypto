import torch
import torch_npu
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import time

class GroupedMLPTorch(nn.Module):
    def __init__(
        self,
        num_local_experts: int,
        hidden_size: int,
        ffn_hidden_size: int,
        gated_linear_unit: bool = False,
        activation_func = F.silu,  # or F.gelu
        use_bias: bool = False,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.gated_linear_unit = gated_linear_unit
        self.activation_func = activation_func
        self.use_bias = use_bias
        self.dtype = dtype
        self.device = device

        # Determine fc1 output size
        fc1_output_size = ffn_hidden_size * num_local_experts
        if gated_linear_unit:
            fc1_output_size *= 2  # e.g., for SwiGLU: [xW, xV] → split and gate

        # Initialize weights (no tensor parallelism here)
        self.weight1 = nn.Parameter(
            torch.empty(hidden_size, fc1_output_size, dtype=dtype, device=device)
        )
        self.weight2 = nn.Parameter(
            torch.empty(ffn_hidden_size * num_local_experts, hidden_size, dtype=dtype, device=device)
        )

        if use_bias:
            self.bias1 = nn.Parameter(torch.zeros(fc1_output_size, dtype=dtype, device=device))
            self.bias2 = nn.Parameter(torch.zeros(hidden_size, dtype=dtype, device=device))
        else:
            self.bias1 = None
            self.bias2 = None

        self.reset_parameters()

    def reset_parameters(self):
        # Simple init: Xavier-like for weight1, scaled for weight2
        nn.init.kaiming_uniform_(self.weight1, a=0, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_uniform_(self.weight2, a=0, mode='fan_in', nonlinearity='linear')

    def _gmm_torch(
        self,
        inputs: torch.Tensor,          # [total_tokens, hidden]
        weights: torch.Tensor,         # [num_experts, hidden, out_features_per_expert]
        tokens_per_expert: torch.Tensor  # [num_experts]
    ) -> torch.Tensor:
        """
        Simulate grouped GEMM using torch native ops.
        Assumes inputs are already permuted so that tokens for expert i are contiguous.
        """
        total_tokens = inputs.size(0)
        num_experts = tokens_per_expert.size(0)
        assert weights.dim() == 3

        if total_tokens == 0:
            out_feat = weights.size(-1)
            return torch.empty(0, out_feat, dtype=inputs.dtype, device=inputs.device)

        outputs = []
        start = 0
        for i in range(num_experts):
            end = start + tokens_per_expert[i].item()
            if end <= start:
                continue
            x = inputs[start:end]  # [n_i, hidden]
            w = weights[i]         # [hidden, out_i]
            out = torch.matmul(x, w)  # [n_i, out_i]
            if self.use_bias and self.bias1 is not None:
                # Note: bias handling depends on which layer; for simplicity assume applied per expert
                b = self.bias1[i * w.size(1): (i+1) * w.size(1)]
                out = out + b
            outputs.append(out)
            start = end
        return torch.cat(outputs, dim=0) if outputs else torch.empty(0, weights.size(-1), device=inputs.device)

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,  # [total_tokens, hidden]
        tokens_per_expert: torch.Tensor,              # [num_local_experts]
        permuted_probs: torch.Tensor,                 # [total_tokens]
    ) -> Tuple[torch.Tensor, None]:
        num_experts = self.num_local_experts
        total_tokens = permuted_local_hidden_states.size(0)

        # Reshape weights to [num_experts, hidden, ffn_dim] or [num_experts, hidden, 2*ffn_dim]
        ffn_dim = self.ffn_hidden_size
        if self.gated_linear_unit:
            w1 = self.weight1.view(num_experts, self.hidden_size, 2 * ffn_dim)
        else:
            w1 = self.weight1.view(num_experts, self.hidden_size, ffn_dim)

        w2 = self.weight2.view(num_experts, ffn_dim, self.hidden_size)

        # First GEMM: x @ W1
        fc1_output = self._gmm_torch(permuted_local_hidden_states, w1, tokens_per_expert)

        # Apply activation (+ gating if GLU)
        if self.gated_linear_unit:
            x1, x2 = torch.chunk(fc1_output, 2, dim=-1)
            activated = self.activation_func(x1) * x2
        else:
            activated = self.activation_func(fc1_output)

        # Apply router probs (if enabled)
        # activated = activated * permuted_probs.unsqueeze(-1)

        # Second GEMM: activated @ W2
        output = self._gmm_torch(activated, w2, tokens_per_expert)

        if self.use_bias and self.bias2 is not None:
            # Add bias per expert — requires splitting output by expert
            # For simplicity in test, we skip bias2 in performance test
            pass

        return output, None


def generate_tokens_per_expert_python(total_tokens: int, num_experts: int, max_dev: int = 50):
    """
    Returns a list of length `num_experts` such that:
      - sum(list) == total_tokens
      - each element is in [max(0, mean - max_dev), mean + max_dev]
    Strategy:
      1. Randomly assign each expert a value in [mean - max_dev, mean + max_dev], clamped to >=0.
      2. Compute current sum and diff = total_tokens - current_sum.
      3. Traverse experts one by one, adjust each by as much as possible (within bounds) to reduce diff.
      4. Stop once diff == 0.
    """
    mean = total_tokens // num_experts
    low_bound = max(0, mean - max_dev)
    high_bound = mean + max_dev

    # Step 1: Initialize with random values in valid range
    import random
    tokens_list = []
    for _ in range(num_experts):
        val = random.randint(low_bound, high_bound)
        tokens_list.append(val)

    current_sum = sum(tokens_list)
    diff = total_tokens - current_sum  # >0 means need to add, <0 means need to remove

    # Step 2: Adjust from first expert onward
    for i in range(num_experts):
        if diff == 0:
            break
        current_val = tokens_list[i]

        if diff > 0:
            # Need to add tokens: increase this expert as much as possible
            can_add = high_bound - current_val
            add = min(diff, can_add)
            tokens_list[i] += add
            diff -= add
        else:
            # Need to remove tokens: decrease this expert as much as possible
            can_remove = current_val - low_bound
            remove = min(-diff, can_remove)
            tokens_list[i] -= remove
            diff += remove  # because diff is negative

    # Final safety: should be zero
    assert diff == 0, f"Failed to balance tokens: remaining diff = {diff}"
    assert sum(tokens_list) == total_tokens
    assert all(low_bound <= x <= high_bound for x in tokens_list)
    assert all(x >= 0 for x in tokens_list)

    return tokens_list

def benchmark_grouped_mlp():
    # Config
    num_local_experts = 8
    hidden_size = 4096
    ffn_hidden_size = 2048
    batch_size = 4
    seq_len = 4096
    total_tokens = batch_size * seq_len  # 16384
    dtype = torch.float16
    device = "npu:0"
    torch.npu.set_device(device)

    model = GroupedMLPTorch(
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        gated_linear_unit=True,
        activation_func=F.silu,
        use_bias=False,
        dtype=dtype,
        device=device,
    )

    # Warmup
    print("Warming up...")
    for _ in range(5):
        token_list = generate_tokens_per_expert_python(total_tokens, num_local_experts, max_dev=50)
        print(f"token_list: {token_list}")
        tokens_per_expert = torch.tensor(token_list, dtype=torch.long, device=device)
        assert tokens_per_expert.sum().item() == total_tokens

        permuted_hidden = torch.randn(total_tokens, hidden_size, dtype=dtype, device=device)
        permuted_probs = torch.rand(total_tokens, dtype=dtype, device=device)

        out, _ = model(permuted_hidden, tokens_per_expert, permuted_probs)
        torch.npu.synchronize()

    # Reset memory stats
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats(device)

    # Benchmark
    print("Running benchmark...")
    total_time = 0.0

    for i in range(20):
        token_list = generate_tokens_per_expert_python(total_tokens, num_local_experts, max_dev=50)
        print(f"token_list: {token_list}")
        tokens_per_expert = torch.tensor(token_list, dtype=torch.long, device=device)
        assert tokens_per_expert.sum().item() == total_tokens

        permuted_hidden = torch.randn(total_tokens, hidden_size, dtype=dtype, device=device)
        permuted_probs = torch.rand(total_tokens, dtype=dtype, device=device)

        torch.npu.synchronize()
        start = time.perf_counter()
        out, _ = model(permuted_hidden, tokens_per_expert, permuted_probs)
        torch.npu.synchronize()
        end = time.perf_counter()

        total_time += (end - start)

    avg_time = total_time / 20
    tokens_per_sec = total_tokens / avg_time

    max_mem = torch.npu.max_memory_allocated(device) / (1024**3)
    # current_mem = torch.npu.memory_allocated(device) / (1024**3)

    print(f"\n=== Results ===")
    print(f"Total tokens per iter: {total_tokens}")
    print(f"Avg time: {avg_time*1000:.2f} ms")
    print(f"Tokens/sec: {tokens_per_sec:,.0f}")
    print(f"Peak NPU Mem: {max_mem:.2f} GB")

    # Optional: show one example distribution
    example_list = generate_tokens_per_expert_python(total_tokens, num_local_experts, max_dev=50)
    print(f"Example tokens_per_expert: {example_list}")
    
    # Explain the throughput formula
    # Note: This is a ROUGH approximation.
    # Real FLOPs/token = 2 * h * (2*f) + 2 * f * h = 6 * h * f  (for GLU)
    # But here they use 4 * h as proxy (assuming f ~ h, which is not true!)
    # We keep it for consistency but note the issue.
    print(f"Peak NPU Memory: {max_mem:.2f} GB")
    # print(f"Current NPU Memory: {current_mem:.2f} GB")

    # Optional: more accurate FLOPs
    real_flops_per_token = 2 * (hidden_size * (2 * ffn_hidden_size) + ffn_hidden_size * hidden_size)
    real_tflops = tokens_per_sec * real_flops_per_token / 1e12
    print(f"More Accurate Est. Throughput: {real_tflops:.2f} TFLOPs/s")

if __name__ == "__main__":
    benchmark_grouped_mlp()