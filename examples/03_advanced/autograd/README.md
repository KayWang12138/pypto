# PyPTO Autograd Examples

C++ AutodiffPass examples for training on Ascend NPU.

## Files

| File | Description |
|------|-------------|
| `train_simple_mlp.py` | MLP training example |
| `benchmark_autograd.py` | Performance benchmark |

## Usage

```python
import pypto

@pypto.jit
def train_step(x, w, loss):
    y = pypto.matmul(x, w)
    loss[:] = pypto.sum(y)

x.requires_grad = True
w.requires_grad = True
loss.is_loss = True

train_step(x, w, loss)

grad_x = x.get_gradient_tensor()
```

## Supported VJP Rules

- **Math**: add, sub, mul, div, neg, abs, exp, log, sqrt, rsqrt, sigmoid, pow
- **Matrix**: matmul, batch_matmul
- **Reduction**: sum, amax, amin, maximum, minimum
- **View**: reshape, transpose, unsqueeze, view, assemble
