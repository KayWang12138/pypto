import os
os.environ['TILE_FWK_DEVICE_ID'] = '1'

print('Testing golden reference...')
from gated_delta_rule_backward_golden import gated_delta_rule_backward_golden
import torch

torch.manual_seed(42)
B, T, H, K, V, bt = 1, 128, 4, 128, 128, 64
q = torch.randn(B, T, H, K, dtype=torch.float32) * 0.5
k = torch.randn(B, T, H, K, dtype=torch.float32) * 0.5
v = torch.randn(B, T, H, V, dtype=torch.float32) * 0.5
g_raw = torch.randn(B, T, H, dtype=torch.float32) * 0.1
beta = torch.rand(B, T, H, dtype=torch.float32)
initial_state = torch.randn(B, H, K, V, dtype=torch.float32) * 0.1
do_t = torch.randn(B, T, H, V, dtype=torch.float32) * 0.5
dht = torch.randn(B, H, K, V, dtype=torch.float32) * 0.1

result = gated_delta_rule_backward_golden(
    q=q, k=k, v=v, g_raw=g_raw, beta=beta,
    initial_state=initial_state, do=do_t, dht=dht,
    bt=bt, use_qk_l2norm_in_kernel=True, l2_eps=1e-6,
)
for name, t in zip(['dq', 'dk', 'dv', 'db', 'dg_raw', 'dh0'], result):
    print(f'{name}: shape={t.shape}, has_nan={torch.isnan(t).any()}, max_abs={t.abs().max():.4f}')
print('Golden OK')

print('Testing PyPTO impl...')
from gated_delta_rule_backward_impl import gated_delta_rule_backward_wrapper
impl_result = gated_delta_rule_backward_wrapper(
    q=q, k=k, v=v, g_raw=g_raw, beta=beta,
    initial_state=initial_state, do=do_t, dht=dht,
    bt=bt, use_qk_l2norm_in_kernel=True, l2_eps=1e-6,
)

import numpy as np
from numpy.testing import assert_allclose

names = ['dq', 'dk', 'dv', 'db', 'dg_raw', 'dh0']
all_pass = True
for name, g, i in zip(names, result, impl_result):
    g_np = g.detach().numpy()
    i_np = i.detach().numpy()
    max_diff = np.abs(g_np - i_np).max()
    has_nan = np.isnan(i_np).any()
    print(f'{name}: max_diff={max_diff:.6e}, nan={has_nan}')
    if has_nan:
        all_pass = False
        continue
    try:
        assert_allclose(i_np, g_np, rtol=1e-3, atol=1e-3)
        print(f'  PASS')
    except AssertionError as e:
        print(f'  FAIL: {str(e)[:200]}')
        all_pass = False

if all_pass:
    print('[PRECISION_PASS]')
else:
    print('[PRECISION_FAIL]')
