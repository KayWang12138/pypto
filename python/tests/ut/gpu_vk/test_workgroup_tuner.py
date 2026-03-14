#!/usr/bin/env python3

import torch

from pypto_gpu_vk.tune import profile_once, tune_workgroup


def test_workgroup_tuner_returns_candidate():
    candidate = tune_workgroup("relu", torch.randn(256))
    assert candidate.local_x in (32, 64, 128)


def test_profile_once_returns_profile_and_tensor():
    profile, output = profile_once("relu", torch.randn(32))
    assert profile.local_x >= 1
    assert output.numel() == 32
