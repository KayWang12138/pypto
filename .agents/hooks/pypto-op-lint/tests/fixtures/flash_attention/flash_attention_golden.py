import torch


def flash_attention_golden(query: torch.Tensor) -> torch.Tensor:
    return query.clone()
