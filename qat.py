def Embedding_Head_Quant_new(weight, scale, eps=1e-4, min_v=-128.0, max_v=127.0):
    # aviod scale <= 0
    eps = torch.tensor(eps, device=scale.device).float()
    scale = torch.where(scale > eps, scale, eps)
    
    # quantization: round + clamp
    weight = weight / scale
    weight = (weight.round() - weight).detach() + weight
    weight = torch.clamp(weight, min_v, max_v) * scale

    return weight
