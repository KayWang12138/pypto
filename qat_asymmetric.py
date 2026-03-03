def Enhanced_LSQ_plus(weight, scale, offset, group_size, bit, eps=1e-4, clip_val=0.99):
    eps = torch.tensor(eps, device=scale.device).float()
    scale = torch.where(scale > eps, scale, eps)
    
    # reshape
    orig_shape = weight.shape
    num_groups = weight.numel() // group_size
    weight = weight.view(num_groups, group_size)
    
    # prepare scale and offset
    n_levels = 2 ** (bit - 1)
    shift = 0.5

    weight = weight - offset
    alpha = scale * n_levels

    # quantization and de-quantize
    weight = torch.clamp(weight / alpha, -clip_val, clip_val) * n_levels - shift
    weight = (weight.round() - weight).detach() + weight
    weight = (weight + shift) / n_levels
    weight = weight + alpha + offset
    
    # reshape back
    weight = weight.view(orig_shape)
    
    return weight