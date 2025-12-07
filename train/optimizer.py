import torch

def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    # 1. Separate parameters into decay/no-decay groups
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    
    # Filters: 2D weights (Linear, Embedding) get decay. 1D weights (Norms, Biases) don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"Optimizer Configured: {len(decay_params)} tensors decayed ({num_decay_params:,} params), {len(nodecay_params)} tensors not decayed ({num_nodecay_params:,} params)")
    
    # 2. Create AdamW Optimizer
    # fused=True uses faster CUDA kernels if available
    use_fused = 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
    extra_args = dict(fused=True) if use_fused and device_type == 'cuda' else dict()
    
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    print(f"Using fused AdamW: {use_fused}")
    
    return optimizer
