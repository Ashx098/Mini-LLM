import math

def get_lr(it, args):
    # args must contain: learning_rate, min_lr, warmup_steps, max_steps
    
    # 1. Linear Warmup Phase
    if it < args.warmup_steps:
        return args.learning_rate * (it + 1) / (args.warmup_steps + 1)
        
    # 2. If we go beyond max_steps, return min_lr
    if it > args.max_steps:
        return args.min_lr
        
    # 3. Cosine Decay Phase
    # Calculate progress (0.0 to 1.0) inside the decay window
    decay_ratio = (it - args.warmup_steps) / (args.max_steps - args.warmup_steps)
    assert 0 <= decay_ratio <= 1
    
    # Calculate coefficient (starts at 1.0, goes to 0.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    
    # Scale range between min_lr and max_lr
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)
