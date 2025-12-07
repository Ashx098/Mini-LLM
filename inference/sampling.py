import torch
import torch.nn.functional as F

def sample_top_p(probs, p):
    """
    Nucleus sampling: filters out tokens that account for the bottom (1-p) probability mass.
    """
    # 1. Sort probabilities in descending order
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    
    # 2. Calculate cumulative sum
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    
    # 3. Create a mask for tokens to remove (where cumsum > p)
    # We shift the mask right by 1 to always keep at least the first token
    mask = probs_sum - probs_sort > p
    
    # 4. Zero out the probabilities of filtered tokens
    probs_sort[mask] = 0.0
    
    # 5. Renormalize so sum is 1.0 again
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    
    # 6. Sample from the filtered distribution
    next_token = torch.multinomial(probs_sort, num_samples=1)
    
    # 7. Retrieve original token ID
    next_token = torch.gather(probs_idx, -1, next_token)
    
    return next_token

def apply_temperature(logits, temperature):
    """
    Adjusts the randomness of predictions.
    Temp < 1.0: Makes peaks pointier (More deterministic)
    Temp > 1.0: Makes peaks flatter (More random)
    """
    return logits / max(temperature, 1e-5)
