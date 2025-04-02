import torch
import torch.nn.functional as F
import logging
import os
from accelerate import Accelerator

def create_kl_loss(looped_logits, original_logits, temperature=1.0):
    """
    Compute KL divergence loss between looped model and original model logits.
    Used for knowledge distillation in Phase 1.
    
    Args:
        looped_logits: Logits from the looped model
        original_logits: Logits from the original model
        temperature: Temperature for softening the distributions
    
    Returns:
        KL divergence loss
    """
    # Only consider the next token predictions for tokens where we have inputs
    shift_looped_logits = looped_logits[..., :-1, :].contiguous()
    shift_original_logits = original_logits[..., :-1, :].contiguous()
    
    # Apply temperature scaling
    scaled_looped_logits = shift_looped_logits / temperature
    scaled_original_logits = shift_original_logits / temperature
    
    # Compute KL divergence
    looped_log_probs = F.log_softmax(scaled_looped_logits, dim=-1)
    original_probs = F.softmax(scaled_original_logits, dim=-1)
    
    kl_div = F.kl_div(
        looped_log_probs.view(-1, looped_log_probs.size(-1)), 
        original_probs.view(-1, original_probs.size(-1)),
        reduction="batchmean"
    )
    
    return kl_div * (temperature ** 2)  # Scale by temperature²

def get_batch_size_per_device(total_batch_size, num_devices):
    """
    Calculate the per-device batch size given the total batch size and number of devices.
    Ensures the per-device batch size is at least 1.
    """
    per_device_batch_size = max(1, total_batch_size // num_devices)
    effective_batch_size = per_device_batch_size * num_devices
    
    if effective_batch_size != total_batch_size:
        print(f"Warning: Effective batch size {effective_batch_size} differs from requested batch size {total_batch_size}")
    
    return per_device_batch_size

def setup_logging(args, accelerator):
    """
    Set up logging for the training process.
    """
    logger = logging.getLogger(__name__)
    
    if accelerator.is_main_process:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(args.output_dir, "training.log"))
            ]
        )
    else:
        logging.basicConfig(level=logging.ERROR)
    
    logger.info(accelerator.state, main_process_only=True)
    logger.info(args, main_process_only=True)
    
    return logger

def generate_with_different_k(model, tokenizer, prompt, max_length=50, k_values=[1, 2, 3, 5]):
    """
    Generate text using the model with different loop counts for comparison.
    
    Args:
        model: The LayerLoopingModel
        tokenizer: Tokenizer for encoding/decoding
        prompt: Text prompt to start generation
        max_length: Maximum length of generated text
        k_values: List of k values to try
    
    Returns:
        Dictionary of generations with k values as keys
    """
    device = next(model.parameters()).device
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    results = {}
    
    for k in k_values:
        output_ids = model.sample_generation(input_ids, max_length=max_length, k_value=k)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        results[k] = output_text
    
    return results

def get_model_parameter_count(model):
    """
    Get the number of parameters in the model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Total parameter count and trainable parameter count
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_millions": total_params / 1e6,
        "trainable_params_millions": trainable_params / 1e6
    }