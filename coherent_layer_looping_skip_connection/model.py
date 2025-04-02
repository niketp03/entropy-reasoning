import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
import os
from safetensors.torch import load_file, save_file

class LayerLoopingModel(nn.Module):
    """
    Model wrapper that implements layer looping for a pretrained transformer model.
    Phase 1 implementation: Simple looping of middle layers without explore/exploit.
    """
    def __init__(
        self, 
        model_name_or_path="Qwen/Qwen2.5-0.5B", 
        n=6,  # Start layer index for looping (0-indexed)
        m=12,  # End layer index for looping (0-indexed)
        max_loop_count=5,  # Maximum number of times to loop during training
        device_map=None  # Changed from "auto" to None
    ):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            device_map=device_map,
            torch_dtype=torch.bfloat16  # Using bfloat16 for efficiency
        )
        
        # Save configuration
        self.n = n
        self.m = m
        self.max_loop_count = max_loop_count
        self.layers = self.model.model.layers  # Get transformer layers
        self.layer_count = len(self.layers)
        
        # Validate indices
        assert 0 < n < m < self.layer_count, f"Invalid layer indices: n={n}, m={m}, total_layers={self.layer_count}"
        
        # Extract layer segments
        self.early_layers = self.layers[:n]  # Layers 0 to n-1
        self.loop_layers = self.layers[n:m+1]  # Layers n to m
        self.late_layers = self.layers[m+1:]  # Layers m+1 to L
        
        print(f"Model initialized with looping configuration:")
        print(f"  - Total layers: {self.layer_count}")
        print(f"  - Early layers: 0 to {n-1}")
        print(f"  - Loop layers: {n} to {m}")
        print(f"  - Late layers: {m+1} to {self.layer_count-1}")
        print(f"  - Max loop count: {max_loop_count}")
    
    @classmethod
    def from_pretrained(cls, checkpoint_path, **kwargs):
        """Load from checkpoint directory"""
        model = cls(**kwargs)
        
        # Try loading from accelerator saved state
        accelerator_state_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.exists(accelerator_state_path):
            state_dict = torch.load(accelerator_state_path)
            model.load_state_dict(state_dict)
            return model
        
        # Try loading from HuggingFace format
        try:
            model.model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                device_map=None
            )
            return model
        except Exception as e:
            print(f"Error loading from HuggingFace format: {str(e)}")
        
        raise FileNotFoundError(
            f"No valid model files found in {checkpoint_path}. "
            "Expected either pytorch_model.bin or HuggingFace model files."
        )

    def save_pretrained(self, save_dir):
        """Save model to directory"""
        os.makedirs(save_dir, exist_ok=True)
        save_file(self.state_dict(), os.path.join(save_dir, "model.safetensors"))

    def forward(self, input_ids, attention_mask=None, labels=None, k=None, return_hidden_states=False):
        """
        Forward pass with layer looping.
        """
        # Initialize list to store hidden states if needed
        hidden_states_history = [] if return_hidden_states else None
        
        # During training, randomly sample loop count if not specified
        if k is None and self.training:
            k = torch.randint(1, self.max_loop_count + 1, (1,)).item()
        elif k is None:
            k = 1  # Default to 1 loop during inference
        
        # Prepare attention mask for Qwen2 format
        if attention_mask is not None:
            # Convert 2D mask to 4D mask for qwen attention
            batch_size = input_ids.shape[0]
            seq_length = input_ids.shape[1]
            # [batch_size, seq_length] -> [batch_size, 1, seq_length, seq_length]
            attention_mask = attention_mask.view(batch_size, 1, 1, seq_length)
            attention_mask = attention_mask.expand(-1, 1, seq_length, -1)
            # Convert to bfloat16 to match model dtype
            attention_mask = attention_mask.to(dtype=torch.bfloat16)
        
        # Get model embeddings and prepare hidden states
        outputs = self.model.model.embed_tokens(input_ids)  
        hidden_states = outputs
        
        # Apply early layers (no looping)
        for i, layer in enumerate(self.early_layers):
            hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
        
        #save skip connection
        skip_connect = hidden_states.clone()

        if return_hidden_states:
            hidden_states_history.append(("early", hidden_states.clone()))
        
        # Apply middle layers with looping k times
        for j in range(k):
            for layer in self.loop_layers:
                hidden_states = layer(hidden_states, attention_mask=attention_mask)[0] + skip_connect
            if return_hidden_states:
                hidden_states_history.append((f"middle_loop_{j}", hidden_states.clone()))
        
        # Apply late layers (no looping)
        for layer in self.late_layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
        
        if return_hidden_states:
            hidden_states_history.append(("late", hidden_states.clone()))
        
        # Apply final layer norm
        hidden_states = self.model.model.norm(hidden_states)
        
        # Compute logits
        lm_logits = self.model.lm_head(hidden_states)
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        if return_hidden_states:
            return {"loss": loss, "logits": lm_logits, "hidden_states": hidden_states_history}
        return {"loss": loss, "logits": lm_logits}
    
    def sample_generation(self, input_ids, return_matrix=False, max_length=100, k_value=1, **kwargs):
        """
        Generate text using the model with a specific loop count.
        
        Args:
            input_ids: Input token IDs
            return_matrix: If True, return hidden states at each stage
            max_length: Maximum length of generated text
            k_value: Number of loops to use during generation
            **kwargs: Additional arguments for generation
        
        Returns:
            If return_matrix=False: generated token ids
            If return_matrix=True: tuple of (generated token ids, list of hidden states)
        """
        self.eval()
        all_hidden_states = [] if return_matrix else None
        
        with torch.no_grad():
            current_ids = input_ids.clone()
            
            # Generate tokens one by one
            for _ in range(max_length - input_ids.size(1)):
                # Forward pass with specific k and hidden states collection
                outputs = self(current_ids, k=k_value, return_hidden_states=return_matrix)
                
                if return_matrix:
                    all_hidden_states.append(outputs["hidden_states"])
                
                # Get next token (simple greedy decoding)
                next_token_logits = outputs["logits"][:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                
                # Concatenate with current tokens
                current_ids = torch.cat([current_ids, next_token], dim=-1)
                
                # Check for EOS token
                if next_token.item() == self.model.config.eos_token_id:
                    break
        
        if return_matrix:
            return current_ids, all_hidden_states
        return current_ids

    def get_original_model_output(self, input_ids, attention_mask=None):
        """
        Get the output from the original model without looping (for distillation)
        """
        with torch.no_grad():
            # Handle both wrapped and unwrapped models
            if hasattr(self, 'module'):
                outputs = self.module.model(input_ids, attention_mask=attention_mask)
            else:
                outputs = self.model(input_ids, attention_mask=attention_mask)
            return outputs.logits