import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig

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
    
    def forward(self, input_ids, attention_mask=None, labels=None, k=None):
        """
        Forward pass with layer looping.
        """
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
        
        # Rest of the forward method remains the same...
        
        # Apply middle layers with looping k times
        for j in range(k):
            for layer in self.loop_layers:
                hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
        
        # Apply late layers (no looping)
        for layer in self.late_layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
        
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
        
        return {"loss": loss, "logits": lm_logits}
    
    def sample_generation(self, input_ids, max_length=100, k_value=1, **kwargs):
        """
        Generate text using the model with a specific loop count.
        
        Args:
            input_ids: Input token IDs
            max_length: Maximum length of generated text
            k_value: Number of loops to use during generation
            **kwargs: Additional arguments for generation
        """
        # Set model to eval mode
        self.eval()
        
        with torch.no_grad():
            # Clone input_ids for generation
            current_ids = input_ids.clone()
            
            # Generate tokens one by one
            for _ in range(max_length - input_ids.size(1)):
                # Forward pass with specific k
                outputs = self(current_ids, k=k_value)
                
                # Get next token (simple greedy decoding)
                next_token_logits = outputs["logits"][:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                
                # Concatenate with current tokens
                current_ids = torch.cat([current_ids, next_token], dim=-1)
                
                # Check for EOS token
                if next_token.item() == self.model.config.eos_token_id:
                    break
                    
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