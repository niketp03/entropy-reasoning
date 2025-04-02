import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    default_data_collator,
    get_scheduler,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from accelerate import Accelerator
from tqdm import tqdm
import wandb
import numpy as np

from model import LayerLoopingModel
from utils import create_kl_loss, get_batch_size_per_device, setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Train a layer looping transformer model")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen1.5-1.8B",
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="Dataset name from Hugging Face datasets",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Dataset configuration name (wikitext-2-raw-v1 or wikitext-103-raw-v1)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=8,
        help="Start layer index for looping (0-indexed)",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=12,
        help="End layer index for looping (0-indexed)",
    )
    parser.add_argument(
        "--max_loop_count",
        type=int,
        default=5,
        help="Maximum number of times to loop during training",
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=4,
        help="Batch size per device for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps (overrides num_train_epochs)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before optimizer step",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use",
    )
    parser.add_argument(
        "--kl_weight",
        type=float,
        default=0.5,
        help="Weight for KL divergence loss for distillation",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Ratio of steps for warmup",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="Number of update steps between evaluations",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Number of update steps between saving checkpoints",
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=10,
        help="Number of update steps between logging",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Whether to use Weights & Biases for logging",
    )
    parser.add_argument(
        "--use_distillation",
        action="store_true",
        help="Whether to use knowledge distillation loss",
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help="Whether to use low CPU memory usage when loading the model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--lm_weight",
        type=float,
        default=1.0,
        help="Weight for language modeling loss",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set up accelerator for distributed training
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb" if args.use_wandb else None
    )
    
    # Make output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize wandb if requested
    if args.use_wandb and accelerator.is_main_process:
        wandb.init(
            project="layer-looping-transformers",
            name=f"phase1-{args.model_name_or_path.split('/')[-1]}-n{args.n}-m{args.m}-k{args.max_loop_count}",
            config=vars(args)
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine total batch size
    effective_batch_size = args.per_device_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    accelerator.print(f"Effective batch size: {effective_batch_size}")
    
    # Initialize the model
    model = LayerLoopingModel(
        model_name_or_path=args.model_name_or_path,
        n=args.n,
        m=args.m,
        max_loop_count=args.max_loop_count,
    )
    
    # Load dataset
    dataset = load_dataset(args.dataset_name, args.dataset_config)
    if "train" not in dataset:
        raise ValueError(f"Dataset {args.dataset_name} does not have a 'train' split")
    
    # Create a validation split if there is none
    if "validation" not in dataset:
        dataset = dataset["train"].train_test_split(test_size=0.05, seed=args.seed)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    else:
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
    
    # Tokenize the datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length",
            truncation=True,
            max_length=args.sequence_length,
            return_tensors="pt"
        )
    
    # Process dataset
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"],
    )
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"],
    )
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal language modeling
    )
    
    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_batch_size,
    )
    
    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    # Determine the number of training steps
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * len(train_dataloader)
    else:
        args.num_train_epochs = args.max_train_steps // len(train_dataloader) + 1
    
    # Learning rate scheduler
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_ratio * args.max_train_steps),
        num_training_steps=args.max_train_steps,
    )
    
    # Prepare model, optimizer, and dataloaders for distributed training
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    
    # Get the total number of steps we'll train for
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    # Training loop
    accelerator.print(f"***** Running training *****")
    accelerator.print(f"  Num examples = {len(train_dataset)}")
    accelerator.print(f"  Num epochs = {args.num_train_epochs}")
    accelerator.print(f"  Per-device batch size = {args.per_device_batch_size}")
    accelerator.print(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    accelerator.print(f"  Total optimization steps = {args.max_train_steps}")
    accelerator.print(f"  Using distillation: {args.use_distillation}")
    
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    best_eval_loss = float('inf')
    
    for epoch in range(args.num_train_epochs):
        model.train()
        total_train_loss = 0
        total_kl_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            # Forward pass through model with random loop count
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["labels"]
            )
            
            # Get the language modeling loss
            lm_loss = outputs["loss"]
            
            # If using distillation, get original model outputs and compute KL loss
            kl_loss = torch.tensor(0.0, device=lm_loss.device)
            if args.use_distillation:
                # Get original model output (no looping)
                if hasattr(model, 'module'):
                    original_logits = model.module.get_original_model_output(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask")
                    )
                else:
                    original_logits = model.get_original_model_output(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask")
                    )
                
                # Compute KL divergence loss
                looped_logits = outputs["logits"]
                kl_loss = create_kl_loss(looped_logits, original_logits)
                
                # Total loss
                loss = args.lm_weight * lm_loss + args.kl_weight * kl_loss
            else:
                loss = args.lm_weight * lm_loss
            
            # Backward pass and optimization
            accelerator.backward(loss / args.gradient_accumulation_steps)
            
            # Accumulate losses for logging
            total_train_loss += lm_loss.detach().float()
            if args.use_distillation:
                total_kl_loss += kl_loss.detach().float()
            
            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
                
                # Log metrics
                if completed_steps % args.log_steps == 0 and accelerator.is_main_process:
                    avg_train_loss = total_train_loss / args.log_steps / args.gradient_accumulation_steps
                    logs = {
                        "train/loss": avg_train_loss.item(),
                        "train/learning_rate": lr_scheduler.get_last_lr()[0],
                        "train/step": completed_steps,
                    }
                    
                    if args.use_distillation:
                        avg_kl_loss = total_kl_loss / args.log_steps / args.gradient_accumulation_steps
                        logs["train/kl_loss"] = avg_kl_loss.item()
                    
                    if args.use_wandb:
                        wandb.log(logs)
                    
                    # Reset tracking variables
                    total_train_loss = 0
                    total_kl_loss = 0
                
                # Evaluate
                if completed_steps % args.eval_steps == 0:
                    eval_loss = evaluate(model, eval_dataloader, accelerator)
                    
                    if accelerator.is_main_process:
                        logs = {"eval/loss": eval_loss}
                        
                        if args.use_wandb:
                            wandb.log(logs)
                        
                        # Save best model
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            accelerator.print(f"New best eval loss: {best_eval_loss:.4f}")
                            
                            # Unwrap and save
                            unwrapped_model = accelerator.unwrap_model(model)
                            unwrapped_model.model.save_pretrained(
                                os.path.join(args.output_dir, "best_model"),
                                is_main_process=accelerator.is_main_process,
                                save_function=accelerator.save
                            )
                            tokenizer.save_pretrained(os.path.join(args.output_dir, "best_model"))
                    
                    model.train()  # Make sure we return to train mode
                
                # Save checkpoint
                if completed_steps % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, f"checkpoint-{completed_steps}")
                    if accelerator.is_main_process:
                        os.makedirs(output_dir, exist_ok=True)
                    
                    # Unwrap and save
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.model.save_pretrained(
                        output_dir,
                        is_main_process=accelerator.is_main_process,
                        save_function=accelerator.save
                    )
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(output_dir)
                
                # Break if we've reached max_train_steps
                if completed_steps >= args.max_train_steps:
                    break
        
        # End of epoch, evaluate
        eval_loss = evaluate(model, eval_dataloader, accelerator)
        if accelerator.is_main_process:
            logs = {
                "eval/loss": eval_loss,
                "eval/epoch": epoch + 1
            }
            
            if args.use_wandb:
                wandb.log(logs)
    
    # Save final model
    if accelerator.is_main_process:
        final_output_dir = os.path.join(args.output_dir, "final_model")
        os.makedirs(final_output_dir, exist_ok=True)
        
        # Unwrap and save
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.model.save_pretrained(
            final_output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save
        )
        tokenizer.save_pretrained(final_output_dir)

def evaluate(model, eval_dataloader, accelerator):
    """Evaluate the model on the evaluation dataset"""
    model.eval()
    eval_loss = 0
    eval_steps = 0
    
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["labels"],
                k=1  # Fixed k=1 for evaluation to match original behavior
            )
        
        loss = outputs["loss"]
        eval_loss += loss.detach().float()
        eval_steps += 1
    
    # Gather losses across all processes
    eval_loss = accelerator.gather(eval_loss).mean().item()
    eval_loss = eval_loss / eval_steps
    
    accelerator.print(f"Evaluation loss: {eval_loss:.4f}")
    return eval_loss

if __name__ == "__main__":
    main()