"""
Online GRPO Training for CUDA Kernel Optimization
==================================================

Architecture: Generate kernels → Benchmark with KernelBench → Train with GRPO → Repeat

This implements true online RL where the model improves between generation cycles,
not supervised learning on a fixed dataset.
"""

import modal
import torch
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json
import subprocess
import tempfile
import os
import re

# ============================================================================
# MODAL SETUP
# ============================================================================

app = modal.App("cuda-kernel-online-grpo")

# Build image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.7.0",
        "transformers==4.51.1",
        "trl==0.19.1",
        "vllm==0.9.1",
        "datasets==3.5.1",
        "peft==0.15.0",
        "accelerate==1.5.1",
        "wandb==0.17.6",
        "ninja",
        "litellm",
        "openai",
        "requests",
    )
    .run_commands(
        "git clone https://github.com/ScalingIntelligence/KernelBench /root/KernelBench",
        "cd /root/KernelBench && pip install -e ."
    )
)

# Persistent volumes
checkpoint_vol = modal.Volume.from_name("grpo-checkpoints", create_if_missing=True)
results_vol = modal.Volume.from_name("grpo-results", create_if_missing=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig:
    """Online GRPO training configuration"""
    
    # Model
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    
    # Online RL parameters
    num_training_iterations: int = 10  # How many generate-train cycles
    num_generations_per_iter: int = 4  # Completions per prompt per iteration
    problems_per_iter: int = 16  # How many problems to sample each iteration
    
    # GRPO parameters
    learning_rate: float = 1e-6
    grpo_epsilon: float = 0.2
    grpo_beta: float = 0.0  # KL penalty coefficient
    loss_type: str = "dapo"  # dapo, grpo, or dr_grpo
    
    # Generation parameters
    max_prompt_length: int = 1024
    max_completion_length: int = 2048
    temperature: float = 0.7
    
    # Training parameters
    mini_epochs: int = 2  # How many gradient updates per generation batch
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    
    # LoRA configuration
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    
    # Checkpointing
    save_every_n_iters: int = 10
    
    # Reward weights
    reward_compilation: float = 0.2
    reward_correctness: float = 0.5
    reward_speedup: float = 0.3
    speedup_target: float = 2.0  # Normalize speedups to this value

# ============================================================================
# KERNELBENCH INTEGRATION
# ============================================================================

@dataclass
class BenchmarkResult:
    """Result from benchmarking a single kernel"""
    compiles: bool
    correct: bool
    speedup: float
    error_message: str = ""
    execution_time: float = 0.0

def get_kernelbench_problems(level: int = 1) -> List[str]:
    """
    Load problems from KernelBench
    
    Returns list of dicts with:
    - problem_id: Unique identifier
    - reference_code: PyTorch implementation
    """
    import sys
    sys.path.insert(0, '/root/KernelBench')
    from src.dataset import construct_kernelbench_dataset
    problems = construct_kernelbench_dataset(level=level)

    return problems

@app.function(
    image=image,
    gpu="L40S",
    timeout=300,
)
def benchmark_kernel(
    kernel_code: str,
    reference_code: str,
    problem_id: str
) -> BenchmarkResult:
    """
    Benchmark a single generated kernel against reference

    Executes in isolated environment with proper error handling
    Uses KernelBench's eval_kernel_against_ref API
    """
    import torch
    import sys
    sys.path.insert(0, '/root/KernelBench')
    from src.eval import eval_kernel_against_ref, get_torch_dtype_from_string
    from src.utils import set_gpu_arch

    result = BenchmarkResult(
        compiles=False,
        correct=False,
        speedup=0.0
    )

    try:
        # Set GPU architecture (Ada for L40S)
        set_gpu_arch(["Ada"])
        device = torch.device("cuda:0")

        # Combine reference code with test inputs
        # KernelBench expects reference code to include Model and get_inputs

        # Use KernelBench's proper evaluation API
        eval_result = eval_kernel_against_ref(
            original_model_src=reference_code,
            custom_model_src=kernel_code,
            measure_performance=True,
            verbose=False,
            num_correct_trials=5,
            num_perf_trials=10,
            device=device,
            backend="cuda",
            precision=get_torch_dtype_from_string("fp32")
        )

        # Map KernelBench result to our BenchmarkResult
        result.compiles = eval_result.compiled
        result.correct = eval_result.correctness

        if eval_result.correctness and eval_result.runtime is not None:
            # Calculate speedup from baseline and runtime
            # eval_result contains runtime for the kernel
            # We need to measure baseline time as well
            from scripts.generate_baseline_time import measure_program_time

            baseline_result = measure_program_time(
                ref_arch_name="Reference",
                ref_arch_src=reference_code,
                num_trials=10,
                use_torch_compile=False,
                device=device
            )

            baseline_time = baseline_result.get("mean", None)
            if baseline_time and baseline_time > 0:
                result.speedup = baseline_time / eval_result.runtime
                result.execution_time = eval_result.runtime
            else:
                result.speedup = 0.0
                result.execution_time = eval_result.runtime
        else:
            result.speedup = 0.0

        # Extract error message from metadata if available
        if hasattr(eval_result, 'metadata') and eval_result.metadata:
            if 'cuda_error' in eval_result.metadata:
                result.error_message = eval_result.metadata['cuda_error']
            elif 'other_error' in eval_result.metadata:
                result.error_message = eval_result.metadata['other_error']
            elif 'correctness_error' in eval_result.metadata:
                result.error_message = eval_result.metadata['correctness_error']

        if not result.compiles:
            result.error_message = result.error_message or "Kernel compilation failed"
        elif not result.correct:
            result.error_message = result.error_message or "Kernel correctness check failed"

        return result

    except Exception as e:
        result.error_message = f"Unexpected error: {str(e)}"
        return result

def compute_fast_p(results: List[BenchmarkResult], p: float) -> float:
    """
    Compute fast_p metric: fraction of samples that are correct AND have speedup > p

    This is the official KernelBench metric.
    """
    total = len(results)
    if total == 0:
        return 0.0

    # Count samples that are both correct and faster than threshold
    fast_and_correct = sum(
        1 for r in results
        if r.correct and r.speedup > p
    )

    return fast_and_correct / total

def compute_reward(result: BenchmarkResult, config: TrainingConfig) -> float:
    """
    Compute scalar reward from benchmark result

    Reward components:
    1. Compilation success (partial credit)
    2. Correctness (major component)
    3. Speedup (performance bonus)
    """
    reward = 0.0

    # Compilation reward
    if result.compiles:
        reward += config.reward_compilation

    # Correctness reward
    if result.correct:
        reward += config.reward_correctness

        # Speedup bonus (only if correct)
        normalized_speedup = min(result.speedup / config.speedup_target, 1.0)
        reward += config.reward_speedup * normalized_speedup

    return reward
def extract_last_code(output_string: str, code_language_types: list[str]) -> str | None:
    """
    Extract last code block from model output, specified by code_language_type
    """
    trimmed = output_string.strip()

    # Find all matches of code blocks
    code_matches = re.finditer(r"```(.*?)```", trimmed, re.DOTALL)
    
    # Get the last match by converting to list and taking the last element
    matches_list = list(code_matches)
    if matches_list:
        last_match = matches_list[-1]
        code = last_match.group(1).strip()

        # Remove language type headers
        for code_type in code_language_types:
            if code.startswith(code_type):
                code = code[len(code_type):].strip()

        return code
    
    return None
# ============================================================================
# PROMPT ENGINEERING
# ============================================================================

def create_kernel_optimization_prompt(reference_code: str, iteration: int) -> str:
    """
    Create prompt for kernel optimization
    
    Includes clear instructions and constraints for CUDA generation
    """
    import sys

    sys.path.insert(0, '/root/KernelBench')
    from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template
    prompt = prompt_generate_custom_cuda_from_prompt_template(
        ref_arch_src=reference_code,
    )
    return prompt
def read_file(file_path) -> str:
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return ""
    
    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

# ============================================================================
# ONLINE RL TRAINING LOOP
# ============================================================================

@app.function(
    gpu="L40S:2",  # 2 GPUs: one for generation (vLLM), one for training
    timeout=86400,  #24 hours
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={
        "/checkpoints": checkpoint_vol,
        "/results": results_vol
    },
    image=image
)
def train_online_grpo():
    """
    Main online GRPO training loop
    
    For each iteration:
    1. Sample problems from KernelBench
    2. Generate multiple kernel candidates with current model
    3. Benchmark all candidates in parallel
    4. Compute rewards from benchmark results
    5. Update model with GRPO on this batch
    6. Save checkpoint and metrics
    7. Repeat with improved model
    """
    import wandb
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import GRPOConfig, GRPOTrainer
    import numpy as np
    
    config = TrainingConfig()
    
    # Initialize wandb
    wandb.init(
        project="cuda-kernel-grpo",
        config=config.__dict__,
        name=f"online_grpo_{config.model_name.split('/')[-1]}"
    )
    
    # Load base model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 1},  # Put on GPU 1 (GPU 0 for vLLM)
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True
    )
    
    # Apply LoRA for efficient training
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Start vLLM server on GPU 0 for fast generation
    print("Starting vLLM server...")
    import time
    import requests

    vllm_env = os.environ.copy()
    vllm_env["CUDA_VISIBLE_DEVICES"] = "0"
    # Stwórz pliki dla logów
    vllm_stdout_file = open("/tmp/vllm_stdout.log", "w")
    vllm_stderr_file = open("/tmp/vllm_stderr.log", "w")

    vllm_process = subprocess.Popen([
    "python", "-m", "vllm.entrypoints.openai.api_server",
    "--model", config.model_name,
    "--gpu-memory-utilization", "0.7",  # ← Zmniejszone!
    "--dtype", "bfloat16",
    "--max-model-len", "4096",  # ← Zmniejszone!
    "--port", "8000",
    "--tensor-parallel-size", "1",
    
    # NOWE - limitują obciążenie:
    "--max-num-seqs", "4",
    "--max-num-batched-tokens", "8192",
    "--disable-log-requests",  # Mniej I/O
    "--enforce-eager",  # Wyłącza CUDA graphs (bardziej stabilne)
    ],
    env=vllm_env,
    stdout=vllm_stdout_file,
    stderr=vllm_stderr_file,
    text=True
    )


    # Wait for vLLM to be ready with health check
    print("Waiting for vLLM server to be ready...")
    max_wait = 180  # 3 minutes
    start_time = time.time()
    server_ready = False

    while time.time() - start_time < max_wait:
        try:
            response = requests.get("http://localhost:8000/health", timeout=1)
            if response.status_code == 200:
                server_ready = True
                print("vLLM server is ready!")
                break
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            pass

        # Check if process died
        if vllm_process.poll() is not None:
            stdout, stderr = vllm_process.communicate()
            print(f"vLLM process died! Exit code: {vllm_process.returncode}")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            raise RuntimeError("vLLM server failed to start")

        time.sleep(2)

    if not server_ready:
        vllm_process.terminate()
        stdout, stderr = vllm_process.communicate(timeout=5)
        print(f"vLLM server failed to start within {max_wait}s")
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
        raise RuntimeError("vLLM server startup timeout")
    
    # Load KernelBench problems
    print("Loading KernelBench problems...")
    curr_level_dataset = get_kernelbench_problems(level=1)
    
    # Training loop
    for iteration in range(config.num_training_iterations):
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration + 1}/{config.num_training_iterations}")
        print(f"{'='*80}\n")
        
        # Sample problems for this iteration
        problem_indices = np.random.choice(
            curr_level_dataset,
            size=min(config.problems_per_iter, len(curr_level_dataset)),
            replace=False
        ).tolist()

        # Phase 1: GENERATION
        print("Phase 1: Generating kernel candidates...")
        all_completions = []
        all_prompts = []
        all_references = []
        all_problem_ids = []
        


        from openai import OpenAI
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="dummy",
            timeout=120.0,  # 2 minute timeout for API calls
            max_retries=2  # Retry up to 2 times on connection errors
        )

        for problem in problem_indices:
            problem_name = os.path.basename(problem)
            ref_arch_src =read_file(problem)
            problem_number = int(problem_name.split("_")[0])
            prompt = create_kernel_optimization_prompt(
                ref_arch_src,
                iteration
            )

            # Generate multiple candidates
            for gen_idx in range(config.num_generations_per_iter):
                print(f"Generating candidate {gen_idx + 1}/{config.num_generations_per_iter} for problem {problem_number}...")
                try:
                    try:
                        response = client.completions.create(
                            model=config.model_name,
                            prompt=prompt,
                            max_tokens=config.max_completion_length,
                            temperature=config.temperature,
                            n=1
                        )
                    except Exception as e:
                        print(f"Generation error for problem {problem_number} (candidate {gen_idx + 1}): {e}")
                        continue
                    completion = response.choices[0].text
                    code = extract_last_code(completion, ["python", "cpp"])
                    if code is None:
                        code = ""
                        print(f"Warning: No code extracted for problem {problem_number} (candidate {gen_idx + 1})")
                    else:
                        print(f"Generated code for problem {problem_number} (candidate {gen_idx + 1})\n")
                    all_completions.append(code)
                    all_prompts.append(prompt)
                    all_references.append(ref_arch_src)
                    all_problem_ids.append(problem_number)
                except Exception as e:
                    print(f"Unexpected error during generation for problem {problem_number} (candidate {gen_idx + 1}): {e}")
                    continue
        
        print(f"Generated {len(all_completions)} kernel candidates")
        
        # Phase 2: BENCHMARKING
        print("Phase 2: Benchmarking kernels in parallel...")
        
        # Parallel benchmark execution
        benchmark_results = list(benchmark_kernel.starmap(
            zip(
                all_completions,
                all_references,
                all_problem_ids
            )
        ))
        
        # Compute rewards
        rewards = [compute_reward(result, config) for result in benchmark_results]

        # Log statistics
        num_compiled = sum(1 for r in benchmark_results if r.compiles)
        num_correct = sum(1 for r in benchmark_results if r.correct)
        speedups = [r.speedup for r in benchmark_results if r.correct]
        avg_speedup = np.mean(speedups) if speedups else 0.0

        # Compute fast_p metrics for different thresholds (KernelBench standard)
        fast_p_thresholds = [0.0, 0.5, 0.8, 1.0, 1.5, 2.0]
        fast_p_scores = {
            f"fast_p_{p}": compute_fast_p(benchmark_results, p)
            for p in fast_p_thresholds
        }

        print(f"Benchmark Results:")
        print(f"  Compiled: {num_compiled}/{len(all_completions)} ({100*num_compiled/len(all_completions):.1f}%)")
        print(f"  Correct: {num_correct}/{len(all_completions)} ({100*num_correct/len(all_completions):.1f}%)")
        print(f"  Avg Speedup (correct only): {avg_speedup:.2f}x")
        print(f"  Avg Reward: {np.mean(rewards):.3f}")
        print(f"  Fast_p Metrics:")
        for p in fast_p_thresholds:
            print(f"    fast_p@{p}: {fast_p_scores[f'fast_p_{p}']:.3f}")

        wandb.log({
            "iteration": iteration,
            "compilation_rate": num_compiled / len(all_completions),
            "correctness_rate": num_correct / len(all_completions),
            "avg_speedup": avg_speedup,
            "avg_reward": np.mean(rewards),
            "reward_std": np.std(rewards),
            **fast_p_scores,  # Add all fast_p scores
        })
        
        # Phase 3: GRPO TRAINING
        print("Phase 3: Training model with GRPO...")
        
        # Prepare training data
        from datasets import Dataset
        
        train_data = {
            "prompt": all_prompts,
            "completion": all_completions,
            "reward": rewards,
        }
        train_dataset = Dataset.from_dict(train_data)
        
        # GRPO configuration for this mini-training
        grpo_config = GRPOConfig(
            output_dir=f"/checkpoints/iter_{iteration}",
            num_train_epochs=config.mini_epochs,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            
            # GRPO parameters
            num_generations=config.num_generations_per_iter,
            epsilon=config.grpo_epsilon,
            beta=config.grpo_beta,
            loss_type=config.loss_type,
            
            # Optimization
            gradient_checkpointing=True,
            bf16=True,
            
            # Logging
            logging_steps=1,
            report_to=["wandb"],
            save_steps=999999,  # Don't auto-save, we'll do it manually
        )
        
        # Create trainer (note: we're using pre-generated completions)
        trainer = GRPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )
        
        # Train on this batch
        trainer.train()
        
        # Phase 4: CHECKPOINTING
        if (iteration + 1) % config.save_every_n_iters == 0:
            checkpoint_path = f"/checkpoints/iteration_{iteration+1}"
            print(f"Saving checkpoint to {checkpoint_path}")
            trainer.save_model(checkpoint_path)
            checkpoint_vol.commit()
            
            # Save metrics
            metrics = {
                "iteration": iteration + 1,
                "compilation_rate": num_compiled / len(all_completions),
                "correctness_rate": num_correct / len(all_completions),
                "avg_speedup": avg_speedup,
                "avg_reward": np.mean(rewards),
                "speedup_distribution": speedups[:100],  # Sample for storage
                "fast_p_scores": fast_p_scores,  # Include fast_p metrics
            }

            with open(f"/results/metrics_iter_{iteration+1}.json", "w") as f:
                json.dump(metrics, f, indent=2)
            results_vol.commit()
    
    # Final save
    print("\nTraining complete! Saving final model...")
    trainer.save_model("/checkpoints/final_model")
    checkpoint_vol.commit()
    
    # Cleanup
    vllm_process.terminate()
    try:
        vllm_stdout_file.close()
        vllm_stderr_file.close()
    except:
        pass
    wandb.finish()

# ============================================================================
# INFERENCE ENDPOINT
# ============================================================================

@app.function(
    gpu="L40S",
    image=image,
    volumes={"/checkpoints": checkpoint_vol},
)
def optimize_kernel(
    pytorch_code: str,
    checkpoint_path: str = "/checkpoints/final_model"
) -> Dict[str, str]:
    """
    Inference endpoint: optimize a PyTorch kernel with trained model
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    # Load trained model
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    # Generate optimized kernel
    prompt = create_kernel_optimization_prompt(pytorch_code, 0)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=768,
        temperature=0.7,
        do_sample=True
    )
    
    optimized_code = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:])
    
    return {
        "original_code": pytorch_code,
        "optimized_code": optimized_code,
        "prompt": prompt
    }

# ============================================================================
# ENTRY POINTS
# ============================================================================

@app.local_entrypoint()
def main(action: str = "train"):
    """
    Entry point for the application
    
    Usage:
        modal run cuda_kernel_online_grpo.py --action train
        modal run cuda_kernel_online_grpo.py --action optimize --code "..."
    """
    if action == "train":
        print("Starting online GRPO training...")
        train_online_grpo.remote()
    
    elif action == "optimize":
        # Example optimization
        example_code = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        return x + y

def get_inputs():
    x = torch.randn(1024, 1024).cuda()
    y = torch.randn(1024, 1024).cuda()
    return [x, y]
"""
        result = optimize_kernel.remote(example_code)
        print("Optimized kernel:")
        print(result['optimized_code'])
    
    else:
        print(f"Unknown action: {action}")
        print("Valid actions: train, optimize")

if __name__ == "__main__":
    main()