import openai
from openai import AsyncOpenAI
import json
import random
import re
from datasets import load_dataset
import pprint
import ast
from tqdm import tqdm
import asyncio
from typing import List, Dict, Any
import time
import requests

API_KEY = 'test'
BASE_URL = '<YOUR_APP_URL>/v1'  # Adjust if using a different endpoint
DEFAULT_MODELS = [
    "Qwen/Qwen3-8B",
    "qwen_v1"
]
NUM_SAMPLES = 500
NUM_COCNURRENT_INFERENCE_CALLS = 20

def load_lora_adapter(lora_path):
    """Load a LoRA adapter using the API."""
    # Use the same value for lora_name as lora_path
    lora_name = lora_path
    
    url = f"{BASE_URL.replace('/v1', '')}/v1/load_lora_adapter"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    data = {
        "lora_name": lora_name,
        "lora_path": lora_path
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        print(f"Successfully loaded LoRA adapter: {lora_name}")
        return True
    except Exception as e:
        print(f"Error loading LoRA adapter: {e}")
        return False

def get_available_models():
    """Query the API to get available models."""
    url = f"{BASE_URL}/models"
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        models_data = response.json()
        
        # Extract model IDs
        model_ids = [model['id'] for model in models_data.get('data', [])]
        return model_ids
    except Exception as e:
        print(f"Error querying models: {e}")
        return DEFAULT_MODELS

def interactive_menu():
    """Interactive menu to choose between loading LoRA or running evaluations."""
    while True:
        print("\nWhat would you like to do?")
        print("1. Load LoRA adapter")
        print("2. Run evaluations")
        choice = input("Enter your choice (1 or 2): ").strip()
        
        if choice == "1":
            lora_path = input("Enter the HuggingFace path for the LoRA adapter: ").strip()
            if lora_path:
                if load_lora_adapter(lora_path):
                    print("LoRA adapter loaded successfully!")
                else:
                    print("Failed to load LoRA adapter.")
            else:
                print("Invalid path provided.")
            return None  # Don't run evaluations
            
        elif choice == "2":
            # Get available models
            available_models = get_available_models()
            
            print("\nAvailable models:")
            for i, model in enumerate(available_models, 1):
                print(f"{i}. {model}")
            print(f"{len(available_models) + 1}. All models")
            
            model_choice = input(f"Choose models (1-{len(available_models) + 1}): ").strip()
            
            try:
                choice_num = int(model_choice)
                if choice_num == len(available_models) + 1:
                    return available_models  # All models
                elif 1 <= choice_num <= len(available_models):
                    return [available_models[choice_num - 1]]  # Single model
                else:
                    print("Invalid choice.")
                    continue
            except ValueError:
                print("Please enter a valid number.")
                continue
        else:
            print("Invalid choice. Please enter 1 or 2.")


def safe_parse_function_call(match_text):
    """Try to parse function call using both JSON and AST literal eval."""
    cleaned_text = match_text.strip()
    
    # First try JSON parsing
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        pass
    
    # Then try AST literal eval (handles Python-style syntax)
    try:
        return ast.literal_eval(cleaned_text)
    except (ValueError, SyntaxError):
        pass
    
    # Both methods failed
    return None



def load_hermes_dataset(max_samples=500):
    """Load Hermes function calling dataset (randomized version)."""
    print("Loading Hermes dataset...")
    dataset = load_dataset("NousResearch/hermes-function-calling-v1", "func_calling_singleturn")
    train = dataset["train"]
    num_total = len(train)
    indices = list(range(num_total))

    # Shuffle all indices
    random.shuffle(indices)

    samples = []
    for i in range(min(max_samples, num_total)):
        idx = indices[i]
        sample = train[idx]
        
        system_prompt = ""
        user_query = ""
        expected_response = ""
        
        for conv in sample["conversations"]:
            if conv["from"] == "system":
                system_prompt = conv["value"]
            elif conv["from"] == "human":
                user_query = conv["value"]
            elif conv["from"] == "gpt":
                expected_response = conv["value"]
        
        if user_query and expected_response:
            samples.append({
                "id": f"sample_{i}",
                "system_prompt": system_prompt,
                "user_query": user_query,
                "expected_response": expected_response
            })
    
    print(f"Loaded {len(samples)} samples")
    return samples


async def get_model_response_async(model, system_prompt, user_query, semaphore):
    """Get OpenAI model response asynchronously with rate limiting."""
    
    async with semaphore:
        client = AsyncOpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,  # Adjust if using a different endpoint
        )
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_query})
        
        try:
            if model in ["o3", "o4-mini"]:
                # For OpenAI's new models, use the chat completion endpoint
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
            else:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=2000,  # Adjust as needed
                )
            return response.choices[0].message.content
        except Exception as e:
            return f"ERROR: {str(e)}"


def extract_function_calls(text):
    """Extract function calls from text response."""
    
    function_calls = []
    
    # Look for <tool_call> format (Hermes style)
    tool_call_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(tool_call_pattern, text, re.DOTALL)
    
    for match in matches:
        func_call = safe_parse_function_call(match)
        if func_call is not None:
            function_calls.append(func_call)
        else:
            print(f"Failed to parse function call from match (tried both JSON and AST):\n{match}")
    
    # Also try to find JSON objects with name and arguments
    if not function_calls:
        json_pattern = r'\{[^{}]*"name"\s*:\s*"[^"]*"[^{}]*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            func_call = safe_parse_function_call(match)
            if func_call is not None:
                function_calls.append(func_call)
    
    return function_calls


def calculate_objective_metrics(expected_response, actual_response):
    """Calculate simple objective metrics."""
    
    # Extract function calls
    expected_calls = extract_function_calls(expected_response)
    actual_calls = extract_function_calls(actual_response)
    
    metrics = {
        "exact_match": 0,
        "function_name_accuracy": 0,
        "parameter_f1": 0,
        "format_valid": 1 if actual_calls else 0
    }
    
    # No function calls expected or generated
    if not expected_calls and not actual_calls:
        metrics["exact_match"] = 1
        metrics["function_name_accuracy"] = 1
        metrics["parameter_f1"] = 1
        return metrics
    
    # One side has calls, other doesn't
    if not expected_calls or not actual_calls:
        return metrics
    
    # Exact match check
    if len(expected_calls) == len(actual_calls):
        exact_match = True
        for exp_call, act_call in zip(expected_calls, actual_calls):
            if (exp_call.get("name") != act_call.get("name") or 
                exp_call.get("arguments") != act_call.get("arguments")):
                exact_match = False
                break
        metrics["exact_match"] = int(exact_match)
    
    # Function name accuracy
    expected_names = [call.get("name", "") for call in expected_calls]
    actual_names = [call.get("name", "") for call in actual_calls]
    
    correct_names = 0
    for exp_name in expected_names:
        if exp_name in actual_names:
            correct_names += 1
    
    if expected_names:
        metrics["function_name_accuracy"] = correct_names / len(expected_names)
    
    # Parameter F1 (simplified)
    expected_params = set()
    actual_params = set()
    
    for call in expected_calls:
        args = call.get("arguments", {})
        if isinstance(args, dict):
            for k, v in args.items():
                expected_params.add(f"{k}:{v}")
    
    for call in actual_calls:
        args = call.get("arguments", {})
        if isinstance(args, dict):
            for k, v in args.items():
                actual_params.add(f"{k}:{v}")
    
    if expected_params or actual_params:
        intersection = len(expected_params & actual_params)
        precision = intersection / len(actual_params) if actual_params else 0
        recall = intersection / len(expected_params) if expected_params else 0
        
        if precision + recall > 0:
            metrics["parameter_f1"] = 2 * precision * recall / (precision + recall)
    
    return metrics


def evaluate_single_sample(sample, model_response, judge_model="gpt-4o"):
    """Evaluate single function calling sample with both objective and LLM judge metrics."""
    
    # Calculate objective metrics
    objective_metrics = calculate_objective_metrics(
        sample["expected_response"], 
        model_response
    )
    
    # Combine all metrics
    result = {
        "sample_id": sample["id"],
        **objective_metrics
    }
    
    return result


def print_model_summary(model, results):
    """Print summary for a single model."""
    
    if not results:
        return
    
    # Calculate averages
    total = len(results)
    exact_match = sum(r["exact_match"] for r in results) / total
    function_name_acc = sum(r["function_name_accuracy"] for r in results) / total
    parameter_f1 = sum(r["parameter_f1"] for r in results) / total
    format_valid = sum(r["format_valid"] for r in results) / total
    
    print(f"\n{model} Summary:")
    print("-" * 40)
    print(f"Total samples: {total}")
    print(f"Exact Match: {exact_match:.3f}")
    print(f"Function Name Accuracy: {function_name_acc:.3f}")
    print(f"Parameter F1: {parameter_f1:.3f}")
    print(f"Format Valid: {format_valid:.3f}")


async def process_single_sample(sample: Dict[str, Any], model: str, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    """Process a single sample asynchronously."""
    
    # Get model response
    response = await get_model_response_async(
        model, sample["system_prompt"], sample["user_query"], semaphore
    )
    
    # print(f"Sample {sample['id']} response: {response}")
    # Evaluate
    eval_result = evaluate_single_sample(sample, response)
    eval_result["model"] = model
    eval_result["actual_response"] = response
    
    return eval_result


class AsyncProgressTracker:
    """Track progress of async tasks with tqdm."""
    
    def __init__(self, total: int, desc: str = "Processing"):
        self.pbar = tqdm(total=total, desc=desc)
        self.completed = 0
    
    def update(self):
        self.completed += 1
        self.pbar.update(1)
    
    def close(self):
        self.pbar.close()


async def run_benchmark_async(eval_samples: List[Dict[str, Any]], models: List[str] = None, max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """Run benchmark on OpenAI models with async parallelization."""
    
    if models is None:
        models = DEFAULT_MODELS
    
    all_results = []
    
    for model in models:
        print(f"\nEvaluating {model}...")
        start_time = time.time()
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Create tasks for all samples
        tasks = []
        for sample in eval_samples:
            task = asyncio.create_task(
                process_single_sample(sample, model, semaphore)
            )
            tasks.append(task)
        
        # Initialize progress tracker
        progress_tracker = AsyncProgressTracker(len(tasks), f"Processing {model}")
        
        # Process tasks as they complete
        model_results = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            model_results.append(result)
            progress_tracker.update()
        
        progress_tracker.close()
        
        # Calculate and print timing
        elapsed_time = time.time() - start_time
        print(f"Completed {len(eval_samples)} samples in {elapsed_time:.2f} seconds")
        print(f"Average time per sample: {elapsed_time/len(eval_samples):.2f} seconds")
        
        all_results.extend(model_results)
        
        # Print model summary
        print_model_summary(model, model_results)
    
    return all_results


def print_final_results(results):
    """Print final benchmark summary."""
    
    # Group by model
    model_results = {}
    for result in results:
        model = result["model"]
        if model not in model_results:
            model_results[model] = []
        model_results[model].append(result)
    
    print("\n" + "="*60)
    print("FINAL BENCHMARK SUMMARY")
    print("="*60)
    print(f"{'Model':<15} {'ExactMatch':<10} {'FuncName':<10} {'ParamF1':<10} {'FormatValid':<10}")
    print("-" * 60)
    
    for model, model_data in model_results.items():
        total = len(model_data)
        exact_match = sum(r["exact_match"] for r in model_data) / total
        func_name = sum(r["function_name_accuracy"] for r in model_data) / total
        param_f1 = sum(r["parameter_f1"] for r in model_data) / total
        format_valid = sum(r["format_valid"] for r in model_data) / total
        
        print(f"{model:<15} {exact_match:<10.3f} {func_name:<10.3f} {param_f1:<10.3f} {format_valid:<10.3f}")


def save_results(results, filename="benchmark_results.json"):
    """Save results to JSON file."""
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filename}")


async def main():
    """Main async function."""
    """Main async function."""
    print("Starting async function calling benchmark...")
    
    # Interactive menu
    selected_models = interactive_menu()
    
    if selected_models is None:
        print("Exiting...")
        return
    
    total_start_time = time.time()
    
    # Load dataset
    dataset = load_hermes_dataset(max_samples=NUM_SAMPLES)
    
    # Run benchmark asynchronously
    # Adjust max_concurrent based on your OpenAI tier:
    # Tier 1: 3-5, Tier 2: 10-15, Tier 3+: 20+
    results = await run_benchmark_async(
        dataset, 
        models=selected_models, 
        max_concurrent=20
    )
    
    # Calculate total time
    total_time = time.time() - total_start_time
    print(f"\nTotal benchmark time: {total_time:.2f} seconds")
    print(f"Average time per model: {total_time/len(selected_models):.2f} seconds")
    
    # Show results
    print_final_results(results)
    
    # Save results
    save_results(results)


if __name__ == "__main__":
    # Run the async main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"Error running benchmark: {e}")
        raise
