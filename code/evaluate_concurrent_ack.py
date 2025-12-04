import os
import json
import time
import boto3
import argparse
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from botocore.exceptions import ClientError
from typing import List, Dict, Tuple

# model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"
# model_id = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

def parse_preference_and_answer(input_string):
    """Parse preference and answer from XML-formatted response."""
    soup = BeautifulSoup(input_string, "html.parser")
    
    preference_tag = soup.find("preference")
    preference = preference_tag.text.strip() if preference_tag else ""
    
    answer_tag = soup.find("answer")
    answer = answer_tag.text.strip() if answer_tag else ""
    
    return preference, answer


def generate_message(bedrock_runtime, model_id, system_prompt, messages, max_tokens, max_retries=20):
    """Generate response using AWS Bedrock with retry logic."""
    retries = 0
    while retries < max_retries:
        try:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": messages,
                "temperature": 0.0,
            })

            response = bedrock_runtime.invoke_model(body=body, modelId=model_id)
            response_body = json.loads(response.get("body").read())
            return response_body
        except ClientError as e:
            retries += 1
            print(f"Error: {str(e)}, retrying: {retries}")
            if retries == 19:
                time.sleep(1)
                retries = 0
            time.sleep(0.5)  # Reduced sleep time for faster retries


def check_acknowledgement_string_match(target, response_text):
    """
    Check if target string exists in response text for acknowledgement evaluation.
    Returns True if target is found (case-insensitive), False otherwise.
    """
    if not target or not response_text:
        return False
    
    # Convert both to lowercase for case-insensitive matching
    target_lower = target.lower().strip()
    response_lower = response_text.lower().strip()
    
    # Check if target exists as substring in response
    return target_lower in response_lower


def load_ack_prompts(base_dir):
    """Load evaluation prompts for acknowledge metric."""
    file_dict = {
        "acknow": "check_acknowledge_with_pref.txt",
    }
    
    eval_message_texts = []
    for metric_name, file_name in file_dict.items():
        file_path = os.path.join(base_dir, file_name)
        try:
            with open(file_path, "r") as f:
                eval_message_texts.append([metric_name, f.read()])
        except FileNotFoundError:
            print(f"Warning: Could not find {file_path}. Skipping {metric_name} evaluation.")
    
    return eval_message_texts


def evaluate_batch_responses(client, model_id, system_prompt, max_tokens, data, eval_message_texts, batch_size=8):
    """
    Evaluate responses for acknowledge metric using concurrent processing.
    
    Uses hybrid evaluation approach:
    1. String matching for acknowledgement: checks if target exists in response_to_q
    2. LLM evaluation for remaining acknowledgement cases where string matching fails
    """
    
    # First pass: Handle acknowledgement with string matching
    string_match_count = 0
    for task_idx, task in enumerate(data):
        if "response_to_q" not in task:
            continue
            
        end_generation = task["response_to_q"]
        target = task.get("target", "")
        error_check = task.get("evaluation_error_analysis", {})
        
        # Check acknowledgement using string matching first
        if "acknow" not in error_check:
            if check_acknowledgement_string_match(target, end_generation):
                # Initialize evaluation_error_analysis if not exists
                if "evaluation_error_analysis" not in data[task_idx]:
                    data[task_idx]["evaluation_error_analysis"] = {}
                
                # Set acknowledgement result based on string match
                data[task_idx]["evaluation_error_analysis"]["acknow"] = {
                    "answer": "Yes",
                    "extract_pref": target,
                    "method": "string_match"  # Track evaluation method
                }
                string_match_count += 1
    
    if string_match_count > 0:
        print(f"String matching found acknowledgement in {string_match_count} responses")
    
    # Collect all requests for LLM evaluation (remaining acknowledgement cases)
    all_requests = []
    
    for task_idx, task in enumerate(data):
        if "response_to_q" not in task:
            continue
            
        preference = task["preference"]
        question = task["question"]
        end_generation = task["response_to_q"]
        error_check = task.get("evaluation_error_analysis", {})
        
        for metric, eval_message_text in eval_message_texts:
            if metric in error_check:
                continue
                
            # Replace placeholders for acknowledge
            eval_text = eval_message_text.replace("{end_generation}", end_generation)
            eval_text = eval_text.replace("{question}", question)
            eval_text = eval_text.replace("{user_preference}", preference)
            
            eval_message = [{"role": "user", "content": eval_text}]
            all_requests.append((task_idx, metric, eval_message))
    
    # Process all requests in batches
    if all_requests:
        print(f"Processing {len(all_requests)} LLM evaluation requests...")
        for i in range(0, len(all_requests), batch_size):
            batch = all_requests[i:i+batch_size]
            message_batch = [req[2] for req in batch]
            
            print(f"Processing batch {i//batch_size + 1}/{(len(all_requests)-1)//batch_size + 1}")
            
            with ThreadPoolExecutor(max_workers=min(batch_size, len(message_batch))) as executor:
                # Submit all requests and maintain order
                futures = []
                for j, messages in enumerate(message_batch):
                    future = executor.submit(generate_message, client, model_id, system_prompt, messages, max_tokens)
                    futures.append((future, batch[j]))
                
                # Process results in submission order to ensure determinism
                for future, (task_idx, metric, _) in futures:
                    try:
                        result = future.result()  # This will block until the specific future completes
                        
                        if "evaluation_error_analysis" not in data[task_idx]:
                            data[task_idx]["evaluation_error_analysis"] = {}
                        
                        eval_response_text = result["content"][0]["text"]
                        extract_preference, answer = parse_preference_and_answer(eval_response_text)
                        data[task_idx]["evaluation_error_analysis"][metric] = {
                            "answer": answer,
                            "extract_pref": extract_preference,
                            "method": "llm_evaluation"  # Track evaluation method
                        }
                    except Exception as e:
                        print(f"Error in {metric} evaluation: {str(e)}")
    
    return data


def analyze_ack_results(data):
    """Analyze acknowledge results and calculate statistics."""
    stats = {
        "acknowledgement": 0,
        "string_match_ack": 0,
        "llm_ack": 0,
    }
    
    valid_entries = 0
    
    for idx, entry in enumerate(data):
        if "evaluation_error_analysis" not in entry:
            print(f"Warning: Entry {idx} has not been evaluated yet!")
            continue
            
        if "response_to_q" not in entry:
            print(f"Warning: Entry {idx} has no response!")
            continue
            
        valid_entries += 1
        error_types = entry["evaluation_error_analysis"]
        
        # Extract evaluation results
        is_acknowledgement = "yes" in error_types.get("acknow", {}).get("answer", "").lower()
        eval_method = error_types.get("acknow", {}).get("method", "unknown")
        
        # Update statistics
        stats["acknowledgement"] += is_acknowledgement
        if is_acknowledgement and eval_method == "string_match":
            stats["string_match_ack"] += 1
        elif is_acknowledgement and eval_method == "llm_evaluation":
            stats["llm_ack"] += 1
    
    return stats, valid_entries


def print_ack_results(stats, total_data, input_file):
    """Print acknowledge evaluation results."""
    print("\n" + "="*60)
    print("ACKNOWLEDGEMENT EVALUATION RESULTS")
    print("="*60)
    print(f"Input File: {input_file}")
    print(f"Total Entries Evaluated: {total_data}")
    if total_data == 0:
        print("No valid entries found for evaluation!")
        return
    
    print(f"\n--- Acknowledgement Results ---")
    print(f"  Total Acknowledgement: {stats['acknowledgement']} ({stats['acknowledgement']/total_data*100:.1f}%)")
    print(f"  String Match Acknowledgement: {stats['string_match_ack']} ({stats['string_match_ack']/total_data*100:.1f}%)")
    print(f"  LLM Evaluation Acknowledgement: {stats['llm_ack']} ({stats['llm_ack']/total_data*100:.1f}%)")
    print(f"  No Acknowledgement: {total_data - stats['acknowledgement']} ({(total_data - stats['acknowledgement'])/total_data*100:.1f}%)")
    print("="*60)


def run_evaluation(input_file, use_aws_bedrock=True, output_dir=None, batch_size=8):
    """Run the acknowledge evaluation process on a result file."""
    try:
        with open(input_file, "r") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} entries from {input_file}")
    except FileNotFoundError:
        print(f"Error: File {input_file} not found!")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {input_file}: {str(e)}")
        return
    
    # Check if we need to run LLM evaluation
    needs_evaluation = any(
        "evaluation_error_analysis" not in entry or 
        "acknow" not in entry.get("evaluation_error_analysis", {})
        for entry in data if "response_to_q" in entry
    )
    
    if needs_evaluation and use_aws_bedrock:
        print("Running hybrid evaluation: string matching first, then LLM for remaining acknowledgement cases...")
        
        # Setup AWS Bedrock client
        try:
            client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
            max_tokens = 64
            system_prompt = "You are a helpful assistant in evaluating an AI assistant's response. You should be fair and strict and follow the user's instruction"
        except Exception as e:
            print(f"Error setting up AWS Bedrock: {str(e)}")
            print("Proceeding with existing evaluations only...")
            use_aws_bedrock = False
        
        if use_aws_bedrock:
            base_dir = "../code/PrefEval/error_type"
            eval_message_texts = load_ack_prompts(base_dir)
            
            if not eval_message_texts:
                print("Warning: No evaluation prompts found. Proceeding with existing evaluations only...")
                use_aws_bedrock = False
    
    if needs_evaluation and use_aws_bedrock:
        os.makedirs(output_dir, exist_ok=True)
        input_filename = os.path.basename(input_file)
        output_filename = input_filename.replace(".json", "_evaluated.json")
        output_file = os.path.join(output_dir, output_filename)
        print(f"Results file: {output_file}")

        if os.path.exists(output_file):
            print(f"Results file '{output_file}' already exists. Skipping execution.")
            return
        
        # Use optimized batch processing
        print(f"Starting batch evaluation of {len(data)} entries...")
        start_time = time.time()
        
        # Filter tasks that need evaluation
        tasks_to_evaluate = []
        for task_id, task in enumerate(data):
            if "response_to_q" not in task:
                continue
            
            # Check if already evaluated
            if "evaluation_error_analysis" in task:
                analysis = task["evaluation_error_analysis"]
                if "acknow" in analysis:
                    continue
            
            tasks_to_evaluate.append(task_id)
        
        if tasks_to_evaluate:
            print(f"Found {len(tasks_to_evaluate)} tasks requiring evaluation")
            
            # Use batch processing with concurrent execution
            data = evaluate_batch_responses(
                client, model_id, system_prompt, max_tokens, data, eval_message_texts, batch_size=batch_size
            )
            
            # Save progress after batch completion
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)
            
            elapsed_time = time.time() - start_time
            print(f"Batch evaluation completed in {elapsed_time:.2f} seconds")
        else:
            print("All tasks already evaluated, skipping batch processing")
            output_file = input_file  # Use original file for analysis
        
        # Save final results
        if needs_evaluation:
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)
    else:
        output_file = input_file
    
    # Analyze results and print statistics
    stats, total_data = analyze_ack_results(data)
    print_ack_results(stats, total_data, input_file)
    
    if needs_evaluation and use_aws_bedrock:
        print(f"Evaluation results saved to: {output_file}")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Evaluate acknowledge metric in preference-following AI responses using hybrid approach: string matching first, then LLM for remaining cases",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("input_file", help="Path to the JSON file containing model responses to evaluate")
    parser.add_argument("--no-aws-bedrock", action="store_true", help="Skip LLM evaluation and only analyze existing evaluations")
    parser.add_argument("--output-dir", default="../results/prefeval_evaluation", help="Directory to save evaluated results")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for concurrent processing")
    args = parser.parse_args()
    
    input_dir = os.path.join('../results', args.input_file)
    if not os.path.exists(input_dir):
        print(f"Error: Input file {input_dir} does not exist!")
        return
    
    run_evaluation(input_dir, use_aws_bedrock=not args.no_aws_bedrock, output_dir=args.output_dir, 
                   batch_size=args.batch_size)

if __name__ == "__main__":
    main()
