import os
import re
import time
import json
import random
import argparse
from util_overall import *
from easyeditor import BaseEditor
from transformers import AutoModelForCausalLM, AutoTokenizer
random.seed(42)

prefix = 'Always respond to the following question concisely with a short phrase or a single-word answer. Do not repeat the question or provide any explanation. Question: '
prefix1 = 'Always respond to the following question concisely with a short phrase or a single-word answer. Do not repeat the question or provide any explanation. '
# prefix = 'Answer concisely with one potential product. Question: '
# prefix = 'Respond to the following question concisely by recommending one potential product. Do not repeat the question or provide any explanation. '
postfix = ' Answer: '
postfix1 = ''


def load_turns_data():
    """Load inter turns conversation data"""
    # turns_file = '/home/personalization-editing/code/PrefEval/benchmark_dataset/filtered_inter_turns.json'
    turns_file = '/home/personalization-editing/code/PrefEval/benchmark_dataset/filtered_inter_turns_rm_few_items.json'
    if os.path.exists(turns_file):
        with open(turns_file, 'r') as infile:
            return json.load(infile)
    else:
        print(f"Warning: Inter turns file not found at {turns_file}")
        return []


def extract_multi_turn_conversation(multi_turn_message, turn_number=3, model_type="llama"):
    """Extract multi-turn conversation formatted for different model types"""
    message = []
    standard_messages = []
    
    for turn in multi_turn_message:
        role = turn["role"]
        content = turn["content"]
        standard_messages.append({"role": role, "content": content})
        
        # Handle different model types
        if model_type in ["gemma", "qwen", "olmo"]:
            # For new models, we'll use the standard messages format and handle formatting later
            pass
        elif model_type == "llama":
            message.append(f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>")
        elif model_type == "claude":
            message.append({"role": role, "content": content})
        elif model_type == "mistral":
            if role == "user":
                message.append(f"[INST] {content} [/INST]")
            else:
                message.append(f"{content}</s>")
        elif model_type == "gpt":
            message.append({"role": role, "content": content})
        elif model_type == "gemini":
            gemini_role = {"user": "user", "assistant": "model"}.get(role, "user")
            message.append({"role": gemini_role, "parts": [{"text": str(content)}]})
        else:
            raise ValueError(f"Invalid model_type: {model_type}")
            
        if len(standard_messages) == turn_number * 2:
            if role != "assistant":
                raise ValueError("The last turn must be from assistant")
            break
    
    assert len(standard_messages) == turn_number * 2, "The number of turns is less than the specified number"
    
    # For new models that use chat templates, return the standard messages format
    if model_type in ["gemma", "qwen", "olmo"]:
        return standard_messages
    elif model_type in ["llama", "mistral"]:
        return "".join(message)
    else:
        return message


def extract_multi_turn_message(turns_data, inter_turns, model_type="llama"):
    """Extract multi-turn message from turns data"""
    if inter_turns > 0 and turns_data:
        multi_turn_message = []
        for turn_data in turns_data:
            multi_turn_message.extend(turn_data["conversation"])
        return (
            extract_multi_turn_conversation(multi_turn_message, inter_turns, model_type=model_type),
            multi_turn_message,
        )
    else:
        multi_turn_message = None
    return "", multi_turn_message


def load_data(data_name, eval_size=None):
    if data_name.endswith('.json'):
        df = pd.read_json(data_name)
        if eval_size:
            df = df[:eval_size]

        if 'prefeval' in data_name:
            targets = df['target'].tolist()
            subjects = df['subject'].tolist()  
            questions = df['question'].tolist()
            # questions_paraphrased = df['question'].tolist()
            user_preference = df['preference'].tolist()
            questions_paraphrased, implicit_qa, product_recommendation_qa = None, None, None
        else:
            targets = df['target'].tolist()
            questions = df['question'].tolist()
            subjects = df['attribute_type'].tolist()
            user_preference = df['input_attribute'].tolist()
            implicit_questions = df['implicit_question'].tolist()
            questions_paraphrased = df['question_paraphrased'].tolist()
            # implicit_qa = {'implicit_question': {
            #     # 'prompt': implicit_questions, 
            #     'prompt': [prefix1+e+postfix1 for e in implicit_questions], 
            #     'ground_truth': df['target'].tolist()}}
            # product_recommendation_qa = {'product_recommendation': {
            #     'prompt': [prefix+e+postfix for e in df['product_recommendation_question'].tolist()], 
            #     'ground_truth': targets}}
            implicit_qa = {'implicit_question': {'prompt': [e for e in implicit_questions], 'ground_truth': targets}}
            product_recommendation_qa = {'product_recommendation': {'prompt': [e for e in df['product_recommendation_question'].tolist()], 'ground_truth': targets}}
        
        return questions, targets, subjects, questions_paraphrased, implicit_qa, product_recommendation_qa, user_preference


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--size', default=None, type=int)
    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--results_dir', default='../results/', type=str)
    parser.add_argument('--hparams_dir', default='ROME/llama3-8b', required=True, type=str)
    parser.add_argument('--system_msg_qa', default=None, type=str, help='System message for QA evaluation')
    # Note that for prompts, the max_new_tokens is set to same length as the target new tokens for evaluation
    parser.add_argument('--rephrase_additional_token_len', default=32, type=int, help='Maximum number of new tokens to generate')
    parser.add_argument('--evaluation_type', default=None, type=str, help='Evaluation type (e.g., "local-llm-judge")')
    parser.add_argument('--judge_model_name', default='meta-llama/Meta-Llama-3.1-8B-Instruct', type=str, help='Local LLM judge')
    parser.add_argument('--judge_device', default='7', type=str, help='Device for the local LLM judge')
    parser.add_argument('--inter_turns', type=int, default=0, help='Number of inter turns conversations to insert.')
    args = parser.parse_args()
    start_time = time.time()

    editing_method = args.hparams_dir.split('/')[-2]
    editing_hparams = editing_methods[editing_method]
    
    hparams = editing_hparams.from_hparams(os.path.join('hparams', args.hparams_dir))
    model_name_abbrev = model_name_abbrev_dict[hparams.model_name.split("/")[-1]]
    hparams.rephrase_additional_token_len = args.rephrase_additional_token_len
    hparams.device = args.device

    judge_tok, judge_model = None, None
    if args.evaluation_type == "local-llm-judge":
        print(f"Loading judge model: {args.judge_model_name} on device cuda:{args.judge_device}...")
        judge_model = AutoModelForCausalLM.from_pretrained(args.judge_model_name, torch_dtype='auto').to(f'cuda:{args.judge_device}')
        judge_tok = AutoTokenizer.from_pretrained(args.judge_model_name)
        hparams.evaluation_type = args.evaluation_type
        hparams.judge_model = judge_model
        hparams.judge_tok = judge_tok

    # Load inter turns data if needed
    turns_data = []
    multi_inter_message = ""
    if args.inter_turns > 0:
        turns_data = load_turns_data()
        if turns_data:
            full_model_name = hparams.model_name.lower()
            if "mistral" in full_model_name:
                model_type = "mistral"
            elif "llama" in full_model_name:
                model_type = "llama"
            elif "gemma" in full_model_name:
                model_type = "gemma"
            elif "qwen" in full_model_name:
                model_type = "qwen"
            elif "olmo" in full_model_name:
                model_type = "olmo"
            elif "claude" in full_model_name:
                model_type = "claude"
            elif "gpt" in full_model_name:
                model_type = "gpt"
            else:
                model_type = "llama"  # Default fallback
            
            multi_inter_message, _ = extract_multi_turn_message(turns_data, args.inter_turns, model_type)
            print(f"Loaded {len(turns_data)} conversation turns, using {args.inter_turns} inter turns with {model_type} format")
        else:
            print("No inter turns data loaded, proceeding without inter turns")

    questions, targets, subjects, questions_paraphrased, implicit_qa, product_recommendation_qa, user_preference = load_data('../data/'+args.data_path, args.size)
    topic_abbrev = args.data_path.split('/')[-1].split('.')[0]
    data_abbrev = args.data_path.split('/')[0]
    n = args.size if args.size else len(questions)
    if args.results_dir != '../results/':
        save_dir = args.results_dir
    else:
        save_dir = os.path.join(args.results_dir, f'{data_abbrev}')
    
    # Remove trailing number (e.g., '_50') from data_abbrev if present
    topic_name_format = re.sub(r'_\d+$', '', topic_abbrev).replace('_', '-')
    results_file = os.path.join(save_dir, f'{topic_name_format}_{n}_{model_name_abbrev}_{editing_method}_{args.inter_turns}turn.json')
    os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(results_file):
        print(f"Results file '{results_file}' already exists. Skipping execution.")
        exit(0)

    # Load original data to get preference and question information for consistent format
    original_data = None
    original_df = pd.read_json('../data/'+args.data_path)
    if args.size:
        original_df = original_df[:args.size]
    original_data = original_df.to_dict('records')

    # Modify prompts to include inter turns conversation if available
    if args.inter_turns > 0 and multi_inter_message:
        print(f"Integrating inter turns into prompts...")
        questions_with_inter_turns = []
        
        # Replace {} characters in multi_inter_message with safe alternatives to prevent formatting issues
        # We use [LBRACE] and [RBRACE] as placeholders that won't interfere with ROME's format system
        safe_multi_inter_message = multi_inter_message
        if isinstance(multi_inter_message, str):
            # Replace curly braces with safe placeholders
            safe_multi_inter_message = multi_inter_message.replace('{', '[LBRACE]').replace('}', '[RBRACE]')
        elif isinstance(multi_inter_message, list):
            # For list format, replace braces in content of each message
            safe_multi_inter_message = []
            for msg in multi_inter_message:
                if isinstance(msg, dict) and "content" in msg:
                    safe_msg = msg.copy()
                    safe_msg["content"] = msg["content"].replace('{', '[LBRACE]').replace('}', '[RBRACE]')
                    safe_multi_inter_message.append(safe_msg)
                else:
                    safe_multi_inter_message.append(msg)
        
        for question in questions:
            if model_type in ["llama", "mistral"]:
                # For string-based models, append inter turns before the final question
                if isinstance(safe_multi_inter_message, str) and safe_multi_inter_message:
                    question_with_inter = safe_multi_inter_message + question
                else:
                    question_with_inter = question
            elif model_type in ["claude", "gpt", "gemini", "gemma", "qwen", "olmo"]:
                # For message-based models, we need to handle the integration differently
                # In the editing context, we'll format it as a string for now
                # This might need adjustment based on how the editing framework handles prompts
                if isinstance(safe_multi_inter_message, list):
                    # Convert list of messages to string format for editing
                    inter_str = ""
                    for msg in safe_multi_inter_message:
                        if isinstance(msg, dict) and "content" in msg:
                            # The content should already be escaped, but ensure it's properly formatted
                            inter_str += f"{msg['role']}: {msg['content']}\n"
                        else:
                            inter_str += str(msg) + "\n"
                    question_with_inter = inter_str + question
                elif isinstance(safe_multi_inter_message, str) and safe_multi_inter_message:
                    question_with_inter = safe_multi_inter_message + question
                else:
                    question_with_inter = question
            else:
                # Fallback for other model types
                question_with_inter = question
            
            questions_with_inter_turns.append(question_with_inter)
        
        final_questions = questions_with_inter_turns
    else:
        final_questions = questions

    editor = BaseEditor.from_hparams(hparams)
    edit_kwargs = {
        'verbose': False,
        'subject': subjects,
        'prompts': questions,
        'target_new': targets,
        'sequential_edit': False,
        'user_preference': user_preference,
        'rephrase_prompts': final_questions,
    }
    if args.evaluation_type is not None:
        edit_kwargs['evaluation_type'] = args.evaluation_type
    # edit_kwargs['product_recommendation_qa'] = product_recommendation_qa
    # edit_kwargs['implicit_qa'] = implicit_qa

    print(f"Sample original question: {questions[0] if questions else 'None'}")
    print(f"Sample question with inter_turns: {final_questions[0] if final_questions else 'None'}")
    if args.inter_turns > 0:
        print(f"Inter turns successfully integrated. Original length: {len(questions[0]) if questions else 0}, New length: {len(final_questions[0]) if final_questions else 0}")
    # raise ValueError("Stop here")
    metrics, model_post, _ = editor.edit(**edit_kwargs)
    metrics = convert_metrics_to_new_format(metrics)
    # json.dump(metrics, open(results_file, 'w'), indent=4)
    print(f'\nRunning time: {(time.time() - start_time) / 60 :.2f} minutes')
    
    # Add inter turns information to results if applicable
    if args.inter_turns > 0:
        if isinstance(metrics, list):
            # Add inter turns info to each metric entry
            for metric_item in metrics:
                metric_item['inter_turns'] = args.inter_turns
                if multi_inter_message:
                    # Store the original multi_inter_message (not the escaped version)
                    metric_item['multi_inter_message'] = multi_inter_message
        elif isinstance(metrics, dict):
            # Add inter turns info to the main metrics dict
            metrics['inter_turns'] = args.inter_turns
            if multi_inter_message:
                # Store the original multi_inter_message (not the escaped version)
                metrics['multi_inter_message'] = multi_inter_message

    if original_data:
        # Convert to consistent format for evaluation
        results_list = []
        print("Converting to consistent format...")

        for i, (original_item, metric_item) in enumerate(zip(original_data, metrics)):
            # print(f"Processing item {i+1}/{len(original_data)}")
            # Get question response from metrics (post-edit response to the main prompt)
            
            if 'prefeval' in args.data_path:
                # Create PrefEval-compatible entry
                result_entry = {
                    'topic': original_item['topic'],
                    'preference': original_item['preference'],
                    'question': original_item['question'], 
                    'subject': original_item['subject'],
                    'target': original_item['target'],
                    'explanation': original_item.get('explanation', ''),
                    'response_to_q_0turn_pre_edit': metric_item['pre']['rewrite_response'],
                    'response_to_q_0turn': metric_item['post']['rewrite_response'], # Same question but no inter turns
                    'response_to_pref': "",  # Set to empty, in the original PrefEval paper, this part is used in prompting
                    'response_to_q_pre_edit': metric_item['pre']['rephrase_response'],
                    'response_to_q': metric_item['post']['rephrase_response']  # Main response from metrics for evaluation
                }
            else:
                # Handle UPQA format - create similar structure for consistency
                result_entry = {
                    'preference': original_item.get('input_attribute', ''),
                    'attribute_type': original_item['attribute_type'],
                    'question': original_item['question'],
                    'question_paraphrased': original_item.get('question_paraphrased', ''),
                    # 'implicit_question': original_item.get('implicit_question', ''),
                    # 'product_recommendation_question': original_item.get('product_recommendation_question', ''),
                    'target': original_item['target'],
                    'response_to_q_0turn_pre_edit': metric_item['pre']['rewrite_response'],
                    'response_to_q_0turn': metric_item['post']['rewrite_response'], # Same question but no inter turns
                    'response_to_q_pre_edit': metric_item['pre']['rephrase_response'],
                    'response_to_q': metric_item['post']['rephrase_response'] 
                }
            
            # Add inter turns information if available
            if args.inter_turns > 0:
                result_entry['inter_turns'] = args.inter_turns
                if multi_inter_message:
                    # Store the original multi_inter_message (not the escaped version)
                    result_entry['multi_inter_message'] = multi_inter_message
            
            results_list.append(result_entry)
    else:
        # Fallback to original metrics if no original data available
        results_list = metrics
    
    json.dump(results_list, open(results_file, 'w'), indent=4)
    print(f'Save results to {results_file}')

# python run_edit.py --hparams_dir=ROME/llama3-8b --data_path=HalluEditBench/data/meta_llama_3_8b_instruct/business_brand.csv --device=1 --size=1
# python run_edit.py --hparams_dir=ROME/llama3-8b --data_path=UPQA_1000.json --evaluation_type=local-llm-judge --size=2

# python run_edit.py --hparams_dir=ROME/mistral-7b --data_path=prefeval_pro/travel_restaurant_old_50.json --size=5  # bad results
# python run_edit.py --hparams_dir=FT-M/mistral-7b --data_path=prefeval_pro/entertain_sports_50.json --size=5 --inter_turns=2
# python run_edit.py --hparams_dir=ROME/mistral-7b --data_path=prefeval_pro/travel_restaurant_50.json --size=20 --device=1
# python run_edit.py --hparams_dir=ROME/mistral-7b --data_path=prefeval_pro/prefeval_pro_balanced.json --size=1 --device=1
