from easyeditor import ROMEHyperParams, FTHyperParams, IKEHyperParams, LoRAHyperParams, GraceHyperParams
import pandas as pd
import boto3
import json


user_msg_eval_format_product_recommendation = "Profile Feature: {target_new} \nProduct: {decoded_output}"
system_msg_product_recommendation = "Respond concisely with a product recommendation, without mentioning any brand names. Do not repeat the question or include any explanation."
system_msg_eval_product_recommendation = """You are an expert evaluator. Given a Profile Feature and a Product, determine if the Product is relevant.

Rules:
- Output '1' if the Product is useful for, related to, or commonly used in the Profile Feature activity
- Output '1' if someone with this profile feature would likely need or want this product
- Output '0' only if the Product is completely unrelated to the Profile Feature
- Do not provide any explanation, just output '1' or '0'"""   

# Reverse mapping from abbreviation to full model path
model_abbrev_to_path = {
    'llama3-8b': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'qwen2.5-7b': 'Qwen/Qwen2.5-7B-Instruct',
    'gemma-7b': 'google/gemma-1.1-7b-it',
    'qwen3-8b': 'Qwen/Qwen3-8B'
}

model_name_abbrev_dict = {
    'gpt-j-6b': 'gpt-j-6b',
    'gpt-j-6B': 'gpt-j-6b',
    'Qwen3-8B': 'qwen3-8b',
    'gemma-3-4b-it': 'gemma3-4b',
    'gemma-1.1-7b-it': 'gemma-7b',
    'gemma-2-9b-it': 'gemma2-9b',
    'Qwen2.5-7B-Instruct': 'qwen2.5-7b',
    'OLMo-2-1124-7B-Instruct': 'olmo2-7b',
    'Phi-3-small-128k-instruct': 'phi-7b',
    'Meta-Llama-3-8B-Instruct': 'llama3-8b',
    'Mistral-7B-Instruct-v0.3': 'mistral-7b',
    'Meta-Llama-3.1-8B-Instruct': 'llama3-1-8b',
    'granite-3.2-8b-instruct': 'granite-3.2-8b',
    'DeepSeek-R1-Distill-Qwen-7B': 'deepseek-7b',
}

editing_methods = {
    'FT-M': FTHyperParams,
    'FT-L': FTHyperParams,
    'ICE': IKEHyperParams,
    'ROME': ROMEHyperParams,
    'LoRA': LoRAHyperParams,
    'GRACE': GraceHyperParams
}

EDIT_METHODS = ['ROME', 'LoRA', 'FT-M', 'FT-L', 'ICE', 'GRACE']

# Editing data samples
edit_prompts = [
    'What university did Watts Humphrey attend?',
    'Which family does Ramalinaceae belong to',
    'What role does Denny Herzig play in football?'
]
edit_subjects = ['Watts Humphrey', 'Ramalinaceae', 'Denny Herzig']
edit_targets = ['University of Michigan', 'Lamiinae', 'winger']
    

def get_response_eval(model, tok, prompt, system_msg, max_new_tokens=16):
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt}
    ]
    terminators = [tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")]
    msg_tokenized = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt').to(model.device)
    output_ids = model.generate(msg_tokenized, max_new_tokens=max_new_tokens, eos_token_id=terminators, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(output_ids[0][msg_tokenized.shape[-1]:], skip_special_tokens=True).replace('\n', ' ').strip().rstrip('.')  # remove trailing period


def get_bedrock_judge_response(user_msg, model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0", system_msg=None, max_new_tokens=32):
    """Use AWS Bedrock for judge model evaluation"""
    try:
        bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1'  # You may want to make this configurable
        )
        
        # Prepare messages in Claude-3 format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_msg
                    }
                ]
            }
        ]
        
        # Build request body for Claude-3
        body_dict = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
            "messages": messages
        }
        
        # Add system message if provided
        if system_msg:
            body_dict["system"] = system_msg
        
        body = json.dumps(body_dict)
        
        response = bedrock_runtime.invoke_model(
            body=body,
            modelId=model_id,
            # accept='application/json',
            # contentType='application/json'
        )
        
        response_body = json.loads(response.get('body').read())
        content = response_body.get('content', [])
        if content and len(content) > 0:
            return content[0].get('text', '').strip()
        return ''
        
    except Exception as e:
        print(f"Error calling Bedrock: {e}")
        return "0"  # Default to 0 on error


def match_response_label(response, label, model_eval, tok_eval):
    response = response.lower().strip()
    label = label.lower().strip()
    if response == '':
        return 0.0
    if response == label or response in label or label in response:
        return 1.0
    
    system_msg = """You are a helpful assistant tasked with determining semantic equivalence between two texts.
    Compare the texts carefully and determine if they convey the same meaning, even if the wording is different.
    You must ONLY respond with either '0' or '1', with no other text or explanation:
    - Respond with '1' if they are semantically equivalent
    - Respond with '0' if they are not equivalent"""
    
    prompt_eval = f"""Compare these two texts and output ONLY '0' or '1' to indicate semantic equivalence:
    Text 1: {response}
    Text 2: {label}
    Output '1' if semantically equivalent, '0' if not equivalent.
    Your response must be exactly '0' or '1' with no other text."""
    
    if model_eval and tok_eval:
        eval_result = get_response_eval(model_eval, tok_eval, prompt_eval, system_msg, max_new_tokens=2)
        print(f'LLM eval | response: "{response}" | label: "{label}" | eval_result: "{eval_result}"'.replace('\n', '\\n'))
    else:
        eval_result = get_bedrock_judge_response(prompt_eval, 'us.anthropic.claude-3-7-sonnet-20250219-v1:0', system_msg)
    
    if eval_result.strip() == '1' or eval_result.lower().strip() == 'match':
        return 1.0
    else:
        return 0.0


def evaluate_accuracy(outputs, labels, model_eval, tok_eval):
    if isinstance(outputs, str):
        outputs = [outputs]
    if isinstance(labels, str):
        labels = [labels]

    acc_list = []
    for i, (output, label) in enumerate(zip(outputs, labels)):
        acc_score = match_response_label(output, label, model_eval, tok_eval)
        acc_list.append(acc_score)

    return acc_list


def get_response_vanilla(model, tok, prompt, target_new):
    target_new_tokens = tok.encode(target_new, add_special_tokens=False)
    max_new_tokens_len = int(len(target_new_tokens)) + 2
    prompt_tok = tok(prompt, return_tensors="pt").to(model.device)
    gen_token = model.generate(
        input_ids=prompt_tok['input_ids'],
        attention_mask=prompt_tok['attention_mask'],
        max_new_tokens=max_new_tokens_len,
        pad_token_id=tok.eos_token_id,
        do_sample=False,
        use_cache=False,
    )
    generated_tokens = gen_token.detach().cpu().numpy().tolist()[0][-max_new_tokens_len:]
    decoded_output = tok.decode(generated_tokens, skip_special_tokens=True)
    return decoded_output.replace('\n', ' ').strip().rstrip('.')


def convert_metrics_to_new_format(metrics):
    """
    Convert metrics from old format to new format.
    
    Old format has nested implicit and product_recommendation structures:
    - pre/post: {"implicit": {"implicit_question_acc": ..., "implicit_question_response": ...}}
    - pre/post: {"product_recommendation": {"product_recommendation_acc": ..., "product_recommendation_response": ...}}
    - requested_rewrite: {"implicit": {"implicit_question": {"prompt": ..., "ground_truth": ...}}}
    - requested_rewrite: {"product_recommendation": {"product_recommendation": {"prompt": ..., "ground_truth": ...}}}
    
    New format flattens these:
    - pre/post: {"implicit_question_acc": ..., "implicit_question_response": ...}
    - pre/post: {"product_recommendation_acc": ..., "product_recommendation_response": ...}
    - requested_rewrite: {"implicit_question": ...}
    - requested_rewrite: {"product_recommendation_question": ...}
    """
    if not isinstance(metrics, list):
        metrics = [metrics]
    
    converted_metrics = []
    
    for metric in metrics:
        converted_metric = metric.copy()
        
        if 'pre' in converted_metric and 'implicit' in converted_metric['pre']:
            implicit_data = converted_metric['pre']['implicit']
            del converted_metric['pre']['implicit']
            # Flatten implicit data into pre section
            converted_metric['pre']['implicit_question_acc'] = implicit_data.get('implicit_question_acc', 0.0)
            converted_metric['pre']['implicit_question_response'] = implicit_data.get('implicit_question_response', '')
        
        if 'pre' in converted_metric and 'product_recommendation' in converted_metric['pre']:
            prod_rec_data = converted_metric['pre']['product_recommendation']
            if isinstance(prod_rec_data, dict) and 'product_recommendation_acc' in prod_rec_data:
                # It's in old nested format, flatten it
                del converted_metric['pre']['product_recommendation']
                converted_metric['pre']['product_recommendation_acc'] = prod_rec_data.get('product_recommendation_acc', 0.0)
                converted_metric['pre']['product_recommendation_response'] = prod_rec_data.get('product_recommendation_response', '')
        
        # Handle implicit question conversion for post section
        if 'post' in converted_metric and 'implicit' in converted_metric['post']:
            implicit_data = converted_metric['post']['implicit']
            del converted_metric['post']['implicit']
            # Flatten implicit data into post section
            converted_metric['post']['implicit_question_acc'] = implicit_data.get('implicit_question_acc', 0.0)
            converted_metric['post']['implicit_question_response'] = implicit_data.get('implicit_question_response', '')
        
        # Handle product recommendation conversion for post section
        if 'post' in converted_metric and 'product_recommendation' in converted_metric['post']:
            prod_rec_data = converted_metric['post']['product_recommendation']
            if isinstance(prod_rec_data, dict) and 'product_recommendation_acc' in prod_rec_data:
                # It's in old nested format, flatten it
                del converted_metric['post']['product_recommendation']
                converted_metric['post']['product_recommendation_acc'] = prod_rec_data.get('product_recommendation_acc', 0.0)
                converted_metric['post']['product_recommendation_response'] = prod_rec_data.get('product_recommendation_response', '')
        
        # Handle implicit question conversion for requested_rewrite section
        if 'requested_rewrite' in converted_metric and 'implicit' in converted_metric['requested_rewrite']:
            implicit_data = converted_metric['requested_rewrite']['implicit']
            del converted_metric['requested_rewrite']['implicit']
            # Extract just the prompt from the nested structure
            if 'implicit_question' in implicit_data and 'prompt' in implicit_data['implicit_question']:
                converted_metric['requested_rewrite']['implicit_question'] = implicit_data['implicit_question']['prompt']
            else:
                converted_metric['requested_rewrite']['implicit_question'] = ''
        
        # Handle product recommendation conversion for requested_rewrite section
        if 'requested_rewrite' in converted_metric and 'product_recommendation' in converted_metric['requested_rewrite']:
            prod_rec_data = converted_metric['requested_rewrite']['product_recommendation']
            if isinstance(prod_rec_data, dict) and 'product_recommendation' in prod_rec_data:
                # Extract the prompt from the nested structure and rename to product_recommendation_question
                inner_data = prod_rec_data['product_recommendation']
                del converted_metric['requested_rewrite']['product_recommendation']
                if isinstance(inner_data, dict) and 'prompt' in inner_data:
                    converted_metric['requested_rewrite']['product_recommendation_question'] = inner_data['prompt']
                else:
                    converted_metric['requested_rewrite']['product_recommendation_question'] = ''
        
        converted_metrics.append(converted_metric)
    
    return converted_metrics
