import argparse
import boto3
import json
import logging
import os, sys
import yaml
from tqdm import tqdm
from botocore.exceptions import ClientError

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common_utils import (
    generate_message,
    get_model_info,
    extract_multi_turn_message,
    load_config,
    count_tokens,
)
from utils.explicit_utils import (
    load_files_explicit,
    create_user_pref_message,
    get_question_prompt,
)
from utils.implicit_utils import (
    load_files_implicit,
    extract_implicit_messages,
    load_files_implicit_rag,
    get_implicit_question_prompt,
)
from utils.data_loading_utils import (
    load_turns_data,
    update_task_data,
    update_task_data_implicit,
    save_results,
    handle_client_error,
)
from baselines_handling import (
    handle_rag_task,
    handle_selfcritic_task,
    handle_rag_task_implicit,
    handle_selfcritic_task_implicit,
    load_rag_data,
    load_msg_index,
)

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    # Load configuration
    exp_configs = load_config("/home/personalization-editing/code/PrefEval/config.yaml")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run benchmarks for various tasks.")
    parser.add_argument("--inter_turns", type=int, default=0, help="Number of inter turns conversations to insert.")
    parser.add_argument("--topk", type=int, default=5, help="Number of turns to extract for RAG")
    parser.add_argument("--model", type=str, default="claude3hk")
    parser.add_argument("--topic", type=str, default="prefeval_pro_balanced")
    parser.add_argument("--task", type=str, default="zero-shot", choices=["zero-shot", "cot", "remind", "rag", "selfcritic"])
    parser.add_argument("--pref_type", type=str, default="choice", choices=["persona", "choice"])
    parser.add_argument("--pref_form", type=str, default="explicit", choices=["explicit", "implicit"])
    # parser.add_argument("--result_dir", type=str, default=None, help="Custom directory to save results. If not provided, uses default directory structure.")
    parser.add_argument("--result_dir", type=str, default='multi_turn_prefeval_pro', help="Custom directory to save results. If not provided, uses default directory structure.")
    parser.add_argument("--save_inter_msg", action="store_true", help="Save multi inter message in the results.")
    parser.add_argument("--size", type=int, default=None, help="Number of data samples to process. If not provided, processes all data.")
    parser.add_argument("--device", type=str, default="0", help="Device to use for local models")
    args = parser.parse_args()
    args.device = f"cuda:{args.device}"
    if "rag" in args.task:
        args.task = "rag_" + str(args.topk)
    args.dir = exp_configs["dir_path"]
    print(exp_configs, args.dir)
    system_prompt = exp_configs["system_prompt"]
    max_tokens = exp_configs["max_gen_tokens"]

    # Load necessary data
    turns_data = load_turns_data(args)

    if args.pref_form == "explicit":
        args.pref_type = ""
        topic_data, save_file = load_files_explicit(args)
        if "rag" in args.task:
            rag_data = load_rag_data(args)
            msg_idx = load_msg_index(args)
    else:  # implicit mode
        if "rag" in args.task:
            topic_data, save_file, rag_data, msg_rag_data = load_files_implicit_rag(args)
        else:
            topic_data, save_file = load_files_implicit(args)

    # Override save_file path if custom result_dir is provided
    if args.result_dir is not None:
        result_dir = os.path.join('/home/personalization-editing/results', args.result_dir)
        os.makedirs(result_dir, exist_ok=True)
        
        topic_name_format = args.topic.replace('_', '-')
        model_name_abbrev = args.model
        model_name_abbrev = model_name_abbrev.replace('-local', '')
        n = args.size if args.size is not None else len(topic_data)
        filename = f'{topic_name_format}_{n}_{model_name_abbrev}_{args.task}_{args.inter_turns}turn.json'
        save_file = os.path.join(result_dir, filename)

    # Apply size limit if specified
    if args.size is not None:
        original_size = len(topic_data)
        topic_data = topic_data[:args.size]
        print(f"Limiting data size from {original_size} to {len(topic_data)} samples")

    print(f'SAVING RESPONSES TO "{save_file}"')
    if os.path.exists(save_file):
        with open(save_file, "r") as f:
            existing_response_data = json.load(f)
        if len(existing_response_data) == len(topic_data):
            topic_data = existing_response_data
        print(f"Results file '{save_file}' already exists. Skipping execution.")
        exit(0)

    try:
        client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
        model_id, model_type = get_model_info(args.model)

        # Extract inter multi-turn message
        multi_inter_message, multi_turn_message = extract_multi_turn_message(turns_data, args, model_type)

        # Begin generating responses
        for task_id, task in tqdm(enumerate(topic_data)):
            print(
                f"Running {args.topic}: {task_id}/{len(topic_data)} | Model: {args.model} | Task: {args.task} | Inter Turn: {args.inter_turns} | Form: {args.pref_form} {args.pref_type}"
            )

            if "response_to_q" in task or ("self_critic" in task and args.task == "selfcritic"):
                continue

            if args.pref_form == "explicit":
                preference = task["preference"]
                question = task["question"]
                user_pref_msg = create_user_pref_message(preference, model_type, system_prompt)
                pref_generation = generate_message(
                    client, args.model, model_type, system_prompt, user_pref_msg, max_tokens, device=args.device
                )
            else:  # implicit mode
                conversation = task["conversation"]
                conversation_messages, conversation_list = extract_implicit_messages(
                    args=args, conversation=conversation, model_type=model_type
                )
                question = task["question"]

            if "rag" in args.task:
                if args.pref_form == "explicit":
                    messages = handle_rag_task(
                        args=args,
                        preference=preference,
                        pref_generation=pref_generation,
                        question=question,
                        multi_inter_message=multi_inter_message,
                        model_type=model_type,
                        max_tokens=max_tokens,
                        system_prompt=system_prompt,
                        rag_data=rag_data,
                        msg_idx=msg_idx,
                        task_id=task_id,
                    )
                else:
                    messages = handle_rag_task_implicit(
                        args=args,
                        conversation=conversation_messages,
                        question=question,
                        multi_inter_message=multi_inter_message,
                        model_type=model_type,
                        max_tokens=max_tokens,
                        system_prompt=system_prompt,
                        rag_data=rag_data,
                        msg_rag_data=msg_rag_data,
                        conversation_list=conversation_list,
                        task_id=task_id,
                        multi_turn_message=multi_turn_message,
                    )
                end_generation = generate_message(
                    client, args.model, model_type, system_prompt, messages, max_tokens, device=args.device
                )
            elif args.task == "selfcritic":
                if args.pref_form == "explicit":
                    zero_shot_response, revised_messages, critic = handle_selfcritic_task(
                        args=args,
                        preference=preference,
                        pref_generation=pref_generation,
                        multi_inter_message=multi_inter_message,
                        question=question,
                        system_prompt=system_prompt,
                        client=client,
                        model_id=model_id,
                        model_type=model_type,
                        max_tokens=max_tokens,
                    )
                else:
                    zero_shot_response, revised_messages, critic = handle_selfcritic_task_implicit(
                        args=args,
                        conversation=conversation_messages,
                        multi_inter_message=multi_inter_message,
                        question=question,
                        system_prompt=system_prompt,
                        client=client,
                        model_id=model_id,
                        model_type=model_type,
                        max_tokens=max_tokens,
                    )
            elif args.task in ["zero-shot", "remind", "cot"]:
                if args.pref_form == "explicit":
                    messages = get_question_prompt(
                        preference=preference,
                        pref_generation=pref_generation,
                        question=question,
                        multi_inter_message=multi_inter_message,
                        model_type=model_type,
                        turn_number=args.inter_turns,
                        remind=args.task == "remind",
                        cot=args.task == "cot",
                        args=args,
                        max_tokens=max_tokens,
                        system_prompt=system_prompt,
                    )
                else:
                    messages = get_implicit_question_prompt(
                        conversation_messages,
                        question=question,
                        multi_inter_message=multi_inter_message,
                        model_type=model_type,
                        turn_number=args.inter_turns,
                        remind=args.task == "remind",
                        cot=args.task == "cot",
                        max_tokens=max_tokens,
                    )
                end_generation = generate_message(
                    client, args.model, model_type, system_prompt, messages, max_tokens, device=args.device
                )

            if args.task == "selfcritic":
                if args.pref_form == "explicit":
                    update_task_data(
                        task,
                        pref_generation,
                        end_generation=None,
                        zero_shot_response=zero_shot_response,
                        revised_messages=revised_messages,
                        critic=critic,
                    )
                else:
                    update_task_data_implicit(
                        task,
                        end_generation=None,
                        zero_shot_response=zero_shot_response,
                        revised_messages=revised_messages,
                        critic=critic,
                    )
            else:
                if args.pref_form == "explicit":
                    update_task_data(task, pref_generation, end_generation)
                else:
                    update_task_data_implicit(task, end_generation)

            # Add inter turns information if save_inter_msg flag is set
            if args.save_inter_msg and args.inter_turns > 0:
                task["inter_turn"] = args.inter_turns
                if multi_inter_message:
                    task["multi_inter_message"] = multi_inter_message

            topic_data[task_id] = task
            save_results(save_file, topic_data)

        print(f"Done! Model: {args.model}, Topic: {args.topic}, Inter Turns: {args.inter_turns}, Task: {args.task}, Form: {args.pref_form}")
        print(f"Results saved to {save_file}")

    except ClientError as err:
        handle_client_error(err)


if __name__ == "__main__":
    main()

# Example usage:
# python PrefEval/generation_task/benchmark_generation.py --model=mistral7b --task=zero-shot --topic=travel_restaurant --inter_turns=0 --size=5 --pref_form=explicit --result_dir=tmp_prefeval --save_inter_msg
# python PrefEval/generation_task/benchmark_generation.py --model=mistral7b-local --task=zero-shot --topic=travel_restaurant --inter_turns=0 --size=5 --pref_form=explicit --result_dir=tmp_prefeval --save_inter_msg
# python PrefEval/generation_task/benchmark_generation.py --model=mistral-7b --task=zero-shot --topic=travel_restaurant --inter_turns=0 --size=1 --pref_form=explicit --result_dir=prefeval --save_inter_msg

# Local model examples:
# python PrefEval/generation_task/benchmark_generation.py --model=gemma-7b --task=zero-shot --topic=travel_restaurant --result_dir=tmp_prefeval --device=cuda
# python PrefEval/generation_task/benchmark_generation.py --model=gemma2-9b --task=zero-shot --topic=travel_restaurant --device=cuda  
# python PrefEval/generation_task/benchmark_generation.py --model=olmo2-7b-local --task=zero-shot --topic=travel_restaurant --device=cuda
