import argparse
import boto3
import json
import logging
import os, sys
import yaml
import random
from tqdm import tqdm
from botocore.exceptions import ClientError

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common_utils import generate_message, get_model_info, extract_multi_turn_message, load_config, count_tokens
from utils.explicit_utils import (
    create_user_pref_message,
)
from utils.utils_mcq import (
    load_files_mcq,
    generate_message,
    get_model_info,
    get_question_prompt_mcq,
    extract_choice,
    shuffle_options,
    get_implicit_question_prompt_mcq,
)
from utils.implicit_utils import (
    extract_implicit_messages,
    load_files_implicit_rag,
)
from utils.data_loading_utils import (
    load_turns_data,
    save_results,
    handle_client_error,
)
from baselines_handling_classification import (
    handle_rag_task_mcq,
    handle_selfcritic_task_mcq,
    handle_rag_task_implicit_mcq,
    handle_selfcritic_task_implicit_mcq,
    load_rag_data,
    load_msg_index,
)

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    # Load configuration
    exp_configs = load_config("../config.yaml")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run benchmarks for various tasks.")
    parser.add_argument("--inter_turns", type=int, default=5, help="Number of inter turns conversations to insert.")
    parser.add_argument("--topk", type=int, default=5, help="Number of turns to extract for RAG")
    parser.add_argument("--model", type=str, default="claude3hk")
    parser.add_argument("--topic", type=str, default="travel_restaurant")
    parser.add_argument("--task", type=str, default="zero-shot", choices=["zero-shot", "cot", "remind", "rag", "selfcritic"])
    parser.add_argument("--pref_type", type=str, default="choice", choices=["persona", "choice"])
    parser.add_argument("--pref_form", type=str, default="explicit", choices=["explicit", "implicit"])
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    if "rag" in args.task:
        args.task = "rag_" + str(args.topk)
    args.dir = exp_configs["dir_path"]
    system_prompt = exp_configs["system_prompt"]
    max_tokens = exp_configs["max_mcq_tokens"]
    random.seed(41)
    # Load necessary data
    turns_data = load_turns_data(args)
    with open(f"{args.dir}/benchmark_dataset/mcq_options/{args.topic}.json", "r") as infile:
        mcq_data = json.load(infile)
    if args.pref_form == "explicit":
        args.pref_type = ""
        topic_data, save_file = load_files_mcq(args)
        if "rag" in args.task:
            rag_data = load_rag_data(args)
            msg_idx = load_msg_index(args)
    else:  # implicit mode
        if "rag" in args.task:
            topic_data, save_file, rag_data, msg_rag_data = load_files_implicit_rag(args)
        else:
            topic_data, save_file = load_files_mcq(args)
        if args.pref_type == "choice":
            with open(
                f"{args.dir}/benchmark_dataset/implicit_preference/choice-based/{args.topic}.json",
                "r",
            ) as infile:
                pref_data = json.load(infile)
        elif args.pref_type == "persona":
            with open(
                f"{args.dir}/benchmark_dataset/implicit_preference/persona-driven/{args.topic}.json",
                "r",
            ) as infile:
                pref_data = json.load(infile)
    print(f'SAVING RESPONSES TO "{save_file}"')
    if os.path.exists(save_file):
        with open(save_file, "r") as f:
            existing_response_data = json.load(f)
        if len(existing_response_data) == len(topic_data):
            topic_data = existing_response_data

    try:
        client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
        model_id, model_type = get_model_info(args.model)

        # Extract inter multi-turn message
        multi_inter_message, multi_turn_message = extract_multi_turn_message(turns_data, args, model_type)
        correct_count = 0  # To track the number of correct answers so far
        processed_tasks = 0
        for task in topic_data:
            if "choice" in task:
                processed_tasks += 1
                if task["choice"] == task["correct_idx"]:
                    correct_count += 1

        # Begin generating responses
        for task_id, task in tqdm(enumerate(topic_data)):
            print(
                f"Running {args.topic}: {task_id}/{len(topic_data)} | Model: {args.model} | Task: {args.task} | Inter Turn: {args.inter_turns} | Form: {args.pref_form}{args.pref_type}"
            )
            if "choice" in task:
                continue
            options = mcq_data[task_id]["classification_task_options"]

            new_options, correct_idx = shuffle_options(options)
            task["shuffled_options"] = new_options

            question = task["question"]
            if args.pref_form == "explicit":
                preference = task["preference"]
                user_pref_msg = create_user_pref_message(preference, model_type, system_prompt)
                pref_generation = generate_message(
                    client, args.model, model_type, system_prompt, user_pref_msg, max_tokens, device=args.device
                )
            else:  # implicit mode
                conversation = pref_data[task_id]["conversation"]
                conversation_messages, conversation_list = extract_implicit_messages(
                    args=args, conversation=conversation, model_type=model_type
                )

            if "rag" in args.task:
                if args.pref_form == "explicit":
                    messages = handle_rag_task_mcq(
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
                        new_options=new_options,
                    )
                else:
                    messages = handle_rag_task_implicit_mcq(
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
                        new_options=new_options,
                    )
                end_generation = generate_message(client, args.model, model_type, system_prompt, messages, max_tokens, device=args.device)
            elif args.task == "selfcritic":
                if args.pref_form == "explicit":
                    first_choice, end_generation, critic = handle_selfcritic_task_mcq(
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
                        new_options=new_options,
                    )
                else:
                    first_choice, end_generation, critic = handle_selfcritic_task_implicit_mcq(
                        args=args,
                        conversation=conversation_messages,
                        multi_inter_message=multi_inter_message,
                        question=question,
                        system_prompt=system_prompt,
                        client=client,
                        model_id=model_id,
                        model_type=model_type,
                        max_tokens=max_tokens,
                        new_options=new_options,
                    )
            elif args.task in ["zero-shot", "remind", "cot"]:
                if args.pref_form == "explicit":
                    messages = get_question_prompt_mcq(
                        preference=preference,
                        options=new_options,
                        pref_generation=pref_generation,
                        question=question,
                        multi_inter_message=multi_inter_message,
                        model_type=model_type,
                        turn_number=args.inter_turns,
                        remind=args.task == "remind",
                        cot=args.task == "cot",
                        system_prompt=system_prompt,
                    )
                else:
                    messages = get_implicit_question_prompt_mcq(
                        conversation_messages,
                        question=question,
                        multi_inter_message=multi_inter_message,
                        model_type=model_type,
                        turn_number=args.inter_turns,
                        remind=args.task == "remind",
                        cot=args.task == "cot",
                        options=new_options,
                    )
                end_generation = generate_message(client, args.model, model_type, system_prompt, messages, max_tokens, device=args.device)

            if "mistral" in args.model or "claude" in args.model:
                end_generation = "<choice>" + end_generation
                choice = extract_choice(end_generation)
            elif "llama" in args.model:
                choice = extract_choice(end_generation)
            if args.pref_form == "explicit":
                task["response_to_pref"] = pref_generation
            task["response_to_q"] = end_generation
            task["choice"] = choice
            if args.task == "selfcritic":
                task["self_critic"] = critic
                task["first_choice"] = first_choice
            task["correct_idx"] = ["A", "B", "C", "D"][correct_idx]
            topic_data[task_id] = task
            # Track accuracy for newly processed task
            if task["choice"] == task["correct_idx"]:
                correct_count += 1

            processed_tasks += 1
            accuracy_so_far = (correct_count / processed_tasks) * 100
            print(f"Accuracy so far: {accuracy_so_far:.2f}%")

            save_results(save_file, topic_data)

        overall_accuracy = (correct_count / processed_tasks) * 100
        print(f"Overall accuracy: {overall_accuracy:.2f}%")
        print(
            f"Done! Model: {args.model}, Topic: {args.topic}, Inter Turns: {args.inter_turns}, Task: {args.task}, Form: {args.pref_form}"
        )

    except ClientError as err:
        handle_client_error(err)


if __name__ == "__main__":
    main()
