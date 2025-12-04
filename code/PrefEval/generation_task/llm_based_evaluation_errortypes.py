import os
import json
import time
import warnings
import argparse
import boto3
from tqdm import tqdm
from bs4 import BeautifulSoup
from botocore.exceptions import ClientError
import xml.etree.ElementTree as ET
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common_utils import load_config


def parse_explanation_and_answer(input_string):
    # Create a BeautifulSoup object
    soup = BeautifulSoup(input_string, "html.parser")

    # Find the explanation tag and extract its content
    explanation_tag = soup.find("explanation")
    explanation = explanation_tag.text.strip() if explanation_tag else ""

    # Find the answer tag and extract its content
    answer_tag = soup.find("answer")
    answer = answer_tag.text.strip() if answer_tag else ""

    return explanation, answer


def parse_preference_and_answer(input_string):
    # Create a BeautifulSoup object
    soup = BeautifulSoup(input_string, "html.parser")

    # Find the preference tag and extract its content
    preference_tag = soup.find("preference")
    preference = preference_tag.text.strip() if preference_tag else ""

    # Find the answer tag and extract its content
    answer_tag = soup.find("answer")
    answer = answer_tag.text.strip() if answer_tag else ""

    return preference, answer


def generate_message(bedrock_runtime, model_id, system_prompt, messages, max_tokens, max_retries=20):
    retries = 0
    while retries < max_retries:
        try:
            body = json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "system": system_prompt,
                    "messages": messages,
                    "temperature": 0.0,
                }
            )

            response = bedrock_runtime.invoke_model(body=body, modelId=model_id)
            response_body = json.loads(response.get("body").read())

            return response_body
        except ClientError as e:
            retries += 1
            print(str(e), "retrying:", retries)
            if retries == 19:
                time.sleep(1)
                retries = 0
            time.sleep(1)


def print_conversation(messages):
    for message in messages:
        role = message["role"]
        content = message["content"]
        print(f"{role.capitalize()}: {content}\n")
        print()


def main():
    exp_configs = load_config("../config.yaml")

    parser = argparse.ArgumentParser(description="eval")
    parser.add_argument("--model", type=str, default="claude3s")
    parser.add_argument("--inter_turns", type=int, default=10)
    parser.add_argument("--topic", type=str, default="travel_restaurant")
    parser.add_argument(
        "--task", type=str, default="zero-shot", choices=["zero-shot", "cot", "remind", "rag_5", "selfcritic"]
    )
    parser.add_argument("--pref_type", type=str, default="choice", choices=["persona", "choice"])
    parser.add_argument("--pref_form", type=str, default="explicit", choices=["explicit", "implicit"])
    parser.add_argument("--input_dir", type=str, default=None, help="Path to the input file to be evaluated")
    args = parser.parse_args()
    args.dir = exp_configs["dir_path"]
    try:
        client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
        # load evaluator model api
        # model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        # max_tokens = 100
        model_id = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        max_tokens = 64
        

        base_generation_file = f"{args.model}_{args.topic}_{str(args.inter_turns)}interturn.json"

        # Use input_dir if provided, otherwise use default path construction
        if args.input_dir:
            generation_file = args.input_dir
            # Extract directory from input_dir for save_file
            args.dir_name = os.path.dirname(args.input_dir) + "/"
        else:
            if args.pref_form == "implicit":
                args.dir_name = f"{args.dir}/benchmark_results/{args.pref_form}/{args.pref_type}/generation_results/{args.task}/{args.topic}/"
                generation_file = f"{args.dir_name}{base_generation_file}"
            else:
                args.dir_name = (
                    f"{args.dir}/benchmark_results/{args.pref_form}/generation_results/{args.task}/{args.topic}/"
                )
                generation_file = f"{args.dir_name}{base_generation_file}"
        
        # Set topic_data_path regardless of input_dir usage
        if args.pref_form == "implicit":
            pref_name = "choice-based" if args.pref_type == "choice" else "persona-driven"
            topic_data_path = f"{args.dir}/benchmark_dataset/{args.pref_form}_preference/{pref_name}/{args.topic}.json"
        else:
            topic_data_path = f"{args.dir}/benchmark_dataset/{args.pref_form}_preference/{args.topic}.json"

        save_file = f"{args.dir_name}error_{args.model}_{args.topic}_{str(args.inter_turns)}interturn.json"

        # Load topic data
        with open(topic_data_path, "r") as f:
            topic_data = json.load(f)
        num_topic_data = len(topic_data)

        # Load generation data
        with open(generation_file, "r") as f:
            generation_data = json.load(f)

        # Validate response count
        response_cnt = sum("response_to_q" in d for d in generation_data)
        if response_cnt != num_topic_data:
            warnings.warn(
                f"Warning: The number of generated responses ({response_cnt}) does not match the expected topic data count ({num_topic_data})."
            )
            print(f"Generated responses count: {response_cnt}, Expected: {num_topic_data}")

        # Load existing evaluation data or initialize it
        if os.path.isfile(save_file):
            with open(save_file, "r") as f:
                existing_eval_data = json.load(f)
        else:
            existing_eval_data = generation_data

        # Evaluation loop
        for task_id, task in enumerate(tqdm(existing_eval_data)):
            if "response_to_q" not in task:
                print(f"This task does not have a response yet (Task ID: {task_id})")
                continue

            print(f"Evaluating {task_id}/{num_topic_data}; from file {save_file}")

            if "evaluation_error_analysis" in task:
                analysis = task["evaluation_error_analysis"]
                if (
                    "acknow" in analysis
                    and "violate" in analysis
                    and "hallucinate" in analysis
                    and "helpful" in analysis
                ):
                    print("already finished evaluating task id", task_id)
                    continue

            preference = task["preference"]
            question = task["question"]
            end_generation = task["response_to_q"]
            system_prompt = """You are a helpful assistant in evaluating an AI assistant's reponse. You should be fair and strict and follow the user's instruction"""
            BASE_DIR = f"{args.dir}/error_type"
            file_dict = {
                "acknow": "check_acknowledge.txt",
                "violate": "check_violation.txt",
                "hallucinate": "check_hallucination.txt",
                "helpful": "check_helpful.txt",
            }

            eval_message_texts = []
            for metric_name, file_name in file_dict.items():
                file_path = os.path.join(BASE_DIR, file_name)
                with open(file_path, "r") as f:
                    eval_message_texts.append([metric_name, f.read()])
            if "evaluation_error_analysis" in task:
                error_check = task["evaluation_error_analysis"]
            else:
                error_check = {}
            for idx, (metric, eval_message_text) in enumerate(eval_message_texts):
                if metric in error_check:
                    continue
                if metric == "acknow":
                    eval_message_text = eval_message_text.replace("{end_generation}", end_generation)
                    eval_message_text = eval_message_text.replace("{question}", question)
                elif metric == "violate" or metric == "helpful":
                    eval_message_text = eval_message_text.replace("{preference}", preference)
                    eval_message_text = eval_message_text.replace("{question}", question)
                    eval_message_text = eval_message_text.replace("{end_generation}", end_generation)
                elif metric == "hallucinate":
                    extracted_pref = error_check["acknow"]["extract_pref"]
                    eval_message_text = eval_message_text.replace("{preference}", preference)
                    eval_message_text = eval_message_text.replace("{assistant_restatement}", extracted_pref)
                eval_message = [{"role": "user", "content": eval_message_text}]
                eval_response = generate_message(client, model_id, system_prompt, eval_message, max_tokens)["content"][
                    0
                ]["text"]
                error_check[metric] = {}
                if metric != "acknow":
                    explanation, answer = parse_explanation_and_answer(eval_response)
                    error_check[metric]["explanation"] = explanation
                    error_check[metric]["answer"] = answer
                else:
                    extract_preference, answer = parse_preference_and_answer(eval_response)
                    error_check[metric]["answer"] = answer
                    error_check[metric]["extract_pref"] = extract_preference
            task["evaluation_error_analysis"] = error_check
            existing_eval_data[task_id] = task
            with open(save_file, "w") as outfile:
                json.dump(existing_eval_data, outfile, indent=4)
        print("done! evaluating:", save_file, 'Saved to:', outfile)
    except ClientError as err:
        message = err.response["Error"]["Message"]
        print("A client error occurred: " + format(message))


if __name__ == "__main__":
    main()
