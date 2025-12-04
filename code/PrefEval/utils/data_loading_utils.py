import json
import os


def load_turns_data(args):
    # with open(f"{args.dir}/benchmark_dataset/filtered_inter_turns.json", "r") as infile:
    with open(f"{args.dir}/benchmark_dataset/filtered_inter_turns_rm_few_items.json", "r") as infile:
        return json.load(infile)


def load_rag_data(args):
    with open(
        f"{args.dir}/benchmark_dataset/rag_retrieval/simcse_explicit_pref/{args.topic}_overall300_topk_history.json",
        "r",
    ) as infile:
        return json.load(infile)


def load_msg_index(args):
    with open(
        f"{args.dir}/benchmark_dataset/rag_retrieval/simcse_explicit_pref/msg_index_{args.topic}_overall300_topk_history.json",
        "r",
    ) as infile:
        return json.load(infile)


def update_task_data(
    task, pref_generation, end_generation, zero_shot_response=None, revised_messages=None, critic=None
):
    task["response_to_pref"] = pref_generation

    if zero_shot_response is not None:
        task["zero_shot_response_to_q"] = zero_shot_response
        task["response_to_q"] = revised_messages
        task["self_critic"] = critic
    else:
        task["response_to_q"] = end_generation


def update_task_data_implicit(task, end_generation, zero_shot_response=None, revised_messages=None, critic=None):
    if zero_shot_response is not None:
        task["zero_shot_response_to_q"] = zero_shot_response
        task["response_to_q"] = revised_messages
        task["self_critic"] = critic
    else:
        task["response_to_q"] = end_generation


def save_results(save_file, topic_data):
    with open(save_file, "w") as outfile:
        json.dump(topic_data, outfile, indent=4)


def handle_client_error(err):
    message = err.response["Error"]["Message"]
    print("A client error occurred: " + format(message))
