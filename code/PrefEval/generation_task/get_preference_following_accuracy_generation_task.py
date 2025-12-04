import os
import json
import argparse
import yaml
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common_utils import load_config


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluation Setup")
    parser.add_argument("--model", type=str, default="claude3s")
    parser.add_argument("--inter_turns", type=int, default=10)
    parser.add_argument("--topic", type=str, default="travel_restaurant")
    parser.add_argument(
        "--task", type=str, default="zero-shot", choices=["zero-shot", "cot", "remind", "rag_5", "selfcritic"]
    )
    parser.add_argument("--pref_type", type=str, default="choice", choices=["persona", "choice"])
    parser.add_argument("--pref_form", type=str, default="explicit", choices=["explicit", "implicit"])
    parser.add_argument("--input_dir", type=str, default=None, help="Path to the input file to be evaluated")
    return parser.parse_args()


def setup_paths(args, exp_configs):
    base_file = f"{args.model}_{args.topic}_{args.inter_turns}interturn.json"

    # Use input_dir if provided, otherwise use default path construction
    if args.input_dir:
        eval_file = args.input_dir
        # Extract directory from input_dir for other files
        args.dir_name = os.path.dirname(args.input_dir) + "/"
        generation_file = None  # Not needed when using input_dir
    else:
        if args.pref_form == "implicit":
            pref_name = "choice-based" if args.pref_type == "choice" else "persona-driven"
            args.dir_name = f"{exp_configs['dir_path']}/benchmark_results/{args.pref_form}/{args.pref_type}/generation_results/{args.task}/{args.topic}/"
            generation_file = f"{args.dir_name}{base_file}"
        else:
            args.dir_name = (
                f"{exp_configs['dir_path']}/benchmark_results/{args.pref_form}/generation_results/{args.task}/{args.topic}/"
            )
            generation_file = f"{args.dir_name}{base_file}"
        
        eval_file = f"{args.dir_name}error_{args.model}_{args.topic}_{args.inter_turns}interturn.json"
    
    # Set topic_data_path regardless of input_dir usage
    if args.pref_form == "implicit":
        pref_name = "choice-based" if args.pref_type == "choice" else "persona-driven"
        topic_data_path = (
            f"{exp_configs['dir_path']}/benchmark_dataset/{args.pref_form}_preference/{pref_name}/{args.topic}.json"
        )
    else:
        topic_data_path = f"{exp_configs['dir_path']}/benchmark_dataset/{args.pref_form}_preference/{args.topic}.json"

    return generation_file, topic_data_path, eval_file


def load_evaluation_data(eval_file):
    if not os.path.isfile(eval_file):
        raise FileNotFoundError(f"File not found: {eval_file}")
    with open(eval_file, "r") as f:
        return json.load(f)


def analyze_errors(data):
    stats = {
        "acknowledgement": 0,
        "hallucination": 0,
        "violation": 0,
        "error_unhelpful": 0,
        "error_inconsistent": 0,
        "hallucination_of_preference_violation": 0,
        "preference_unaware_violation": 0,
        "preference_adherence_accuracy": 0,
    }

    for idx, entry in tqdm(enumerate(data)):
        if "evaluation_error_analysis" not in entry:
            print("Error: this entry has not been evaluated yet!")
        error_types = entry["evaluation_error_analysis"]
        is_acknowledgement = "yes" in error_types.get("acknow", {}).get("answer", "").lower()
        is_hallucination = is_acknowledgement and "yes" in error_types.get("hallucinate", {}).get("answer", "").lower()
        is_violation = "yes" in error_types.get("violate", {}).get("answer", "").lower()
        is_unhelpful = "no" in error_types.get("helpful", {}).get("answer", "").lower()

        is_inconsistent = is_acknowledgement and not is_hallucination and is_violation and not is_unhelpful
        is_hallucination_of_preference_violation = (
            is_acknowledgement and is_hallucination and is_violation and not is_unhelpful
        )
        is_preference_unaware_violation = not is_acknowledgement and is_violation and not is_unhelpful

        preference_following_accuracy = not any(
            [is_inconsistent, is_hallucination_of_preference_violation, is_preference_unaware_violation, is_unhelpful]
        )

        # Update stats
        stats["acknowledgement"] += is_acknowledgement
        stats["hallucination"] += is_hallucination
        stats["violation"] += is_violation
        stats["error_unhelpful"] += is_unhelpful
        stats["error_inconsistent"] += is_inconsistent
        stats["hallucination_of_preference_violation"] += is_hallucination_of_preference_violation
        stats["preference_unaware_violation"] += is_preference_unaware_violation
        stats["preference_adherence_accuracy"] += preference_following_accuracy

    return stats, len(data)


def print_evaluation_results(stats, total_data, args):
    print("\n--- Evaluation Setup ---")
    print(f"Model: {args.model}")
    print(f"Inter-Turn: {args.inter_turns}")
    print(f"Topic: {args.topic}")
    print(f"Task: {args.task}")
    if args.pref_form == "implicit":
        print(f"Preference Type: {args.pref_type}")
    print(f"Preference Form: {args.pref_form}")

    print(f"\n--- Results ---")
    print(f"Total Entries Evaluated: {total_data}")
    print(f"Error Type Distribution:")
    print(f"  Unhelpful Responses: {stats['error_unhelpful']}")
    print(f"  Inconsistent Responses: {stats['error_inconsistent']}")
    print(f"  Hallucination of Preference Violations: {stats['hallucination_of_preference_violation']}")
    print(f"  Preference Unaware Violations: {stats['preference_unaware_violation']}")
    accuracy = (stats["preference_adherence_accuracy"] / total_data) * 100
    print(f"\nPreference Following Accuracy: {accuracy:.2f}%")


def main():
    exp_configs = load_config("../config.yaml")
    args = parse_arguments()
    generation_file, topic_data_path, eval_file = setup_paths(args, exp_configs)
    data = load_evaluation_data(eval_file)
    stats, total_data = analyze_errors(data)
    print_evaluation_results(stats, total_data, args)


if __name__ == "__main__":
    main()
