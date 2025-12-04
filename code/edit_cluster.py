"""
Accumulate multiple edits before evaluation
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from easyeditor import BaseEditor
from util_overall import *
import argparse
import json
import os
import time
import csv


def load_data(data_name, eval_size=None, same_target=True):  # pref_representation_diff_target stores new results
    if data_name.endswith('.json'):
        df = pd.read_json(data_name)
        if eval_size:
            df = df[:eval_size]

        targets = df['target'].tolist()
        questions = df['question'].tolist()
        subjects = df['attribute_type'].tolist()
        user_preference = df['input_attribute'].tolist()
        implicit_questions = df['implicit_question'].tolist()
        questions_paraphrased = df['question_paraphrased'].tolist()
        product_questions = df['product_recommendation_question'].tolist()
        implicit_qa = {'implicit_question': {'prompt': [e for e in implicit_questions], 'ground_truth': targets}}
        product_recommendation_qa = {'product_recommendation': {'prompt': [e for e in product_questions], 'ground_truth': targets}}

        # For new format, each subject, question, and target should be a list of cluster_size elements
        if len(subjects) > 0 and isinstance(subjects[0], list) and len(subjects[0]) > 1:
            print("Loading 2D cluster format data")
            # expand implicit_questions, single questions_paraphrased to lists matching the cluster size
            questions_paraphrased_expanded, targets_expanded, implicit_questions_expanded, product_questions_expanded = [], [], [], []
            for i, qp in enumerate(questions_paraphrased):
                # expand question for each element in the cluster
                cluster_size = len(subjects[i])
                questions_paraphrased_expanded.append([qp] * cluster_size)
                targets_expanded.append([targets[i][-1]] * cluster_size)
                implicit_questions_expanded.append([implicit_questions[i]] * cluster_size)
                product_questions_expanded.append([product_questions[i]] * cluster_size)
            questions_paraphrased = questions_paraphrased_expanded
            targets = targets_expanded if same_target else targets
            # Update implicit_qa structure for 2D format
            implicit_qa = {'implicit_question': {'prompt': [e for e in implicit_questions_expanded], 'ground_truth': targets}}
            product_recommendation_qa = {'product_recommendation': {'prompt': [e for e in product_questions_expanded], 'ground_truth': targets}}
        
        return questions, targets, subjects, questions_paraphrased, implicit_qa, product_recommendation_qa, same_target, user_preference


def write_runtime_to_csv(data_abbrev, editing_method, model_name, total_runtime, avg_edit_time, total_edits, cluster_size, data_size):
    """Write runtime data to CSV file"""
    csv_filename = f'../results/time_{data_abbrev}.csv'
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.exists(csv_filename)
    
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['editing_method', 'model_name', 'total_file_runtime_seconds', 'average_edit_time_seconds', 
                     'total_edits', 'cluster_size', 'data_size', 'timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write data row
        writer.writerow({
            'editing_method': editing_method,
            'model_name': model_name,
            'total_file_runtime_seconds': total_runtime,
            'average_edit_time_seconds': avg_edit_time,
            'total_edits': total_edits,
            'cluster_size': cluster_size,
            'data_size': data_size,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        })


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Test edit cluster with configurable cluster size')
    parser.add_argument('--device', type=int, default=1, help='GPU device to use')
    parser.add_argument('--hparams_dir', default='ROME/llama3-8b', required=True, type=str)
    parser.add_argument('--data_size', type=int, default=20, help='Number of data points to process')
    parser.add_argument('--cluster_size', type=int, default=1, help='Number of subjects in a cluster for sequential editing')
    # parser.add_argument('--data_path', type=str, default='../data/UPQA/UPQA_50_cluster9.json', help='Path to the data file')
    parser.add_argument('--data_path', type=str, default='../data/UPQA/balanced_subset_cluster9.json', help='Path to the data file')
    parser.add_argument('--same_target', action='store_true', default=True, help='If set, use same target for each cluster')
    parser.add_argument('--diff_target', dest='same_target', action='store_false', help='If set, use different targets for each cluster')
    args = parser.parse_args()

    editing_method = args.hparams_dir.split('/')[-2]
    editing_hparams = editing_methods[editing_method]

    hparams = editing_hparams.from_hparams(os.path.join('hparams', args.hparams_dir))
    model_name_abbrev = model_name_abbrev_dict[hparams.model_name.split("/")[-1]]
    hparams.rephrase_additional_token_len = 8
    hparams.implicit_additional_token_len = 8
    hparams.product_additional_token_len = 8
    
    n = args.data_size
    hparams.device = args.device
    cluster_size = args.cluster_size
    # hparams.evaluation_type = "local-llm-judge"
    # judge_model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    # hparams.judge_tok = AutoTokenizer.from_pretrained(judge_model_name)
    # hparams.judge_model = AutoModelForCausalLM.from_pretrained(judge_model_name, torch_dtype='auto').to(f'cuda:7')
    hparams.evaluation_type = "claude-judge"
    
    questions, targets, subjects, questions_paraphrased, implicit_qa, product_recommendation_qa, same_target, user_preference = load_data(args.data_path, n, args.same_target)
    
    # Check if results file already exists BEFORE loading the model
    data_abbrev, file_name = args.data_path.split('/')[-2], args.data_path.split('/')[-1]
    if 'cluster' in file_name:
        output_dir = '../results/pref_representation' if args.same_target else '../results/pref_representation_diff_target'
        output_filename = f'{output_dir}/{model_name_abbrev}_{editing_method}_{n}_cs{cluster_size}.json'
    else:
        output_dir = f'../results/{data_abbrev}'
        output_filename = f'{output_dir}/{model_name_abbrev}_{editing_method}_{n}.json'
    
    if os.path.exists(output_filename):
        print(f"Results file '{output_filename}' already exists. Skipping execution.")
        exit(0)
    
    # Extract implicit_questions from implicit_qa for cluster processing
    implicit_questions = implicit_qa['implicit_question']['prompt']

    editor = BaseEditor.from_hparams(hparams)
    
    print(f"Using cluster_size={cluster_size}, processing {n} data points")
    
    # Check if we have 2D data (cluster format)
    is_2d_data = len(subjects) > 0 and isinstance(subjects[0], list)
    if cluster_size < 1:
        raise ValueError("cluster_size must be >= 1")
    
    if is_2d_data and cluster_size > 1:
        min_cluster_size = min(len(subj) for subj in subjects[:n])
        if cluster_size > min_cluster_size:
            print(f"Warning: cluster_size ({cluster_size}) is larger than the smallest cluster size ({min_cluster_size})")
            print("Some clusters will be padded or truncated")
    
    print(f"Data format: {'2D (cluster format)' if is_2d_data else '1D (individual format)'}")
    
    if cluster_size == 1:
        print("========= Processing with cluster_size=1 - editing one at a time")
        if is_2d_data:
            # For 2D data with cluster_size=1, only use the last element of each cluster
            single_targets = [t[-1] if isinstance(t, list) else t for t in targets]
            single_subjects = [s[-1] if isinstance(s, list) else s for s in subjects]
            single_questions = [q[-1] if isinstance(q, list) else q for q in questions]
            single_implicit_questions = [iq[-1] if isinstance(iq, list) else iq for iq in implicit_questions]
            single_questions_paraphrased = [qp[-1] if isinstance(qp, list) else qp for qp in questions_paraphrased]
            
            # Create implicit_qa structure for evaluation
            single_implicit_qa = {'implicit_question': {
                'prompt': single_implicit_questions[:n], 
                'ground_truth': single_targets[:n]
            }}
            
            single_user_preference = [up[-1] if isinstance(up, list) else up for up in user_preference]
            
            metrics, model_post_edit, _ = editor.edit( 
                prompts=single_questions[:n],
                target_new=single_targets[:n],
                subject=single_subjects[:n],
                implicit_qa=single_implicit_qa,
                rephrase_prompts=single_questions_paraphrased[:n],
                user_preference=single_user_preference[:n],
                sequential_edit=False,
                verbose=False
            )
        else:
            metrics, model_post_edit, _ = editor.edit( 
                prompts=questions,
                target_new=targets,
                subject=subjects,
                implicit_qa=implicit_qa,
                product_recommendation_qa=product_recommendation_qa,
                rephrase_prompts=questions_paraphrased,
                user_preference=user_preference,
                sequential_edit=False,
                verbose=False
            )
    else:
        print(f"========= Processing with cluster_size={cluster_size} - sequential editing in clusters")
        if not is_2d_data:
            raise ValueError("cluster_size > 1 requires 2D data format (lists of subjects/targets)")
        
        all_metrics = []
        # Process each data point with cluster_size sequential edits
        for data_idx in range(min(n, len(questions))):
            print(f"\nProcessing data point {data_idx + 1}/{min(n, len(questions))}")
            # Get the cluster data and limit to cluster_size
            cluster_targets = targets[data_idx][-cluster_size:] 
            cluster_subjects = subjects[data_idx][-cluster_size:]
            cluster_questions = questions[data_idx][-cluster_size:]
            cluster_implicit_questions = implicit_questions[data_idx][-cluster_size:]
            cluster_questions_paraphrased = questions_paraphrased[data_idx][-cluster_size:]
            print(f"Cluster questions: {cluster_questions}")
            print(f"Cluster targets: {cluster_targets}")
            
            cluster_implicit_qa = {'implicit_question': {
                'prompt': cluster_implicit_questions, 
                'ground_truth': cluster_targets
            }}
            
            cluster_user_preference = [user_preference[data_idx]] * cluster_size
            
            # Edit cluster_size elements sequentially, then evaluate
            # The editor should return metrics for all edits, but we only keep the last one
            cluster_metrics, model_post_edit, _ = editor.edit(
                prompts=cluster_questions,
                target_new=cluster_targets,
                subject=cluster_subjects,
                implicit_qa=cluster_implicit_qa,
                rephrase_prompts=cluster_questions_paraphrased,
                user_preference=cluster_user_preference,
                sequential_edit=True,  # Edit sequentially within cluster
                verbose=True
            )
            
            # Only keep the metrics from the last edit in the cluster
            if cluster_metrics:
                final_metric = cluster_metrics[-1]  # Last edit's result
                all_metrics.append(final_metric)
                print(f"Completed data point {data_idx + 1}, kept final result after {cluster_size} edits")
        
        metrics = all_metrics

    metrics = convert_metrics_to_new_format(metrics)
    
    # Calculate runtime statistics
    total_runtime = round(time.time() - start_time, 2)
    avg_edit_time = round(sum(metric.get('time', 0) for metric in metrics) / len(metrics) if metrics else 0, 2)
    
    # Add runtime metadata to the results
    runtime_metadata = {
        "total_file_runtime_seconds": total_runtime,
        "average_edit_time_seconds": avg_edit_time,
        "total_edits": len(metrics),
        "editing_method": editing_method,
        "model_name": model_name_abbrev,
        "cluster_size": cluster_size,
        "data_size": n,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Create final results structure
    final_results = {
        "runtime_metadata": runtime_metadata,
        "metrics": metrics
    }
    
    os.makedirs(output_dir, exist_ok=True)
    json.dump(metrics, open(output_filename, 'w'), indent=4)
    print(f"\nCompleted processing. Total metrics: {len(metrics)}")
    # json.dump(final_results, open(output_filename, 'w'), indent=4)
    
    # Write runtime data to CSV
    write_runtime_to_csv(data_abbrev, editing_method, model_name_abbrev, total_runtime, avg_edit_time, len(metrics), cluster_size, n)
    
    print(f"\n=== RUNTIME SUMMARY ===")
    print(f"Editing Method: {editing_method} Model: {model_name_abbrev} Total Edits: {len(metrics)}")
    print(f"Total File Runtime: {total_runtime}s")
    print(f"Average Edit Time: {avg_edit_time}s")
    print(f"Completed processing. Results saved to {output_filename}")
    print(f"Runtime data appended to ../results/time_{data_abbrev}.csv")
    # - cluster_size=1, n=20: 20 individual edits → 20 metrics (one per edit)
    # - cluster_size=3, n=20: 20×3=60 total edits → 20 metrics (final result after every 3 edits)


if __name__ == "__main__":
    main()

# python edit_cluster.py --hparams_dir ROME/llama3-8b --cluster_size 3 --data_size 5 --device 2
