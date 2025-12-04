#!/bin/bash
# Disable core dumps to prevent core.NUMBER files when interrupted
ulimit -c 0
trap "echo 'Caught signal, killing children'; kill 0; exit 1" SIGINT SIGTERM
start_time=$(date +%s)

size=100
for cluster_size in 9; do
    for editing_method in LoRA FT-L ROME FT-M; do
        python edit_cluster.py --hparams_dir=$editing_method/deepseek-qwen-7b --cluster_size=$cluster_size --data_size=$size --device=0 &
        python edit_cluster.py --hparams_dir=$editing_method/llama3-8b --cluster_size=$cluster_size --data_size=$size --device=1 &
        python edit_cluster.py --hparams_dir=$editing_method/olmo2-7b --cluster_size=$cluster_size --data_size=$size --device=2 &
        python edit_cluster.py --hparams_dir=$editing_method/qwen3-8b --cluster_size=$cluster_size --data_size=$size --device=3 &
        wait
    done
done


# ICE only cluster size = 1
cluster_size=1
for editing_method in ICE; do
    python edit_cluster.py --hparams_dir=$editing_method/qwen3-8b --cluster_size=$cluster_size --data_size=$size --device=0 &
    python edit_cluster.py --hparams_dir=$editing_method/gpt-j-6b --cluster_size=$cluster_size --data_size=$size --device=1 &
    python edit_cluster.py --hparams_dir=$editing_method/llama3-8b --cluster_size=$cluster_size --data_size=$size --device=2 &
    python edit_cluster.py --hparams_dir=$editing_method/mistral-7b --cluster_size=$cluster_size --data_size=$size --device=3 &
    python edit_cluster.py --hparams_dir=$editing_method/olmo2-7b --cluster_size=$cluster_size --data_size=$size --device=4 &
    python edit_cluster.py --hparams_dir=$editing_method/deepseek-qwen-7b --cluster_size=$cluster_size --data_size=$size --device=25&
    wait
done

size=200
path=../data/UPQA/balanced_subset_200.json
for editing_method in ROME FT-M ICE LoRA FT-L GRACE; do
    python edit_cluster.py --hparams_dir=$editing_method/olmo2-7b --data_path=$path --data_size=$size --device=0 &
    python edit_cluster.py --hparams_dir=$editing_method/qwen3-8b --data_path=$path --data_size=$size --device=1 &
    python edit_cluster.py --hparams_dir=$editing_method/llama3-8b --data_path=$path --data_size=$size --device=2 &
    python edit_cluster.py --hparams_dir=$editing_method/deepseek-qwen-7b --data_path=$path --data_size=$size --device=3 &
    wait
    echo "Completed editing method: $editing_method"
done


end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Total runtime: $((runtime / 60)) minutes and $((runtime % 60)) seconds"