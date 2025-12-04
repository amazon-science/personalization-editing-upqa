#!/bin/bash
ulimit -c 0
trap "echo 'Caught signal, killing children'; kill 0; exit 1" SIGINT SIGTERM
start_time=$(date +%s)

size=100
path=prefeval_pro/prefeval_pro_balanced.json
results_dir=../results/prefeval_pro  
for editing_method in LoRA ROME FT-M; do
    for turn in 2 4 6 8 10; do
        python run_edit.py --hparams_dir=$editing_method/olmo2-7b --data_path=$path --size=$size --inter_turns=$turn --results_dir=$results_dir --device=0 &
        python run_edit.py --hparams_dir=$editing_method/qwen3-8b --data_path=$path --size=$size --inter_turns=$turn --results_dir=$results_dir --device=1 &
        wait
    done
done


end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Total runtime: $((runtime / 60)) minutes and $((runtime % 60)) seconds"

# Model Context Windows:
# mistral-7b: 32,768 tokens
# llama3-8b: 8,192 tokens
# gemma-7b: 8,192 tokens
# deepseek-7b: 4,096 tokens
# gemma2-9b: 8,192 tokens
# gpt-j-6b: 2,048 tokens, Critical Issue: GPT-J-6B exceeds context window at 6+ inter-turns 6 turns: 2,343 tokens > 2,048 limit (295 tokens over)
# olmo2-7b: 4,096 tokens
# qwen3-8b: 32,768 tokens
