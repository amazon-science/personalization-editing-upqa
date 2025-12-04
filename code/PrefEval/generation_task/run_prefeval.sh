#!/bin/bash
ulimit -c 0
trap "echo 'Caught signal, killing children'; kill 0; exit 1" SIGINT SIGTERM
start_time=$(date +%s)

size=100
result_dir=prefeval_pro
for task in zero-shot cot; do
    for turn in 0 4 8; do
        python benchmark_generation.py --model=olmo2-7b-local --task=$task --inter_turns=$turn --size=$size --save_inter_msg --result_dir=$result_dir --device=0 &
        python benchmark_generation.py --model=qwen3-8b-local --task=$task --inter_turns=$turn --size=$size --save_inter_msg --result_dir=$result_dir --device=1 &
        python benchmark_generation.py --model=gpt-j-6b-local --task=$task --inter_turns=$turn --size=$size --save_inter_msg --result_dir=$result_dir --device=2 &
        python benchmark_generation.py --model=mistral-7b-local --task=$task --inter_turns=$turn --size=$size --save_inter_msg --result_dir=$result_dir --device=3 &

        python benchmark_generation.py --model=olmo2-7b-local --task=$task --inter_turns=$((turn+2)) --size=$size --save_inter_msg --result_dir=$result_dir --device=4 &
        python benchmark_generation.py --model=qwen3-8b-local --task=$task --inter_turns=$((turn+2)) --size=$size --save_inter_msg --result_dir=$result_dir --device=5 &
        python benchmark_generation.py --model=gpt-j-6b-local --task=$task --inter_turns=$((turn+2)) --size=$size --save_inter_msg --result_dir=$result_dir --device=6 &
        python benchmark_generation.py --model=mistral-7b-local --task=$task --inter_turns=$((turn+2)) --size=$size --save_inter_msg --result_dir=$result_dir --device=7 &
        wait
    done
done

# python benchmark_generation.py --model=llama3-8b-local --task=zero-shot --inter_turns=2 --size=2 --save_inter_msg --result_dir=tmp --device=5

end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Total runtime: $((runtime / 60)) minutes and $((runtime % 60)) seconds"
