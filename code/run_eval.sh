#!/bin/bash
ulimit -c 0
trap "echo 'Caught signal, killing children'; kill 0; exit 1" SIGINT SIGTERM
start_time=$(date +%s)

file_dir=prefeval_pro
for file in ../results/$file_dir/*.json; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        echo "Processing: $filename"
        python evaluate_concurrent_ack.py "$file_dir/$filename" --output-dir=../results/prefeval_pro_evaluated_ack_olmo_qwen
    fi
done


end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Total runtime: $((runtime / 60)) minutes and $((runtime % 60)) seconds"
