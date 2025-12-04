#!/bin/bash
ulimit -c 0
trap "echo 'Caught signal, killing children'; kill 0; exit 1" SIGINT SIGTERM
start_time=$(date +%s)
model_name=llama3-8b

python general_capability.py --hparams_dir=ROME/$model_name --device_pre=0 --device_post=0 --task_name=nli &
python general_capability.py --hparams_dir=ROME/$model_name --device_pre=1 --device_post=1 --task_name=boolq &
python general_capability.py --hparams_dir=ROME/$model_name --device_pre=2 --device_post=2 --task_name=gsm8k &
python general_capability.py --hparams_dir=ROME/$model_name --device_pre=3 --device_post=3 --task_name=natural_questions &

python general_capability.py --hparams_dir=FT-M/$model_name --device_pre=4 --device_post=4 --task_name=nli &
python general_capability.py --hparams_dir=FT-M/$model_name --device_pre=5 --device_post=5 --task_name=boolq &
python general_capability.py --hparams_dir=FT-M/$model_name --device_pre=6 --device_post=6 --task_name=gsm8k &
python general_capability.py --hparams_dir=FT-M/$model_name --device_pre=7 --device_post=7 --task_name=natural_questions &
wait

python general_capability.py --hparams_dir=LoRA/$model_name --device_pre=0 --device_post=0 --task_name=nli &
python general_capability.py --hparams_dir=LoRA/$model_name --device_pre=1 --device_post=1 --task_name=boolq &
python general_capability.py --hparams_dir=LoRA/$model_name --device_pre=2 --device_post=2 --task_name=gsm8k &
python general_capability.py --hparams_dir=LoRA/$model_name --device_pre=3 --device_post=3 --task_name=natural_questions &


end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Total runtime: $((runtime / 60)) minutes and $((runtime % 60)) seconds"