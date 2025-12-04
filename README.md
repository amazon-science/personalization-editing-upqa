# Personalization Editing

**Repository Overview**: This repository contains the code and data for the paper *"Towards Effective Model Editing for LLM Personalization"*
<!-- <img src="./data/fig1.png" width=55%> -->


## Table of Contents
1. [Overview](#overview)  
2. [Repository Structure](#repository-structure)  
3. [Installation](#installation)  
4. [Usage](#usage)  
    - [Data Preparation](#data-preparation)
    - [Running Experiments](#running-experiments) 
5. [Citation](#citation)  


## Repository Structure
- `data/`: Contains the datasets used in Personalization Editing.
- `code/`: Includes scripts and code to perform Personalization Editing and reproduce the results in the paper.
<!-- - `results/`: Results of the experiments that we report in the paper. -->


## Installation
To set up the environment for running the code, follow these steps:

1. Clone the repository

2. Create a virtual environment and activate it:
    ```bash
    conda create -n edit python=3.9 -y
    conda activate edit
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```


## Usage

### Data Preparation

1. Datasets are stored in the `data/` directory. There are following files: 
```bash
data/
    .
    ├── prefeval_pro
    └── UPQA
```


## Data Format

Each generated entry contains:
- `input_attribute`: Original persona text
- `attribute_type`: High-level category (e.g., "hobby", "profession", "pet", "location")
- `question`: Direct question using the attribute_type (e.g., "What's my hobby?")
- `question_paraphrased`: Natural rewording of the direct question
- `implicit_question`: Conversational question that guides toward the target without naming the attribute
- `product_recommendation_question`: Product suggestion question relevant to the attribute_type
- `target`: Concise description extracted from the persona

<!-- - `general_capability` contains data to evaluate general knowledge and reasoning capacities before and after editing. To run evaluation first download data to this folder from the following data sources: [GSM8K](https://github.com/openai/gsm8k), [BoolQ](https://github.com/google-research-datasets/boolean-questions), [NLI](https://github.com/hendrycks/nli), [Natural Questions](https://github.com/google-research-datasets/natural-questions). -->


### Running Experiments

**Quick start test run**: To get started (e.g. using ROME to edit llama3-8b on UPQA), run:

```bash
cd ./code
python3 edit_cluster.py \
    --hparams_dir=ROME/llama3-8b \
    --data_path=../data/UPQA/balanced_subset.json \
    --device=0 \
    --size=100 \
```


To run the multi-turn evaluation, here is an example:
```bash
cd ./code
python run_edit.py \
    --hparams_dir=ROME/olmo2-7b \
    --data_path=prefeval_pro/prefeval_pro_balanced.json \
    --size=100 \
    --inter_turns=2 \
    --results_dir=prefeval_multi_turn \
    --device=0 
```

- Use `--inter_turns` to set the number of turns for multi-turn evaluation.

<!-- Note:
If you want to run GRACE, please download their code to 'code/easyeditor/models/grace', we didn't include their code because the authors didn't indicate the license of their code
To run multi-turn evaluation, first download data from https://github.com/amazon-science/PrefEval/blob/main/benchmark_dataset/filtered_inter_turns.json to 'PrefEval/benchmark_dataset/filtered_inter_turns.json' -->

We use claude-3-7-sonnet as the evaluator to assess if model responses match the labels, switch to a local LLM (e.g., Llama3-8b) with ''. For experiments, we recommend using at least one GPU with 48 GB of memory (e.g., NVIDIA RTX A6000) Adjust the device number and evaluation model using `--model_eval` and `--device_eval` as shown in the example above.

For full experiments to reproduce the results in the paper:
1. Experiments for clustering-based preference representations:
    ```bash
    ./run_edit_cluster.sh
    ```

2. Experiments for multi-turn:
    ```bash
    ./run_edit.sh
    ./run_eval.sh
    ```

We evaluate models including `Llama-3-8B-Instruct`, `OLMo-7B-Instruct-hf`, `Qwen3-8B`, `DeepSeek-R1-Distill-Qwen-7B`, `GPT-J-6B` and `Mistral-7B-v0.3`. All parameters are in the `code/hparams/<method_name>/<model_name>`. 

<!-- Results are stored at `specific`, `impact`, `impact-api` under the `results` folder. -->

<!-- To summarize the results, use the jupyter notebook `code/result_table.ipynb` -->


## Acknowledgements
We gratefully acknowledge the use of code and data from the following projects: 
 [EasyEdit](https://github.com/zjunlp/EasyEdit), [ROME](https://github.com/kmeng01/rome), and [PrefEval](https://github.com/amazon-science/PrefEval).

<!-- [GSM8K](https://github.com/openai/gsm8k), [BoolQ](https://github.com/google-research-datasets/boolean-questions), [NLI](https://github.com/hendrycks/nli), [Natural Questions](https://github.com/google-research-datasets/natural-questions), [GRACE](https://github.com/thartvigsen/grace), -->
