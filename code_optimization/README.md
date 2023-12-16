## Code Optimization

### 0. Data
The code optimization dataset is at `data/code_optimization_data.jsonl`. We explain the fields of the data below:
| field | description |
| :---: | :---: |
| id | the local id of items in the dataset |
| src_uid | the codeforce's id of the given coding problem |
| description | the nl description of the given coding problem |
| lang | the programming language of the baseline codes that solve the coding problem |
| memory_baseline_source_code | the human-submitted code solution that have high memory usage |
| time_baseline_source_code | the human-submitted code solution that have high execution time |
| testcases | a list of testcases of the coding problem, each testcase contains two fields: "input" and "output" |

### 1. Installation

1. `cd code_optimization`
2. install `python>=3.9` (we only guarantee the code works on python 3.9), `GNU C 9.4.0`, `GNU C++ 9.4.0`, `Mono C# 6.12.0.200` environment on your Linux machine. (here are some useful links for installing [GCC](https://linuxize.com/post/how-to-install-gcc-on-ubuntu-20-04/) and [Mono C#](https://linuxize.com/post/how-to-install-mono-on-ubuntu-20-04/) on Ubuntu 20.04)
3. install `torch` (we suggest `torch==2.1.1`) based on your cuda version
4. `pip install -r requirements.txt`

### 2. Inference

`cd inference`

Run the inference scripts to get the inference results of the targeted llms. The inference results `{time/mem}_code_opt_data_{modelname}.jsonl` will be saved under the `inference/results` folder. Specifically,

#### 2.1 PaLM

Replace "google_api_key" with your own Google API key.

`python run_palm.py --api_key google_api_key --data_load_name code_optimization_data.jsonl --result_save_name code_opt_infer_palm.jsonl --log_file_name code_opt_infer_palm.log`

#### 2.2 GPT-3.5

Replace "openai_api_key" with your own OpenAI API key.

`python run_gpt.py --api_key openai_api_key --model gpt-3.5-turbo-0613 --data_load_name code_optimization_data.jsonl --result_save_name code_opt_infer_gpt3.jsonl --log_file_name code_opt_infer_gpt3.log`

#### 2.3 GPT-4

Replace "openai_api_key" with your own OpenAI API key.
`python run_gpt.py --api_key openai_api_key --model gpt-4-0613 --data_load_name code_optimization_data.jsonl --result_save_name code_opt_infer_gpt4.jsonl --log_file_name code_opt_infer_gpt4.log`

#### 2.4 Huggingface Models
For huggingface models, you can run the following command by replacing "access_token" with your own HuggingFace access token, "cache_dir" with path to a directory in which a downloaded pretrained model should be cached, "model_checkpoint" with the name of the targeted model, and "{model_name}" with the name of the targeted huggingface model's name.

`python run_{model_name}.py --access_token access_token --cache_dir cache_dir --checkpoint model_checkpoint --data_load_name code_optimization_data.jsonl --result_save_name code_opt_infer_{model_name}.jsonl --log_file_name code_opt_infer_{model_name}.log`

We provide the following huggingface models inference scripts for you:

| Model Name | Model Checkpoint | Script Name |
| :---: | :---: | :---: |
| codellama | codellama/CodeLlama-34b-Instruct-hf | run_codellama.py |
| vicuna | lmsys/vicuna-13b-v1.5-16k | run_vicuna.py |
| llama2 | meta-llama/Llama-2-70b-chat-hf | run_llama2.py |
| wizardcoder | WizardLM/WizardCoder-15B-V1.0 | run_wizardcoder.py |
| starcoder | HuggingFaceH4/starchat-beta | run_starcoder.py |

An example of running the codellama model is:

`python run_codellama.py --access_token access_token --cache_dir cache_dir --checkpoint codellama/CodeLlama-34b-Instruct-hf --data_load_name code_optimization_data.jsonl --result_save_name code_opt_infer_codellama.jsonl --log_file_name code_opt_infer_codellama.log`

We welcome you to try other huggingface models on our benchmark and share your results with us!

### 3. Evaluation

After getting the inference results, go through the following steps to parse the code, execute and get the efficiency performance, and finally get the evaluation metrics.

Replace "{modelname}" with the name of the targeted model.

1. `cd ../evaluator`
2. Run `python save_opt_codes.py --code_test_data_name mem_code_opt_eval_{modelname}.jsonl --codes_dir_name {modelname}_opt_codes --opt_type mem --parse_code` and `python save_opt_codes.py --code_test_data_name time_code_opt_eval_{modelname}.jsonl --codes_dir_name {modelname}_opt_codes --opt_type time --parse_code` to parse the optimized codes into code files under `evaluator/codes/{modelname}_opt_codes/` folder.
3. Run `python test_opt_codes.py --code_test_data_name mem_code_opt_eval_{modelname}.jsonl --codes_dir_name {modelname}_opt_codes --opt_type mem` and `python test_opt_codes.py --code_test_data_name time_code_opt_eval_{modelname}.jsonl --codes_dir_name {modelname}_opt_codes --opt_type time` to execute the codes and obtain the execution time and memory usage performance. The performance metrics of each code snippet will be saved together with the code file under `results/ans/{modelname}_opt_codes`.
4. Run `python eval_code_optimization.py --codes_dir_name opt_{modelname}_codes > opt_scores/cal_{modelname}_metrics.log`. This will calculate the pass@5 and opt@5 scores of the targeted llm's optimization results. And these scores will be placed under  `evaluator/opt_scores/` folder.
