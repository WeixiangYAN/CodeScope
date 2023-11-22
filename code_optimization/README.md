# Code Optimization

## Data
The code optimization dataset is located in `data/code_optimization_data.jsonl`. The fields of the data are explained below:

| field | description |
| :---: | :---: |
| id | the local id of items in the dataset |
| src_uid | the codeforce's id of the given coding problem |
| description | the nl description of the given coding problem |
| lang | the programming language of the baseline codes that solve the coding problem |
| memory_baseline_source_code | the human-submitted code solution that have high memory usage |
| time_baseline_source_code | the human-submitted code solution that have high execution time |
| testcases | a list of testcases of the coding problem, each testcase contains two fields: "input" and "output" |

## Dependence

1. `cd code_optimization`
2. install `python>=3.9` (we only guarantee the code works on python 3.9), `GNU C 9.4.0`, `GNU C++ 9.4.0`, `Mono C# 6.12.0.200` environment on your Linux machine. (here are some useful links for installing [GCC](https://linuxize.com/post/how-to-install-gcc-on-ubuntu-20-04/) and [Mono C#](https://linuxize.com/post/how-to-install-mono-on-ubuntu-20-04/) on Ubuntu 20.04)
3. install `torch` (we suggest `torch==2.1.1`) based on your cuda version
4. `pip install -r requirements.txt`

## Inference

Run the inference scripts to get the inference results of the targeted LLMs. The inference results `code_opt_result_{model_name}.jsonl` will be saved under the `inference/results` folder. The inference logs `code_opt_log_{model_name}.log` will be saved under the `inference/logs` folder.

### Closed-sourced LLMs

We provide the following closed-sourced LLMs inference scripts for you:


| Model Name | Model Version      | Script Name  |
| ---------- | ------------------ | ------------ |
| PaLM 2     | text-bison-001     | run_palm2.py |
| GPT-4      | gpt-4-0613         | run_gpt.py   |
| GPT-3.5    | gpt-3.5-turbo-0613 | run_gpt.py   |

For PaLM 2, you can run the following command by replacing `google_api_key` with your own Google API key. 

`python run_palm.py --api_key google_api_key --data_load_name code_optimization_data.jsonl --result_save_name code_opt_infer_palm.jsonl --log_file_name code_opt_infer_palm.log`


For GPT-4 and GPT-3.5, you can run the following command by replacing `openai_api_key` with your own OpenAI API key, `model_version` with specific model version.

`python run_gpt.py --api_key openai_api_key --model gpt-4-0613 --data_load_name code_optimization_data.jsonl --result_save_name code_opt_infer_gpt4.jsonl --log_file_name code_opt_infer_gpt4.log`

`python run_gpt.py --api_key openai_api_key --model gpt-3.5-turbo-0613 --data_load_name code_optimization_data.jsonl --result_save_name code_opt_infer_gpt3.jsonl --log_file_name code_opt_infer_gpt3.log`


### Open-sourced LLMs

We provide the following open-sourced LLMs inference scripts for you:


| Model Name  | Model Checkpoint                    | Script Name        |
| ----------- | ----------------------------------- | ------------------ |
| Code LLaMA  | codellama/CodeLlama-34b-Instruct-hf | run_codellama.py   |
| LLaMA 2     | meta-llama/Llama-2-70b-chat-hf      | run_llama2.py      |
| StarCoder   | HuggingFaceH4/starchat-beta         | run_starcoder.py   |
| Vicuna      | lmsys/vicuna-13b-v1.5-16k           | run_vicuna.py      |
| WizardCoder | WizardLM/WizardCoder-15B-V1.0       | run_wizardcoder.py |

For HuggingFace models, you can run the following command by replacing `huggingface_access_token` with your own HuggingFace access token, `cache_dir` with path to a directory in which a downloaded pretrained model and tokenizer should be cached, `model_checkpoint` with specific model checkpoint.

`python run_{model_name}.py 
	--access_token access_token 
	--cache_dir cache_dir 
	--checkpoint model_checkpoint 
	--data_load_name code_optimization_data.jsonl 
	--result_save_name code_opt_infer_{model_name}.jsonl 
	--log_file_name code_opt_infer_{model_name}.log`

An example of running the codellama model is:

`python run_codellama.py 
	--access_token access_token 
	--cache_dir cache_dir 
	--checkpoint codellama/CodeLlama-34b-Instruct-hf 
	--data_load_name code_optimization_data.jsonl 
	--result_save_name code_opt_infer_codellama.jsonl 
	--log_file_name code_opt_infer_codellama.log`


## Evaluation

After getting the inference results, go through the following steps to parse the code, execute and get the efficiency performance, and finally get the evaluation metrics.

Replace `{modelname}` with the name of the targeted model.

1. `cd ../evaluator`
2. Run `python save_opt_codes.py --code_test_data_name mem_code_opt_eval_{modelname}.jsonl --codes_dir_name {modelname}_opt_codes --opt_type mem --parse_code` and `python save_opt_codes.py --code_test_data_name time_code_opt_eval_{modelname}.jsonl --codes_dir_name {modelname}_opt_codes --opt_type time --parse_code` to parse the optimized codes into code files under `evaluator/codes/{modelname}_opt_codes/` folder.
3. Run `python test_opt_codes.py --code_test_data_name mem_code_opt_eval_{modelname}.jsonl --codes_dir_name {modelname}_opt_codes --opt_type mem` and `python test_opt_codes.py --code_test_data_name time_code_opt_eval_{modelname}.jsonl --codes_dir_name {modelname}_opt_codes --opt_type time` to execute the codes and obtain the execution time and memory usage performance. The performance metrics of each code snippet will be saved together with the code file under `results/ans/{modelname}_opt_codes`.
4. Run `python eval_code_optimization.py --codes_dir_name opt_{modelname}_codes > opt_scores/cal_{modelname}_metrics.log`. This will calculate the pass@5 and opt@5 scores of the targeted LLM's optimization results. And these scores will be placed under  `evaluator/opt_scores/` folder.
