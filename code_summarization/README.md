# Code Summarization

## Data
The code summarization dataset is located in `data/code_summarization_data.jsonl`. The fields of the data are explained below:

| field | description |
| :---: | :---: |
| id | the local id of items in the dataset |
| source_code | a code snippet that perform some functionaliy |
| lang_cluster | the programming language of the source code |
| human_summarization | a piece of reference nl summarization for the source code |

## Installation

1. `cd code_summarization`
2. install `python>=3.9` (we use `python==3.9`)
3. install `torch` (we suggest `torch==2.1.1`) based on your cuda version
4. `pip install -r requirements.txt`

## Inference

Run the inference scripts to get the inference results of the targeted LLMs. The inference results `code_summ_data_{modelname}.jsonl` will be saved under the `inference/results` folder. The inference logs `code_summ_log_{model_name}.log` will be saved under the `inference/logs` folder. 

### Closed-sourced LLMs

We provide the following closed-sourced LLMs inference scripts for you:


| Model Name | Model Version      | Script Name  |
| ---------- | ------------------ | ------------ |
| PaLM 2     | text-bison-001     | run_palm2.py |
| GPT-4      | gpt-4-0613         | run_gpt.py   |
| GPT-3.5    | gpt-3.5-turbo-0613 | run_gpt.py   |

For PaLM 2, you can run the following command by replacing `google_api_key` with your own PaLM API key.

`python run_palm.py --api_key google_api_key --data_load_name data/code_summarization_data.jsonl --result_save_name code_summ_infer_palm.jsonl --log_file_name code_summ_infer_palm.log`

For GPT-4 and GPT-3.5, you can run the following command by replacing `openai_api_key` with your own OpenAI API key, `model_version` with specific model version.

`python run_gpt.py --api_key openai_api_key --model gpt-4-0613 --data_load_name data/code_summarization_data.jsonl --result_save_name code_summ_infer_gpt4.jsonl --log_file_name code_summ_infer_gpt4.log`

`python run_gpt.py --api_key openai_api_key --model gpt-3.5-turbo-0613 --data_load_name data/code_summarization_data.jsonl --result_save_name code_summ_infer_gpt3.jsonl --log_file_name code_summ_infer_gpt3.log`



### Open-sourced LLMs

We provide the following open-sourced LLMs inference scripts for you:

| Model Name  | Model Checkpoint                    | Script Name        |
| ----------- | ----------------------------------- | ------------------ |
| Code LLaMA  | codellama/CodeLlama-34b-Instruct-hf | run_codellama.py   |
| LLaMA 2     | meta-llama/Llama-2-70b-chat-hf      | run_llama2.py      |
| StarCoder   | HuggingFaceH4/starchat-beta         | run_starcoder.py   |
| Vicuna      | lmsys/vicuna-13b-v1.5-16k           | run_vicuna.py      |
| WizardCoder | WizardLM/WizardCoder-15B-V1.0       | run_wizardcoder.py |


For open-sourced HuggingFace models, you can run the following command by replacing `huggingface_access_token` with your own HuggingFace access token, `cache_dir` with path to a directory in which a downloaded pretrained model and tokenizer should be cached, `model_checkpoint` with specific model checkpoint.

`python inference/run_{model_name}.py --access_token huggingface_access_token --cache_dir cache_dir --checkpoint model_checkpoint --data_load_name data/code_summarization_data.jsonl --result_save_name code_summ_infer_{model_name}.jsonl --log_file_name code_summ_infer_{model_name}.log`

An example of running the Code LLaMA model is:
`python run_codellama.py --access_token access_token --cache_dir cache_dir --checkpoint codellama/CodeLlama-34b-Instruct-hf --data_load_name data/code_summarization_data.jsonl --result_save_name code_summ_infer_codellama.jsonl --log_file_name code_summ_infer_codellama.log`

## Evaluation

1. `cd ../evaluator` 
2. Execute the command `python eval_code_summarization.py --llm_infer_result infer_file`, substituting 'infer_file' with the name of the llm's inference file (for example, 'code_summ_infer_palm.jsonl'). This will generate scores for the targeted llm's inference results, which will be saved in the directory `evaluator/summ_scores/`.
