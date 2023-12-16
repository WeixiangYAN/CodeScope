# Code Smell

## Data

The code smell dataset is located in `data/code_smell_data.jsonl`. The fields of the data are explained below:


|    Field    |                 Description                 |
| :----------: | :------------------------------------------: |
|      id      |     the local id of items in the dataset     |
| lang_cluster | the programming language of the source code |
| source_code |     code containing certain code smell     |
|  smell_code  |   smelly code snippet in the source code   |
|    smell    | code smell type of the smelly code snippet |

## Dependence

1. `cd code_smell`
2. install `python>=3.9` (we use `python==3.9`)
3. install `torch` (we suggest `torch==2.1.1`) based on your cuda version
4. `pip install -r requirements.txt`

## Inference

Run the inference scripts to get the inference results of the targeted LLMs. The inference results `code_smell_result_{model_name}.jsonl` will be saved under the `inference/results` folder. The inference logs `code_smell_log_{model_name}.log` will be saved under the `inference/logs` folder.

### Closed-sourced LLMs

We provide the following closed-sourced LLMs inference scripts for you:


| Model Name | Model Version      | Script Name  |
| ---------- | ------------------ | ------------ |
| PaLM 2     | text-bison-001     | run_palm2.py |
| GPT-4      | gpt-4-0613         | run_gpt.py   |
| GPT-3.5    | gpt-3.5-turbo-0613 | run_gpt.py   |

For PaLM 2, you can run the following command by replacing `google_api_key` with your own Google API key.

`python inference/run_palm2.py --api_key google_api_key`

For GPT, you can run the following command by replacing `openai_api_key` with your own OpenAI API key, `model_version` with specific model version.

`python inference/run_gpt.py --api_key openai_api_key --model model_version`

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

`python inference/run_{model_name}.py --access_token huggingface_access_token --cache_dir cache_dir --checkpoint model_checkpoint`

## Evaluation

Run `python evaluator/score.py` to get the scores of the targeted LLMs' inference results. The scores `code_smell_score.json` will be saved under the `evaluator/scores` folder.
