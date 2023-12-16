# Automated Testing

## Data

The automated testing dataset is located in `data/automated_testing_data.jsonl`. The fields of the data are explained below:


|              Field              |                                                 Description                                                 |
| :------------------------------: | :----------------------------------------------------------------------------------------------------------: |
|                id                |                                     the local id of items in the dataset                                     |
|           lang_cluster           |                                 the programming language of the source code                                 |
|           source_code           |                                      the human-submitted code solution                                      |
|             src_uid             |                                      the codeforce's id of the problem                                      |
|           description           |                                             problem description                                             |
|       input_specification       | how and in what order the input will be given to the program, also includes the date range, types, and sizes |
|       output_specification       |                                      how the outputs should be printed                                      |
|          sample_inputs          |                     sample inputs for theprogram that is expected to solve the problem                     |
|          sample_outputs          |              the expected output for the sample inputs that is expected to solve the problem              |
|              notes              |                              explanation of sample inputs and sample outputs                              |
|         human_testcases         |                                           human-written testcases                                           |
|         human_pass_rate         |                            pass rate of human-written testcases in source code                            |
|       human_line_coverage       |                          line coverage of human-written testcases in source code                          |
|      human_branch_coverage      |                         branch coverage of human-written testcases in source code                         |
|    human_sample_testcases_1~5    |                     5 testcases randomly selected among human-written testcases 5 times                     |
|    human_sample_pass_rate_1~5    |                    pass rate of sample human-written testcases in source code 5 times                    |
|  human_sample_line_coverage_1~5  |                  line coverage of sample human-written testcases in source code 5 times                  |
| human_sample_branch_coverage_1~5 |                 branch coverage of sample human-written testcases in source code 5 times                 |
|      human_sample_pass_rate      |                            average of 5 sample human-written testcases pass rate                            |
|    human_sample_line_coverage    |                          average of 5 sample human-written testcases line coverage                          |
|   human_sample_branch_coverage   |                         average of 5 sample human-written testcases branch coverage                         |

## Dependence

1. `cd automated_testing`
2. install `python>=3.9` (we use `python==3.9`)
3. install [GCC](https://linuxize.com/post/how-to-install-gcc-on-ubuntu-20-04/) on your Linux machine or [MinGW](https://sourceforge.net/projects/mingw-w64/files/mingw-w64/mingw-w64-release/) on your Windows machine
4. install [Java8](https://www.oracle.com/java/technologies/downloads/#java8-linux) on your Linux machine or [Java8](https://www.oracle.com/java/technologies/downloads/#java8-windows) on your Windows machine
5. install `torch` (we suggest `torch==2.1.1`) based on your cuda version
6. `pip install -r requirements.txt`

## Inference

Run the inference scripts to get the inference results of the targeted LLMs. The inference results `automated_testing_result_{model_name}.jsonl` will be saved under the `inference/results` folder. The inference logs `automated_testing_log_{model_name}.log` will be saved under the `inference/logs` folder.

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

1. Run `python evaluator/save_codes.py` to parse the source codes into code files under `evaluator/codes` folder.
2. Run `python evaluator/test_codes.py --result_name automated_testing_result_{model_name}.jsonl` to test the source codes using predicted testcases and obtain the corresponding pass rate, line coverage, branch coverage.
3. Run `python evaluator/score.py` to get the scores of the targeted LLMs' inference results. The scores `automated_testing_score.json` will be saved under the `evaluator/scores` folder.
