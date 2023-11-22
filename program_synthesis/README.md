# Program Synthesis

## Data
The program synthesis dataset is located in `data/program_synthesis_data.jsonl`. The fields of the data are explained below:

| Field                	| Explanation                                          	         |
|----------------------	|----------------------------------------------------------------|
| `description`          	| The original problem description in natural language 	         |
| `input_specification`  	| Description of the form of input data                	         |
| `output_specification` 	| Description of the form of output data               	         |
| `sample_inputs`        	| Sample inputs                                        	         |
| `sample_outputs`       	| Sample outputs                                       	         |
| `notes`                	| Additional note for the problem                              	 |
| `src_uid`              	| Unique identifier of the problem                     	         |
| `lang_cluster`         	| The programming language to use                      	         |
| `difficulty`           	| Difficulty of the problem                            	         |
| `human_solution`       	| Accepted human solution                              	         |
| `testcases`            	| List of testcases of the coding problem           	         |
| `id`                   	| The local ID in the task                             	         |

## Dependence (same as code repair and code translation)
1. `cd program_synthesis`
2. install `python>=3.9` (we use `python==3.9`)
3. install `pytorch` (we use `pytorch==2.1.1`) based on your cuda version
4. ``pip install -r requirement.txt``

### Executor Dependence 
#### Perl Dependence
```
conda install -c conda-forge perl
```
Validate the correctness of installation:
```
perl -v
touch myscript.pl
perl myscript.pl
```

Programs written in D, and Delphi that need to run on **Windows** require the following dependencies to be installed:

#### D Dependencies:

Download [dmd 2.105.0](https://downloads.dlang.org/releases/2.x/2.105.0/) for windows and unzip it to a suitable location. Replace `d_path` in run.py

#### Delphi Dependencies:

Download [delphi 7](http://altd.embarcadero.com/download/delphi/d7/english/ent/delphi_7_ent_en.iso) and install it to a suitable location. Replace `delphi_path` in run.py

***

Programs written in **other languages** need to be run using the ExecEval (under the project root directory), and the following dependencies need to be installed:

### ExecEval Dependencies:

1. Install [docker-ce](https://docs.docker.com/engine/install/)
2. `cd ExecEval`
3. `docker build . -t exec-eval:1.0`

## Inference
Run the inference scripts to get the inference results of the targeted LLMs. The inference results `program_synthesis_result_{model_name}.jsonl` will be saved under the `inference/results` folder. The inference logs `program_synthesis_log_{model_name}.log` will be saved under the `inference/logs` folder.

### Closed-sourced LLMs

We provide the following closed-sourced LLMs inference scripts for you:


| Model Name | Model Version      | Script Name  |
| ---------- | ------------------ | ------------ |
| PaLM 2     | text-bison-001     | run_palm2.py |
| GPT-4      | gpt-4-0613         | run_gpt.py   |
| GPT-3.5    | gpt-3.5-turbo-0613 | run_gpt.py   |

For PaLM 2, you can run the following command by replacing `google_api_key` with your own Google API key. 

```angular2html
python run_palm.py
    --api_key your_palm_api_key
    --data_load_name program_synthesis_data.jsonl
    --candidate_num 5
    --result_save_name program_synthesis_run_palm.jsonl
    --log_file_name program_synthesis_run_palm.log
```

For GPT-4 and GPT-3.5, you can run the following command by replacing `openai_api_key` with your own OpenAI API key, `model_version` with specific model version.

```angular2html
python run_gpt.py
    --api_key your_openai_apikey
    --model model_specific_version
    --data_load_name program_synthesis_data.jsonl
    --candidate_num 5
    --result_save_name program_synthesis_run_{model_name}}.jsonl
    --log_file_name program_synthesis_run_\{model_name\}.log
```


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

```angular2html
python run_{model_name}.py 
    --access_token access_token
    --cache_dir cache_dir 
    --checkpoint your_model_ckpt
    --data_load_name program_synthesis_data.jsonl
    --candidate_num 5
    --result_save_name program_synthesis_run_{model_name}.jsonl
    --log_file_name program_synthesis_run_{model_name}.log
```


## Evaluator (executor & scorer)


The code ready for testing should be stored line by line in your\_codes.jsonl and the file should be placed in your\_codes\_dir. A typical code record is shown below and should contain at least the following keys:

```
{
    "lang_cluster": "{model_name}",
    "lang": "{model_name}",
    "source_code": "{model_name}",
    "src_uid": "{model_name}",
    "difficulty": 800,
    "testcases": "[{'input': 'input1', 'output': ['output1']}, {'input': 'input2', 'output': ['output2']}]"
}
```

* For all programming languages except Perl, D, and Delphi, example of most typical usage:

1. `docker run -it -p x:y -e NUM_WORKERS=n exec-eval:1.0.` This will expose port y (default 5000) as http://localhost:y on the local machine whereas port x is used within the docker container which can be set by environment variable GUNICORN_PORT. It is recommended to not use all cpus, as if cpu goes into 100% load it might affect execution speed of the codes uncontrollably, and keeping some cpus free for evaluation script. A valid example assuming less cpus available: `docker run -it -p 5000:5000 -e NUM_WORKERS=5 exec-eval:1.0`
   
2. `python run_execeval.py --codes_dir your_codes_dir --results_dir your_results_dir --code_filename your_codes.jsonl`

    The results of the run are output to `your_results_dir`, forming a jsonl file, which compares the input jsonl, with each new entry adding the results of each test case run, stored in the `testcases`

* For Perl, D, and Delphi, example of most typical usage:

    `python run.py  --code_path your_codes_{program_language}.jsonl --output_path result/results.json --cmd_path your_cmd_path `
  
    Please change the `--code_path` with `perl/d/delphi` code files. The execute results are saved to `--output_path`, which records the results of `accepted`, `wrong`, and `error` for each key, and each output records the possible error outputs and the type of error.

## Evaluation
After the execution, we provide a *scorer* script to count the number of correct solutions around different languages and difficulties. 

Please put all your executed results into `--result_dir`, include `d/perl/delphi` and the rest. Then run following command to count the results generated by `{model_name}`: `python score_program_synthesis.py --result_dir your_result_dir --model_name model_name`
