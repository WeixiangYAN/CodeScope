import re
import json
import torch
import logging
import argparse
import warnings

from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tenacity import retry, stop_after_attempt, wait_random_exponential


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--access_token', default=None, type=str)
    parser.add_argument('--cache_dir', default=None, type=str)
    parser.add_argument('--checkpoint', default='meta-llama/Llama-2-70b-chat-hf',
                        choices=[
                            'meta-llama/Llama-2-7b-chat-hf',
                            'meta-llama/Llama-2-13b-chat-hf',
                            'meta-llama/Llama-2-70b-chat-hf'
                        ],
                        type=str)
    args = parser.parse_args()

    return args


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_text(prompt, temperature, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(device)
    outputs = model.generate(
        inputs['input_ids'],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    ).to('cpu')
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response.split('[/INST]')[-1].strip()


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def count_message_tokens(content):
    tokens = tokenizer(content)['input_ids']
    num_tokens = len(tokens)

    return num_tokens


def add_testcases(example):
    id = example['id']
    description = example['description']
    input_specification = example['input_specification']
    output_specification = example['output_specification']
    sample_inputs = example['sample_inputs']
    sample_outputs = example['sample_outputs']
    notes = example['notes']
    lang_cluster = example['lang_cluster']
    source_code = example['source_code']
    user_message = f"""As an expert code test developer with years of experience, please provide multiple test cases for a given problem along and its solution.
The detailed information are as follows:
1. Problem description: {description}
2. Input specification: {input_specification}
3. Output specification: {output_specification}
4. Sample inputs: {sample_inputs}
5. Sample outputs: {sample_outputs}
6. Notes: {notes}
7. Programming language: {lang_cluster} 
8. Solution source code: 
```
{source_code.strip()}
```
Craft 5 test cases with these criteria:
1. Each test case contains a string for both input and output.
2. The solution source code successfully processes the test case's input with no errors.
3. The solution source code's outcome aligns with the test case's output.
4. All test cases are simple and achieve optimal branch and line coverage.
Respond only with a string in the following JSON format:
[{{"input": input string, "output": output string}}]"""
    prompt = f'<s>[INST] {user_message.strip()} [/INST]'

    logging.info('sample id: ' + str(id))

    input_tokens = count_message_tokens(prompt)
    logging.info('input tokens: ' + str(input_tokens))
    if input_tokens > max_input_tokens:
        logging.warning('Over input tokens limit: ' + str(id))

    try:
        response = generate_text(
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )
        logging.info('response: ' + str(response))

        if response is not None:
            output_tokens = count_message_tokens(response)
            logging.info('output tokens: ' + str(output_tokens))
            if output_tokens > max_new_tokens:
                logging.warning('Over output tokens limit ' + str(id))

            pattern = r'\[\s*\{.*?\}\s*\]'
            matches = re.search(pattern, response, re.DOTALL)
            if matches:
                json_array_string = matches.group().replace("'", '"')
                try:
                    json_array = json.loads(json_array_string, strict=False)
                    if isinstance(json_array, list):
                        for json_item in json_array:
                            if isinstance(json_item['input'], list):
                                json_item['input'] = str(json_item['input'][0])
                            if isinstance(json_item['output'], str):
                                json_item['output'] = [json_item['output']]
                        testcases = str(json_array)
                    else:
                        logging.warning('Respond content is not a list.')
                        testcases = "[{'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}]"
                except json.JSONDecodeError as e:
                    logging.warning('Failed to load json:', e)
                    testcases = "[{'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}]"
            else:
                logging.warning('JSON array object not found.')
                testcases = "[{'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}]"

        else:
            logging.warning('Respond content is none.')
            testcases = "[{'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}]"

    except Exception as e:
        logging.error('Failed to generate text: ' + e.__str__())
        testcases = "[{'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}]"

    logging.info('testcases: ' + str(testcases))
    example['predicted_testcases'] = testcases

    return example


def main():
    load_data_path = Path(__file__).parent.parent.parent / Path('data') / Path('automated_testing_data.jsonl')
    save_data_path = Path(__file__).parent / Path('results') / Path('automated_testing_result_llama2.jsonl')

    dataset = load_dataset('json', split='train', data_files=str(load_data_path))
    dataset.cleanup_cache_files()
    print(dataset)

    dataset = dataset.map(add_testcases)
    print(dataset)

    dataset.to_json(save_data_path, lines=True)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = parse_arguments()

    log_dir = Path(__file__).parent / Path('logs')
    if not log_dir.is_dir():
        log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / Path('automated_testing_log_llama2.log')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s - %(filename)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(filename=log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        use_fast=True,
        trust_remote_code=True,
        token=args.access_token,
        cache_dir=args.cache_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map='auto',
        token=args.access_token,
        cache_dir=args.cache_dir
    )
    print(f'Memory footprint: {model.get_memory_footprint() / 1e6:.2f} MB')
    temperature = 0
    max_input_tokens = tokenizer.model_max_length  # 1000000000000000019884624838656
    max_new_tokens = 1024

    main()
