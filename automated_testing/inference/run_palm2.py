import re
import json
import logging
import argparse
import sys

import google.generativeai as palm

from pathlib import Path
from datasets import load_dataset
from google.api_core import retry


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', default=None, type=str)
    args = parser.parse_args()

    return args


@retry.Retry()
def generate_text(*args, **kwargs):
    return palm.generate_text(*args, **kwargs)


@retry.Retry()
def count_message_tokens(*args, **kwargs):
    return palm.count_message_tokens(*args, **kwargs)


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
    prompt = f"""As an expert code test developer with years of experience, please provide multiple test cases for a given problem along and its solution.
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

    logging.info('sample id: ' + str(id))

    input_tokens = count_message_tokens(prompt=prompt)['token_count']
    logging.info('input tokens: ' + str(input_tokens))
    if input_tokens > max_input_tokens:
        logging.warning('Over input tokens limit: ' + str(id))

    try:
        response = generate_text(
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )
        logging.info('response: ' + str(response.result))

        if response.result is not None:
            output_tokens = count_message_tokens(prompt=response.result)['token_count']
            logging.info('output tokens: ' + str(output_tokens))
            if output_tokens > max_output_tokens:
                logging.warning('Over output tokens limit ' + str(id))

            pattern = r'\[\s*\{.*?\}\s*\]'
            matches = re.search(pattern, response.result, re.DOTALL)
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
    save_data_path = Path(__file__).parent / Path('results') / Path('automated_testing_result_palm2.jsonl')

    dataset = load_dataset('json', split='train', data_files=str(load_data_path))
    dataset.cleanup_cache_files()
    print(dataset)

    dataset = dataset.map(add_testcases)
    print(dataset)

    dataset.to_json(save_data_path, lines=True)


if __name__ == '__main__':
    args = parse_arguments()

    log_dir = Path(__file__).parent / Path('logs')
    if not log_dir.is_dir():
        log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / Path('automated_testing_log_palm2.log')
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

    palm.configure(api_key=args.api_key, transport='rest')
    models = [model for model in palm.list_models() if 'generateText' in model.supported_generation_methods]
    temperature = 0
    max_input_tokens = models[0].input_token_limit  # 8192
    max_output_tokens = models[0].output_token_limit  # 1024

    main()
