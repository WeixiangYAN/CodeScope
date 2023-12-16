import re
import json
import openai
import backoff
import logging
import tiktoken
import argparse

from pathlib import Path
from datasets import load_dataset


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', default=None, type=str)
    parser.add_argument('--model', default='gpt-3.5-turbo',
                        choices=[
                            'gpt-3.5-turbo',
                            'gpt-3.5-turbo-16k',
                            'gpt-3.5-turbo-0613',
                            'gpt-3.5-turbo-16k-0613',
                            'gpt-3.5-turbo-0301',
                            'gpt-4-1106-preview',
                            'gpt-4',
                            'gpt-4-0613',
                            'gpt-4-32k',
                            'gpt-4-32k-0613',
                            'gpt-4-0314',
                            'gpt-4-32k-0314'
                        ],
                        type=str)
    args = parser.parse_args()

    return args


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def generate_text(model, prompt, temperature):
    messages = [{'role': 'user', 'content': prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature
    )

    return response['choices'][0]['message']['content']


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def count_message_tokens(content, model, type):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print('Model not found, using cl100k_base encoding.')
        encoding = tiktoken.get_encoding('cl100k_base')

    num_tokens = 0
    if type == 'input':
        messages = [{'role': 'user', 'content': content}]
        tokens_per_message = 4
        tokens_per_name = -1
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == 'name':
                    num_tokens += tokens_per_name
        num_tokens += 3
    elif type == 'output':
        num_tokens = len(encoding.encode(content))

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

    input_tokens = count_message_tokens(prompt, args.model, 'input')
    logging.info('input tokens: ' + str(input_tokens))

    try:
        response = generate_text(
            model=args.model,
            prompt=prompt,
            temperature=temperature
        )
        logging.info('response: ' + str(response))

        if response is not None:
            output_tokens = count_message_tokens(response, args.model, 'output')
            logging.info('output tokens: ' + str(output_tokens))
            if input_tokens + output_tokens > max_tokens:
                logging.warning('Over total tokens limit ' + str(id))

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
    save_data_path = Path(__file__).parent / Path('results') / Path(f'automated_testing_result_{model_abbr_mapping[args.model]}.jsonl')

    dataset = load_dataset('json', split='train', data_files=str(load_data_path))
    dataset.cleanup_cache_files()
    print(dataset)

    dataset = dataset.map(add_testcases)
    print(dataset)

    dataset.to_json(save_data_path, lines=True)


if __name__ == '__main__':
    args = parse_arguments()

    model_abbr_mapping = {
        'gpt-3.5-turbo': 'gpt3-5',
        'gpt-3.5-turbo-16k': 'gpt3-5',
        'gpt-3.5-turbo-0613': 'gpt3-5',
        'gpt-3.5-turbo-16k-0613': 'gpt3-5',
        'gpt-3.5-turbo-0301': 'gpt3-5',
        'gpt-4-1106-preview': 'gpt4',
        'gpt-4': 'gpt4',
        'gpt-4-0613': 'gpt4',
        'gpt-4-32k': 'gpt4',
        'gpt-4-32k-0613': 'gpt4',
        'gpt-4-0314': 'gpt4',
        'gpt-4-32k-0314': 'gpt4'
    }

    log_dir = Path(__file__).parent / Path('logs')
    if not log_dir.is_dir():
        log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / Path(f'automated_testing_log_{model_abbr_mapping[args.model]}.log')
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

    openai.api_key = args.api_key
    model_max_tokens_mapping = {
        'gpt-3.5-turbo': 4097,
        'gpt-3.5-turbo-16k': 16385,
        'gpt-3.5-turbo-0613': 4097,
        'gpt-3.5-turbo-16k-0613': 16385,
        'gpt-3.5-turbo-0301': 4097,
        'gpt-4-1106-preview': 128000,
        'gpt-4': 8192,
        'gpt-4-0613': 8192,
        'gpt-4-32k': 32768,
        'gpt-4-32k-0613': 32768,
        'gpt-4-0314': 8192,
        'gpt-4-32k-0314': 32768
    }
    temperature = 0
    max_tokens = model_max_tokens_mapping.get(args.model) if model_max_tokens_mapping.get(args.model) is not None else 0

    main()
