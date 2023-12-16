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
    parser.add_argument('--model', default='gpt-3.5-turbo-0613',
                        choices=['gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-16k-0613',
                                 'gpt-3.5-turbo-0301', 'gpt-4', 'gpt-4-0613', 'gpt-4-32k', 'gpt-4-32k-0613',
                                 'gpt-4-0314', 'gpt-4-32k-0314'],
                        type=str)
    parser.add_argument('--data_load_name', default='code_optimiztion_data.jsonl', type=str)
    parser.add_argument('--result_save_name', default='code_opt_infer_gpt3.jsonl', type=str)
    parser.add_argument('--log_file_name', default='code_opt_infer_gpt3.log', type=str)
    parser.add_argument('--temperature', default=0.7, type=float)
    parser.add_argument('--candidate_num', default=5, type=int)
    args = parser.parse_args()

    return args


# References: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def generate_text(model, prompt, temperature, candidate_num):
    assert temperature > 0 if candidate_num > 1 else True
    messages = [{'role': 'user', 'content': prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        n=candidate_num
    )
    results = []
    for i in range(candidate_num):
        results.append(response['choices'][i]['message']['content'])
    return results


# References: https://github.com/openai/openai-cookbook/blob/5783656852d507c335955d14875ebc9902f628ef/examples/How_to_count_tokens_with_tiktoken.ipynb
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

def add_time_optimization(example):
    src_uid = example['src_uid']
    task_description = example['description']
    baseline_code = example['time_baseline_source_code']
    testcases = example['testcases']
    lang = example['lang']
    example_input = testcases[0]['input']
    example_output = testcases[0]['output'][0]

    user_message = f"""As an expert software developer with years of experience, please meticulously inspect the following unoptimized inefficient code and give an optimized version of the code, making it solve the same exact problem while achieving faster execution time.
To pass the testcases, the generated optimized code should strictly follow the same input/output format as the original unoptimized code.
The detailed information are as follows:
1. Description of the problem: {task_description}
2. Programming language: {lang}
3. Unoptimized code: 
```
{baseline_code}
```
4. Example testcase input: {example_input}
5. Example testcase output: {example_output}

Respond only the optimized code in the following JSON format:
{{"optimized_code": code string}}"""
    prompt = user_message

    logging.info(f"\nstart inferencing for src_uid={src_uid}, lang={lang}")
    logging.info(f"unoptimized code:\n {baseline_code}")
    input_tokens = count_message_tokens(prompt, args.model, 'input')
    logging.info('input tokens: ' + str(input_tokens))
    try:
        responses = generate_text(
            model=args.model,
            prompt=prompt,
            temperature=temperature,
            candidate_num=candidate_num
        )
    except Exception as e:
        logging.error('Failed to generate text: ' + e.__str__())
        responses = [None]*candidate_num
        # print('response: ' + str(response))
    finally:
        for i in range(candidate_num):
            if len(responses) <= i:
                logging.error(f"response number is {len(responses)}, optimization_{i} is set to empty string")
                optimization = ''
            elif responses[i] is None:
                logging.error(f"the {i}th response is None, optimization_{i} is set to empty string")
                optimization = ''
            else:
                output_tokens = count_message_tokens(responses[i], args.model, 'output')
                logging.info(f'the {i}th response tokens: ' + str(output_tokens))
                if input_tokens + output_tokens > max_tokens:
                    logging.warning(f'Over output tokens limit ---- lang: {lang}, src_uid: {src_uid}')
                optimization = responses[i]
            example[f'optimization_{i}'] = optimization
            logging.info(f'optimization_{i}: {str(optimization)}')
    return example

def add_mem_optimization(example):
    src_uid = example['src_uid']
    task_description = example['description']
    baseline_code = example['memory_baseline_source_code']
    testcases = example['testcases']
    lang = example['lang']
    example_input = testcases[0]['input']
    example_output = testcases[0]['output'][0]

    user_message = f"""As an expert software developer with years of experience, please meticulously inspect the following the following unoptimized inefficient code and give an optimized version of the code, making it solve the same exact problem while achieving smaller memory usage.
To pass the testcases, the generated optimized code should strictly follow the same input/output format as the original unoptimized code.
The detailed information are as follows:
1. Description of the problem: {task_description}
2. Programming language: {lang}
3. Unoptimized code: 
```
{baseline_code}
```
4. Example testcase input: {example_input}
5. Example testcase output: {example_output}

Respond only the optimized code in the following JSON format:
{{"optimized_code": code string}}"""
    prompt = user_message

    logging.info(f"\nstart mem optimizing for src_uid={src_uid}, lang={lang}")
    logging.info(f"unoptimized code:\n {baseline_code}")
    input_tokens = count_message_tokens(prompt, args.model, 'input')
    logging.info('input tokens: ' + str(input_tokens))
    try:
        responses = generate_text(
            model=args.model,
            prompt=prompt,
            temperature=temperature,
            candidate_num=candidate_num
        )
    except Exception as e:
        logging.error('Failed to generate text: ' + e.__str__())
        responses = [None]*candidate_num
        # print('response: ' + str(response))
    finally:
        for i in range(candidate_num):
            if len(responses) <= i:
                logging.error(f"response number is {len(responses)}, optimization_{i} is set to empty string")
                optimization = ''
            elif responses[i] is None:
                logging.error(f"the {i}th response is None, optimization_{i} is set to empty string")
                optimization = ''
            else:
                output_tokens = count_message_tokens(responses[i], args.model, 'output')
                logging.info(f'the {i}th response tokens: ' + str(output_tokens))
                if input_tokens + output_tokens > max_tokens:
                    logging.warning(f'Over output tokens limit ---- lang: {lang}, src_uid: {src_uid}')
                optimization = responses[i]
            example[f'optimization_{i}'] = optimization
            logging.info(f'optimization_{i}: {str(optimization)}')
    return example

def main():
    load_path = Path(__file__).parent.parent.parent / Path('data') / Path(args.data_load_name)
    save_dir = Path(__file__).parent / Path('results')
    dataset = load_dataset('json', split='train', data_files=str(load_path))
    dataset.cleanup_cache_files() 
    mem_save_path = save_dir / Path(f"mem_{args.result_save_name}")
    time_save_path = save_dir / Path(f"time_{args.result_save_name}")
    logging.info("=====start mem optimiing=====")
    mem_ds = dataset.map(add_mem_optimization)
    mem_ds.to_json(mem_save_path, lines=True)
    logging.info("=====start time optimiing=====")
    time_ds = dataset.map(add_time_optimization)
    time_ds.to_json(time_save_path, lines=True)



if __name__ == '__main__':
    args = parse_arguments()

    log_file_path = Path(__file__).parent / Path('logs') / Path(args.log_file_name)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(filename=log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # References: https://platform.openai.com/docs/api-reference/authentication
    openai.api_key = args.api_key
    model_max_tokens = {
        'gpt-3.5-turbo': 4097,
        'gpt-3.5-turbo-16k': 16385,
        'gpt-3.5-turbo-0613': 4097,
        'gpt-3.5-turbo-16k-0613': 16385,
        'gpt-3.5-turbo-0301': 4097,
        'gpt-4': 8192,
        'gpt-4-0613': 8192,
        'gpt-4-32k': 32768,
        'gpt-4-32k-0613': 32768,
        'gpt-4-0314': 8192,
        'gpt-4-32k-0314': 32768
    }
    candidate_num = args.candidate_num
    temperature = args.temperature
    max_tokens = model_max_tokens.get(args.model) if model_max_tokens.get(args.model) is not None else 0

    main()
