import re
import json
import logging
import argparse
import google.generativeai as palm

from pathlib import Path
from datasets import load_dataset
from google.api_core import retry


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', default=None, type=str)
    parser.add_argument('--data_load_name', default='code_optimization_data.jsonl', type=str)
    parser.add_argument('--result_save_name', default='code_opt_infer_palm.jsonl', type=str)
    parser.add_argument('--log_file_name', default='code_opt_infer_palm.log', type=str)
    parser.add_argument('--temperature', default=0.7, type=float)
    parser.add_argument('--candidate_num', default=5, type=int)
    args = parser.parse_args()

    return args


@retry.Retry()
def generate_text(*args, **kwargs):
    response = palm.generate_text(*args, **kwargs).candidates
    return [output['output'] for output in response]


@retry.Retry()
def count_message_tokens(*args, **kwargs):
    return palm.count_message_tokens(*args, **kwargs)


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

    input_tokens = count_message_tokens(prompt=prompt)['token_count']
    logging.info('input tokens: ' + str(input_tokens))
    if input_tokens > max_input_tokens:
        logging.warning('Over input tokens limit: src_uid=' + str(src_uid))

    try:
        responses = generate_text(
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            candidate_count=candidate_num
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
                output_tokens = count_message_tokens(prompt=responses[i])['token_count']
                logging.info(f'the {i}th response tokens: ' + str(output_tokens))
                if output_tokens > max_output_tokens:
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
    logging.info(f"\nstart inferencing for src_uid={src_uid}, lang={lang}")
    logging.info(f"unoptimized code:\n {baseline_code}")

    input_tokens = count_message_tokens(prompt=prompt)['token_count']
    logging.info('input tokens: ' + str(input_tokens))
    if input_tokens > max_input_tokens:
        logging.warning('Over input tokens limit: src_uid=' + str(src_uid))
    try:
        responses = generate_text(
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
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
                output_tokens = count_message_tokens(responses[i])
                logging.info(f'the {i}th response tokens: ' + str(output_tokens))
                if output_tokens > max_output_tokens:
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

    # References: https://github.com/google/generative-ai-python/issues/29
    palm.configure(api_key=args.api_key, transport='rest')
    models = [model for model in palm.list_models() if 'generateText' in model.supported_generation_methods]
    temperature = args.temperature
    candidate_num = args.candidate_num
    max_input_tokens = models[0].input_token_limit  # 8192
    max_output_tokens = models[0].output_token_limit  # 1024

    main()
