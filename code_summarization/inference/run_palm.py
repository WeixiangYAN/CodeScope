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
    parser.add_argument('--data_load_name', default='code_summarization_data.jsonl', type=str)
    parser.add_argument('--result_save_name', default='code_summ_infer_palm.jsonl', type=str)
    parser.add_argument('--log_file_name', default='code_summ_infer_palm.log', type=str)
    parser.add_argument('--temperature', default=0, type=float)
    parser.add_argument('--candidate_num', default=1, type=int)
    args = parser.parse_args()

    return args


@retry.Retry()
def generate_text(*args, **kwargs):
    response = palm.generate_text(*args, **kwargs).candidates
    return [output['output'] for output in response]


@retry.Retry()
def count_message_tokens(*args, **kwargs):
    return palm.count_message_tokens(*args, **kwargs)


def add_code_summ(example):
    code = example['source_code']
    lang = example['lang_cluster']
    id = example['id']

    user_message = f'Please generate a short summarization for the following codes:\n{code}'
    prompt = user_message

    logging.info(f'lang: {lang}, id: {id}')

    input_tokens = count_message_tokens(prompt=prompt)['token_count']
    logging.info('input tokens: ' + str(input_tokens))
    if input_tokens > max_input_tokens:
        logging.warning(f'Over input tokens limit ---- lang: {lang}, id: {id}')

    try:
        response = generate_text(
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            candidate_count=candidate_num
        )[0]
        logging.info('response: ' + str(response))

        if response is not None:
            output_tokens = count_message_tokens(prompt=response)['token_count']
            logging.info('output tokens: ' + str(output_tokens))
            if output_tokens > max_output_tokens:
                logging.warning(f'Over output tokens limit ---- lang: {lang}, id: {id}')
            code_summ = response
        else:
            logging.warning('Respond content is none.')
            code_summ = ''
    except Exception as e:
        logging.error('Failed to generate text: ' + e.__str__())
        code_summ = ''

    logging.info('code_sum_candidate: ' + str(code_summ))
    example['code_sum_candidate'] = code_summ

    return example

def main():
    load_path = Path(__file__).parent.parent.parent / Path('data') / Path(args.data_load_name)
    save_dir = Path(__file__).parent / Path('results')
    dataset = load_dataset('json', split='train', data_files=str(load_path))
    dataset.cleanup_cache_files() 
    save_path = save_dir / Path(args.result_save_name)
    dataset = dataset.map(add_code_summ)
    dataset.to_json(save_path, lines=True)


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
