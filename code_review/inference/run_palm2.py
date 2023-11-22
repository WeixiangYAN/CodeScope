import logging
import argparse
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


def add_diff_tag(example):
    id = example['id']
    lang_cluster = example['lang_cluster']
    source_code = example['source_code']
    diff_hunk = example['diff_hunk']
    prompt = f"""As an expert code reviewer with years of experience, please meticulously inspect the following code change and categorize its quality into one of the following categories:
- 0: Good quality that no review comments required.
- 1: Poor quality that requires review comments.
The detailed information are as follows:
1. Programming language: {lang_cluster} 
2. Original version code: 
```
{source_code.strip()}
```
3. Code diff chunk:
```
{diff_hunk.strip()}
```
Respond only with the number: 0 or 1."""

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

            supported_diff_tags = ['0', '1']
            if all(supported_diff_tag not in response.result for supported_diff_tag in supported_diff_tags):
                logging.warning('Respond content is invalid value.')
                diff_tag = 2
            else:
                diff_tag = 2
                min_index = float('inf')
                for supported_diff_tag in supported_diff_tags:
                    first_index = response.result.find(supported_diff_tag)
                    if first_index != -1 and first_index < min_index:
                        min_index = first_index
                        diff_tag = int(supported_diff_tag)
        else:
            logging.warning('Respond content is none.')
            diff_tag = 2

    except Exception as e:
        logging.error('Failed to generate text: ' + e.__str__())
        diff_tag = 2

    logging.info('diff_tag: ' + str(diff_tag))
    example['predicted_diff_tag'] = diff_tag

    return example


def add_review_comment(example):
    id = example['id']
    lang_cluster = example['lang_cluster']
    source_code = example['source_code']
    diff_hunk = example['diff_hunk']
    prompt = f"""As an expert code reviewer with years of experience, please meticulously inspect the following code change and provide a concise review comment.
The detailed information are as follows:
1. Programming language: {lang_cluster} 
2. Original version code: 
```
{source_code.strip()}
```
3. Code diff chunk:
```
{diff_hunk.strip()}
```
Respond only with a string that represents review comment."""

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

            review_comment = response.result
        else:
            logging.warning('Respond content is none.')
            review_comment = ''

    except Exception as e:
        logging.error('Failed to generate text: ' + e.__str__())
        review_comment = ''

    logging.info('review_comment: ' + str(review_comment))
    example['predicted_review_comment'] = review_comment

    return example


def main():
    load_data_path = Path(__file__).parent.parent.parent / Path('data') / Path('code_review_data.jsonl')
    save_data_path = Path(__file__).parent / Path('results') / Path('code_review_result_palm2.jsonl')

    dataset = load_dataset('json', split='train', data_files=str(load_data_path))
    dataset.cleanup_cache_files()
    print(dataset)

    dataset = dataset.map(add_diff_tag)
    dataset = dataset.map(add_review_comment)
    print(dataset)

    dataset.to_json(save_data_path, lines=True)


if __name__ == '__main__':
    args = parse_arguments()

    log_dir = Path(__file__).parent / Path('logs')
    if not log_dir.is_dir():
        log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / Path('code_review_log_palm2.log')
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
