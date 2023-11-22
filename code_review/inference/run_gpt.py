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
    parser.add_argument('--model', default='gpt-4',
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

            supported_diff_tags = ['0', '1']
            if all(supported_diff_tag not in response for supported_diff_tag in supported_diff_tags):
                logging.warning('Respond content is invalid value.')
                diff_tag = 2
            else:
                diff_tag = 2
                min_index = float('inf')
                for supported_diff_tag in supported_diff_tags:
                    first_index = response.find(supported_diff_tag)
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

            review_comment = response
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
    save_data_path = Path(__file__).parent / Path('results') / Path(f'code_review_result_{model_abbr_mapping[args.model]}.jsonl')

    dataset = load_dataset('json', split='train', data_files=str(load_data_path))
    dataset.cleanup_cache_files()
    print(dataset)

    dataset = dataset.map(add_diff_tag)
    dataset = dataset.map(add_review_comment)
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
    log_file_path = log_dir / Path(f'code_review_log_{model_abbr_mapping[args.model]}.log')
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
