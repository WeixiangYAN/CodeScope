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


def add_smell(example):
    id = example['id']
    lang_cluster = example['lang_cluster']
    smell_code = example['smell_code']
    source_code = example['source_code']
    prompt = f"""As an expert software developer with years of experience, please meticulously inspect the following smell code snippet and categorize it into one of the following categories:
- large class
- data class
- blob
- feature envy
- long method
The detailed information are as follows:
1. Programming language: {lang_cluster} 
2. Smell code snippet: 
```
{smell_code.strip()}
```
3. Source code containing code smells:
```
{source_code.strip()}
```
Respond only with one of the specified categories."""

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

            supported_smells = ['large class', 'long method', 'data class', 'blob', 'feature envy']
            if all(supported_smell not in response.lower() for supported_smell in supported_smells):
                logging.warning('Respond content is invalid value.')
                smell = ''
            else:
                smell = ''
                min_index = float('inf')
                for supported_smell in supported_smells:
                    first_index = response.lower().find(supported_smell)
                    if first_index != -1 and first_index < min_index:
                        min_index = first_index
                        smell = supported_smell
        else:
            logging.warning('Respond content is none.')
            smell = ''

    except Exception as e:
        logging.error('Failed to generate text: ' + e.__str__())
        smell = ''

    logging.info('smell: ' + str(smell))
    example['predicted_smell'] = smell

    return example


def main():
    load_data_path = Path(__file__).parent.parent.parent / Path('data') / Path('code_smell_data.jsonl')
    save_data_path = Path(__file__).parent / Path('results') / Path(f'code_smell_result_{model_abbr_mapping[args.model]}.jsonl')

    dataset = load_dataset('json', split='train', data_files=str(load_data_path))
    dataset.cleanup_cache_files()
    print(dataset)

    dataset = dataset.map(add_smell)
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
    log_file_path = log_dir / Path(f'code_smell_log_{model_abbr_mapping[args.model]}.log')
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
