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

            supported_smells = ['large class', 'long method', 'data class', 'blob', 'feature envy']
            if all(supported_smell not in response.result.lower() for supported_smell in supported_smells):
                logging.warning('Respond content is invalid value.')
                smell = ''
            else:
                smell = ''
                min_index = float('inf')
                for supported_smell in supported_smells:
                    first_index = response.result.lower().find(supported_smell)
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
    save_data_path = Path(__file__).parent / Path('results') / Path('code_smell_result_palm2.jsonl')

    dataset = load_dataset('json', split='train', data_files=str(load_data_path))
    dataset.cleanup_cache_files()
    print(dataset)

    dataset = dataset.map(add_smell)
    print(dataset)

    dataset.to_json(save_data_path, lines=True)


if __name__ == '__main__':
    args = parse_arguments()

    log_dir = Path(__file__).parent / Path('logs')
    if not log_dir.is_dir():
        log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / Path('code_smell_log_palm2.log')
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
