import torch
import logging
import argparse
import warnings

from typing import List
from pathlib import Path
from datasets import load_dataset
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--access_token', default=None, type=str)
    parser.add_argument('--cache_dir', default=None, type=str)
    parser.add_argument('--checkpoint', default='HuggingFaceH4/starchat-beta',
                        choices=[
                            'HuggingFaceH4/starchat-alpha',
                            'HuggingFaceH4/starchat-beta'
                        ],
                        type=str)
    args = parser.parse_args()

    return args


class StopAtSpecificTokenCriteria(StoppingCriteria):

    def __init__(self, token_id_list: List[int] = None):
        self.token_id_list = token_id_list

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0][-1].detach().cpu().numpy() in self.token_id_list


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_text(prompt, temperature, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(device)
    outputs = model.generate(
        inputs['input_ids'],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=StoppingCriteriaList([
            StopAtSpecificTokenCriteria(token_id_list=[
                tokenizer.encode("<|end|>", return_tensors='pt').tolist()[0][0]
            ])
        ])
    ).to('cpu')
    response = tokenizer.decode(outputs[0]).split('<|assistant|>')[-1].replace('<|end|>', '')

    return response.strip()


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def count_message_tokens(content):
    tokens = tokenizer(content)['input_ids']
    num_tokens = len(tokens)

    return num_tokens


def add_diff_tag(example):
    id = example['id']
    lang_cluster = example['lang_cluster']
    source_code = example['source_code']
    diff_hunk = example['diff_hunk']
    user_message = f"""As an expert code reviewer with years of experience, please meticulously inspect the following code change and categorize its quality into one of the following categories:
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
    system_message = 'Below is a dialogue between a human and an AI assistant called StarChat.'
    prompt = f'<|system|>\n{system_message.strip()}<|end|>\n<|user|>\n{user_message.strip()}<|end|>\n<|assistant|>'

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
    user_message = f"""As an expert code reviewer with years of experience, please meticulously inspect the following code change and provide a concise review comment.
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
    system_message = 'Below is a dialogue between a human and an AI assistant called StarChat.'
    prompt = f'<|system|>\n{system_message.strip()}<|end|>\n<|user|>\n{user_message.strip()}<|end|>\n<|assistant|>'

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
    save_data_path = Path(__file__).parent / Path('results') / Path('code_review_result_starcoder.jsonl')

    dataset = load_dataset('json', split='train', data_files=str(load_data_path))
    dataset.cleanup_cache_files()
    print(dataset)

    dataset = dataset.map(add_diff_tag)
    dataset = dataset.map(add_review_comment)
    print(dataset)

    dataset.to_json(save_data_path, lines=True)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = parse_arguments()

    log_dir = Path(__file__).parent / Path('logs')
    if not log_dir.is_dir():
        log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / Path('code_review_log_starcoder.log')
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
