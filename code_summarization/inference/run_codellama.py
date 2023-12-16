import re
import json
import torch
import logging
import argparse
import warnings

from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tenacity import retry, stop_after_attempt, wait_random_exponential


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--access_token', default=None, type=str)
    parser.add_argument('--cache_dir', default=None, type=str)
    parser.add_argument('--checkpoint', default='codellama/CodeLlama-7b-Instruct-hf', type=str)
    parser.add_argument('--data_load_name', default='code_summarization_data.jsonl', type=str)
    parser.add_argument('--result_save_name', default='code_summ_infer_codellama.jsonl', type=str)
    parser.add_argument('--log_file_name', default='code_summ_infer_codellama.log', type=str)
    parser.add_argument('--temperature', default=0, type=float)
    parser.add_argument('--candidate_num', default=1, type=int)
    args = parser.parse_args()

    return args


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_text(prompt, temperature, max_new_tokens, candidate_num):
    assert temperature > 0 if candidate_num > 1 else True
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(device)
    outputs = model.generate(
        inputs['input_ids'],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True if temperature > 0 else False,
        top_k=50,
        top_p=0.95,
        num_return_sequences=candidate_num,
        pad_token_id=tokenizer.eos_token_id
    ).to('cpu')
    responses = [tokenizer.decode(output, skip_special_tokens=True)
                 .split('[/INST]')[-1].strip().replace('</s>','')
                  for output in outputs]

    return responses


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def count_message_tokens(content):
    tokens = tokenizer(content)['input_ids']
    num_tokens = len(tokens)

    return num_tokens

def add_code_summ(example):
    code = example['source_code']
    lang = example['lang_cluster']
    id = example['id']

    user_message = f'Please generate a short summarization for the following codes:\n{code}'
    prompt = f'<s>[INST] {user_message.strip()} [/INST]'

    logging.info(f'lang: {lang}, id: {id}')

    input_tokens = count_message_tokens(prompt)
    logging.info('input tokens: ' + str(input_tokens))
    if input_tokens > max_input_tokens:
        logging.warning(f'Over input tokens limit ---- lang: {lang}, id: {id}')

    try:
        response = generate_text(
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            candidate_num=candidate_num
        )[0]
        logging.info('response: ' + str(response))

        if response is not None:
            output_tokens = count_message_tokens(response)
            logging.info('output tokens: ' + str(output_tokens))
            if output_tokens > max_new_tokens:
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
    print(dataset)
    save_path = save_dir / Path(args.result_save_name)
    dataset = dataset.map(add_code_summ)
    dataset.to_json(save_path, lines=True)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # References: https://huggingface.co/codellama
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
        # load_in_4bit=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map='auto',
        token=args.access_token,
        cache_dir=args.cache_dir
    )
    print(f'Memory footprint: {model.get_memory_footprint() / 1e6:.2f} MB')
    candidate_num = args.candidate_num
    temperature = args.temperature
    max_input_tokens = tokenizer.model_max_length  # 1000000000000000019884624838656
    # The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
    max_new_tokens = 2048

    main()
