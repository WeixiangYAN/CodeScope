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
from transformers import StoppingCriteria, StoppingCriteriaList
from typing import List


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--access_token', default=None, type=str)
    parser.add_argument('--cache_dir', default=None, type=str)
    parser.add_argument('--checkpoint', default='HuggingFaceH4/starchat-beta', type=str)
    parser.add_argument('--data_load_name', default='code_summarization_data.jsonl', type=str)
    parser.add_argument('--result_save_name', default='code_summ_infer_starcoder.jsonl', type=str)
    parser.add_argument('--log_file_name', default='code_summ_infer_starcoder.log', type=str)
    parser.add_argument('--temperature', default=0, type=float)
    parser.add_argument('--candidate_num', default=1, type=int)
    args = parser.parse_args()

    return args


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_text(prompt, temperature, max_new_tokens, candidate_num):
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(device)
    outputs = model.generate(
        inputs['input_ids'],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=candidate_num,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=StoppingCriteriaList([# https://github.com/bigcode-project/starcoder/issues/73
            StopAtSpecificTokenCriteria(token_id_list=[
                tokenizer.encode("<|end|>", return_tensors='pt').tolist()[0][0]
            ]) # tokenizer.encode("<|end|>", return_tensors='pt') = tensor([[49155]])
        ])
    ).to('cpu')
    responses = [tokenizer.decode(output)
                 .split('<|assistant|>')[-1].replace('<|end|>', '')
                  for output in outputs]

    return responses

class StopAtSpecificTokenCriteria(StoppingCriteria):
    """
    当生成出第一个指定token时，立即停止生成
    """
    def __init__(self, token_id_list: List[int] = None):
        """
        :param token_id_list: 停止生成的指定token的id的列表
        """
        self.token_id_list = token_id_list

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0][-1].detach().cpu().numpy() in self.token_id_list
    
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def count_message_tokens(content):
    tokens = tokenizer(content)['input_ids']
    num_tokens = len(tokens)

    return num_tokens


def add_code_summ(example):
    code = example['source_code']
    lang = example['lang']
    id = example['id']

    user_message = f'Please generate a short summarization for the following codes:\n{code}'
    prompt = f'<|system|>\n<|end|>\n<|user|>\n{user_message.strip()}<|end|>\n<|assistant|>'

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

    # References: https://huggingface.co/blog/starcoder
    # References: https://huggingface.co/datasets/bigcode/ta-prompt
    # References: https://github.com/bigcode-project/starcoder/issues/101
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
    max_new_tokens = 5120

    main()
