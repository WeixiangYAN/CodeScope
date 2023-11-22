import torch
import logging
import argparse
import warnings

from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

from typing import List

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--access_token', default=None, type=str)
    parser.add_argument('--cache_dir', default=None, type=str)
    parser.add_argument('--checkpoint', default='HuggingFaceH4/starchat-beta',
                        choices=['HuggingFaceH4/starchat-alpha', 'HuggingFaceH4/starchat-beta'], type=str)
    parser.add_argument('--data_load_name', default='code_repair_data.jsonl', type=str)
    parser.add_argument('--result_save_name', default='code_repair_eval_starcoder.jsonl',type=str)
    parser.add_argument('--log_file_name', default='code_repair_eval_starcoder.logs',type=str),
    parser.add_argument('--temperature', default=0.5, type=float)
    parser.add_argument('--candidate_num', default=1, type=int)
    args = parser.parse_args()

    return args

lang_cluster = ['c++', 'java', 'python', 'c', 'c#', 'ruby', 'delphi', 'go',
                'javascript', 'kotlin', 'php', 'd', 'perl', 'rust']

class StopAtSpecificTokenCriteria(StoppingCriteria):

    def __init__(self, token_id_list: List[int] = None):
        self.token_id_list = token_id_list

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0][-1].detach().cpu().numpy() in self.token_id_list


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_text(prompt, temperature, max_new_tokens,candidate_num):
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(device)
    outputs = model.generate(
        inputs['input_ids'],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=50,
        top_p=0.95,
        num_return_sequences=candidate_num,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        # References: https://github.com/bigcode-project/starcoder/issues/73
        stopping_criteria=StoppingCriteriaList([
            StopAtSpecificTokenCriteria(token_id_list=[
                tokenizer.encode("<|end|>", return_tensors='pt').tolist()[0][0]
            ])
        ])
    ).to('cpu')
    response = [tokenizer.decode(output).split('<|assistant|>')[-1].replace('<|end|>', '').strip()
                for output in outputs]

    return response


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def count_message_tokens(content):
    tokens = tokenizer(content)['input_ids']
    num_tokens = len(tokens)

    return num_tokens


env_map = {
    'c++': ['GNU C++11', 'GNU C++14', 'MS C++', 'GNU C++0x', 'GNU C++', 'MS C++ 2017', 'Clang++17 Diagnostics',
            'GNU C++17'],
    'c#': ['MS C#', 'Mono C#', '.NET Core C#'],
    'java': ['Java 11', 'Java 7', 'Java 6', 'Java 8'],
    'javascript': ['JavaScript', 'Node.js'],
    'c': ['GNU C', 'GNU C11'],
    'python': ['Python 2', 'PyPy 3', 'Python 3', 'PyPy 2'],
    'php': ['PHP'],
    'ruby': ['Ruby'],
    'kotlin': ['Kotlin'],
    'rust': ['Rust'],
    'go': ['Go'],
    'd': ['dmd 2.105.0 win32'],
    'delphi': ['Delphi7 win32'],
    'perl': ['Perl v5.20.3']
}


def add_code_repairing(example):
    """
     Code repair: Generate corresponding code based on the problem description and buggy code

     """

    source_lang = example['lang']

    prob_uid = example['src_uid']
    source_code = example['source_code']
    prob_desc_description = example['description']
    prob_desc_input_spec = example['input_specification']
    prob_desc_output_spec = example['output_specification']
    prob_desc_sample_inputs = example['sample_inputs']
    prob_desc_sample_outputs = example['sample_outputs']
    error_msg = example['execute_outcome']
    user_message = f"""As an expert code developer with years of experience, please debug the source code in {source_lang} based on the corresponding problem description and show the correct code. 
The detailed information are shown as follows: 
1. Problem description: {prob_desc_description} 
2. Input specification: {prob_desc_input_spec}
3. Output specification: {prob_desc_output_spec}
4. Sample inputs: {prob_desc_sample_inputs}
5. Sample outputs: {prob_desc_sample_outputs}
6. Programming language: {source_lang}
7. Buggy code :\n {source_code}
8. Error message: {error_msg}
Please note that use complex header files as little as possible. 

Respond should only with a string in the following JSON format:
[{{"version": specific version used in the programming language, "target code":  the code you produced in the respective programming language version."}}] """

    system_message = 'Below is a dialogue between a human and an AI assistant called StarChat.'
    prompt = f'<|system|>\n{system_message.strip()}<|end|>\n<|user|>\n{user_message.strip()}<|end|>\n<|assistant|>'

    logging.info('problem src_id: ' + str(prob_uid))

    input_tokens = count_message_tokens(prompt)
    logging.info('input tokens: ' + str(input_tokens))
    if input_tokens > max_input_tokens:
        logging.warning('Over input tokens limit: ' + str(prob_uid))
    try:
        response = generate_text(
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            candidate_num=candidate_num
        )
        logging.info('response: ' + str(response))

        if response is not None:
            repair_outcome = response
        else:
            logging.warning('Respond content is none.')
            repair_outcome = []

    except Exception as e:
        logging.error('Failed to generate text: ' + e.__str__())
        repair_outcome = []

    for i, generated_code in enumerate(repair_outcome):
        output_tokens = count_message_tokens(generated_code)
        logging.info('output tokens: ' + str(output_tokens))
        if output_tokens > max_new_tokens:
            logging.warning('Over total tokens limit ' + str(prob_uid) + ' lang: ' + str(source_lang))
            generated_code = ''
        logging.info('Code repairing in: ' + source_lang + ' :' + generated_code)
        example['code_repairing_' + str(i)] = generated_code
    if len(repair_outcome) < candidate_num:
        for i in range(candidate_num - len(repair_outcome)):
            example['code_repairing_' + str(i + len(repair_outcome))] = ''

    return example


def main():
    load_path = Path(__file__).parent.parent.parent / Path('data') / Path(args.data_load_name)
    save_path = Path(__file__).parent / Path('result') / Path(args.result_save_name)

    dataset = load_dataset('json', split='train', data_files=str(load_path))
    dataset.cleanup_cache_files()  # for multiple evaluation

    dataset = dataset.map(add_code_repairing)

    dataset.to_json(save_path, lines=True)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = parse_arguments()

    log_file_path = Path(__file__).parent / Path('log') / Path(args.log_file_name)
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
    max_new_tokens = 2048

    main()