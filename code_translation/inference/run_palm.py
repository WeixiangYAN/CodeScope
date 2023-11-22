import logging
import argparse
import google.generativeai as palm

from pathlib import Path
from datasets import load_dataset, Dataset
from google.api_core import retry


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', default='AIzaSyCKgHPsJ6kyJa0BBsY2QTBWZVBJcZRi5fY', type=str)
    parser.add_argument('--candidate_num', default=1, type=int)
    parser.add_argument('--temperature', default=0.7, type=float)
    parser.add_argument('--data_load_name', default='code_translation_data.jsonl', type=str)
    parser.add_argument('--result_save_name', default='code_translation_eval_palm.jsonl', type=str)
    parser.add_argument('--log_file_name', default='code_translation_eval_palm.logs', type=str)
    args = parser.parse_args()

    return args

lang_cluster = ['c++', 'java', 'python', 'c', 'c#', 'ruby', 'delphi', 'go',
                'javascript', 'kotlin', 'php', 'd', 'perl', 'rust']

@retry.Retry()
def generate_text(*args, **kwargs):
    response = palm.generate_text(*args, **kwargs).candidates
    return [output['output'] for output in response]


@retry.Retry()
def count_message_tokens(*args, **kwargs):
    return palm.count_message_tokens(*args, **kwargs)


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


def add_code_translation(example):
    """

    Translation: Generate corresponding code in specific language based on the given code

    """

    source_lang = example['source_lang_cluster'].lower()

    prob_uid = example['src_uid']
    source_code = example['source_code']
    target_lang = example['target_lang_cluster'].lower()

    prompt = f"""As an expert code developer proficient in multiple programming languages with years of experience, please translate the source code in {source_lang} to programming language {target_lang} within our supported version. 
The detailed information are as follows:
1. Target programming language: {target_lang}
2. support programming language version: {env_map[target_lang]}
3. Source code\n: {source_code}
    
Respond should only with a string in the following JSON format:
[{{"version": specific version used in the programming language, "target code":  the code you produced in the respective programming language version."}}] """

    logging.info('problem src_id: ' + str(prob_uid))

    input_tokens = count_message_tokens(prompt=prompt)['token_count']
    logging.info('input tokens: ' + str(input_tokens))

    try:
        response = generate_text(
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            candidate_count=candidate_num
        )
        logging.info('response: ' + str(response))

        if response is not None:
            translation_outcome = response
        else:
            logging.warning('Respond content is none.')
            translation_outcome = []

    except Exception as e:
        logging.error('Failed to generate text: ' + e.__str__())
        translation_outcome = []

    for i, generated_code in enumerate(translation_outcome):
        output_tokens = count_message_tokens(prompt=generated_code)['token_count']
        logging.info('output tokens: ' + str(output_tokens))
        if output_tokens > max_output_tokens:
            logging.warning('Over total tokens limit ' + str(prob_uid) + ' lang: ' + str(target_lang))
            generated_code = ''
        logging.info('code translation  in: ' + target_lang + ' :' + generated_code)
        example['code_translation_' + str(i)] = generated_code
    if len(translation_outcome) < candidate_num:
        for i in range(candidate_num-len(translation_outcome)):
            example['code_translation_' + str(i+len(translation_outcome))] = ''
    return example


def main():

    load_path = Path(__file__).parent.parent.parent / Path('data') / Path(args.data_load_name)
    save_path = Path(__file__).parent / Path('result') / Path(args.result_save_name)

    dataset = load_dataset('json', split='train', data_files=str(load_path))
    dataset.cleanup_cache_files()  # for multiple evaluation
    dataset = dataset.select([0])
    dataset = dataset.map(add_code_translation)
    dataset.to_json(save_path, lines=True)


if __name__ == '__main__':
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

    # References: https://github.com/google/generative-ai-python/issues/29
    palm.configure(api_key=args.api_key, transport='rest')
    models = [model for model in palm.list_models() if 'generateText' in model.supported_generation_methods]
    temperature = args.temperature
    candidate_num = args.candidate_num
    max_input_tokens = models[0].input_token_limit  # 8192
    max_output_tokens = models[0].output_token_limit  # 1024

    main()
