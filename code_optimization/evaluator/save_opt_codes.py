import re
import json
import logging
import argparse

from tqdm import tqdm
from pathlib import Path
from collections import Counter
from datasets import load_dataset

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_infer_result', default='mem_code_opt_infer_palm.jsonl', type=str)
    parser.add_argument('--codes_dir_name', default='palm_opt_codes', type=str, choices=['vicuna_opt_codes', 'wizardcoder_opt_codes', 'codellama_opt_codes', 'gpt4_opt_codes', 'gpt3_opt_codes', 'starcoder_opt_codes', 'llama2_opt_codes', 'palm_opt_codes'])
    parser.add_argument('--opt_type', default='mem', choices=['mem', 'time'], type=str)
    parser.add_argument('--parse_code', action='store_true', default=False)
    args = parser.parse_args()

    return args

def main():
    load_path = Path(__file__).parent.parent / Path('inference') / Path('results') / Path(args.llm_infer_result)
    codes_dir = Path(__file__).parent / Path('codes') / Path(args.codes_dir_name)
    codes_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset('json', split='train', data_files=str(load_path))
    print(dataset)

    for example in tqdm(dataset):
        opt_type = args.opt_type
        lang = example['lang_cluster']
        src_uid = example['src_uid']
        unopt_code = example[f'{opt_type}_baseline_source_code']
        opt0_code = example[f'optimization_0'].split('{"optimized_code": code string}')[-1]
        opt1_code = example[f'optimization_1'].split('{"optimized_code": code string}')[-1]
        opt2_code = example[f'optimization_2'].split('{"optimized_code": code string}')[-1]
        opt3_code = example[f'optimization_3'].split('{"optimized_code": code string}')[-1]
        opt4_code = example[f'optimization_4'].split('{"optimized_code": code string}')[-1]

        # create saved directory of four language clusters codes
        if lang == 'GNU C':
            lang_dir = codes_dir / Path(opt_type) / Path('c')
        elif lang == 'GNU C++':
            lang_dir = codes_dir / Path(opt_type) / Path('cpp')
        elif lang == 'Python 3':
            lang_dir = codes_dir / Path(opt_type) / Path('python')
        elif lang == 'Mono C#':
            lang_dir = codes_dir / Path(opt_type) / Path('cs')
        else:
            print('Language cluster not found, use default language cluster directory.')
            lang_dir = codes_dir
        file_dir = lang_dir / Path(src_uid)
        if not file_dir.is_dir():
            file_dir.mkdir(parents=True, exist_ok=True)
        if args.parse_code==True:
            parselog_path = file_dir / Path('parse.log')
            file = logging.FileHandler(filename=parselog_path, mode='w', encoding='utf-8')
            fmt = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
            file.setFormatter(fmt)
            logger = logging.Logger(name='parse log', level=logging.DEBUG)
            logger.addHandler(file)
            opt0_code = parse_code(opt0_code, logger)
            opt1_code = parse_code(opt1_code, logger)
            opt2_code = parse_code(opt2_code, logger)
            opt3_code = parse_code(opt3_code, logger)
            opt4_code = parse_code(opt4_code, logger)

        # create saved path of four language clusters codes
        if lang == 'GNU C':
            unopt_file_path = file_dir / Path('unopt.c')
            opt0_file_path = file_dir / Path('opt0.c')
            opt1_file_path = file_dir / Path('opt1.c')
            opt2_file_path = file_dir / Path('opt2.c')
            opt3_file_path = file_dir / Path('opt3.c')
            opt4_file_path = file_dir / Path('opt4.c')
        elif lang == 'GNU C++':
            unopt_file_path = file_dir / Path('unopt.cpp')
            opt0_file_path = file_dir / Path('opt0.cpp')
            opt1_file_path = file_dir / Path('opt1.cpp')
            opt2_file_path = file_dir / Path('opt2.cpp')
            opt3_file_path = file_dir / Path('opt3.cpp')
            opt4_file_path = file_dir / Path('opt4.cpp')
        elif lang == 'Python 3':
            unopt_file_path = file_dir / Path('unopt.py')
            opt0_file_path = file_dir / Path('opt0.py')
            opt1_file_path = file_dir / Path('opt1.py')
            opt2_file_path = file_dir / Path('opt2.py')
            opt3_file_path = file_dir / Path('opt3.py')
            opt4_file_path = file_dir / Path('opt4.py')
        elif lang == 'Mono C#':
            unopt_file_path = file_dir / Path('unopt.cs')
            opt0_file_path = file_dir / Path('opt0.cs')
            opt1_file_path = file_dir / Path('opt1.cs')
            opt2_file_path = file_dir / Path('opt2.cs')
            opt3_file_path = file_dir / Path('opt3.cs')
            opt4_file_path = file_dir / Path('opt4.cs')
        else:
            print('Language cluster not found, use default language cluster path.')

        with open(str(unopt_file_path), mode='w', encoding='utf-8') as file:
            file.write(unopt_code)
        with open(str(opt0_file_path), mode='w', encoding='utf-8') as file:
            file.write(opt0_code)
        with open(str(opt1_file_path), mode='w', encoding='utf-8') as file:
            file.write(opt1_code)
        with open(str(opt2_file_path), mode='w', encoding='utf-8') as file:
            file.write(opt2_code)
        with open(str(opt3_file_path), mode='w', encoding='utf-8') as file:
            file.write(opt3_code)
        with open(str(opt4_file_path), mode='w', encoding='utf-8') as file:
            file.write(opt4_code)

def contains_json(text):
    brackets_pattern = r".*\{\s*\"optimized_code\":.*\}.*"
    return re.match(brackets_pattern, text, re.DOTALL)!=None

def get_json(text):
    lpos = text.find("\"optimized_code\"")
    rpos = lpos
    while text.find("}", rpos+1)!=-1:
        rpos = text.find("}", rpos+1)
    json_ret = "{"+text[lpos:rpos].strip()+"}"
    return json_ret

def contain_tick(text):
    tick_pattern = r".*?(`.*`).*?"
    return re.match(tick_pattern, text, re.DOTALL)!=None

def get_tick(text):
    tick_pattern = r".*?(`.*`).*?"
    return re.findall(tick_pattern, text, re.DOTALL)[0][1:-1]

def contains_code_snippets(text):
    pattern = r"```(.+?)```"
    results = re.findall(pattern, text, re.DOTALL)
    if len(results)==0:
        return False
    else:
        return True

def get_code_content(text):
    pattern = r"```(.+?)```"
    results = re.findall(pattern, text, re.DOTALL)
    lang_patterns = [r"^python", r"^java", r"^cpp", r"^csharp", r"^c\+\+", r"^c#", r"^C#", r"^c", r"^C"]
    if any(re.match(lang_pattern, result) 
           for result in results 
           for lang_pattern in lang_patterns):
        for lang_pattern in lang_patterns:
            results = [re.sub(lang_pattern, '', result) for result in results]
    return results

def parse_code(text, logger):
    logging.info(f"start parsing code for:\n{text}")
    if contains_json(text):
        logger.debug("text contains json")
        json_ret = get_json(text)
        try:
            logger.debug(f"try to parse json...")
            code = json.loads(json_ret)['optimized_code']
            logger.debug(f"succeed to parse json")
        except Exception as e:
            logger.debug(f"failed to parse json")
            if contains_code_snippets(json_ret):
                logger.debug("json text contains tripple backtick:```code```")
                code = get_code_content(text)[0].replace("\\\"","\"").replace("\\\\n","\\n").replace("\\\\t","\\t")
            elif contain_tick(json_ret):
                logger.debug("json text contains backtick:`code`")
                code = get_tick(json_ret)
                code = """ """.replace(" ", code)
            else:
                logger.debug("json text does not contain tick, try to select quoted code")
                tmp = json_ret.replace("\"optimized_code\"", "").replace("\"\"\"", "\"")
                lpos = tmp.find("\"")
                rpos = lpos
                while tmp.find("\"", rpos+1)!=-1:
                    rpos = tmp.find("\"", rpos+1)
                code = tmp[lpos+1:rpos].strip()
                code = """ """.replace(" ", code).replace("\\\"","\"").replace("\\\\n","\\n").replace("\\\\t","\\t")
    elif contains_code_snippets(text):
        logger.debug("text contains ```code snippets```")
        code = get_code_content(text)[0]
    else:
        logger.warning("unknown pattern")
        code = text
    logger.info(f"parsed code:\n{code}")
    return code
        
if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    main()
