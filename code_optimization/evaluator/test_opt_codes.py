import os
import re
import argparse
import subprocess
import time
import psutil
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_dataset

import json

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_infer_result', default='mem_code_opt_eval_palm.jsonl', type=str) # will read the inference results that is to be evaluated from '/inference/results/{llm_infer_result}'
    parser.add_argument('--codes_dir_name', default='palm_opt_codes', type=str, choices=['vicuna_opt_codes', 'wizardcoder_opt_codes', 'codellama_opt_codes', 'gpt4_opt_codes', 'gpt3_opt_codes', 'starcoder_opt_codes', 'llama2_opt_codes', 'palm_opt_codes']) # same as --codes_dir_name in save_codes. source codes files to be evaluated should be stored at './codes/{args.codes_dir_name}/{lang_cluster}/{code_uid}
    parser.add_argument('--opt_type', default='time', choices=['mem','time'] ,type=str) 
    parser.add_argument('--retest', default=50, type=int) # number of times to retest the codes' performance
    args = parser.parse_args()
    return args


def count_memory_and_time(command, input=None):
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, shell=True)
    process.stdin.write(input)
    process.stdin.flush()
    cpu_time = 0
    peak_memory = 0
    timeout_flag = False
    timeout_cnt = 0
    while process.poll() is None:
        try:
            p = psutil.Process(process.pid)
            current_memory = p.memory_full_info().uss / 1024
            cpu_time = float(sum(p.cpu_times()[:4]))
        except:
            current_memory = 0
        if current_memory > peak_memory:
            peak_memory = current_memory
        time.sleep(0.0002)
        timeout_cnt += 1
        if timeout_cnt > 25000:
            print("Time limit exceeded!")
            process.kill()
            timeout_flag = True
            break

    return peak_memory, cpu_time, timeout_flag


def execute_command(command, input=None):
    if input is not None:
        input = input.replace('\r\n', '\n')
    try:
        outcome = subprocess.run(command, input=input, capture_output=True, text=True, timeout=20, shell=True)
    except Exception as e:
        print('Error occurred while executing command:', e)
        outcome = subprocess.CompletedProcess(args=command, returncode=-1, stdout='', stderr=str(e))
    return outcome

def test_opt(compile_command, test_command, hidden_unit_tests, src_uid, lang, code_name):
    if compile_command is not None:
        outcome = execute_command(compile_command)
    num_passed = 0
    tmp_testcases_perf = []
    for index, hidden_unit_test in enumerate(hidden_unit_tests):
        input = hidden_unit_test['input']
        output = hidden_unit_test['output'][0]
        outcome = execute_command(test_command, input)
        is_passed = True if outcome.returncode == 0 and (
                outcome.stdout in output 
                or outcome.stdout.rstrip() in output 
                or outcome.stdout.replace('\n','\r\n') in output 
                or outcome.stdout.replace('\n', '\r\n').rstrip() in output
                ) else False
        if is_passed is True:
            num_passed += 1
        peak_mem, cpu_time, timeout_flag = count_memory_and_time(test_command, input)
        if timeout_flag == True:
            timeout_code_uids.append(f'{src_uid}_{code_name}')
            peak_mem = 0
            cpu_time = 0
            tmp_testcases_perf.append([
                code_name, index, src_uid, lang, None, is_passed, cpu_time, peak_mem
            ])
            break
        tmp_testcases_perf.append([
            code_name, index, src_uid, lang, None, is_passed, cpu_time, peak_mem
        ])
    pass_rate = round(100. * num_passed / len(hidden_unit_tests), 2)
    print(f'{code_name} pass rate: {pass_rate}% [{num_passed}/{len(hidden_unit_tests)}]')
    mean_time = np.mean(np.array(tmp_testcases_perf).T[-2]).tolist()
    mean_mem = np.mean(np.array(tmp_testcases_perf).T[-1]).tolist()
    
    return pass_rate, mean_time, mean_mem, tmp_testcases_perf

def test_unopt(compile_command, test_command, hidden_unit_tests, src_uid, lang, code_uid, retest):
    if compile_command is not None:
        outcome = execute_command(compile_command)
    num_passed = 0
    mean_mem_li, mean_time_li = [], []
    tmp_testcases_perf = []
    for index, hidden_unit_test in enumerate(hidden_unit_tests):
        input = hidden_unit_test['input']
        output = hidden_unit_test['output'][0]
        outcome = execute_command(test_command, input)
        is_passed = True if outcome.returncode == 0 and (
                outcome.stdout in output 
                or outcome.stdout.rstrip() in output 
                or outcome.stdout.replace('\n','\r\n') in output 
                or outcome.stdout.replace('\n', '\r\n').rstrip() in output
                ) else False
        if is_passed is True:
            num_passed += 1
        peak_mem_li, cpu_time_li = [], []
        for i in range(retest):
            peak_mem, cpu_time, timeout_flag = count_memory_and_time(test_command, input)
            if timeout_flag == True:
                timeout_code_uids.append(f'{src_uid}_{code_uid}')
                peak_mem_li = [0]*retest
                cpu_time_li = [0]*retest
                break
            peak_mem_li.append(float(peak_mem))
            cpu_time_li.append(float(cpu_time))
        tmp_testcases_perf.append([
            'unopt', index, src_uid, lang, code_uid, is_passed, ','.join([str(x) for x in cpu_time_li]), ','.join([str(x) for x in peak_mem_li])
        ])
        mean_mem_li.append(peak_mem_li)
        mean_time_li.append(cpu_time_li)
    pass_rate = round(100. * num_passed / len(hidden_unit_tests), 2)
    print(f'unopt pass rate: {pass_rate}% [{num_passed}/{len(hidden_unit_tests)}]')
    mean_mem_li = np.mean(mean_mem_li, axis=0) # shape: (n_retest,), calculate the mean testcases peak memory value for each run in the n_retest runs
    mean_time_li = np.mean(mean_time_li, axis=0) # shape: (n_retest,), calculate the mean testcases cpu time value for each run in the n_retest runs
    
    return pass_rate, mean_time_li, mean_mem_li, tmp_testcases_perf

def cal_passrate_perfmetrcs(example):
    src_uid = example['src_uid']
    lang = example['lang_cluster']
    code_uid = f"{example['id']}_{args.opt_type}"
    hidden_unit_tests = example['testcases']
    num_hidden_unit_tests = len(hidden_unit_tests)
    testcases_perf = []
    code_perf = []

    if num_hidden_unit_tests == 0:
        print('Failed to generate hidden unit tests:', code_uid)
    else:
        if lang == 'GNU C':
            os.chdir(str(code_dir / Path(args.opt_type) / Path('c') / Path(src_uid)))
            print(os.getcwd())
            compile_command_template = 'gcc -fno-optimize-sibling-calls -w -fno-strict-aliasing -DONLINE_JUDGE -include limits.h -fno-asm -s -O2 -DONLINE_JUDGE -include math.h -static -lm -o {code_name} {code_name}.c'
            test_command_template = './{code_name}'
        elif lang == 'GNU C++':
            os.chdir(str(code_dir / Path(args.opt_type) / Path('cpp') / Path(src_uid)))
            print(os.getcwd())
            compile_command_template = 'g++ -s -x c++ -O2 -w -DONLINE_JUDGE -include math.h -include limits.h -static -lm -o {code_name} {code_name}.cpp'
            test_command_template = './{code_name}'
        elif lang == 'Mono C#':
            os.chdir(str(code_dir / Path(args.opt_type) / Path('cs') / Path(src_uid)))
            print(os.getcwd())
            compile_command_template = 'csc /out:{code_name} {code_name}.cs'
            test_command_template = 'mono {code_name}'
        elif lang == 'Python 3':
            os.chdir(str(code_dir / Path(args.opt_type) / Path('python') / Path(src_uid)))
            print(os.getcwd())
            compile_command_template = None
            test_command_template = 'python {code_name}.py'
        else:
            print('Unsupported language:', lang)
            return example
        
        if os.path.exists(f'./codes_perf.csv'):
            return example
        
        compile_command = compile_command_template.format(code_name='unopt') if compile_command_template is not None else None
        test_command = test_command_template.format(code_name='unopt')
        pass_rate, mean_time_li, mean_mem_li, tmp_testcases_perf = test_unopt(compile_command, test_command, hidden_unit_tests, src_uid, lang, code_uid, retest)
        testcases_perf.extend(tmp_testcases_perf)
        mean_time_str = ','.join([str(x) for x in mean_time_li])
        mean_mem_str = ','.join([str(x) for x in mean_mem_li])
        code_perf.append([
            'unopt', src_uid, lang, code_uid, pass_rate, mean_time_str, mean_mem_str
        ])

        for code in ['opt0','opt1','opt2','opt3','opt4']:
            compile_command = compile_command_template.format(code_name=code) if compile_command_template is not None else None
            test_command = test_command_template.format(code_name=code)
            pass_rate, mean_time, mean_mem, tmp_testcases_perf = test_opt(compile_command, test_command, hidden_unit_tests, src_uid, lang, code)
            testcases_perf.extend(tmp_testcases_perf)
            code_perf.append([
                code, src_uid, lang, code_uid, pass_rate, mean_time, mean_mem
            ])
        testcases_perf_df = pd.DataFrame(testcases_perf, columns=['code_name','test_idx','src_uid','lang','code_uid','is_passed','cpu_time','peak_mem'])
        testcases_perf_df.to_csv("./testcases_perf.csv", index=False)# stored the testcase-level performance of the code under its code_uid dir
        code_perf_df = pd.DataFrame(code_perf, columns=['code_name','src_uid','lang','code_uid','pass_rate','mean_cpu_time','mean_peak_mem'])
        code_perf_df.to_csv("./codes_perf.csv", index=False)# stored the code-level performance of the code under its code_uid dir
    
    return example


def main():
    dataset = load_dataset('json', split='train', data_files=str(load_path)).select(range(10))
    dataset = dataset.map(cal_passrate_perfmetrcs)



if __name__ == '__main__':
    args = parse_arguments()
    load_path = Path(__file__).parent.parent.parent / Path('results') / Path('raw') / Path(args.llm_infer_result)
    code_dir = Path(__file__).parent.parent.parent / Path('results') / Path('ans') / Path(args.codes_dir_name)
    retest = args.retest
    timeout_code_uids = []
    main()
