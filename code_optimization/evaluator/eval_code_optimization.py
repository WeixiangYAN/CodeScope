import os
from pathlib import Path
import pandas as pd
import re
import argparse
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--codes_dir_name', default='palm_opt_codes', type=str, choices=['vicuna_opt_codes', 'wizardcoder_opt_codes', 'codellama_opt_codes', 'gpt4_opt_codes', 'gpt3_opt_codes', 'starcoder_opt_codes', 'llama2_opt_codes', 'palm_opt_codes']) 
    args = parser.parse_args()
    return args

def main():
    metrcs_df = []
    print(f'Start to evaluate optimized codes performance for {model_name}...\n')
    for lang in ['c','cpp','python','cs']:
        for opt_type in ['mem','time']:
            perf_key = 'mean_peak_mem' if opt_type == 'mem' else 'mean_cpu_time'
            perf_unit = 'kb' if opt_type == 'mem' else 's'
            err_flag = False# In case some codes are not fully tested in test_opt_codes.py (determined by whether codes_perf.csv exists)
            pass_rate = 0
            success_opt_rate = 0
            lang_dir = load_dir / Path(model_name) / Path(opt_type) / Path(lang)
            src_dirs = os.listdir(lang_dir)
            for src_dir in src_dirs:
                print(f">> calculating {lang}-{src_dir} performance metrics:")
                if not os.path.exists(lang_dir / Path(src_dir) / Path('codes_perf.csv')):
                    print('------------------------------------')
                    print(f"{lang}-{opt_type} is not fully tested")
                    print('------------------------------------')
                    err_flag = True
                    break
                df = pd.read_csv(lang_dir / Path(src_dir) / Path('codes_perf.csv'),index_col=0)

                unopt = df.loc['unopt']
                if opt_type == 'mem':
                    perf_li = [float(x) for x in unopt['mean_peak_mem'].split(',')]
                elif opt_type == 'time':
                    perf_li = [float(x) for x in unopt['mean_cpu_time'].split(',')]
                unopt_low_bound = min(perf_li)
                unopt_perf = sum(perf_li)/len(perf_li)
                unopt_perf_dev = np.std(perf_li)

                print(f"unopt {opt_type} performance: {unopt_perf}+-{unopt_perf_dev} {perf_unit}")
                
                # obtain the performance of generated optimized code，calculate the pass rate and success optimization rate
                pass_flag = False
                opt_flag = False
                for opt_idx in range(5):
                    opt = df.loc[f'opt{opt_idx}']
                    if opt['pass_rate'] == 100.0:
                        pass_flag = True
                        opt_perf = float(opt[perf_key])
                        if opt_perf < unopt_low_bound:
                            print(f'opt{opt_idx} {opt_type} performance: {opt_perf}. success to opt {unopt_perf - opt_perf} {perf_unit}, opt rate: {(unopt_perf - opt_perf) / unopt_perf}%')
                            opt_flag = True
                        else:
                            print(f'opt{opt_idx} {opt_type} performance: {opt_perf}. failed to opt')
                if pass_flag:
                    pass_rate += 1
                if opt_flag:
                    success_opt_rate += 1
            if err_flag:
                continue
            print('------------------------------------')
            print(f'{lang} {opt_type} pass_rate: {pass_rate/len(src_dirs)}')
            print(f'{lang} {opt_type} success_opt_rate: {success_opt_rate/len(src_dirs)}')
            print('------------------------------------')
            metrcs_df.append([model_name,lang,opt_type,pass_rate/len(src_dirs),success_opt_rate/len(src_dirs)])
    metrcs_df = pd.DataFrame(metrcs_df,columns=['model_name','lang','opt_type','pass_rate','success_opt_rate'])
    metrcs_df.to_csv(output_dir / Path(f'{model_name}_metrics.csv'),index=False)
if __name__ == '__main__':
    args = parse_arguments()
    model_name = args.codes_dir_name
    load_dir = Path(__file__).parent / Path('codes')
    output_dir = Path(__file__).parent / Path('opt_scores')
    output_dir.mkdir(exist_ok=True)
    main()


