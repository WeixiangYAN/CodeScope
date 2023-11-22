import os.path
import json

import argparse


def count_passed_problems(support_lang_clusters, results_path, results_perl_path='', results_d_path='', results_delphi_path=''):
    record_dict = {}
    for lang in support_lang_clusters:
        record_dict[lang.lower()] = [[], []]

    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            result_content = json.loads(line)
            pass_flag = 1
            for outcome in result_content["testcases"]:
                if outcome["exec_outcome"] != "PASSED":
                    pass_flag = 0
                    break
            if pass_flag:
                if result_content["src_uid"] not in record_dict[result_content["lang_cluster"]][0]:
                    record_dict[result_content["lang_cluster"]][0].append(result_content["src_uid"])
                    record_dict[result_content["lang_cluster"]][1].append(result_content["difficulty"])
    if os.path.exists(results_perl_path):
        with open(results_perl_path, 'r', encoding='utf-8') as rf:
            content = json.load(rf)
            for item in content["accepted"]:
                if content["accepted"][item]["src_uid"] not in record_dict["perl"][0]:
                    record_dict["perl"][0].append(content["accepted"][item]["src_uid"])
                    record_dict["perl"][1].append(int(content["accepted"][item]["difficulty"]))
                else:
                    continue
    if os.path.exists(results_d_path):
        with open(results_d_path, 'r', encoding='utf-8') as rf:
            content = json.load(rf)
            for item in content["accepted"]:
                if content["accepted"][item]["src_uid"] not in record_dict["d"][0]:
                    record_dict["d"][0].append(content["accepted"][item]["src_uid"])
                    record_dict["d"][1].append(int(content["accepted"][item]["difficulty"]))
                else:
                    continue
    if os.path.exists(results_delphi_path):
        with open(results_delphi_path, 'r', encoding='utf-8') as rf:
            content = json.load(rf)
            for item in content["accepted"]:
                if content["accepted"][item]["src_uid"] not in record_dict["delphi"][0]:
                    record_dict["delphi"][0].append(content["accepted"][item]["src_uid"])
                    record_dict["delphi"][1].append(int(content["accepted"][item]["difficulty"]))
                else:
                    continue

    HARD_BAR = 1501
    NON_BAR = 2701
    print("Number of problems solved by:\n")
    for lang in record_dict.keys():
        E_count = 0
        H_count = 0
        for record in record_dict[lang][1]:
            if record < HARD_BAR:
                E_count += 1
            elif record < NON_BAR:
                H_count += 1

        print(lang, " Easy:", E_count, " Hard:", H_count)


def main():

    dir_name = args.result_dir
    name = args.model_name

    results_path = dir_name + "code_repair_eval_" + name + ".jsonl"
    results_perl_path = dir_name + "code_repair_eval_" + name + "_perl.json"
    results_d_path = dir_name + "code_repair_eval_" + name + "_d.json"
    results_delphi_path = dir_name + "code_repair_eval_" + name + "_delphi.json"

    support_lang_clusters = ['C++', 'Java', 'Python', 'C', 'C#', 'Ruby', 'Go', 'JavaScript', 'Kotlin', 'PHP', 'Rust',
                             'Perl', 'D', 'Delphi']

    count_passed_problems(support_lang_clusters, results_path, results_perl_path, results_d_path, results_delphi_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default="execute_results/")
    parser.add_argument('--model_name', type=str, default='gpt4')

    args = parser.parse_args()
    main()