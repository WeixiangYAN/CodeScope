import os.path
import json
import pandas as pd
import argparse


def count_translation_passed_problems(results_path, results_perl_path, results_d_path, results_delphi_path,
                                      chart_E_path, chart_H_path, support_lang_clusters):
    record_dict = {}
    for lang in support_lang_clusters:
        record_dict[lang.lower()] = {}
        for from_lang in support_lang_clusters:
            record_dict[lang.lower()][from_lang.lower()] = [[], []]

    result_E_dict = {}
    for lang in support_lang_clusters:
        result_E_dict[lang.lower()] = {}
        for from_lang in support_lang_clusters:
            result_E_dict[lang.lower()][from_lang.lower()] = 0

    result_M_dict = {}
    for lang in support_lang_clusters:
        result_M_dict[lang.lower()] = {}
        for from_lang in support_lang_clusters:
            result_M_dict[lang.lower()][from_lang.lower()] = 0

    HARD_BAR = 1501
    NON_BAR = 2701


    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            result_content = json.loads(line)
            pass_flag = 1
            for outcome in result_content["testcases"]:
                if outcome["exec_outcome"] != "PASSED":
                    pass_flag = 0
            if pass_flag:
                from_lang = result_content["code_uid"].split("*")[1]
                if result_content["src_uid"] not in record_dict[result_content["lang_cluster"]][from_lang][0]:
                    record_dict[result_content["lang_cluster"]][from_lang][0].append(result_content["src_uid"])
                    record_dict[result_content["lang_cluster"]][from_lang][1].append(result_content["difficulty"])
                    if result_content["difficulty"] < HARD_BAR:
                        result_E_dict[result_content["lang_cluster"]][from_lang] += 1
                    elif result_content["difficulty"] < NON_BAR:
                        result_M_dict[result_content["lang_cluster"]][from_lang] += 1

    if os.path.exists(results_perl_path):
        with open(results_perl_path, 'r', encoding='utf-8') as rf:
            content = json.load(rf)
            for item in content["accepted"]:
                lang_cluster = content["accepted"][item]["submission_id"].split("*")[0]
                from_lang = content["accepted"][item]["submission_id"].split("*")[1]
                if int(content["accepted"][item]["difficulty"]) < HARD_BAR:
                    result_E_dict[lang_cluster][from_lang] += 1
                elif int(content["accepted"][item]["difficulty"]) < NON_BAR:
                    result_M_dict[lang_cluster][from_lang] += 1


    if os.path.exists(results_d_path):
        with open(results_d_path, 'r', encoding='utf-8') as rf:
            content = json.load(rf)
            for item in content["accepted"]:
                lang_cluster = content["accepted"][item]["submission_id"].split("*")[0]
                from_lang = content["accepted"][item]["submission_id"].split("*")[1]
                if int(content["accepted"][item]["difficulty"]) < HARD_BAR:
                    result_E_dict[lang_cluster][from_lang] += 1
                elif int(content["accepted"][item]["difficulty"]) < NON_BAR:
                    result_M_dict[lang_cluster][from_lang] += 1


    if os.path.exists(results_delphi_path):
        with open(results_delphi_path, 'r', encoding='utf-8') as rf:
            content = json.load(rf)
            for item in content["accepted"]:
                lang_cluster = content["accepted"][item]["submission_id"].split("*")[0]
                from_lang = content["accepted"][item]["submission_id"].split("*")[1]
                if int(content["accepted"][item]["difficulty"]) < HARD_BAR:
                    result_E_dict[lang_cluster][from_lang] += 1
                elif int(content["accepted"][item]["difficulty"]) < NON_BAR:
                    result_M_dict[lang_cluster][from_lang] += 1

    df = pd.DataFrame(result_E_dict)
    df.to_csv(chart_E_path)

    df = pd.DataFrame(result_M_dict)
    df.to_csv(chart_H_path)

    return record_dict


def main():

    dir_name = args.result_dir
    name = args.model_name

    results_path = dir_name + "code_translation_eval_" + name + ".jsonl"
    results_perl_path = dir_name + "code_translation_eval_" + name + "_perl.json"
    results_d_path = dir_name + "code_translation_eval_" + name + "_d.json"
    results_delphi_path = dir_name + "code_translation_eval_" + name + "_delphi.json"
    chart_E_path = dir_name + "code_translation_eval_" + name + "_E.csv"
    chart_H_path =  dir_name + "code_translation_eval_" + name + "_H.csv"
    support_lang_clusters = ['C++', 'Java', 'Python', 'C', 'C#', 'Ruby', 'Go', 'JavaScript', 'Kotlin', 'PHP', 'Rust',
                             'Perl', 'D', 'Delphi']

    count_translation_passed_problems(results_path, results_perl_path, results_d_path, results_delphi_path,
                                      chart_E_path, chart_H_path, support_lang_clusters)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default="execute_results/")
    parser.add_argument('--model_name', type=str, default='gpt4')

    args = parser.parse_args()
    main()