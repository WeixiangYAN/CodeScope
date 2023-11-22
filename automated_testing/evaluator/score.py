import json
import warnings
import numpy as np

from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset


def main():
    lang_cluster_list = ['C', 'C++', 'Java', 'Python']

    load_result_name_list = [
        'automated_testing_result_codellama.jsonl',
        'automated_testing_result_gpt3-5.jsonl',
        'automated_testing_result_gpt4.jsonl',
        'automated_testing_result_llama2.jsonl',
        'automated_testing_result_palm2.jsonl',
        'automated_testing_result_starcoder.jsonl',
        'automated_testing_result_vicuna.jsonl',
        'automated_testing_result_wizardcoder.jsonl'
    ]

    model_name_mapping = {
        'codellama': 'Code LLaMA',
        'gpt3-5': 'GPT-3.5',
        'gpt4': 'GPT-4',
        'llama2': 'LLaMA 2',
        'palm2': 'PaLM 2',
        'starcoder': 'StarCoder',
        'vicuna': 'Vicuna',
        'wizardcoder': 'WizardCoder',
    }

    score_dict = {}
    score_dict['code'] = 0
    score_dict['data'] = []
    for index, load_result_name in tqdm(enumerate(load_result_name_list), total=len(load_result_name_list)):
        load_result_path = Path(__file__).parent.parent / Path('inference') / Path('results') / Path(load_result_name)
        dataset = load_dataset('json', split='train', data_files=str(load_result_path))

        lang_cluster_dataset_list = [dataset.filter(lambda example: example['lang_cluster'] == lang_cluster) for lang_cluster in lang_cluster_list]

        if index == 0:
            score_item = {}
            score_item['model'] = 'Human'
            evaluation_metrics = []
            for lang_cluster, lang_cluster_dataset in zip(lang_cluster_list, lang_cluster_dataset_list):
                pass_rate = round(float(np.mean(lang_cluster_dataset['human_sample_pass_rate'])), 2)
                evaluation_metrics.append(pass_rate)
                score_item[f'{lang_cluster.lower()}_pass_rate'] = str(pass_rate)

                line_coverage = round(float(np.mean(lang_cluster_dataset['human_sample_line_coverage'])), 2)
                evaluation_metrics.append(line_coverage)
                score_item[f'{lang_cluster.lower()}_line_coverage'] = str(line_coverage)

                branch_coverage = round(float(np.mean(lang_cluster_dataset['human_sample_branch_coverage'])), 2)
                evaluation_metrics.append(branch_coverage)
                score_item[f'{lang_cluster.lower()}_branch_coverage'] = str(branch_coverage)

            overall_score = round(float(np.mean(evaluation_metrics)), 2)
            score_item['overall'] = str(overall_score)

            score_dict['data'].append(score_item)

        score_item = {}
        score_item['model'] = model_name_mapping[load_result_name.split('_')[-1].split('.')[0]]
        evaluation_metrics = []
        for lang_cluster, lang_cluster_dataset in zip(lang_cluster_list, lang_cluster_dataset_list):
            pass_rate = round(float(np.mean(lang_cluster_dataset['predicted_pass_rate'])), 2)
            evaluation_metrics.append(pass_rate)
            score_item[f'{lang_cluster.lower()}_pass_rate'] = str(pass_rate)

            line_coverage = round(float(np.mean(lang_cluster_dataset['predicted_line_coverage'])), 2)
            evaluation_metrics.append(line_coverage)
            score_item[f'{lang_cluster.lower()}_line_coverage'] = str(line_coverage)

            branch_coverage = round(float(np.mean(lang_cluster_dataset['predicted_branch_coverage'])), 2)
            evaluation_metrics.append(branch_coverage)
            score_item[f'{lang_cluster.lower()}_branch_coverage'] = str(branch_coverage)

        overall_score = round(float(np.mean(evaluation_metrics)), 2)
        score_item['overall'] = str(overall_score)

        score_dict['data'].append(score_item)

    score_dict['data'].sort(key=lambda x: x['overall'], reverse=True)
    print(score_dict)

    score_dir = Path(__file__).parent / Path('scores')
    if not score_dir.is_dir():
        score_dir.mkdir(parents=True, exist_ok=True)
    save_score_path = score_dir / Path('automated_testing_score.json')
    with open(str(save_score_path), mode='w', encoding='utf-8') as file:
        json.dump(score_dict, file, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
