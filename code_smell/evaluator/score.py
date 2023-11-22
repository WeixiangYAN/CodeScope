import json
import warnings
import numpy as np

from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():
    average = 'weighted'
    lang_cluster_list = ['Java', 'C#']
    smell_list = ['large class', 'long method', 'data class', 'blob', 'feature envy', '']

    load_result_name_list = [
        'code_smell_result_codellama.jsonl',
        'code_smell_result_gpt3-5.jsonl',
        'code_smell_result_gpt4.jsonl',
        'code_smell_result_llama2.jsonl',
        'code_smell_result_palm2.jsonl',
        'code_smell_result_starcoder.jsonl',
        'code_smell_result_vicuna.jsonl',
        'code_smell_result_wizardcoder.jsonl'
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
    for load_result_name in tqdm(load_result_name_list):
        load_result_path = Path(__file__).parent.parent / Path('inference') / Path('results') / Path(load_result_name)
        dataset = load_dataset('json', split='train', data_files=str(load_result_path))

        lang_cluster_dataset_list = [dataset.filter(lambda example: example['lang_cluster'] == lang_cluster) for lang_cluster in lang_cluster_list]

        score_item = {}
        score_item['model'] = model_name_mapping[load_result_name.split('_')[-1].split('.')[0]]
        evaluation_metrics = []
        for lang_cluster, lang_cluster_dataset in zip(lang_cluster_list, lang_cluster_dataset_list):
            references = lang_cluster_dataset['smell']
            predictions = lang_cluster_dataset['predicted_smell']

            accuracy = round(accuracy_score(y_true=references, y_pred=predictions) * 100, 2)
            evaluation_metrics.append(accuracy)
            score_item[f'{lang_cluster.lower()}_accuracy'] = str(accuracy)

            precision = round(precision_score(y_true=references, y_pred=predictions, labels=smell_list, average=average) * 100, 2)
            evaluation_metrics.append(precision)
            score_item[f'{lang_cluster.lower()}_precision'] = str(precision)

            recall = round(recall_score(y_true=references, y_pred=predictions, labels=smell_list, average=average) * 100, 2)
            evaluation_metrics.append(recall)
            score_item[f'{lang_cluster.lower()}_recall'] = str(recall)

            f1 = round(f1_score(y_true=references, y_pred=predictions, labels=smell_list, average=average) * 100, 2)
            evaluation_metrics.append(f1)
            score_item[f'{lang_cluster.lower()}_f1'] = str(f1)

        overall_score = round(float(np.mean(evaluation_metrics)), 2)
        score_item['overall'] = str(overall_score)

        score_dict['data'].append(score_item)

    score_dict['data'].sort(key=lambda x: x['overall'], reverse=True)
    print(score_dict)

    score_dir = Path(__file__).parent / Path('scores')
    if not score_dir.is_dir():
        score_dir.mkdir(parents=True, exist_ok=True)
    save_score_path = score_dir / Path('code_smell_score.json')
    with open(str(save_score_path), mode='w', encoding='utf-8') as file:
        json.dump(score_dict, file, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
