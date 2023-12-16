import json
import warnings
import tiktoken
import numpy as np

from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def count_tokens(content):
    encoding = tiktoken.get_encoding('cl100k_base')
    num_tokens = len(encoding.encode(content))

    return num_tokens


def add_tokens(example):
    example['tokens'] = count_tokens(example['source_code'])

    return example


def main():
    average = 'weighted'
    lang_cluster_list = ['Java', 'C#']
    length_list = ['short', 'medium', 'long']
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

    print('Table 1:')
    score_dict = {}
    score_dict['code'] = 0
    score_dict['data'] = []
    for load_result_name in tqdm(load_result_name_list):
        load_result_path = Path(__file__).parent.parent / Path('inference') / Path('results') / Path(load_result_name)
        dataset = load_dataset('json', split='train', data_files=str(load_result_path))
        dataset = dataset.map(add_tokens)

        lang_cluster_dataset_list = [dataset.filter(lambda example: example['lang_cluster'] == lang_cluster) for lang_cluster in lang_cluster_list]

        score_item = {}
        score_item['model'] = model_name_mapping[load_result_name.split('_')[-1].split('.')[0]]
        score_item['metrics'] = {}
        for lang_cluster, lang_cluster_dataset in zip(lang_cluster_list, lang_cluster_dataset_list):
            tokens = lang_cluster_dataset['tokens']

            import matplotlib.pyplot as plt
            sorted_tokens = sorted(tokens)
            print(sorted_tokens)
            plt.bar(range(len(sorted_tokens)), sorted_tokens)
            plt.show()

            if lang_cluster == 'Java':
                min_tokens = 205
                max_tokens = 1585
            elif lang_cluster == 'C#':
                min_tokens = 165
                max_tokens = 1572
            else:
                min_tokens = min(tokens)
                max_tokens = max(tokens)
            interval_length = int((max_tokens - min_tokens) / 3)
            interval_list = [
                range(min_tokens, min_tokens + interval_length),
                range(min_tokens + interval_length, min_tokens + interval_length + interval_length),
                range(min_tokens + interval_length + interval_length, max_tokens + 1)
            ]
            print(interval_list)
            final_interval_list = [
                range(min(tokens), min_tokens + interval_length),
                range(min_tokens + interval_length, min_tokens + interval_length + interval_length),
                range(min_tokens + interval_length + interval_length, max(tokens))
            ]
            print(final_interval_list)

            length_dataset_list = [lang_cluster_dataset.filter(lambda example: example['tokens'] in interval) for interval in interval_list]
            if lang_cluster == 'Java':
                length_dataset_list[0] = concatenate_datasets(
                    [length_dataset_list[0], lang_cluster_dataset.filter(lambda example: example['tokens'] in [22, 38, 79, 100])]
                )
                length_dataset_list[-1] = concatenate_datasets(
                    [length_dataset_list[-1], lang_cluster_dataset.filter(lambda example: example['tokens'] in [1725])]
                )
            elif lang_cluster == 'C#':
                length_dataset_list[0] = concatenate_datasets(
                    [length_dataset_list[0], lang_cluster_dataset.filter(lambda example: example['tokens'] in [22, 44, 63, 64, 98, 98, 103])]
                )
                length_dataset_list[-1] = concatenate_datasets(
                    [length_dataset_list[-1], lang_cluster_dataset.filter(lambda example: example['tokens'] in [2113])]
                )
            print(length_dataset_list)

            num_rows = 0
            for length_dataset in length_dataset_list:
                num_rows += length_dataset.num_rows
            if num_rows != lang_cluster_dataset.num_rows:
                raise Exception

            score_item['metrics'][lang_cluster.lower()] = {}
            for length, length_dataset in zip(length_list, length_dataset_list):
                score_item['metrics'][lang_cluster.lower()][length] = {}

                references = length_dataset['smell']
                predictions = length_dataset['predicted_smell']

                accuracy = round(accuracy_score(y_true=references, y_pred=predictions) * 100, 2)
                score_item['metrics'][lang_cluster.lower()][length]['accuracy'] = str(accuracy)

                precision = round(precision_score(y_true=references, y_pred=predictions, labels=smell_list, average=average) * 100, 2)
                score_item['metrics'][lang_cluster.lower()][length]['precision'] = str(precision)

                recall = round(recall_score(y_true=references, y_pred=predictions, labels=smell_list, average=average) * 100, 2)
                score_item['metrics'][lang_cluster.lower()][length]['recall'] = str(recall)

                f1 = round(f1_score(y_true=references, y_pred=predictions, labels=smell_list, average=average) * 100, 2)
                score_item['metrics'][lang_cluster.lower()][length]['f1'] = str(f1)

        evaluation_metrics = []
        for length in length_list:
            length_metrics = []

            for lang_cluster in lang_cluster_list:
                length_metrics.append(float(score_item['metrics'][lang_cluster.lower()][length]['accuracy']))
                length_metrics.append(float(score_item['metrics'][lang_cluster.lower()][length]['precision']))
                length_metrics.append(float(score_item['metrics'][lang_cluster.lower()][length]['recall']))
                length_metrics.append(float(score_item['metrics'][lang_cluster.lower()][length]['f1']))

            length_score = round(float(np.mean(length_metrics)), 2)
            evaluation_metrics.append(length_score)
            score_item[f'{length}'] = str(length_score)

        print(score_item)
        overall_score = round(float(np.mean(evaluation_metrics)), 2)
        score_item['overall'] = str(overall_score)
        del score_item['metrics']

        score_dict['data'].append(score_item)

    score_dict['data'].sort(key=lambda x: x['overall'], reverse=True)
    print(score_dict)

    score_dir = Path(__file__).parent / Path('scores')
    if not score_dir.is_dir():
        score_dir.mkdir(parents=True, exist_ok=True)
    save_score_path = score_dir / Path('code_smell_score_1.json')
    with open(str(save_score_path), mode='w', encoding='utf-8') as file:
        json.dump(score_dict, file, ensure_ascii=False, indent=2)

    print('Table 2:')
    score_dict = {}
    score_dict['code'] = 0
    score_dict['data'] = []
    for load_result_name in tqdm(load_result_name_list):
        load_result_path = Path(__file__).parent.parent / Path('inference') / Path('results') / Path(load_result_name)
        dataset = load_dataset('json', split='train', data_files=str(load_result_path))
        dataset = dataset.map(add_tokens)

        lang_cluster_dataset_list = [dataset.filter(lambda example: example['lang_cluster'] == lang_cluster) for lang_cluster in lang_cluster_list]

        score_item = {}
        score_item['model'] = model_name_mapping[load_result_name.split('_')[-1].split('.')[0]]
        evaluation_metrics = []
        for lang_cluster, lang_cluster_dataset in zip(lang_cluster_list, lang_cluster_dataset_list):
            tokens = lang_cluster_dataset['tokens']

            # import matplotlib.pyplot as plt
            # sorted_tokens = sorted(tokens)
            # print(sorted_tokens)
            # plt.bar(range(len(sorted_tokens)), sorted_tokens)
            # plt.show()

            if lang_cluster == 'Java':
                min_tokens = 205
                max_tokens = 1585
            elif lang_cluster == 'C#':
                min_tokens = 165
                max_tokens = 1572
            else:
                min_tokens = min(tokens)
                max_tokens = max(tokens)
            interval_length = int((max_tokens - min_tokens) / 3)
            interval_list = [
                range(min_tokens, min_tokens + interval_length),
                range(min_tokens + interval_length, min_tokens + interval_length + interval_length),
                range(min_tokens + interval_length + interval_length, max_tokens + 1)
            ]
            print(interval_list)
            final_interval_list = [
                range(min(tokens), min_tokens + interval_length),
                range(min_tokens + interval_length, min_tokens + interval_length + interval_length),
                range(min_tokens + interval_length + interval_length, max(tokens))
            ]
            print(final_interval_list)

            length_dataset_list = [lang_cluster_dataset.filter(lambda example: example['tokens'] in interval) for interval in interval_list]
            if lang_cluster == 'Java':
                length_dataset_list[0] = concatenate_datasets(
                    [length_dataset_list[0], lang_cluster_dataset.filter(lambda example: example['tokens'] in [22, 38, 79, 100])]
                )
                length_dataset_list[-1] = concatenate_datasets(
                    [length_dataset_list[-1], lang_cluster_dataset.filter(lambda example: example['tokens'] in [1725])]
                )
            elif lang_cluster == 'C#':
                length_dataset_list[0] = concatenate_datasets(
                    [length_dataset_list[0], lang_cluster_dataset.filter(lambda example: example['tokens'] in [22, 44, 63, 64, 98, 98, 103])]
                )
                length_dataset_list[-1] = concatenate_datasets(
                    [length_dataset_list[-1], lang_cluster_dataset.filter(lambda example: example['tokens'] in [2113])]
                )
            print(length_dataset_list)

            num_rows = 0
            for length_dataset in length_dataset_list:
                num_rows += length_dataset.num_rows
            if num_rows != lang_cluster_dataset.num_rows:
                raise Exception

            for length, length_dataset in zip(length_list, length_dataset_list):
                references = length_dataset['smell']
                predictions = length_dataset['predicted_smell']

                accuracy = round(accuracy_score(y_true=references, y_pred=predictions) * 100, 2)
                evaluation_metrics.append(accuracy)
                score_item[f'{lang_cluster.lower()}_{length}_accuracy'] = str(accuracy)

                precision = round(precision_score(y_true=references, y_pred=predictions, labels=smell_list, average=average) * 100, 2)
                evaluation_metrics.append(precision)
                score_item[f'{lang_cluster.lower()}_{length}_precision'] = str(precision)

                recall = round(recall_score(y_true=references, y_pred=predictions, labels=smell_list, average=average) * 100, 2)
                evaluation_metrics.append(recall)
                score_item[f'{lang_cluster.lower()}_{length}_recall'] = str(recall)

                f1 = round(f1_score(y_true=references, y_pred=predictions, labels=smell_list, average=average) * 100, 2)
                evaluation_metrics.append(f1)
                score_item[f'{lang_cluster.lower()}_{length}_f1'] = str(f1)

        print(score_item)
        overall_score = round(float(np.mean(evaluation_metrics)), 2)
        score_item['overall'] = str(overall_score)

        score_dict['data'].append(score_item)

    score_dict['data'].sort(key=lambda x: x['overall'], reverse=True)
    print(score_dict)

    score_dir = Path(__file__).parent / Path('scores')
    if not score_dir.is_dir():
        score_dir.mkdir(parents=True, exist_ok=True)
    save_score_path = score_dir / Path('code_smell_score_2.json')
    with open(str(save_score_path), mode='w', encoding='utf-8') as file:
        json.dump(score_dict, file, ensure_ascii=False, indent=2)

    print('Table 3:')
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
    save_score_path = score_dir / Path('code_smell_score_3.json')
    with open(str(save_score_path), mode='w', encoding='utf-8') as file:
        json.dump(score_dict, file, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
