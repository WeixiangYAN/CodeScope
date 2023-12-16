import json
import warnings
import tiktoken
import evaluate
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
    bleu_metric = evaluate.load('bleu')
    rouge_metric = evaluate.load('rouge')
    bertscore_metric = evaluate.load('bertscore')
    average = 'weighted'
    lang_cluster_list = ['C', 'C#', 'C++', 'Go', 'Java', 'Javascript', 'PHP', 'Python', 'Ruby']
    length_list = ['short', 'medium', 'long']
    diff_tag_list = [0, 1, 2]

    load_result_name_list = [
        'code_review_result_codellama.jsonl',
        'code_review_result_gpt3-5.jsonl',
        'code_review_result_gpt4.jsonl',
        'code_review_result_llama2.jsonl',
        'code_review_result_palm2.jsonl',
        'code_review_result_starcoder.jsonl',
        'code_review_result_vicuna.jsonl',
        'code_review_result_wizardcoder.jsonl'
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

            # import matplotlib.pyplot as plt
            # sorted_tokens = sorted(tokens)
            # print(sorted_tokens)
            # plt.bar(range(len(sorted_tokens)), sorted_tokens)
            # plt.show()

            if lang_cluster == 'C':
                min_tokens = min(tokens)
                max_tokens = 1847
            elif lang_cluster == 'C#':
                min_tokens = 120
                max_tokens = 1634
            elif lang_cluster == 'C++':
                min_tokens = 246
                max_tokens = max(tokens)
            elif lang_cluster == 'Go':
                min_tokens = 184
                max_tokens = 1490
            elif lang_cluster == 'Java':
                min_tokens = 184
                max_tokens = 1617
            elif lang_cluster == 'Javascript':
                min_tokens = min(tokens)
                max_tokens = 1571
            elif lang_cluster == 'PHP':
                min_tokens = min(tokens)
                max_tokens = 1388
            elif lang_cluster == 'Python':
                min_tokens = 168
                max_tokens = max(tokens)
            elif lang_cluster == 'Ruby':
                min_tokens = 93
                max_tokens = 1604
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
            if lang_cluster == 'C':
                length_dataset_list[-1] = concatenate_datasets(
                    [length_dataset_list[-1], lang_cluster_dataset.filter(lambda example: example['tokens'] in [1972, 2062])]
                )
            elif lang_cluster == 'C#':
                length_dataset_list[0] = concatenate_datasets(
                    [length_dataset_list[0], lang_cluster_dataset.filter(lambda example: example['tokens'] in [61, 78, 82])]
                )
                length_dataset_list[-1] = concatenate_datasets(
                    [length_dataset_list[-1], lang_cluster_dataset.filter(lambda example: example['tokens'] in [1778, 1809])]
                )
            elif lang_cluster == 'C++':
                length_dataset_list[0] = concatenate_datasets(
                    [length_dataset_list[0], lang_cluster_dataset.filter(lambda example: example['tokens'] in [139])]
                )
            elif lang_cluster == 'Go':
                length_dataset_list[0] = concatenate_datasets(
                    [length_dataset_list[0], lang_cluster_dataset.filter(lambda example: example['tokens'] in [125])]
                )
                length_dataset_list[-1] = concatenate_datasets(
                    [length_dataset_list[-1], lang_cluster_dataset.filter(lambda example: example['tokens'] in [1606, 1766, 1781, 1969, 2164, 2242])]
                )
            elif lang_cluster == 'Java':
                length_dataset_list[0] = concatenate_datasets(
                    [length_dataset_list[0], lang_cluster_dataset.filter(lambda example: example['tokens'] in [81])]
                )
                length_dataset_list[-1] = concatenate_datasets(
                    [length_dataset_list[-1], lang_cluster_dataset.filter(lambda example: example['tokens'] in [1707, 1782, 1852, 1869])]
                )
            elif lang_cluster == 'Javascript':
                length_dataset_list[-1] = concatenate_datasets(
                    [length_dataset_list[-1], lang_cluster_dataset.filter(lambda example: example['tokens'] in [1975])]
                )
            elif lang_cluster == 'PHP':
                length_dataset_list[-1] = concatenate_datasets(
                    [length_dataset_list[-1], lang_cluster_dataset.filter(lambda example: example['tokens'] in [1512, 1643, 1646, 1890, 2246, 2256, 2397])]
                )
            elif lang_cluster == 'Python':
                length_dataset_list[0] = concatenate_datasets(
                    [length_dataset_list[0], lang_cluster_dataset.filter(lambda example: example['tokens'] in [31])]
                )
            elif lang_cluster == 'Ruby':
                length_dataset_list[0] = concatenate_datasets(
                    [length_dataset_list[0], lang_cluster_dataset.filter(lambda example: example['tokens'] in [8, 20])]
                )
                length_dataset_list[-1] = concatenate_datasets(
                    [length_dataset_list[-1], lang_cluster_dataset.filter(lambda example: example['tokens'] in [1919, 2573])]
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

                diff_tag_references = length_dataset['diff_tag']
                diff_tag_predictions = length_dataset['predicted_diff_tag']

                accuracy = round(accuracy_score(y_true=diff_tag_references, y_pred=diff_tag_predictions) * 100, 2)
                score_item['metrics'][lang_cluster.lower()][length]['accuracy'] = str(accuracy)

                precision = round(precision_score(y_true=diff_tag_references, y_pred=diff_tag_predictions, labels=diff_tag_list, average=average) * 100, 2)
                score_item['metrics'][lang_cluster.lower()][length]['precision'] = str(precision)

                recall = round(recall_score(y_true=diff_tag_references, y_pred=diff_tag_predictions, labels=diff_tag_list, average=average) * 100, 2)
                score_item['metrics'][lang_cluster.lower()][length]['recall'] = str(recall)

                f1 = round(f1_score(y_true=diff_tag_references, y_pred=diff_tag_predictions, labels=diff_tag_list, average=average) * 100, 2)
                score_item['metrics'][lang_cluster.lower()][length]['f1'] = str(f1)

                filtered_dataset = length_dataset.filter(lambda example: example['diff_tag'] == 1)
                review_comment_references = filtered_dataset['review_comment']
                review_comment_predictions = filtered_dataset['predicted_review_comment']

                bleu_results = bleu_metric.compute(predictions=review_comment_predictions, references=review_comment_references)
                bleu = round(bleu_results['bleu'] * 100, 2)
                score_item['metrics'][lang_cluster.lower()][length]['bleu'] = str(bleu)

                rouge_results = rouge_metric.compute(predictions=review_comment_predictions, references=review_comment_references)
                rouge = round(rouge_results['rougeL'] * 100, 2)
                score_item['metrics'][lang_cluster.lower()][length]['rouge'] = str(rouge)

                bertscore_results = bertscore_metric.compute(predictions=review_comment_predictions, references=review_comment_references, lang='en')
                bertscore = round(np.mean(bertscore_results['f1']) * 100, 2)
                score_item['metrics'][lang_cluster.lower()][length]['bertscore'] = str(bertscore)

        evaluation_metrics = []
        for length in length_list:
            length_metrics = []

            for lang_cluster in lang_cluster_list:
                length_metrics.append(float(score_item['metrics'][lang_cluster.lower()][length]['accuracy']))
                length_metrics.append(float(score_item['metrics'][lang_cluster.lower()][length]['precision']))
                length_metrics.append(float(score_item['metrics'][lang_cluster.lower()][length]['recall']))
                length_metrics.append(float(score_item['metrics'][lang_cluster.lower()][length]['f1']))
                length_metrics.append(float(score_item['metrics'][lang_cluster.lower()][length]['bleu']))
                length_metrics.append(float(score_item['metrics'][lang_cluster.lower()][length]['rouge']))
                length_metrics.append(float(score_item['metrics'][lang_cluster.lower()][length]['bertscore']))

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
    save_score_path = score_dir / Path('code_review_score_1.json')
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

            if lang_cluster == 'C':
                min_tokens = min(tokens)
                max_tokens = 1847
            elif lang_cluster == 'C#':
                min_tokens = 120
                max_tokens = 1634
            elif lang_cluster == 'C++':
                min_tokens = 246
                max_tokens = max(tokens)
            elif lang_cluster == 'Go':
                min_tokens = 184
                max_tokens = 1490
            elif lang_cluster == 'Java':
                min_tokens = 184
                max_tokens = 1617
            elif lang_cluster == 'Javascript':
                min_tokens = min(tokens)
                max_tokens = 1571
            elif lang_cluster == 'PHP':
                min_tokens = min(tokens)
                max_tokens = 1388
            elif lang_cluster == 'Python':
                min_tokens = 168
                max_tokens = max(tokens)
            elif lang_cluster == 'Ruby':
                min_tokens = 93
                max_tokens = 1604
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
            if lang_cluster == 'C':
                length_dataset_list[-1] = concatenate_datasets(
                    [length_dataset_list[-1], lang_cluster_dataset.filter(lambda example: example['tokens'] in [1972, 2062])]
                )
            elif lang_cluster == 'C#':
                length_dataset_list[0] = concatenate_datasets(
                    [length_dataset_list[0], lang_cluster_dataset.filter(lambda example: example['tokens'] in [61, 78, 82])]
                )
                length_dataset_list[-1] = concatenate_datasets(
                    [length_dataset_list[-1], lang_cluster_dataset.filter(lambda example: example['tokens'] in [1778, 1809])]
                )
            elif lang_cluster == 'C++':
                length_dataset_list[0] = concatenate_datasets(
                    [length_dataset_list[0], lang_cluster_dataset.filter(lambda example: example['tokens'] in [139])]
                )
            elif lang_cluster == 'Go':
                length_dataset_list[0] = concatenate_datasets(
                    [length_dataset_list[0], lang_cluster_dataset.filter(lambda example: example['tokens'] in [125])]
                )
                length_dataset_list[-1] = concatenate_datasets(
                    [length_dataset_list[-1], lang_cluster_dataset.filter(lambda example: example['tokens'] in [1606, 1766, 1781, 1969, 2164, 2242])]
                )
            elif lang_cluster == 'Java':
                length_dataset_list[0] = concatenate_datasets(
                    [length_dataset_list[0], lang_cluster_dataset.filter(lambda example: example['tokens'] in [81])]
                )
                length_dataset_list[-1] = concatenate_datasets(
                    [length_dataset_list[-1], lang_cluster_dataset.filter(lambda example: example['tokens'] in [1707, 1782, 1852, 1869])]
                )
            elif lang_cluster == 'Javascript':
                length_dataset_list[-1] = concatenate_datasets(
                    [length_dataset_list[-1], lang_cluster_dataset.filter(lambda example: example['tokens'] in [1975])]
                )
            elif lang_cluster == 'PHP':
                length_dataset_list[-1] = concatenate_datasets(
                    [length_dataset_list[-1], lang_cluster_dataset.filter(lambda example: example['tokens'] in [1512, 1643, 1646, 1890, 2246, 2256, 2397])]
                )
            elif lang_cluster == 'Python':
                length_dataset_list[0] = concatenate_datasets(
                    [length_dataset_list[0], lang_cluster_dataset.filter(lambda example: example['tokens'] in [31])]
                )
            elif lang_cluster == 'Ruby':
                length_dataset_list[0] = concatenate_datasets(
                    [length_dataset_list[0], lang_cluster_dataset.filter(lambda example: example['tokens'] in [8, 20])]
                )
                length_dataset_list[-1] = concatenate_datasets(
                    [length_dataset_list[-1], lang_cluster_dataset.filter(lambda example: example['tokens'] in [1919, 2573])]
                )
            print(length_dataset_list)

            num_rows = 0
            for length_dataset in length_dataset_list:
                num_rows += length_dataset.num_rows
            if num_rows != lang_cluster_dataset.num_rows:
                raise Exception

            for length, length_dataset in zip(length_list, length_dataset_list):
                diff_tag_references = length_dataset['diff_tag']
                diff_tag_predictions = length_dataset['predicted_diff_tag']

                accuracy = round(accuracy_score(y_true=diff_tag_references, y_pred=diff_tag_predictions) * 100, 2)
                evaluation_metrics.append(accuracy)
                score_item[f'{lang_cluster.lower()}_{length}_accuracy'] = str(accuracy)

                precision = round(precision_score(y_true=diff_tag_references, y_pred=diff_tag_predictions, labels=diff_tag_list, average=average) * 100, 2)
                evaluation_metrics.append(precision)
                score_item[f'{lang_cluster.lower()}_{length}_precision'] = str(precision)

                recall = round(recall_score(y_true=diff_tag_references, y_pred=diff_tag_predictions, labels=diff_tag_list, average=average) * 100, 2)
                evaluation_metrics.append(recall)
                score_item[f'{lang_cluster.lower()}_{length}_recall'] = str(recall)

                f1 = round(f1_score(y_true=diff_tag_references, y_pred=diff_tag_predictions, labels=diff_tag_list, average=average) * 100, 2)
                evaluation_metrics.append(f1)
                score_item[f'{lang_cluster.lower()}_{length}_f1'] = str(f1)

                filtered_dataset = length_dataset.filter(lambda example: example['diff_tag'] == 1)
                review_comment_references = filtered_dataset['review_comment']
                review_comment_predictions = filtered_dataset['predicted_review_comment']

                bleu_results = bleu_metric.compute(predictions=review_comment_predictions, references=review_comment_references)
                bleu = round(bleu_results['bleu'] * 100, 2)
                evaluation_metrics.append(bleu)
                score_item[f'{lang_cluster.lower()}_{length}_bleu'] = str(bleu)

                rouge_results = rouge_metric.compute(predictions=review_comment_predictions, references=review_comment_references)
                rouge = round(rouge_results['rougeL'] * 100, 2)
                evaluation_metrics.append(rouge)
                score_item[f'{lang_cluster.lower()}_{length}_rouge'] = str(rouge)

                bertscore_results = bertscore_metric.compute(predictions=review_comment_predictions, references=review_comment_references, lang='en')
                bertscore = round(np.mean(bertscore_results['f1']) * 100, 2)
                evaluation_metrics.append(bertscore)
                score_item[f'{lang_cluster.lower()}_{length}_bertscore'] = str(bertscore)

        print(score_item)
        overall_score = round(float(np.mean(evaluation_metrics)), 2)
        score_item['overall'] = str(overall_score)

        score_dict['data'].append(score_item)

    score_dict['data'].sort(key=lambda x: x['overall'], reverse=True)
    print(score_dict)

    score_dir = Path(__file__).parent / Path('scores')
    if not score_dir.is_dir():
        score_dir.mkdir(parents=True, exist_ok=True)
    save_score_path = score_dir / Path('code_review_score_2.json')
    with open(str(save_score_path), mode='w', encoding='utf-8') as file:
        json.dump(score_dict, file, ensure_ascii=False, indent=2)

    print('Table 3:')
    score_dict = {}
    score_dict['code'] = 0
    score_dict['data'] = []
    for load_result_name in tqdm(load_result_name_list):
        load_result_path = Path(__file__).parent.parent / Path('inference') / Path('results') / Path(load_result_name)
        dataset = load_dataset('json', split='train', data_files=str(load_result_path))

        score_item = {}
        score_item['model'] = model_name_mapping[load_result_name.split('_')[-1].split('.')[0]]
        evaluation_metrics = []

        diff_tag_references = dataset['diff_tag']
        diff_tag_predictions = dataset['predicted_diff_tag']

        accuracy = round(accuracy_score(y_true=diff_tag_references, y_pred=diff_tag_predictions) * 100, 2)
        evaluation_metrics.append(accuracy)
        score_item['accuracy'] = str(accuracy)

        precision = round(precision_score(y_true=diff_tag_references, y_pred=diff_tag_predictions, labels=diff_tag_list, average=average) * 100, 2)
        evaluation_metrics.append(precision)
        score_item['precision'] = str(precision)

        recall = round(recall_score(y_true=diff_tag_references, y_pred=diff_tag_predictions, labels=diff_tag_list, average=average) * 100, 2)
        evaluation_metrics.append(recall)
        score_item['recall'] = str(recall)

        f1 = round(f1_score(y_true=diff_tag_references, y_pred=diff_tag_predictions, labels=diff_tag_list, average=average) * 100, 2)
        evaluation_metrics.append(f1)
        score_item['f1'] = str(f1)

        filtered_dataset = dataset.filter(lambda example: example['diff_tag'] == 1)
        review_comment_references = filtered_dataset['review_comment']
        review_comment_predictions = filtered_dataset['predicted_review_comment']

        bleu_results = bleu_metric.compute(predictions=review_comment_predictions, references=review_comment_references)
        bleu = round(bleu_results['bleu'] * 100, 2)
        evaluation_metrics.append(bleu)
        score_item['bleu'] = str(bleu)

        rouge_results = rouge_metric.compute(predictions=review_comment_predictions, references=review_comment_references)
        rouge = round(rouge_results['rougeL'] * 100, 2)
        evaluation_metrics.append(rouge)
        score_item['rouge'] = str(rouge)

        bertscore_results = bertscore_metric.compute(predictions=review_comment_predictions, references=review_comment_references, lang='en')
        bertscore = round(np.mean(bertscore_results['f1']) * 100, 2)
        evaluation_metrics.append(bertscore)
        score_item['bertscore'] = str(bertscore)

        overall_score = round(float(np.mean(evaluation_metrics)), 2)
        score_item['overall'] = str(overall_score)

        score_dict['data'].append(score_item)

    score_dict['data'].sort(key=lambda x: x['overall'], reverse=True)
    print(score_dict)

    score_dir = Path(__file__).parent / Path('scores')
    if not score_dir.is_dir():
        score_dir.mkdir(parents=True, exist_ok=True)
    save_score_path = score_dir / Path('code_review_score_3.json')
    with open(str(save_score_path), mode='w', encoding='utf-8') as file:
        json.dump(score_dict, file, ensure_ascii=False, indent=2)

    print('Table 4:')
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
            diff_tag_references = lang_cluster_dataset['diff_tag']
            diff_tag_predictions = lang_cluster_dataset['predicted_diff_tag']

            accuracy = round(accuracy_score(y_true=diff_tag_references, y_pred=diff_tag_predictions) * 100, 2)
            evaluation_metrics.append(accuracy)
            score_item[f'{lang_cluster.lower()}_accuracy'] = str(accuracy)

            precision = round(precision_score(y_true=diff_tag_references, y_pred=diff_tag_predictions, labels=diff_tag_list, average=average) * 100, 2)
            evaluation_metrics.append(precision)
            score_item[f'{lang_cluster.lower()}_precision'] = str(precision)

            recall = round(recall_score(y_true=diff_tag_references, y_pred=diff_tag_predictions, labels=diff_tag_list, average=average) * 100, 2)
            evaluation_metrics.append(recall)
            score_item[f'{lang_cluster.lower()}_recall'] = str(recall)

            f1 = round(f1_score(y_true=diff_tag_references, y_pred=diff_tag_predictions, labels=diff_tag_list, average=average) * 100, 2)
            evaluation_metrics.append(f1)
            score_item[f'{lang_cluster.lower()}_f1'] = str(f1)

            filtered_lang_cluster_dataset = lang_cluster_dataset.filter(lambda example: example['diff_tag'] == 1)
            review_comment_references = filtered_lang_cluster_dataset['review_comment']
            review_comment_predictions = filtered_lang_cluster_dataset['predicted_review_comment']

            bleu_results = bleu_metric.compute(predictions=review_comment_predictions, references=review_comment_references)
            bleu = round(bleu_results['bleu'] * 100, 2)
            evaluation_metrics.append(bleu)
            score_item[f'{lang_cluster.lower()}_bleu'] = str(bleu)

            rouge_results = rouge_metric.compute(predictions=review_comment_predictions, references=review_comment_references)
            rouge = round(rouge_results['rougeL'] * 100, 2)
            evaluation_metrics.append(rouge)
            score_item[f'{lang_cluster.lower()}_rouge'] = str(rouge)

            bertscore_results = bertscore_metric.compute(predictions=review_comment_predictions, references=review_comment_references, lang='en')
            bertscore = round(np.mean(bertscore_results['f1']) * 100, 2)
            evaluation_metrics.append(bertscore)
            score_item[f'{lang_cluster.lower()}_bertscore'] = str(bertscore)

        overall_score = round(float(np.mean(evaluation_metrics)), 2)
        score_item['overall'] = str(overall_score)

        score_dict['data'].append(score_item)

    score_dict['data'].sort(key=lambda x: x['overall'], reverse=True)
    print(score_dict)

    score_dir = Path(__file__).parent / Path('scores')
    if not score_dir.is_dir():
        score_dir.mkdir(parents=True, exist_ok=True)
    save_score_path = score_dir / Path('code_review_score_4.json')
    with open(str(save_score_path), mode='w', encoding='utf-8') as file:
        json.dump(score_dict, file, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
