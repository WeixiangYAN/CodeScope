import json
import warnings
import tiktoken
import numpy as np

from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset, concatenate_datasets


def count_tokens(content):
    encoding = tiktoken.get_encoding('cl100k_base')
    num_tokens = len(encoding.encode(content))

    return num_tokens


def add_tokens(example):
    example['tokens'] = count_tokens(example['source_code'])

    return example


def main():
    lang_cluster_list = ['Python', 'Java', 'C', 'C++']
    length_list = ['short', 'medium', 'long']

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

    print('Table 1:')
    score_dict = {}
    score_dict['code'] = 0
    score_dict['data'] = []
    for index, load_result_name in tqdm(enumerate(load_result_name_list), total=len(load_result_name_list)):
        load_result_path = Path(__file__).parent.parent / Path('inference') / Path('results') / Path(load_result_name)
        dataset = load_dataset('json', split='train', data_files=str(load_result_path))
        dataset = dataset.map(add_tokens)

        lang_cluster_dataset_list = [dataset.filter(lambda example: example['lang_cluster'] == lang_cluster) for lang_cluster in lang_cluster_list]

        if index == 0:
            score_item = {}
            score_item['model'] = 'Human'
            score_item['metrics'] = {}
            for lang_cluster, lang_cluster_dataset in zip(lang_cluster_list, lang_cluster_dataset_list):
                tokens = lang_cluster_dataset['tokens']

                # import matplotlib.pyplot as plt
                # sorted_tokens = sorted(tokens)
                # print(sorted_tokens)
                # plt.bar(range(len(sorted_tokens)), sorted_tokens)
                # plt.show()

                if lang_cluster == 'Python':
                    min_tokens = 23
                    max_tokens = 422
                elif lang_cluster == 'Java':
                    min_tokens = min(tokens)
                    max_tokens = 1197
                elif lang_cluster == 'C':
                    min_tokens = min(tokens)
                    max_tokens = 572
                elif lang_cluster == 'C++':
                    min_tokens = 52
                    max_tokens = 782
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
                if lang_cluster == 'Python':
                    length_dataset_list[0] = concatenate_datasets(
                        [length_dataset_list[0], lang_cluster_dataset.filter(lambda example: example['tokens'] in [8])]
                    )
                    length_dataset_list[-1] = concatenate_datasets(
                        [length_dataset_list[-1], lang_cluster_dataset.filter(lambda example: example['tokens'] in [682, 1166])]
                    )
                elif lang_cluster == 'Java':
                    length_dataset_list[-1] = concatenate_datasets(
                        [length_dataset_list[-1], lang_cluster_dataset.filter(lambda example: example['tokens'] in [1438, 1490])]
                    )
                elif lang_cluster == 'C':
                    length_dataset_list[-1] = concatenate_datasets(
                        [length_dataset_list[-1], lang_cluster_dataset.filter(lambda example: example['tokens'] in [676])]
                    )
                elif lang_cluster == 'C++':
                    length_dataset_list[0] = concatenate_datasets(
                        [length_dataset_list[0], lang_cluster_dataset.filter(lambda example: example['tokens'] in [21, 36])]
                    )
                    length_dataset_list[-1] = concatenate_datasets(
                        [length_dataset_list[-1], lang_cluster_dataset.filter(lambda example: example['tokens'] in [1293, 1596])]
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

                    pass_rate = round(float(np.mean(length_dataset['human_sample_pass_rate'])), 2)
                    score_item['metrics'][lang_cluster.lower()][length]['pass_rate'] = str(pass_rate)

                    line_coverage = round(float(np.mean(length_dataset['human_sample_line_coverage'])), 2)
                    score_item['metrics'][lang_cluster.lower()][length]['line_coverage'] = str(line_coverage)

                    branch_coverage = round(float(np.mean(length_dataset['human_sample_branch_coverage'])), 2)
                    score_item['metrics'][lang_cluster.lower()][length]['branch_coverage'] = str(branch_coverage)

            evaluation_metrics = []
            for length in length_list:
                length_metrics = []

                for lang_cluster in lang_cluster_list:
                    length_metrics.append(float(score_item['metrics'][lang_cluster.lower()][length]['pass_rate']))
                    length_metrics.append(float(score_item['metrics'][lang_cluster.lower()][length]['line_coverage']))
                    length_metrics.append(float(score_item['metrics'][lang_cluster.lower()][length]['branch_coverage']))

                length_score = round(float(np.mean(length_metrics)), 2)
                evaluation_metrics.append(length_score)
                score_item[f'{length}'] = str(length_score)

            print(score_item)
            overall_score = round(float(np.mean(evaluation_metrics)), 2)
            score_item['overall'] = str(overall_score)
            del score_item['metrics']

            score_dict['data'].append(score_item)

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

            if lang_cluster == 'Python':
                min_tokens = 23
                max_tokens = 422
            elif lang_cluster == 'Java':
                min_tokens = min(tokens)
                max_tokens = 1197
            elif lang_cluster == 'C':
                min_tokens = min(tokens)
                max_tokens = 572
            elif lang_cluster == 'C++':
                min_tokens = 52
                max_tokens = 782
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
            if lang_cluster == 'Python':
                length_dataset_list[0] = concatenate_datasets(
                    [length_dataset_list[0], lang_cluster_dataset.filter(lambda example: example['tokens'] in [8])]
                )
                length_dataset_list[-1] = concatenate_datasets(
                    [length_dataset_list[-1], lang_cluster_dataset.filter(lambda example: example['tokens'] in [682, 1166])]
                )
            elif lang_cluster == 'Java':
                length_dataset_list[-1] = concatenate_datasets(
                    [length_dataset_list[-1], lang_cluster_dataset.filter(lambda example: example['tokens'] in [1438, 1490])]
                )
            elif lang_cluster == 'C':
                length_dataset_list[-1] = concatenate_datasets(
                    [length_dataset_list[-1], lang_cluster_dataset.filter(lambda example: example['tokens'] in [676])]
                )
            elif lang_cluster == 'C++':
                length_dataset_list[0] = concatenate_datasets(
                    [length_dataset_list[0], lang_cluster_dataset.filter(lambda example: example['tokens'] in [21, 36])]
                )
                length_dataset_list[-1] = concatenate_datasets(
                    [length_dataset_list[-1], lang_cluster_dataset.filter(lambda example: example['tokens'] in [1293, 1596])]
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

                pass_rate = round(float(np.mean(length_dataset['predicted_pass_rate'])), 2)
                score_item['metrics'][lang_cluster.lower()][length]['pass_rate'] = str(pass_rate)

                line_coverage = round(float(np.mean(length_dataset['predicted_line_coverage'])), 2)
                score_item['metrics'][lang_cluster.lower()][length]['line_coverage'] = str(line_coverage)

                branch_coverage = round(float(np.mean(length_dataset['predicted_branch_coverage'])), 2)
                score_item['metrics'][lang_cluster.lower()][length]['branch_coverage'] = str(branch_coverage)

        evaluation_metrics = []
        for length in length_list:
            length_metrics = []

            for lang_cluster in lang_cluster_list:
                length_metrics.append(float(score_item['metrics'][lang_cluster.lower()][length]['pass_rate']))
                length_metrics.append(float(score_item['metrics'][lang_cluster.lower()][length]['line_coverage']))
                length_metrics.append(float(score_item['metrics'][lang_cluster.lower()][length]['branch_coverage']))

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
    save_score_path = score_dir / Path('automated_testing_score_1.json')
    with open(str(save_score_path), mode='w', encoding='utf-8') as file:
        json.dump(score_dict, file, ensure_ascii=False, indent=2)

    print('Table 2:')
    score_dict = {}
    score_dict['code'] = 0
    score_dict['data'] = []
    for index, load_result_name in tqdm(enumerate(load_result_name_list), total=len(load_result_name_list)):
        load_result_path = Path(__file__).parent.parent / Path('inference') / Path('results') / Path(load_result_name)
        dataset = load_dataset('json', split='train', data_files=str(load_result_path))
        dataset = dataset.map(add_tokens)

        lang_cluster_dataset_list = [dataset.filter(lambda example: example['lang_cluster'] == lang_cluster) for lang_cluster in lang_cluster_list]

        if index == 0:
            score_item = {}
            score_item['model'] = 'Human'
            evaluation_metrics = []
            for lang_cluster, lang_cluster_dataset in zip(lang_cluster_list, lang_cluster_dataset_list):
                tokens = lang_cluster_dataset['tokens']

                # import matplotlib.pyplot as plt
                # sorted_tokens = sorted(tokens)
                # print(sorted_tokens)
                # plt.bar(range(len(sorted_tokens)), sorted_tokens)
                # plt.show()

                if lang_cluster == 'Python':
                    min_tokens = 23
                    max_tokens = 422
                elif lang_cluster == 'Java':
                    min_tokens = min(tokens)
                    max_tokens = 1197
                elif lang_cluster == 'C':
                    min_tokens = min(tokens)
                    max_tokens = 572
                elif lang_cluster == 'C++':
                    min_tokens = 52
                    max_tokens = 782
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
                if lang_cluster == 'Python':
                    length_dataset_list[0] = concatenate_datasets(
                        [length_dataset_list[0], lang_cluster_dataset.filter(lambda example: example['tokens'] in [8])]
                    )
                    length_dataset_list[-1] = concatenate_datasets(
                        [length_dataset_list[-1], lang_cluster_dataset.filter(lambda example: example['tokens'] in [682, 1166])]
                    )
                elif lang_cluster == 'Java':
                    length_dataset_list[-1] = concatenate_datasets(
                        [length_dataset_list[-1], lang_cluster_dataset.filter(lambda example: example['tokens'] in [1438, 1490])]
                    )
                elif lang_cluster == 'C':
                    length_dataset_list[-1] = concatenate_datasets(
                        [length_dataset_list[-1], lang_cluster_dataset.filter(lambda example: example['tokens'] in [676])]
                    )
                elif lang_cluster == 'C++':
                    length_dataset_list[0] = concatenate_datasets(
                        [length_dataset_list[0], lang_cluster_dataset.filter(lambda example: example['tokens'] in [21, 36])]
                    )
                    length_dataset_list[-1] = concatenate_datasets(
                        [length_dataset_list[-1], lang_cluster_dataset.filter(lambda example: example['tokens'] in [1293, 1596])]
                    )
                print(length_dataset_list)

                num_rows = 0
                for length_dataset in length_dataset_list:
                    num_rows += length_dataset.num_rows
                if num_rows != lang_cluster_dataset.num_rows:
                    raise Exception

                for length, length_dataset in zip(length_list, length_dataset_list):
                    pass_rate = round(float(np.mean(length_dataset['human_sample_pass_rate'])), 2)
                    evaluation_metrics.append(pass_rate)
                    score_item[f'{lang_cluster.lower()}_{length}_pass_rate'] = str(pass_rate)

                    line_coverage = round(float(np.mean(length_dataset['human_sample_line_coverage'])), 2)
                    evaluation_metrics.append(line_coverage)
                    score_item[f'{lang_cluster.lower()}_{length}_line_coverage'] = str(line_coverage)

                    branch_coverage = round(float(np.mean(length_dataset['human_sample_branch_coverage'])), 2)
                    evaluation_metrics.append(branch_coverage)
                    score_item[f'{lang_cluster.lower()}_{length}_branch_coverage'] = str(branch_coverage)

            print(score_item)
            overall_score = round(float(np.mean(evaluation_metrics)), 2)
            score_item['overall'] = str(overall_score)

            score_dict['data'].append(score_item)

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

            if lang_cluster == 'Python':
                min_tokens = 23
                max_tokens = 422
            elif lang_cluster == 'Java':
                min_tokens = min(tokens)
                max_tokens = 1197
            elif lang_cluster == 'C':
                min_tokens = min(tokens)
                max_tokens = 572
            elif lang_cluster == 'C++':
                min_tokens = 52
                max_tokens = 782
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
            if lang_cluster == 'Python':
                length_dataset_list[0] = concatenate_datasets(
                    [length_dataset_list[0], lang_cluster_dataset.filter(lambda example: example['tokens'] in [8])]
                )
                length_dataset_list[-1] = concatenate_datasets(
                    [length_dataset_list[-1], lang_cluster_dataset.filter(lambda example: example['tokens'] in [682, 1166])]
                )
            elif lang_cluster == 'Java':
                length_dataset_list[-1] = concatenate_datasets(
                    [length_dataset_list[-1], lang_cluster_dataset.filter(lambda example: example['tokens'] in [1438, 1490])]
                )
            elif lang_cluster == 'C':
                length_dataset_list[-1] = concatenate_datasets(
                    [length_dataset_list[-1], lang_cluster_dataset.filter(lambda example: example['tokens'] in [676])]
                )
            elif lang_cluster == 'C++':
                length_dataset_list[0] = concatenate_datasets(
                    [length_dataset_list[0], lang_cluster_dataset.filter(lambda example: example['tokens'] in [21, 36])]
                )
                length_dataset_list[-1] = concatenate_datasets(
                    [length_dataset_list[-1], lang_cluster_dataset.filter(lambda example: example['tokens'] in [1293, 1596])]
                )
            print(length_dataset_list)

            num_rows = 0
            for length_dataset in length_dataset_list:
                num_rows += length_dataset.num_rows
            if num_rows != lang_cluster_dataset.num_rows:
                raise Exception

            for length, length_dataset in zip(length_list, length_dataset_list):
                pass_rate = round(float(np.mean(length_dataset['predicted_pass_rate'])), 2)
                evaluation_metrics.append(pass_rate)
                score_item[f'{lang_cluster.lower()}_{length}_pass_rate'] = str(pass_rate)

                line_coverage = round(float(np.mean(length_dataset['predicted_line_coverage'])), 2)
                evaluation_metrics.append(line_coverage)
                score_item[f'{lang_cluster.lower()}_{length}_line_coverage'] = str(line_coverage)

                branch_coverage = round(float(np.mean(length_dataset['predicted_branch_coverage'])), 2)
                evaluation_metrics.append(branch_coverage)
                score_item[f'{lang_cluster.lower()}_{length}_branch_coverage'] = str(branch_coverage)

        print(score_item)
        overall_score = round(float(np.mean(evaluation_metrics)), 2)
        score_item['overall'] = str(overall_score)

        score_dict['data'].append(score_item)

    score_dict['data'].sort(key=lambda x: x['overall'], reverse=True)
    print(score_dict)

    score_dir = Path(__file__).parent / Path('scores')
    if not score_dir.is_dir():
        score_dir.mkdir(parents=True, exist_ok=True)
    save_score_path = score_dir / Path('automated_testing_score_2.json')
    with open(str(save_score_path), mode='w', encoding='utf-8') as file:
        json.dump(score_dict, file, ensure_ascii=False, indent=2)

    print('Table 3:')
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
    save_score_path = score_dir / Path('automated_testing_score_3.json')
    with open(str(save_score_path), mode='w', encoding='utf-8') as file:
        json.dump(score_dict, file, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
