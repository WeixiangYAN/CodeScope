from collections import Counter
from datasets import load_dataset
from api_comm import APICommunication, ExtendedUnittest
import argparse


def add_exec_outcome(example):
    language = example['lang']
    source_code = example['source_code']
    hidden_unit_tests = eval(example['testcases'])

    unittests = []
    for hidden_unit_test in hidden_unit_tests:
        unittests.append(
            ExtendedUnittest(
                input=hidden_unit_test['input'],
                output=hidden_unit_test['output']
            ).json()
        )

    api = APICommunication()
    response = api.execute_code(
        language=language,
        source_code=source_code,
        unittests=unittests
    )
    print(response)

    example['exec_outcome'] = response[0]

    return example


def main(args):
    dataset = load_dataset('json', split='train', data_files=args.codes_dir + args.path)
    dataset = dataset.map(add_exec_outcome)

    lang_counts = Counter(dataset['lang'])
    for lang, count in lang_counts.items():
        print(f'{lang}: {count}')

    lang_cluster_counts = Counter(dataset['lang_cluster'])
    for lang_cluster, count in lang_cluster_counts.items():
        print(f'{lang_cluster}: {count}')

    dataset.to_json(args.results_dir + args.path, lines=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--codes_dir', type=str, default='../inference/results', help='The folder where you store your code files')
    parser.add_argument('--results_dir', type=str, default='execute_results/', help='The folder where you store your run results')
    parser.add_argument('--code_filename', type=str, default='program_synthesis_eval_gpt4.jsonl', help='code data')

    args = parser.parse_args()
    main(args)
