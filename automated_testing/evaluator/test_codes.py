import os
import re
import argparse
import subprocess
import pandas as pd

from pathlib import Path
from datasets import load_dataset


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_name', default='automated_testing_result_palm2.jsonl',
                        choices=[
                            'automated_testing_result_codellama.jsonl',
                            'automated_testing_result_gpt3-5.jsonl',
                            'automated_testing_result_gpt4.jsonl',
                            'automated_testing_result_llama2.jsonl',
                            'automated_testing_result_palm2.jsonl',
                            'automated_testing_result_starcoder.jsonl',
                            'automated_testing_result_vicuna.jsonl',
                            'automated_testing_result_wizardcoder.jsonl'
                        ], type=str)
    args = parser.parse_args()

    return args


def execute_command(command, input=None):
    if input is not None:
        input = input.replace('\r\n', '\n')
    try:
        outcome = subprocess.run(command, input=input, capture_output=True, text=True, timeout=20, shell=False)
    except Exception as e:
        print('Error occurred while executing command:', e)
        outcome = subprocess.CompletedProcess(args=command, returncode=-1, stdout='', stderr=str(e))
    return outcome


def add_pass_rate(example):
    id = example['id']
    lang_cluster = example['lang_cluster']
    source_code = example['source_code']
    predicted_testcases = eval(example['predicted_testcases'])
    num_predicted_testcases = len(predicted_testcases)

    if num_predicted_testcases == 0:
        print('Failed to generate testcases:', id)
        example['predicted_pass_rate'] = 0.00
    else:
        if lang_cluster == 'C':
            os.chdir(f'codes/c/{id}')
            print(os.getcwd())

            compile_command = 'gcc -fprofile-arcs -ftest-coverage -fPIC -O0 code.c -o code'
            outcome = execute_command(compile_command)
            print(outcome)

            num_passed = 0
            for index, predicted_testcase in enumerate(predicted_testcases):
                input = predicted_testcase['input']
                output = predicted_testcase['output']

                test_command = 'code'
                outcome = execute_command(test_command, input)
                print(outcome)

                is_passed = True if outcome.returncode == 0 and (
                        outcome.stdout in output or outcome.stdout.rstrip() in output or outcome.stdout.replace('\n', '\r\n') in output or outcome.stdout.replace('\n',
                                                                                                                                                                  '\r\n').rstrip() in output) else False
                if is_passed is True:
                    num_passed += 1
                print(is_passed)

                coverage_command = f'gcovr --json test-{index}.json'
                outcome = execute_command(coverage_command)
                print(outcome)

            pass_rate = round(100. * num_passed / num_predicted_testcases, 2)
            print(f'Pass rate: {pass_rate}% [{num_passed}/{num_predicted_testcases}]')

            line_coverage_report_text_command = 'gcovr --add-tracefile "test-*.json" --txt'
            outcome = execute_command(line_coverage_report_text_command)
            print(outcome.stdout)

            branch_coverage_report_text_command = 'gcovr --add-tracefile "test-*.json" --branches --txt'
            outcome = execute_command(branch_coverage_report_text_command)
            print(outcome.stdout)

            coverage_report_json_command = 'gcovr --add-tracefile "test-*.json" --json-summary coverage.json'
            outcome = execute_command(coverage_report_json_command)
            print(outcome)

            coverage_report_html_command = 'gcovr --add-tracefile "test-*.json" --html-details coverage.html'
            outcome = execute_command(coverage_report_html_command)
            print(outcome)

            os.chdir('../../..')
            print(os.getcwd())

            example['predicted_pass_rate'] = pass_rate

        elif lang_cluster == 'C++':
            os.chdir(f'codes/cpp/{id}')
            print(os.getcwd())

            compile_command = 'g++ -fprofile-arcs -ftest-coverage -fPIC -O0 code.cpp -o code'
            outcome = execute_command(compile_command)
            print(outcome)

            num_passed = 0
            for index, predicted_testcase in enumerate(predicted_testcases):
                input = predicted_testcase['input']
                output = predicted_testcase['output']

                test_command = 'code'
                outcome = execute_command(test_command, input)
                print(outcome)

                is_passed = True if outcome.returncode == 0 and (
                        outcome.stdout in output or outcome.stdout.rstrip() in output or outcome.stdout.replace('\n', '\r\n') in output or outcome.stdout.replace('\n',
                                                                                                                                                                  '\r\n').rstrip() in output) else False
                if is_passed is True:
                    num_passed += 1
                print(is_passed)

                coverage_command = f'gcovr --json test-{index}.json'
                outcome = execute_command(coverage_command)
                print(outcome)

            pass_rate = round(100. * num_passed / num_predicted_testcases, 2)
            print(f'Pass rate: {pass_rate}% [{num_passed}/{num_predicted_testcases}]')

            line_coverage_report_text_command = 'gcovr --add-tracefile "test-*.json" --txt'
            outcome = execute_command(line_coverage_report_text_command)
            print(outcome.stdout)

            branch_coverage_report_text_command = 'gcovr --add-tracefile "test-*.json" --branches --txt'
            outcome = execute_command(branch_coverage_report_text_command)
            print(outcome.stdout)

            coverage_report_json_command = 'gcovr --add-tracefile "test-*.json" --json-summary coverage.json'
            outcome = execute_command(coverage_report_json_command)
            print(outcome)

            coverage_report_html_command = 'gcovr --add-tracefile "test-*.json" --html-details coverage.html'
            outcome = execute_command(coverage_report_html_command)
            print(outcome)

            os.chdir('../../..')
            print(os.getcwd())

            example['predicted_pass_rate'] = pass_rate

        elif lang_cluster == 'Java':
            os.chdir(f'codes/java/{id}')
            print(os.getcwd())

            pattern = r'public\s+(?:final\s+)?class\s+(\w+)'
            matches = re.search(pattern, source_code)
            if matches:
                class_name = matches.group(1)
            else:
                print('Class name not found, use default class name.')
                class_name = 'code'

            compile_command = f'javac {class_name}.java'
            outcome = execute_command(compile_command)
            print(outcome)

            num_passed = 0
            for index, predicted_testcase in enumerate(predicted_testcases):
                input = predicted_testcase['input']
                output = predicted_testcase['output']

                test_command = f'java -javaagent:../../../jars/jacocoagent.jar=destfile=test.exec,append=true {class_name}'
                outcome = execute_command(test_command, input)
                print(outcome)

                is_passed = True if outcome.returncode == 0 and (
                        outcome.stdout in output or outcome.stdout.rstrip() in output or outcome.stdout.replace('\n', '\r\n') in output or outcome.stdout.replace('\n',
                                                                                                                                                                  '\r\n').rstrip() in output) else False
                if is_passed is True:
                    num_passed += 1
                print(is_passed)

            pass_rate = round(100. * num_passed / num_predicted_testcases, 2)
            print(f'Pass rate: {pass_rate}% [{num_passed}/{num_predicted_testcases}]')

            coverage_report_csv_command = 'java -jar ../../../jars/jacococli.jar report test.exec --classfiles . --sourcefiles . --csv coverage.csv'
            outcome = execute_command(coverage_report_csv_command)
            print(outcome)

            coverage_report_html_command = 'java -jar ../../../jars/jacococli.jar report test.exec --classfiles . --sourcefiles . --html coverage'
            outcome = execute_command(coverage_report_html_command)
            print(outcome)

            os.chdir('../../..')
            print(os.getcwd())

            example['predicted_pass_rate'] = pass_rate

        elif lang_cluster == 'Python':
            os.chdir(f'codes/python/{id}')
            print(os.getcwd())

            num_passed = 0
            for index, predicted_testcase in enumerate(predicted_testcases):
                input = predicted_testcase['input']
                output = predicted_testcase['output']

                test_command = 'python code.py'
                outcome = execute_command(test_command, input)
                print(outcome)

                is_passed = True if outcome.returncode == 0 and (
                        outcome.stdout in output or outcome.stdout.rstrip() in output or outcome.stdout.replace('\n', '\r\n') in output or outcome.stdout.replace('\n',
                                                                                                                                                                  '\r\n').rstrip() in output) else False
                if is_passed is True:
                    num_passed += 1
                print(is_passed)

                coverage_command = f'coverage run --branch --append code.py'
                outcome = execute_command(coverage_command, input)
                print(outcome)

            pass_rate = round(100. * num_passed / num_predicted_testcases, 2)
            print(f'Pass rate: {pass_rate}% [{num_passed}/{num_predicted_testcases}]')

            coverage_report_text_command = 'coverage report --format=text --show-missing --precision=2'
            outcome = execute_command(coverage_report_text_command)
            print(outcome.stdout)

            coverage_report_json_command = 'coverage json -o coverage.json'
            outcome = execute_command(coverage_report_json_command)
            print(outcome)

            coverage_report_html_command = 'coverage html -d coverage'
            outcome = execute_command(coverage_report_html_command)
            print(outcome)

            os.chdir('../../..')
            print(os.getcwd())

            example['predicted_pass_rate'] = pass_rate

    return example


def add_coverage(example):
    id = example['id']
    lang_cluster = example['lang_cluster']
    predicted_testcases = eval(example['predicted_testcases'])
    num_predicted_testcases = len(predicted_testcases)

    if num_predicted_testcases == 0:
        print('Failed to generate testcases:', id)
        example['predicted_line_coverage'] = 0.00
        example['predicted_branch_coverage'] = 0.00
    else:
        if lang_cluster == 'C':
            os.chdir(f'codes/c/{id}')
            print(os.getcwd())

            try:
                coverage_data = pd.read_json('coverage.json')
                line_covered = coverage_data.loc[0, 'line_covered']
                line_total = coverage_data.loc[0, 'line_total']
                if line_total == 0:
                    line_coverage = 100.00
                else:
                    line_coverage = round(100. * line_covered / line_total, 2)
                branch_covered = coverage_data.loc[0, 'branch_covered']
                branch_total = coverage_data.loc[0, 'branch_total']
                if branch_total == 0:
                    branch_coverage = 100.00
                else:
                    branch_coverage = round(100. * branch_covered / branch_total, 2)
            except Exception as e:
                print('Error occurred while reading the JSON file:', e)
                line_coverage = 0.00
                branch_coverage = 0.00

            print(f'Line Coverage: {line_coverage}%')
            print(f'Branch Coverage: {branch_coverage}%')

            os.chdir('../../..')
            print(os.getcwd())

            example['predicted_line_coverage'] = line_coverage
            example['predicted_branch_coverage'] = branch_coverage

        elif lang_cluster == 'C++':
            os.chdir(f'codes/cpp/{id}')
            print(os.getcwd())

            try:
                coverage_data = pd.read_json('coverage.json')
                line_covered = coverage_data.loc[0, 'line_covered']
                line_total = coverage_data.loc[0, 'line_total']
                if line_total == 0:
                    line_coverage = 100.00
                else:
                    line_coverage = round(100. * line_covered / line_total, 2)
                branch_covered = coverage_data.loc[0, 'branch_covered']
                branch_total = coverage_data.loc[0, 'branch_total']
                if branch_total == 0:
                    branch_coverage = 100.00
                else:
                    branch_coverage = round(100. * branch_covered / branch_total, 2)
            except Exception as e:
                print('Error occurred while reading the JSON file:', e)
                line_coverage = 0.00
                branch_coverage = 0.00

            print(f'Line Coverage: {line_coverage}%')
            print(f'Branch Coverage: {branch_coverage}%')

            os.chdir('../../..')
            print(os.getcwd())

            example['predicted_line_coverage'] = line_coverage
            example['predicted_branch_coverage'] = branch_coverage

        elif lang_cluster == 'Java':
            os.chdir(f'codes/java/{id}')
            print(os.getcwd())

            try:
                coverage_data = pd.read_csv('coverage.csv')
                line_covered = coverage_data.loc[0, 'LINE_COVERED']
                line_missed = coverage_data.loc[0, 'LINE_MISSED']
                line_total = line_covered + line_missed
                if line_total == 0:
                    line_coverage = 100.00
                else:
                    line_coverage = round(100. * line_covered / line_total, 2)
                branch_covered = coverage_data.loc[0, 'BRANCH_COVERED']
                branch_missed = coverage_data.loc[0, 'BRANCH_MISSED']
                branch_total = branch_covered + branch_missed
                if branch_total == 0:
                    branch_coverage = 100.00
                else:
                    branch_coverage = round(100. * branch_covered / branch_total, 2)
            except Exception as e:
                print('Error occurred while reading the CSV file:', e)
                line_coverage = 0.00
                branch_coverage = 0.00

            print(f'Line Coverage: {line_coverage}%')
            print(f'Branch Coverage: {branch_coverage}%')

            os.chdir('../../..')
            print(os.getcwd())

            example['predicted_line_coverage'] = line_coverage
            example['predicted_branch_coverage'] = branch_coverage

        elif lang_cluster == 'Python':
            os.chdir(f'codes/python/{id}')
            print(os.getcwd())

            try:
                coverage_data = pd.read_json('coverage.json')
                line_covered = int(coverage_data.loc['covered_lines', 'totals'])
                line_missed = int(coverage_data.loc['missing_lines', 'totals'])
                line_total = line_covered + line_missed
                if line_total == 0:
                    line_coverage = 100.00
                else:
                    line_coverage = round(100. * line_covered / line_total, 2)
                branch_covered = int(coverage_data.loc['covered_branches', 'totals'])
                branch_missed = int(coverage_data.loc['missing_branches', 'totals'])
                branch_total = branch_covered + branch_missed
                if branch_total == 0:
                    branch_coverage = 100.00
                else:
                    branch_coverage = round(100. * branch_covered / branch_total, 2)
            except Exception as e:
                print('Error occurred while reading the JSON file:', e)
                line_coverage = 0.00
                branch_coverage = 0.00

            print(f'Line Coverage: {line_coverage}%')
            print(f'Branch Coverage: {branch_coverage}%')

            os.chdir('../../..')
            print(os.getcwd())

            example['predicted_line_coverage'] = line_coverage
            example['predicted_branch_coverage'] = branch_coverage

    return example


def main():
    result_path = Path(__file__).parent.parent / Path('inference') / Path('results') / Path(args.result_name)
    dataset = load_dataset('json', split='train', data_files=str(result_path))
    dataset.cleanup_cache_files()
    print(dataset)

    dataset = dataset.map(add_pass_rate)
    dataset = dataset.map(add_coverage)
    print(dataset)

    dataset.to_json(result_path, lines=True)


if __name__ == '__main__':
    args = parse_arguments()
    main()
