import re

from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset


def main():
    lang_cluster_list = ['c', 'cpp', 'java', 'python']

    codes_dir = Path(__file__).parent / Path('codes')
    if not codes_dir.is_dir():
        codes_dir.mkdir(parents=True, exist_ok=True)
    for lang_cluster in lang_cluster_list:
        lang_cluster_codes_dir = codes_dir / Path(lang_cluster)
        if not lang_cluster_codes_dir.is_dir():
            lang_cluster_codes_dir.mkdir(parents=True, exist_ok=True)

    load_data_path = Path(__file__).parent.parent.parent / Path('data') / Path('automated_testing_data.jsonl')
    dataset = load_dataset('json', split='train', data_files=str(load_data_path))
    dataset.cleanup_cache_files()
    print(dataset)

    for example in tqdm(dataset):
        id = example['id']
        lang_cluster = example['lang_cluster']
        source_code = example['source_code']

        if lang_cluster == 'C':
            lang_cluster_codes_dir = codes_dir / Path('c')
        elif lang_cluster == 'C++':
            lang_cluster_codes_dir = codes_dir / Path('cpp')
        elif lang_cluster == 'Java':
            lang_cluster_codes_dir = codes_dir / Path('java')
        elif lang_cluster == 'Python':
            lang_cluster_codes_dir = codes_dir / Path('python')
        else:
            print('Language cluster not found, use default language cluster directory.')
            lang_cluster_codes_dir = codes_dir

        file_dir = lang_cluster_codes_dir / Path(str(id))
        if not file_dir.is_dir():
            file_dir.mkdir(parents=True, exist_ok=True)

        if lang_cluster == 'C':
            file_path = file_dir / Path('code.c')
        elif lang_cluster == 'C++':
            file_path = file_dir / Path('code.cpp')
        elif lang_cluster == 'Java':
            pattern = r'public\s+(?:final\s+)?class\s+(\w+)'
            matches = re.search(pattern, source_code)
            if matches:
                class_name = matches.group(1)
            else:
                print('Class name not found, use default class name.')
                class_name = 'code'
            constructor_code = f'\n\n\tprivate {class_name}() {{}}\n'
            pattern = r'public\s+(?:final\s+)?class\s+' + class_name + r'(\s+\w+\s+\w+\s*)?(\s+//implements\s+Runnable)?\s*{'
            matches = re.search(pattern, source_code)
            if matches:
                class_definition = matches.group(0)
                source_code = source_code.replace(class_definition, class_definition + constructor_code)
            else:
                print('Class definition not found, use default source code.')
            file_path = file_dir / Path(f'{class_name}.java')
        elif lang_cluster == 'Python':
            file_path = file_dir / Path('code.py')
        else:
            print('Language cluster not found, use default language cluster path.')
            file_path = file_dir / Path('code')

        with open(file_path, mode='w', encoding='utf-8') as file:
            file.write(source_code)


if __name__ == '__main__':
    main()
