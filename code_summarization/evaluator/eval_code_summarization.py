import argparse
import json
import evaluate
from datasets import load_dataset
from nltk.translate.meteor_score import meteor_score
import logging
from pathlib import Path
import statistics
import backoff
import tiktoken
import openai
import math

import json
def cal_bleu_iter(load_path, output_dir, pred_field, ref_field):
    dataset = load_dataset("json", data_files=load_path, split="train")
    for d in dataset:
        try:
            bleu_result = bleu.compute(predictions=[d[pred_field]], references=[d[ref_field]])
        except Exception as e:
            bleu_result = {"error": str(e)}
        finally:
            with open(str(output_dir / "avg_bleu_itr_result.jsonl"), "a") as f:
                json.dump(bleu_result, f)
                f.write("\n")
    return bleu_result
def cal_rouge_iter(load_path, output_dir, pred_field, ref_field):
    dataset = load_dataset("json", data_files=load_path, split="train")
    for d in dataset:
        try:
            rouge_result = rouge.compute(predictions=[d[pred_field]], references=[d[ref_field]])
        except Exception as e:
            rouge_result = {"error": str(e)}
        finally:
            with open(str(output_dir / "avg_rouge_itr_result.jsonl"), "a") as f:
                json.dump(rouge_result, f)
                f.write("\n")
    return rouge_result
def cal_bleu(load_path, output_dir, pred_field, ref_field):
    try:
        dataset = load_dataset("json", data_files=load_path, split="train")
        bleu_result = bleu.compute(predictions=dataset[pred_field], references=dataset[ref_field])
    except Exception as e:
        bleu_result = {"error": str(e)}
    finally:
        with open(str(output_dir / "avg_bleu_result.json"), "w") as f:
            json.dump(bleu_result, f)
        return bleu_result
def cal_rouge(load_path, output_dir, pred_field, ref_field):
    try:
        dataset = load_dataset("json", data_files=load_path, split="train")
        rouge_result = rouge.compute(predictions=dataset[pred_field], references=dataset[ref_field])
    except Exception as e:
        rouge_result = {"error": str(e)}
    finally:
        with open(str(output_dir / "avg_rouge_result.json"), "w") as f:
            json.dump(rouge_result, f)
        return rouge_result
def cal_bertscore(load_path, output_dir, pred_field, ref_field):
    try:
        dataset = load_dataset("json", data_files=load_path, split="train")
        bertscore_result = bertscore.compute(lang="en", predictions=dataset[pred_field], references=dataset[ref_field])
        bertscore_result = {"avg_precision": sum(bertscore_result["precision"])/len(bertscore_result["precision"]), "avg_recall": sum(bertscore_result["recall"])/len(bertscore_result["recall"]), "avg_f1": sum(bertscore_result["f1"])/len(bertscore_result["f1"]), **bertscore_result}
    except Exception as e:
        bertscore_result = {"error": str(e)}
    finally:
        with open(str(output_dir / "avg_bertscore_result.json"), "w") as f:
            json.dump(bertscore_result, f)
    return bertscore_result
def cal_meteor(load_path, output_dir, pred_field, ref_field):
    dataset = load_dataset("json", data_files=load_path, split="train")
    meteor_results = []
    try:
        for d in dataset:
            references = [d[ref_field].split()]
            candidate = d[pred_field].split()
            score = meteor_score(references, candidate)
            meteor_results.append(score)
        results = {"avg": sum(meteor_results)/len(meteor_results), "meteor_results": meteor_results}
    except Exception as e:
        results = {"error": str(e)}
    finally:
        with open(str(output_dir / "avg_meteor_result.json"), "w") as f:
            json.dump(results, f)
        return results



def cal_bleu_by_lang(load_path, output_dir, pred_field, ref_field):
    dataset = load_dataset("json", data_files=load_path, split="train")
    langs = set(dataset['lang_cluster'])
    print(langs)
    for lang in langs:
        lang_dataset = dataset.filter(lambda x:x['lang_cluster']==lang)
        try:
            logging.info(f"start computing bleu results for {lang}")
            bleu_result = bleu.compute(predictions=lang_dataset[pred_field], references=lang_dataset[ref_field])
        except Exception as e:
            logging.error(f"error at {lang}: {e}")
        with open(str(output_dir / f"{lang}.json"), "w") as f:
            json.dump(bleu_result, f)
def cal_rouge_by_lang(load_path, output_dir, pred_field, ref_field):
    dataset = load_dataset("json", data_files=load_path, split="train")
    langs = set(dataset['lang_cluster'])
    print(langs)
    for lang in langs:
        lang_dataset = dataset.filter(lambda x:x['lang_cluster']==lang)
        try:
            logging.info(f"start computing bleu results for {lang}")
            rouge_result = rouge.compute(predictions=lang_dataset[pred_field], references=lang_dataset[ref_field])
        except Exception as e:
            logging.error(f"error at {lang}: {e}")
        with open(str(output_dir / f"{lang}.json"), "w") as f:
            json.dump(rouge_result, f)
def cal_bertscore_by_lang(load_path, output_dir, pred_field, ref_field):
    dataset = load_dataset("json", data_files=load_path, split="train")
    langs = set(dataset['lang_cluster'])
    print(langs)
    for lang in langs:
        lang_dataset = dataset.filter(lambda x:x['lang_cluster']==lang)
        try:
            logging.info(f"start computing bleu results for {lang}")
            bertscore_result = bertscore.compute(lang="en", predictions=lang_dataset[pred_field], references=lang_dataset[ref_field])
            precisions = bertscore_result['precision']
            recalls = bertscore_result['recall']
            f1s = bertscore_result['f1']
            result = {'avg_precision': sum(precisions)/len(precisions), 'avg_recall': sum(recalls)/len(recalls), 'avg_f1': sum(f1s)/len(f1s), **bertscore_result}
        except Exception as e:
            logging.error(f"error at {lang}: {e}")
            result = {"error": e}
        with open(str(output_dir / f"{lang}.json"), "w") as f:
            json.dump(result, f)
def cal_meteor_by_lang(load_path, output_dir, pred_field, ref_field):
    dataset = load_dataset("json", data_files=load_path, split="train")
    langs = set(dataset['lang_cluster'])
    for lang in langs:
        lang_dataset = dataset.filter(lambda x:x['lang_cluster']==lang)
        meteor_results = []
        try:
            logging.info(f"start computing bleu results for {lang}")
            for d in lang_dataset:
                references = [d[ref_field].split()]
                candidate = d[pred_field].split()
                score = meteor_score(references, candidate)
                meteor_results.append(score)
            results = {"avg": sum(meteor_results)/len(meteor_results), "meteor_results": meteor_results}
        except Exception as e:
            logging.error(f"error at {lang}: {e}")
        with open(str(output_dir / f"{lang}.json"), "w") as f:
            json.dump(results, f)

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def count_tokens(content):
    encoding = tiktoken.get_encoding('cl100k_base')
    num_tokens = len(encoding.encode(content))
    return num_tokens

def add_ntokens(example):
    example['ntokens'] = count_tokens(example['source_code'])
    return example
def cal_bleu_by_ntokens(load_path, output_dir, pred_field, ref_field):
    output_dir = output_dir / Path(f'split_3')
    output_dir.mkdir(exist_ok=True, parents=True)
    ds = load_dataset("json", data_files=load_path, split="train").map(add_ntokens)
    langs = ds.unique('lang_cluster')
    for lang in langs:
        temp_ds = ds.filter(lambda x: x['lang_cluster'] == lang).sort('ntokens')
        outlier_threshold_up = statistics.quantiles(temp_ds['ntokens'], n=4)[2] + 1.5 * (statistics.quantiles(temp_ds['ntokens'], n=4)[2] - statistics.quantiles(temp_ds['ntokens'], n=4)[0])
        outlier_threshold_low = statistics.quantiles(temp_ds['ntokens'], n=4)[0] - 1.5 * (statistics.quantiles(temp_ds['ntokens'], n=4)[2] - statistics.quantiles(temp_ds['ntokens'], n=4)[0])
        in_distr_lower_bound = max(outlier_threshold_low, min(temp_ds['ntokens']))
        in_distr_upper_bound = min(outlier_threshold_up, max(temp_ds['ntokens']))
        interval_length = int((in_distr_upper_bound - in_distr_lower_bound) / 3)
        interval_short = (min(temp_ds['ntokens']), in_distr_lower_bound+interval_length-1)
        interval_medium = (in_distr_lower_bound+interval_length, in_distr_lower_bound+2*interval_length-1)
        interval_long = (in_distr_lower_bound+2*interval_length, max(temp_ds['ntokens']))
        short_codes_ds = temp_ds.filter(lambda x: x['ntokens'] >= interval_short[0] and x['ntokens'] <= interval_short[1])
        medium_codes_ds = temp_ds.filter(lambda x: x['ntokens'] >= interval_medium[0] and x['ntokens'] <= interval_medium[1])
        long_codes_ds = temp_ds.filter(lambda x: x['ntokens'] >= interval_long[0] and x['ntokens'] <= interval_long[1])
        try:
            print(f"start computing bleu results for {lang}")
            long_bleu_result = bleu.compute(predictions=long_codes_ds[pred_field], references=long_codes_ds[ref_field])
            medium_bleu_result = bleu.compute(predictions=medium_codes_ds[pred_field], references=medium_codes_ds[ref_field])
            short_bleu_result = bleu.compute(predictions=short_codes_ds[pred_field], references=short_codes_ds[ref_field])
        except Exception as e:
            logging.error(f"error at {lang}: {e}")
        with open(str(output_dir / f"{lang}_tk_{math.ceil(interval_long[0])}_{interval_long[1]}.json"), "w") as f:
            json.dump({"num":len(long_codes_ds), **long_bleu_result}, f)
        with open(str(output_dir / f"{lang}_tk_{math.ceil(interval_medium[0])}_{interval_medium[1]}.json"), "w") as f:
            json.dump({"num":len(medium_codes_ds), **medium_bleu_result}, f)
        with open(str(output_dir / f"{lang}_tk_{math.ceil(interval_short[0])}_{interval_short[1]}.json"), "w") as f:
            json.dump({"num":len(short_codes_ds), **short_bleu_result}, f)
def cal_rouge_by_ntokens(load_path, output_dir, pred_field, ref_field):
    output_dir = output_dir / Path(f'split_3')
    output_dir.mkdir(exist_ok=True, parents=True)
    ds = load_dataset("json", data_files=load_path, split="train").map(add_ntokens)
    langs = ds.unique('lang_cluster')
    for lang in langs:
        temp_ds = ds.filter(lambda x: x['lang_cluster'] == lang).sort('ntokens')
        outlier_threshold_up = statistics.quantiles(temp_ds['ntokens'], n=4)[2] + 1.5 * (statistics.quantiles(temp_ds['ntokens'], n=4)[2] - statistics.quantiles(temp_ds['ntokens'], n=4)[0])
        outlier_threshold_low = statistics.quantiles(temp_ds['ntokens'], n=4)[0] - 1.5 * (statistics.quantiles(temp_ds['ntokens'], n=4)[2] - statistics.quantiles(temp_ds['ntokens'], n=4)[0])
        in_distr_lower_bound = max(outlier_threshold_low, min(temp_ds['ntokens']))
        in_distr_upper_bound = min(outlier_threshold_up, max(temp_ds['ntokens']))
        interval_length = int((in_distr_upper_bound - in_distr_lower_bound) / 3)
        interval_short = (min(temp_ds['ntokens']), in_distr_lower_bound+interval_length-1)
        interval_medium = (in_distr_lower_bound+interval_length, in_distr_lower_bound+2*interval_length-1)
        interval_long = (in_distr_lower_bound+2*interval_length, max(temp_ds['ntokens']))
        short_codes_ds = temp_ds.filter(lambda x: x['ntokens'] >= interval_short[0] and x['ntokens'] <= interval_short[1])
        medium_codes_ds = temp_ds.filter(lambda x: x['ntokens'] >= interval_medium[0] and x['ntokens'] <= interval_medium[1])
        long_codes_ds = temp_ds.filter(lambda x: x['ntokens'] >= interval_long[0] and x['ntokens'] <= interval_long[1])
        try:
            print(f"start computing rouge results for {lang}")
            long_rouge_result = rouge.compute(predictions=long_codes_ds[pred_field], references=long_codes_ds[ref_field])
            medium_rouge_result = rouge.compute(predictions=medium_codes_ds[pred_field], references=medium_codes_ds[ref_field])
            short_rouge_result = rouge.compute(predictions=short_codes_ds[pred_field], references=short_codes_ds[ref_field])
        except Exception as e:
            logging.error(f"error at {lang}: {e}")
        with open(str(output_dir / f"{lang}_tk_{math.ceil(interval_long[0])}_{interval_long[1]}.json"), "w") as f:
            json.dump({"num":len(long_codes_ds), **long_rouge_result}, f)
        with open(str(output_dir / f"{lang}_tk_{math.ceil(interval_medium[0])}_{interval_medium[1]}.json"), "w") as f:
            json.dump({"num":len(medium_codes_ds), **medium_rouge_result}, f)
        with open(str(output_dir / f"{lang}_tk_{math.ceil(interval_short[0])}_{interval_short[1]}.json"), "w") as f:
            json.dump({"num":len(short_codes_ds), **short_rouge_result}, f)
def cal_bertscore_by_ntokens(load_path, output_dir, pred_field, ref_field):
    output_dir = output_dir / Path(f'split_3')
    output_dir.mkdir(exist_ok=True, parents=True)
    ds = load_dataset("json", data_files=load_path, split="train").map(add_ntokens)
    langs = ds.unique('lang_cluster')
    for lang in langs:
        temp_ds = ds.filter(lambda x: x['lang_cluster'] == lang).sort('ntokens')
        outlier_threshold_up = statistics.quantiles(temp_ds['ntokens'], n=4)[2] + 1.5 * (statistics.quantiles(temp_ds['ntokens'], n=4)[2] - statistics.quantiles(temp_ds['ntokens'], n=4)[0])
        outlier_threshold_low = statistics.quantiles(temp_ds['ntokens'], n=4)[0] - 1.5 * (statistics.quantiles(temp_ds['ntokens'], n=4)[2] - statistics.quantiles(temp_ds['ntokens'], n=4)[0])
        in_distr_lower_bound = max(outlier_threshold_low, min(temp_ds['ntokens']))
        in_distr_upper_bound = min(outlier_threshold_up, max(temp_ds['ntokens']))
        interval_length = int((in_distr_upper_bound - in_distr_lower_bound) / 3)
        interval_short = (min(temp_ds['ntokens']), in_distr_lower_bound+interval_length-1)
        interval_medium = (in_distr_lower_bound+interval_length, in_distr_lower_bound+2*interval_length-1)
        interval_long = (in_distr_lower_bound+2*interval_length, max(temp_ds['ntokens']))
        short_codes_ds = temp_ds.filter(lambda x: x['ntokens'] >= interval_short[0] and x['ntokens'] <= interval_short[1])
        medium_codes_ds = temp_ds.filter(lambda x: x['ntokens'] >= interval_medium[0] and x['ntokens'] <= interval_medium[1])
        long_codes_ds = temp_ds.filter(lambda x: x['ntokens'] >= interval_long[0] and x['ntokens'] <= interval_long[1])
        try:
            print(f"start computing bertscore results for {lang}")
            long_bertscore_result = bertscore.compute(lang="en", predictions=long_codes_ds[pred_field], references=long_codes_ds[ref_field])
            medium_bertscore_result = bertscore.compute(lang="en", predictions=medium_codes_ds[pred_field], references=medium_codes_ds[ref_field])
            short_bertscore_result = bertscore.compute(lang="en", predictions=short_codes_ds[pred_field], references=short_codes_ds[ref_field])
        except Exception as e:
            logging.error(f"error at {lang}: {e}")
        with open(str(output_dir / f"{lang}_tk_{math.ceil(interval_long[0])}_{interval_long[1]}.json"), "w") as f:
            precisions = long_bertscore_result['precision']
            recalls = long_bertscore_result['recall']
            f1s = long_bertscore_result['f1']
            json.dump({"num":len(long_codes_ds), 'avg_precision': sum(precisions)/len(precisions), 'avg_recall': sum(recalls)/len(recalls), 'avg_f1': sum(f1s)/len(f1s), **long_bertscore_result}, f)
        with open(str(output_dir / f"{lang}_tk_{math.ceil(interval_medium[0])}_{interval_medium[1]}.json"), "w") as f:
            precisions = medium_bertscore_result['precision']
            recalls = medium_bertscore_result['recall']
            f1s = medium_bertscore_result['f1']
            json.dump({"num":len(medium_codes_ds), 'avg_precision': sum(precisions)/len(precisions), 'avg_recall': sum(recalls)/len(recalls), 'avg_f1': sum(f1s)/len(f1s), **medium_bertscore_result}, f)
        with open(str(output_dir / f"{lang}_tk_{math.ceil(interval_short[0])}_{interval_short[1]}.json"), "w") as f:
            precisions = short_bertscore_result['precision']
            recalls = short_bertscore_result['recall']
            f1s = short_bertscore_result['f1']
            json.dump({"num":len(short_codes_ds), 'avg_precision': sum(precisions)/len(precisions), 'avg_recall': sum(recalls)/len(recalls), 'avg_f1': sum(f1s)/len(f1s), **short_bertscore_result}, f)
def cal_meteor_by_ntokens(load_path, output_dir, pred_field, ref_field):
    output_dir = output_dir / Path(f'split_3')
    output_dir.mkdir(exist_ok=True, parents=True)
    ds = load_dataset("json", data_files=load_path, split="train").map(add_ntokens)
    langs = ds.unique('lang_cluster')
    for lang in langs:
        temp_ds = ds.filter(lambda x: x['lang_cluster'] == lang).sort('ntokens')
        outlier_threshold_up = statistics.quantiles(temp_ds['ntokens'], n=4)[2] + 1.5 * (statistics.quantiles(temp_ds['ntokens'], n=4)[2] - statistics.quantiles(temp_ds['ntokens'], n=4)[0])
        outlier_threshold_low = statistics.quantiles(temp_ds['ntokens'], n=4)[0] - 1.5 * (statistics.quantiles(temp_ds['ntokens'], n=4)[2] - statistics.quantiles(temp_ds['ntokens'], n=4)[0])
        in_distr_lower_bound = max(outlier_threshold_low, min(temp_ds['ntokens']))
        in_distr_upper_bound = min(outlier_threshold_up, max(temp_ds['ntokens']))
        interval_length = int((in_distr_upper_bound - in_distr_lower_bound) / 3)
        interval_short = (min(temp_ds['ntokens']), in_distr_lower_bound+interval_length-1)
        interval_medium = (in_distr_lower_bound+interval_length, in_distr_lower_bound+2*interval_length-1)
        interval_long = (in_distr_lower_bound+2*interval_length, max(temp_ds['ntokens']))
        short_codes_ds = temp_ds.filter(lambda x: x['ntokens'] >= interval_short[0] and x['ntokens'] <= interval_short[1])
        medium_codes_ds = temp_ds.filter(lambda x: x['ntokens'] >= interval_medium[0] and x['ntokens'] <= interval_medium[1])
        long_codes_ds = temp_ds.filter(lambda x: x['ntokens'] >= interval_long[0] and x['ntokens'] <= interval_long[1])
        long_meteor_results = []
        medium_meteor_results = []
        short_meteor_results = []
        try:
            logging.info(f"start computing bleu results for {lang}")
            for d in long_codes_ds:
                references = [d[ref_field].split()]
                candidate = d[pred_field].split()
                score = meteor_score(references, candidate)
                long_meteor_results.append(score)
            long_results = {"num":len(long_codes_ds), "min_ntokens":min(long_codes_ds['ntokens']), "max_ntokens":max(long_codes_ds['ntokens']), "avg": sum(long_meteor_results)/len(long_meteor_results), "meteor_results": long_meteor_results}
            for d in medium_codes_ds:
                references = [d[ref_field].split()]
                candidate = d[pred_field].split()
                score = meteor_score(references, candidate)
                medium_meteor_results.append(score)
            medium_results = {"num":len(medium_codes_ds), "min_ntokens":min(medium_codes_ds['ntokens']), "max_ntokens":max(medium_codes_ds['ntokens']), "avg": sum(medium_meteor_results)/len(medium_meteor_results), "meteor_results": medium_meteor_results}
            for d in short_codes_ds:
                references = [d[ref_field].split()]
                candidate = d[pred_field].split()
                score = meteor_score(references, candidate)
                short_meteor_results.append(score)
            short_results = {"num":len(short_codes_ds), "min_ntokens":min(short_codes_ds['ntokens']), "max_ntokens":max(short_codes_ds['ntokens']), "avg": sum(short_meteor_results)/len(short_meteor_results), "meteor_results": short_meteor_results}
        except Exception as e:
            logging.error(f"error at {lang}: {e}")
        with open(str(output_dir / f"{lang}_tk_{math.ceil(interval_long[0])}_{interval_long[1]}.json"), "w") as f:
            json.dump(long_results, f)
        with open(str(output_dir / f"{lang}_tk_{math.ceil(interval_medium[0])}_{interval_medium[1]}.json"), "w") as f:
            json.dump(medium_results, f)
        with open(str(output_dir / f"{lang}_tk_{math.ceil(interval_short[0])}_{interval_short[1]}.json"), "w") as f:
            json.dump(short_results, f)

def main():
    llm_infer_result = args.llm_infer_result
    model_name = llm_infer_result.split('_')[-1].split('.')[0]
    load_path = str(Path(__file__).parent.parent / Path("inference") / Path("results") / Path(llm_infer_result))
    output_dir = Path(__file__).parent / Path('summ_scores') / Path(model_name)
    output_dir.mkdir(exist_ok=True, parents=True)
    # 1. calculate overall code summarization performance
    avg_bertscore = cal_bertscore(load_path, output_dir, args.pred_field, args.ref_field)['avg_f1']
    avg_meteor = cal_meteor(load_path, output_dir, args.pred_field, args.ref_field)['avg']
    avg_bleu = cal_bleu(load_path, output_dir, args.pred_field, args.ref_field)['bleu']
    avg_rouge = cal_rouge(load_path, output_dir, args.pred_field, args.ref_field)['rougeL']
    overall = (avg_bertscore + avg_meteor + avg_bleu + avg_rouge) / 4
    print(f"calculating {model_name}'s performance\n- bleu: {avg_bleu}\n- rouge: {avg_rouge}\n- bertscore: {avg_bertscore}\n- meteor: {avg_meteor}\n-overall: {overall}")
    # 2. calculate per-language code summarization performance
    meteor_output_dir = output_dir / Path("meteor")
    bleu_output_dir = output_dir / Path("bleu")
    rouge_output_dir = output_dir / Path("rouge")
    bertscore_output_dir = output_dir / Path("bertscore")
    meteor_output_dir.mkdir(exist_ok=True, parents=True)
    bleu_output_dir.mkdir(exist_ok=True, parents=True)
    rouge_output_dir.mkdir(exist_ok=True, parents=True)
    bertscore_output_dir.mkdir(exist_ok=True, parents=True)
    cal_bleu_by_lang(load_path, bleu_output_dir, args.pred_field, args.ref_field)
    cal_rouge_by_lang(load_path, rouge_output_dir, args.pred_field, args.ref_field)
    cal_meteor_by_lang(load_path, meteor_output_dir, args.pred_field, args.ref_field)
    cal_bertscore_by_lang(load_path, bertscore_output_dir, args.pred_field, args.ref_field)
    # 3. calculate per-language code summarization performance by code length
    bleu_output_dir = output_dir / Path("bleu")
    rouge_output_dir = output_dir / Path("rouge")
    meteor_output_dir = output_dir / Path("meteor")
    bertscore_output_dir = output_dir / Path("bertscore")
    bleu_output_dir.mkdir(exist_ok=True, parents=True)
    rouge_output_dir.mkdir(exist_ok=True, parents=True)
    meteor_output_dir.mkdir(exist_ok=True, parents=True)
    bertscore_output_dir.mkdir(exist_ok=True, parents=True)
    cal_bleu_by_ntokens(load_path, bleu_output_dir, args.pred_field, args.ref_field)
    cal_rouge_by_ntokens(load_path, rouge_output_dir, args.pred_field, args.ref_field)
    cal_meteor_by_ntokens(load_path, meteor_output_dir, args.pred_field, args.ref_field)
    cal_bertscore_by_ntokens(load_path, bertscore_output_dir, args.pred_field, args.ref_field)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_field", type=str, default="code_sum_candidate")
    parser.add_argument("--ref_field", type=str, default="human_summarization")
    parser.add_argument("--llm_infer_result", type=str, default="code_summ_infer_codellama.jsonl")
    args = parser.parse_args()
    bleu = evaluate.load('bleu')
    rouge = evaluate.load('rouge')
    bertscore = evaluate.load("bertscore")
    main()

