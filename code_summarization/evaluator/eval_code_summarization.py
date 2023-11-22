import json
import logging
import evaluate
import argparse
import statistics
from pathlib import Path
from datasets import load_dataset
from nltk.translate.meteor_score import meteor_score

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
    langs = set(dataset['lang'])
    print(langs)
    for lang in langs:
        lang_dataset = dataset.filter(lambda x:x['lang']==lang)
        try:
            logging.info(f"start computing bleu results for {lang}")
            bleu_result = bleu.compute(predictions=lang_dataset[pred_field], references=lang_dataset[ref_field])
        except Exception as e:
            logging.error(f"error at {lang}: {e}")
        with open(str(output_dir / f"{lang}.json"), "w") as f:
            json.dump(bleu_result, f)

def cal_rouge_by_lang(load_path, output_dir, pred_field, ref_field):
    dataset = load_dataset("json", data_files=load_path, split="train")
    langs = set(dataset['lang'])
    print(langs)
    for lang in langs:
        lang_dataset = dataset.filter(lambda x:x['lang']==lang)
        try:
            logging.info(f"start computing bleu results for {lang}")
            rouge_result = rouge.compute(predictions=lang_dataset[pred_field], references=lang_dataset[ref_field])
        except Exception as e:
            logging.error(f"error at {lang}: {e}")
        with open(str(output_dir / f"{lang}.json"), "w") as f:
            json.dump(rouge_result, f)

def cal_bertscore_by_lang(load_path, output_dir, pred_field, ref_field):
    dataset = load_dataset("json", data_files=load_path, split="train")
    langs = set(dataset['lang'])
    print(langs)
    for lang in langs:
        lang_dataset = dataset.filter(lambda x:x['lang']==lang)
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
    langs = set(dataset['lang'])
    for lang in langs:
        lang_dataset = dataset.filter(lambda x:x['lang']==lang)
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

