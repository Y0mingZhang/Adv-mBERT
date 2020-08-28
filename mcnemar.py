import os
import sys
import glob
from seqeval.metrics.sequence_labeling import get_entities
from statsmodels.stats.contingency_tables import mcnemar


def parse_from_path(path):
    pred_files = glob.glob(os.path.join(path,"**/test_*_predictions.txt"), recursive=True)
    return parse_predictions_all_langs(pred_files)

def parse_predictions_all_langs(pred_files):
    all_preds = {}
    for f in pred_files:
        lang = f.split('/')[-1].split('_')[1]
        if lang == 'en': continue
        labels = []
        curr = []
        for line in open(f):
            if not line.strip() and curr:
                labels.append(curr)
                curr = []
            else:
                curr.append(line.strip())
        all_preds[lang] = set(get_entities(labels))
    return all_preds

def read_gt(path, langset):
    all_preds = {}
    for f in glob.glob(os.path.join(path, 'test*.tsv')):
        lang = f.split('/')[-1].split('-')[-1].split('.')[0]
        if lang not in langset: continue
        labels = []
        curr = []
        for line in open(f):
            if not line.strip() and curr:
                labels.append(curr)
                curr = []
            else:
                curr.append(line.strip().split('\t')[-1])
        all_preds[lang] = set(get_entities(labels))
    return all_preds

def print_p_values(A, B, GT, langset):
    # [[A & B, A & ~B], [~A & B, ~A & ~B]]
    print('LANG', 'PVALUE', sep='\t')
    for lang in langset:
        contingency = [[0,0],[0,0]]
        for item in GT[lang]:
            i = 0 if item in A[lang] else 1
            j = 0 if item in B[lang] else 1
            contingency[i][j] += 1
        print(lang, mcnemar(contingency).pvalue, sep='\t')

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 mcnemar.py PREDS1_PATH PREDS2_PATH GROUNDTRUTH_PATH")
        exit(1)

    preds1_path, preds2_path, gt_path = sys.argv[1:]
    preds1 = parse_from_path(preds1_path)
    preds2 = parse_from_path(preds2_path)
    langset = sorted(list(set(preds1.keys()) & set(preds2.keys())))
    print("Common languages:", ','.join(langset))

    gt = read_gt(gt_path, langset)
    print_p_values(preds1, preds2, gt, langset) 

if __name__ == '__main__':
    main()