from tqdm.auto import tqdm
import os
import pickle
import shutil
import unicodedata
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score
from nltk.corpus import stopwords
from collections import defaultdict as dd

def get_lexicon_matching(args, tokenizer):
    cache_path = os.path.join(args.cache_dir,'lexicon_cache', '{}-{}.bin'.format(args.src, args.tgt))

    if os.path.exists(cache_path):
        return pickle.load(open(cache_path, 'rb'))

    if not os.path.isdir(os.path.join(args.cache_dir,'lexicon_cache')):
        os.mkdir(os.path.join(args.cache_dir,'lexicon_cache'))

    matching = dd(list)
    sw = set(stopwords.words())

    with open(args.lexicon_path, 'r', encoding="utf-8") as f:
        for line in tqdm(f):
            line = line.strip()
            word, word_trans = line.split('\t') if '\t' in line else line.split()
            if word == word_trans: continue
            if word.lower() in sw or word_trans.lower() in sw: continue
            word_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
            word_trans_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word_trans))

            if len(word_id) > 1 or len(word_trans_id) > 1: continue
            word_id = word_id[0]
            word_trans_id = word_trans_id[0]
                
            matching[word_id].append(word_trans_id)
            matching[word_trans_id].append(word_id)

    pickle.dump(matching, open(cache_path, 'wb'))
    return matching


def lowercase_and_remove_accent(text):
    """
    Lowercase and strips accents from a piece of text based on
    https://github.com/facebookresearch/XLM/blob/master/tools/lowercase_and_remove_accent.py
    """
    
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return ''.join(output)


def score_predictions(preds, labels):
    return (accuracy_score(labels, preds),
        precision_score(labels, preds), 
        recall_score(labels, preds),
        f1_score(labels, preds))