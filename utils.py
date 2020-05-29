from tqdm.auto import tqdm
import os
import pickle
import shutil
import unicodedata
from seqeval.metrics import precision_score, recall_score, f1_score

def get_lexicon_matching(args, tokenizer):
    cache_path = os.path.join(args.cache_dir,'lexicon_cache', '{}-{}.bin'.format(args.src, args.tgt))

    if os.path.exists(cache_path):
        return pickle.load(open(cache_path, 'rb'))

    if not os.path.isdir(os.path.join(args.cache_dir,'lexicon_cache')):
        os.mkdir(os.path.join(args.cache_dir,'lexicon_cache'))

    forward_matching, backward_matching = {}, {}


    with open(args.lexicon_path, 'r') as f:
        for line in tqdm(f):
            line = line.strip()
            word, word_trans = line.split('\t') if '\t' in line else line.split()
            if word == word_trans: continue
            word_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
            word_trans_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word_trans))

            if len(word_id) == 1 and len(word_trans_id) == 1:
                word_id = word_id[0]
                word_trans_id = word_trans_id[0]
                
                if word_id not in forward_matching:
                    forward_matching[word_id] = word_trans_id
                
                if word_trans_id not in backward_matching:
                    backward_matching[word_trans_id] = word_id
    bidir = (forward_matching, backward_matching)    
    pickle.dump(bidir, open(cache_path, 'wb'))
    return bidir


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
    return (precision_score(labels, preds), 
        recall_score(labels, preds),
        f1_score(labels, preds))