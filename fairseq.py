import torch
import os
import matplotlib.pyplot as plt

def load_lm(lang):
    assert lang in ["en", "de", "ru"]
    # Load an lang LM trained on WMT'19 News Crawl data
    return torch.hub.load('pytorch/fairseq', 'transformer_lm.wmt19.'+lang, tokenizer='moses', bpe='fastbpe')

def eval_lm(lm, lang, filepath, src_lang):

    assert lang in ["en", "de", "ru"]
    assert lang != src_lang
    assert os.path.isfile(filepath)

    lm.eval()  # disable dropout
    if torch.cuda.is_available() :
        # Move model to GPU
        lm.cuda()
    #else :
    #    lm.cpu()
  
    scores_per_lenght = {}
    positional_scores = []
    raw_scores        = []
    scores            = []

    with open(filepath, 'r', encoding="utf-8") as datasetfile :
        for line in datasetfile.readlines():
            # todo
            line = line[:512]
            
            if len(lm.score(line)["tokens"]) > 10 :
                while len(lm.score(line)["tokens"]) % 10 != 0 :
                    line = line[:len(line)-1]

            scores_tmp = lm.score(line)

            pos_scores = scores_tmp['positional_scores']
            positional_scores.append(pos_scores)
            raw_scores.append(scores_tmp["score"]) # scores_tmp["score"] is equal to scores_tmp['positional_scores'].mean()

            pos_score = pos_scores.mean().neg().exp()
            scores.append(pos_score)

            lenght = len(scores_tmp["tokens"])
            try :
                scores_per_lenght[lenght].append(pos_score)
            except KeyError:
                scores_per_lenght[lenght] = []
                scores_per_lenght[lenght].append(pos_score)
    scores_per_lenght = {lenght : torch.tensor(pos_score).mean() for lenght, pos_score in scores_per_lenght.items()}
    return scores_per_lenght, positional_scores, raw_scores, scores