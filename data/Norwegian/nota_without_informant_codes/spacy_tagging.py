# via terminal: pip install spacy
# via terminal: pip install pandas
# via terminal: pip install https://github.com/explosion/spacy-models/releases/download/nb_core_news_md-3.7.0/nb_core_news_md-3.7.0.tar.gz

import spacy
import pandas as pd 
import os

# load language model
nlp = spacy.load("nb_core_news_md")

# read data
tx = pd.read_csv("nota_all.csv")

# list with only texts
tx = tx["utterance"].values.tolist()

# iterate over all list items
for i in tx:

    # apply model
    doc = nlp(str(i))

    #for token in doc:
     #   print(token.lemma_)

    # lists: entities, iob, pos, lemma
    entities = []
    iob      = []
    pos      = []
    lemma    = []
    enttype  = []

    # get annotations
    for tok in doc:
        entities.append(tok.ent_type_)
        iob.append(tok.ent_iob_)
        pos.append(tok.pos_)
        lemma.append(tok.lemma_)
        enttype.append(tok.ent_type)

    # list of tokens
    toks = list(tok.doc)

    # combine lists  to df
    d = pd.DataFrame(data = {'tok': toks,
                             'pos': pos,
                             'lemma': lemma,
                             'entities': entities,
                            'iob': iob,
                            'enttype': enttype})


    # export
    with open('nota_tagged.csv', 'a') as f:
        d.to_csv(f, header=False)
