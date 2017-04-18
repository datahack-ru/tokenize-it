from xml.etree import ElementTree as ET
from os import listdir as ls
from os.path import isfile, isdir, join
from fnmatch import fnmatch
import regex
import random

import grmodel
import natcorp
import opencorpora
import tok

import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn.preprocessing

from scipy import interpolate

from profilestats import profile

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score

import logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


def maketypo(sentence):
    ws = [w['w'] for w in sentence]
    if len(ws) - 1 > 1:
        i = int(len(ws) / 2)
        ws[i] = ws[i] + ws[i + 1]
        del ws[i + 1]
    return ' '.join(ws)

def join_dicts(dicts):
    res = {}
    for i in range(0, len(dicts)):
        d = dicts[i]
        j = i + 1
        for key in d.keys():
            res[key + str(j)] = d[key]
    return res

def wordchunks(sentences, dist=1, chunks=False, limit=None):
    res = []
    for sentence in sentences:
        srange = [i for i in range(len(sentence))]
        for i in srange:
            j = i + dist
            if j in srange:
                if chunks:
                    pair = join_dicts(sentence[i:j+1])
                else:
                    pair = join_dicts([sentence[i], sentence[j]])
                pair['d'] = dist
                res.append(pair)
        if limit:
            if len(res) > limit:
                break
    res = pd.DataFrame(res, dtype=int)
    if res.size > 0:
        return res.drop_duplicates()
    else:
        return res

def compute_training_set(nc, oc, chunks=True, limit=None, path="training_set.h5"):
    if not limit:
        limit = len(nc)

    log.info("collecting pairs from national corpora")
    nc_pairs = pd.DataFrame()
    if chunks:
        nc_pairs = nc_pairs.append(wordchunks(nc[:limit], chunks=True, dist=2), ignore_index=True)
    else:
        nc_pairs = nc_pairs.append(wordchunks(nc[:limit], 1), ignore_index=True)
        nc_pairs = nc_pairs.append(wordchunks(nc[:limit], 2), ignore_index=True)
        nc_pairs = nc_pairs.append(wordchunks(nc[:limit], 3), ignore_index=True)

    log.info("collecting tokenization pairs")
    in_pairs_list = []
    t = tok.Tok(oc)
    t.enable_boom_protection()
    for i in range(0, limit):
        if i % 100 == 0:
            log.info("tokenized %s of sentences", '{:.1%}'.format(i / limit))
        s = nc[i][:4]
        tokenizations = t.tok(maketypo(s), gr=True, fuzzylimit=10)
        if len(tokenizations) > 15:
            tokenizations = tokenizations[10:42]
        else:
            continue
        if chunks:
            in_pairs_list.append(wordchunks(tokenizations, chunks=True, dist=2))
        else:
            in_pairs_list.append(wordchunks(tokenizations, 1))
            in_pairs_list.append(wordchunks(tokenizations, 2))
            in_pairs_list.append(wordchunks(tokenizations, 3))

    log.info("concatinating tok pairs dataframes into one")
    in_pairs = pd.concat(in_pairs_list)

    log.info("building training set")
    columns = set(nc_pairs.columns).intersection(set(in_pairs.columns))
    for c in [j + str(i) for i in range(1, 6) for j in ['l', 'w']]:
        if c in columns:
            columns.remove(c)

    # log.info("drop duplicates for national corpora")
    # nc_pairs = nc_pairs.drop_duplicates(subset=columns)
    log.info("reset index for national corpora")
    nc_pairs.reset_index()
    log.info("settng national corpora word pairs class to True")
    nc_pairs['c'] = True

    log.info("drop duplicates for tok pairs")
    in_pairs = in_pairs.drop_duplicates(subset=columns)
    log.info("reset index for tok pairs")
    in_pairs.reset_index()
    log.info("settng tok word pairs class to False")
    in_pairs['c'] = False

    log.info("extract incorrect pairs from tok pairs")
    in_pairs = in_pairs.append(nc_pairs, ignore_index=True).drop_duplicates(subset=columns, keep=False)
    in_pairs = in_pairs[~in_pairs.c]

    log.info("composing whole set of all pairs")
    all_pairs = nc_pairs.append(in_pairs)
    log.info("resetting index")
    all_pairs.reset_index()

    log.info("presisting computed set to disk: %s", path)
    store = pd.HDFStore(path)
    store['training_set'] = all_pairs
    store.close()

def common_columns(df1, df2, exclude=['w1', 'w2']):
    columns = set(df1.columns).intersection(set(df2.columns))
    for c in exclude: columns.remove(c)
    return list(columns)

@profile()
def tok_lookup(vdict, oc, text):
    vdict.replace(np.NaN, 0)
    t = tok.Tok(oc)
    sentences = []
    for s in t.tok(text, gr=True):
        if not s:
            continue
        pairs = pd.DataFrame()
        pairs = pairs.append(wordpairs([s], 1), ignore_index=True)
        pairs.replace(np.NaN, 0)
        cols = common_columns(vdict, pairs)
        score = 0
        for row in pairs.iterrows():
            if (vdict[cols] == row[1][cols]).all(1).any():
                score = score + 1
        sentences.append((score, s, pairs))
    sentences.sort(key=lambda x:x[0])
    return sentences


def unique_pairs(nc, dist=1):
    wp = natcorp.wordpairs(nc, dist).drop(['w1_wf', 'w2_wf'], axis=1)

    nc_shuffled = nc[:]
    random.shuffle(nc_shuffled)
    wp_shuffled = natcorp.wordpairs(nc_shuffled, dist).drop(['w1_wf', 'w2_wf'], axis=1)

    l = len(wp.index)
    x = [i for i in range(10, l, 50000)] + [l]

    y1 = []
    y2 = []

    for i in x:
        print(i)
        y1.append(len(wp.loc[:i].drop_duplicates().index))
        y2.append(len(wp_shuffled.loc[:i].drop_duplicates().index))

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.plot(x, y1, ls='-', color='red', linewidth=0.9)
    ax.plot(x, y2, ls='-', color='black', linewidth=0.9)
    plt.savefig("nc_uniq_pairs" + str(dist) + ".png", bbox_inches='tight')

class LearnExp(object):
    def __init__(self, ts, columns=('Gender', 'Case', 'PoS', 'Number', 'Shortness', 'VerbForm', 'Comp', 'Tense')):
        self.feature_columns = [c for c in ts.columns if c.startswith(columns)]
        self.ts = ts[ts.c].iloc[:len(ts[~ts.c].index)].append(ts[~ts.c], ignore_index=True)
        self.ts = self.ts.sample(n=200000).reset_index(drop=True)
        self.targets = self.ts.c
        self.samples = self.ts[self.feature_columns].replace(np.NaN, 0)
        self.enc = sklearn.preprocessing.OneHotEncoder()
        self.enc.fit(self.samples[:6000])
        self.samples = self.enc.transform(self.samples).toarray()

    def predict(self, df):
        s = df[self.feature_columns].replace(np.NaN, 0)
        s = self.enc.transform(s)

        return self.model.predict_proba(s.toarray())

    def learn(self):
        self.model = svm.SVC()
        self.model.fit(self.samples, self.targets)

    def learn_dtc(self):
        self.model = DecisionTreeClassifier()
        log.debug("Samples: %i", len(self.samples))
        log.debug("Targets: %i", len(self.targets.index))
        self.model.fit(self.samples, self.targets)

    def cv(self):#GaussianNB
        return cross_val_score(DecisionTreeClassifier(), self.samples, self.targets, cv=5)

    def cv_gnb(self):
        return cross_val_score(GaussianNB(), self.samples, self.targets, cv=5)

    def cv_knn(self):
        return cross_val_score(KNeighborsClassifier(n_neighbors=2), self.samples, self.targets, cv=5)

    def cv_mlp(self):
        mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        return cross_val_score(mlp, self.samples, self.targets, cv=5)
