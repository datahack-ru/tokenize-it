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

def join_dicts(dict1, suffix1, dict2, suffix2):
    res = {}
    for key in dict1.keys():
        res[key + suffix1] = dict1[key]
    for key in dict2.keys():
        res[key + suffix2] = dict2[key]
    return res

def wordpairs(sentences, dist=1, limit=None):
    res = []
    for sentence in sentences:
        srange = [i for i in range(len(sentence))]
        for i in srange:
            j = i + dist
            if i + dist in srange:
                pair = join_dicts(sentence[i], "1", sentence[j], "2")
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

def compute_training_set(nc, oc, limit=None, path="training_set.h5"):
    if not limit:
        limit = len(nc)

    log.info("collecting pairs from national corpora")
    nc_pairs = pd.DataFrame()
    nc_pairs = nc_pairs.append(wordpairs(nc[:limit], 1), ignore_index=True)
    nc_pairs = nc_pairs.append(wordpairs(nc[:limit], 2), ignore_index=True)
    nc_pairs = nc_pairs.append(wordpairs(nc[:limit], 3), ignore_index=True)

    log.info("collecting tokenization pairs")
    in_pairs = pd.DataFrame()
    t = tok.Tok(oc)
    for i in range(0, limit):
        if i % 100 == 0:
            log.info("tokenized %s of sentences", '{:.1%}'.format(i / limit))
        s = nc[i][:4]
        tokenizations = t.tok(maketypo(s), gr=True, fuzzylimit=5)
        if len(tokenizations) > 15:
            tokenizations = tokenizations[15:25]
        else:
            continue
        in_pairs = in_pairs.append(wordpairs(tokenizations, 1), ignore_index=True)
        in_pairs = in_pairs.append(wordpairs(tokenizations, 2), ignore_index=True)
        in_pairs = in_pairs.append(wordpairs(tokenizations, 3), ignore_index=True)
    in_pairs = in_pairs

    log.info("building training set")
    columns = set(nc_pairs.columns).intersection(set(in_pairs.columns))
    for c in ['w1', 'w2']: columns.remove(c)

    log.info("drop duplicates for national corpora")
    nc_pairs = nc_pairs.drop_duplicates(subset=columns)
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
