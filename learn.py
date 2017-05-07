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

import numpy as np
import scipy.stats
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

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


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

def compute_training_set(nc, oc, dist=2, chunks=True, limit=None, path="training_set.h5"):
    if not limit:
        limit = len(nc)

    log.info("collecting pairs from national corpora")
    nc_pairs = pd.DataFrame()
    if chunks:
        nc_pairs = nc_pairs.append(wordchunks(nc[:limit], chunks=True, dist=dist), ignore_index=True)
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
            in_pairs_list.append(wordchunks(tokenizations, chunks=True, dist=dist))
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
    store['chunks' + str(dist)] = all_pairs
    store.close()

def opencorpora_todf(oc):
    return pd.DataFrame([entry for key in oc for entry in oc[key]])

def common_columns(df1, df2, exclude=['w1', 'w2']):
    columns = set(df1.columns).intersection(set(df2.columns))
    for c in exclude: columns.remove(c)
    return list(columns)

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

def tok_learn(l, oc, text):
    t = tok.Tok(oc)
    t.enable_boom_protection()

    sentences = []
    for s in t.tok(text, gr=True):
        if not s or len(s) < 3:
            continue
        pairs = pd.DataFrame(wordchunks([s], chunks=True, dist=2))
        score = np.mean(l.predict(pairs), axis=0)
        sentences.append((score[1], s))

    sentences.sort(key=lambda x:x[0])
    return sentences


# def unique_pairs(nc, dist=1):
#     wp = natcorp.wordpairs(nc, dist).drop(['w1_wf', 'w2_wf'], axis=1)
#
#     nc_shuffled = nc[:]
#     random.shuffle(nc_shuffled)
#     wp_shuffled = natcorp.wordpairs(nc_shuffled, dist).drop(['w1_wf', 'w2_wf'], axis=1)
#
#     l = len(wp.index)
#     x = [i for i in range(10, l, 50000)] + [l]
#
#     y1 = []
#     y2 = []
#
#     for i in x:
#         print(i)
#         y1.append(len(wp.loc[:i].drop_duplicates().index))
#         y2.append(len(wp_shuffled.loc[:i].drop_duplicates().index))
#
#     f = plt.figure()
#     ax = f.add_subplot(111)
#     ax.plot(x, y1, ls='-', color='red', linewidth=0.9)
#     ax.plot(x, y2, ls='-', color='black', linewidth=0.9)
#     plt.savefig("nc_uniq_pairs" + str(dist) + ".png", bbox_inches='tight')

def config_ax(ax, xname=None, yname=None, legend=False, grid=True, xgr=False, ygr=False):
    sns.reset_orig()
    fp = fm.FontProperties(fname='Times_New_Roman.ttf', size=14)
    fp_ticks = fm.FontProperties(fname='Times_New_Roman.ttf', size=14)

    if not legend:
        if ax.legend_:
            ax.legend_.remove()

    if xname:
        ax.set_xlabel(xname)
        ax.xaxis.label.set_font_properties(fp)
    if yname:
        ax.set_ylabel(yname)
        ax.yaxis.label.set_font_properties(fp)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in [ax.spines[k] for k in ['left', 'bottom']]:
        spine.set_visible(True)
        spine.set_fill(True)
        spine.set_color('black')
        spine.set_linestyle('-')
        spine.set_linewidth(1.5)

    ax.tick_params(direction='out', length=5, width=0.5, colors='black')
    if grid:
        ax.grid(color='black', linestyle=':', linewidth=0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    grlabel = lambda tick, gr: grmodel.labels[gr][gr(int(float(tick.get_text())))]
    if ygr:
        ax.set_yticklabels([grlabel(t, ygr) for t in ax.get_yticklabels()])
    if xgr:
        ax.set_xticklabels([grlabel(t, xgr) for t in ax.get_xticklabels()])

    for tick in ax.get_yticklabels():
        tick.set_font_properties(fp_ticks)
        tick.set_rotation(0)

    for tick in ax.get_xticklabels():
        tick.set_font_properties(fp_ticks)
        tick.set_rotation(45)

class WordsLenDist:
    def __init__(self, oc):
        df = pd.DataFrame([x for l in oc.values() for x in l])
        df['l'] = df.w.map(lambda x: len(x))
        total = df.shape[0]
        ranges = [0,2,4,6,8,10,12,14,16,18,20,22,50]
        self.c = df.groupby(pd.cut(df.l, ranges)).count()
        self.c.l = self.c.l / total
        #self.c = df.l.value_counts(normalize=True, sort=False)

    def plot(self):
        f = plt.figure(figsize=(6, 4))
        ax = f.add_subplot(111)
        self.c.l.plot(kind='bar', color='black', edgecolor="none", label="jui", ax=ax)
        config_ax(ax, xname='Количество букв в словоформе', yname='Доля словоформ')
        plt.savefig("images/" + self.__class__.__name__ + ".png", bbox_inches='tight')
        plt.savefig("images/" + self.__class__.__name__ + ".eps", format='eps', dpi=1000, bbox_inches='tight')

class AmountOfDictionaryEntriesDist:
    def __init__(self, oc):
        df = pd.DataFrame([x for l in oc.values() for x in l])
        df['l'] = df.w.map(lambda x: len(x))
        df = df.groupby('w').size().to_frame('cnt').reset_index()
        total = df.shape[0]
        ranges = [0,1,2,3,4,5,40]
        self.c = df.groupby(pd.cut(df.cnt, ranges)).count()
        self.c.cnt = self.c.cnt / total
        #self.c = df.cnt.value_counts(normalize=True, sort=False)

    def plot(self):
        f = plt.figure(figsize=(6, 4))
        ax = f.add_subplot(111)
        self.c.cnt.plot(kind='bar', color='black', edgecolor="none", label="jui", ax=ax)
        config_ax(ax, xname='Количество статей, приходящихся на словоформу', yname='Доля словоформ')
        plt.savefig("images/" + self.__class__.__name__ + ".png", bbox_inches='tight')
        plt.savefig("images/" + self.__class__.__name__ + ".eps", format='eps', dpi=1000, bbox_inches='tight')

class AmountOfDictionaryEntriesWordLenCrossDist:
    def __init__(self, oc):
        df = pd.DataFrame([x for l in oc.values() for x in l])
        df['l'] = df.w.map(lambda x: len(x))
        df = df.groupby('w').size().to_frame('cnt').reset_index()
        self.c = df
        self.c['l'] = self.c.w.map(lambda x: len(x))
        self.c = self.c[['cnt', 'l']].drop_duplicates()

    def plot(self):
        sns.set(font='Times New Roman')
        g = sns.jointplot("l", "cnt", data=self.c.iloc[1:1000000], kind='hex', stat_func=scipy.stats.kendalltau)
        g.set_axis_labels("Длина словоформы", "Количество словарных статей")
        g.savefig("images/" + self.__class__.__name__ + ".png", bbox_inches='tight', size=5)
        g.savefig("images/" + self.__class__.__name__ + ".eps", format='eps', dpi=1000, bbox_inches='tight', size=5, aspect=1)

class SentenceLenDist:
    def __init__(self, nc):
        df = pd.DataFrame({"l": [len(x) for x in nc]})
        total = df.shape[0]
        ranges = [0,4,8,12,16,20,24,28,32,36,40,44,100]
        self.c = df.groupby(pd.cut(df.l, ranges)).count()
        self.c.l = self.c.l / total

    def plot(self):
        f = plt.figure(figsize=(6, 4))
        ax = f.add_subplot(111)
        self.c.l.plot(kind='bar', color='black', edgecolor="none", ax=ax)
        config_ax(ax, xname='Количество слов в предложений', yname='Доля предложений в НКРЯ')
        plt.savefig("images/" + self.__class__.__name__ + ".png", bbox_inches='tight')
        plt.savefig("images/" + self.__class__.__name__ + ".eps", format='eps', dpi=1000, bbox_inches='tight')

class GrCorellation:
    def __init__(self, nc, dist):
        self.wp = wordchunks(nc, dist=dist)
        self.dist = dist

    def plot(self):
        ctxs = []
        ctxs.append((self.wp.PoS1, self.wp.PoS2, grmodel.PoS, (6, 4)))
        ctxs.append((self.wp.Case1, self.wp.Case2, grmodel.Case, (6, 4)))
        ctxs.append((self.wp.Number1, self.wp.Number2, grmodel.Number, (4, 3)))
        ctxs.append((self.wp.Gender1, self.wp.Gender2, grmodel.Gender, (4, 3)))
        ctxs.append((self.wp.Person1, self.wp.Person2, grmodel.Person, (4, 3)))
        ctxs.append((self.wp.Tense1, self.wp.Tense2, grmodel.Tense, (4, 3)))

        for ctx in ctxs:
            pt = pd.crosstab(ctx[0], ctx[1], margins=False, normalize='index')
            f = plt.figure(figsize=ctx[3])
            ax = f.add_subplot(111)
            sns.heatmap(pt, linewidths=0.1, ax=ax, cmap="spectral_r", annot=True, fmt=".2f", annot_kws={"size": 11})
            config_ax(ax, xname='Значение для второго слова', yname='Значение для первого слова', grid=False, xgr=ctx[2], ygr = ctx[2])
            name = 'images/Corellation_{0}_{1}'.format(ctx[2].__name__, self.dist)
            plt.savefig(name + ".png", bbox_inches='tight', size=5)
            plt.savefig(name + ".eps", format='eps', dpi=1000, bbox_inches='tight')

class MyStemPreciesness:
    def __init__(self, nc):
        for s in nc:
            for w in s:
                


# def tok_results_amount(nc, oc):
#     limit = 100
#
#     df = pd.DataFrame()
#
#     t = tok.Tok(oc)
#     #t.enable_boom_protection()
#
#     for i in range(0, limit):
#         s = nc[i]
#         tokenizations = t.tok(maketypo(s), gr=True, fuzzylimit=100)
#         df = df.append({'n': len(tokenizations)}, ignore_index=True)
#
#     f = plt.figure()
#     ax = f.add_subplot(111)
#
#     ranges = range(0, 1000, 50)
#     cnt = df.groupby(pd.cut(df.n, ranges)).count().plot(kind='bar', color='black', edgecolor="none", ax=ax)
#     config_ax(ax, xname='Sentence length', yname='Количество предложений в НКРЯ')
#     plt.savefig("tok_results_amount.png", bbox_inches='tight')


class LearnExp(object):
    def __init__(self, ts, columns=('Gender', 'Case', 'PoS', 'Number')):
        self.feature_columns = [c for c in ts.columns if c.startswith(columns)]
        self.ts = ts[ts.c].iloc[:len(ts[~ts.c].index)].append(ts[~ts.c], ignore_index=True)
        self.ts = self.ts.sample(n=100000).reset_index(drop=True)
        self.targets = self.ts.c
        self.samples = self.ts[self.feature_columns].replace(np.NaN, 0)
        self.enc = sklearn.preprocessing.OneHotEncoder()
        self.enc.fit(self.samples[:6000])
        self.samples = self.enc.transform(self.samples).toarray()

    def predict(self, df):
        df_columns = set(df.columns)
        for c in self.feature_columns:
            if c not in df_columns:
                df[c] = np.NaN
        s = df[self.feature_columns].replace(np.NaN, 0)
        s = self.enc.transform(s)

        return self.model.predict_proba(s.toarray())

    def learn(self):
        self.model = DecisionTreeClassifier()
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

    def cv_svc(self):
        return cross_val_score(svm.SVC(), self.samples, self.targets, cv=5)

    def cv_knn(self):
        return cross_val_score(KNeighborsClassifier(n_neighbors=2), self.samples, self.targets, cv=5)

    def cv_mlp(self):
        mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(90,), random_state=1)
        return cross_val_score(mlp, self.samples, self.targets, cv=5)
