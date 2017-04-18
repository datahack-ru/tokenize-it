from xml.etree import ElementTree as ET
from os import listdir as ls
from os.path import isfile, isdir, join
from fnmatch import fnmatch
import regex
import grmodel

import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy
import sklearn.preprocessing

import logging

log = logging.getLogger(__name__)

def parse(p='./../sample_ar/', w=True, misses=None):
    res = []
    re = regex.compile('^(?P<S>[\w-,]+)=?(\(?((?P<F>[\w,]+)\|?)+\)?)?(=[\w,]+)*$')
    cre = regex.compile('[^\w]')
    for f in [join(p, f) for f in ls(p)]:
        if isfile(f) and fnmatch(f, '*.xhtml'):
            print(f)
            for s in ET.parse(f).getroot().iter('se'):
                tokens = []
                for word in s.iter('w'):
                    token = cre.sub('', ' '.join(word.itertext())).lower()

                    for ana in word.iter('ana'):
                        lexem = ana.attrib['lex']
                        gr = ana.attrib['gr']
                        match = re.match(gr)
                        if not match:
                            log.warn('Match failed for "%s"', gr)
                        else:
                            # НКРЯ размечен людьми. Разметка всегда однозначна.
                            grammems = grmodel.get_grammems(match, misses)[0]
                            if w:
                                grammems['w'] = token
                            grammems['l'] = len(token)
                            tokens.append(grammems)
                if(len(tokens) > 0):
                    res.append(tokens)
        elif isdir(f):
            res = res + parse(f, w, misses)
    return res

def words(sentences):
    res = []
    for sentence in sentences:
        for word in sentence:
            res.append(word)
    return pd.DataFrame(res, dtype="category")

def plot_wordpairs(nc):
    for i in range(1, 5):
        wp = wordpairs(nc, i)
        pt = pd.crosstab(wp.w1_PoS, wp.w2_PoS, margins=False, normalize='index')
        f = plt.figure()
        ax = f.add_subplot(111)
        sns.set_palette(sns.color_palette("OrRd", 10))
        sns.heatmap(pt, linewidths=0.1, ax=ax, cmap="YlGnBu")
        figure_config(f)
        plt.savefig("wp_pos_" + str(i) + ".png", bbox_inches='tight')
        plt.close(f)

def figure_config(f):
    for ax in f.axes:
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        for tick in ax.get_yticklabels():
            tick.set_rotation(0)

def prep_noun(sentences):
    res = []
    for sentence in sentences:
        prep = None
        for word in sentence:
            if 'PoS' in word:
                if word['PoS'] == grmodel.PoS.PREP:
                    prep = word
                if word['PoS'] == grmodel.PoS.NOUN:
                    if prep != None:
                        word['prep'] = prep['lex'].lower()
                        res.append(word)
                        prep = None
    return pd.DataFrame(res, dtype="category")[['Case', 'prep']]

def pn_plot(pn):
    f = plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
    ax = f.add_subplot(111)

    p_ = pn.loc[pn.prep.isin(['к', 'о', 'от', 'для', 'без','о'])]

    x = sns.countplot(x="Case", hue="prep", data=p_)

    # for p in ['к', 'о', 'от', 'для', 'без','о']:
    #     pn[pn.prep == p].Case.value_counts(normalize=True, sort=False).plot(ax=ax, kind='bar')

    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    return f


def plot_1(words):
    plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
    plt.xticks(rotation=45)
    return sns.countplot(x = 'PoS', data=words, color="black")

def plot_2(words):
    plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
    plt.xticks(rotation=45)
    return sns.countplot(x = 'Case', data=words[words.PoS == grmodel.PoS.NOUN], color="black")

def enum_value(obj):
    if obj is numpy.NaN:
        return 0
    else:
        return obj.value

class LearnExp(object):
    def __init__(self, ncpath='sample_ar'):
        self.nc = parse(ncpath)
        self.enc = sklearn.preprocessing.OneHotEncoder()

    def build_wp(self, distance=1):
        self.wp = wordpairs(self.nc, distance)
        self.wp_int = self.wp.drop(['w1_lex', 'w2_lex'], axis=1).applymap(lambda x: enum_value(x))
        self.enc.fit(self.wp_int)
        self.learn_set = self.enc.transform(self.wp_int)
