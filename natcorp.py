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

import logging

log = logging.getLogger(__name__)

def parse(p, lex=True, misses=None):
    res = []
    re = regex.compile('^(?P<S>[\w-,]+)=?(\(?((?P<F>[\w,]+)\|?)+\)?)?(=[\w,]+)*$')
    for f in [join(p, f) for f in ls(p)]:
        if isfile(f) and fnmatch(f, '*.xhtml'):
            print(f)
            for s in ET.parse(f).getroot().iter('se'):
                tokens = []
                for wordform in s.iter('ana'):
                    lexem = wordform.attrib['lex']
                    gr = wordform.attrib['gr']
                    match = re.match(gr)
                    if not match:
                        log.warn('Match failed for "%s"', gr)
                    else:
                        # НКРЯ размечен людьми. Разметка всегда однозначна.
                        grammems = grmodel.get_grammems(match, misses)[0]
                        if lex:
                            grammems['lex'] = lexem
                        tokens.append(grammems)
                res.append(tokens)
        elif isdir(f):
            res = res + parse(f, lex, misses)
    return res

def words(sentences):
    res = []
    for sentence in sentences:
        for word in sentence:
            res.append(word)
    return pd.DataFrame(res, dtype="category")

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
    pn[pn.prep == 'к'].plot(x='Case', kind='hist')

def plot_1(words):
    plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
    plt.xticks(rotation=45)
    return sns.countplot(x = 'PoS', data=words, color="black")

def plot_2(words):
    plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
    plt.xticks(rotation=45)
    return sns.countplot(x = 'Case', data=words[words.PoS == grmodel.PoS.NOUN], color="black")
