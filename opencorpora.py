from xml.etree import ElementTree as ET
from os import listdir as ls
from os.path import isfile, isdir, join
from fnmatch import fnmatch
import regex
import grmodel

import pandas as pd


import logging

log = logging.getLogger(__name__)

def parse(path='../dict.opcorpora.txt', misses=None):
    res = {}
    re = regex.compile('^(?P<W>.+(?=\t))\t+(?P<S>[\w-,]+)( (?P<F>[\w-,]+))? ?$')
    with open(path) as f:
        for line in f.readlines():
            match = re.match(line)
            if not match:
                pass # TODO: add some handling
            else:
                grammems = grmodel.get_grammems(match, misses)[0]
                if grmodel.Abbr.__name__ in grammems.keys():
                    continue
                lemma = match.group('W').lower()
                if lemma not in res.keys():
                    res[lemma] = []
                grammems['w'] = lemma
                f = False
                for g in res[lemma]:
                    if grammems == g:
                        f = True
                        break
                if not f: res[lemma].append(grammems)
    return res

def words(oc):
    return pd.DataFrame([v for vs in oc.values() for v in vs])
