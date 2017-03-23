from xml.etree import ElementTree as ET
from os import listdir as ls
from os.path import isfile, isdir, join
from fnmatch import fnmatch
import regex
import grmodel

import pandas as pd
import seaborn as sns

import logging

log = logging.getLogger(__name__)

def parse(p, lex=False, misses=None):
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
