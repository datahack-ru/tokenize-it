from xml.etree import ElementTree as ET
from os import listdir as ls
from os.path import isfile, isdir, join
from fnmatch import fnmatch
import regex
import grmodel


import logging

log = logging.getLogger(__name__)

def parse(path='dict.opcorpora.txt', misses=None):
    res = []
    re = regex.compile('^(?P<W>.+)\t+(?P<S>[\w-,]+)( (?P<F>[\w-,]+))? ?$')
    with open(path) as f:
        for line in f.readlines():
            match = re.match(line)
            if not match:
                pass # TODO: add some handling
            else:
                grammems = grmodel.get_grammems(match, misses)[0]
                lemma = match.group('W').lower()
                print(lemma)
    return res
