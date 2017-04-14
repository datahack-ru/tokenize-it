from xml.etree import ElementTree as ET
from os import listdir as ls
from os.path import isfile, isdir, join
from fnmatch import fnmatch
import regex
import grmodel
import itertools

import logging

log = logging.getLogger(__name__)

class Tok:
    def __init__(self, dict):
        self.dict = dict

    def words(self, text):
        res = []
        for i in range(1, len(text) + 1):
            token = text[:i]
            if token in self.dict.keys():
                res.append([token, text[i:]])
        return res

    def tokenize(self, text):
        text.lower()
        res = []
        queue = self.words(text)
        while queue:
            seq = queue.pop()
            tail = seq.pop()
            if tail:
                for s in self.words(tail):
                    queue = queue + [seq + s]
            else:
                res.append(seq)
        return res

    def addgr(self, sentence):
        res = []
        for word in sentence:
            entries = self.dict[word]
            for entry in entries:
                entry['wf'] = word
            res.append(entries)
        return itertools.product(*res)
