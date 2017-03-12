from xml.etree import ElementTree as ET
from os import listdir as ls
from os.path import isfile, join
import regex
import grmodel

import logging

log = logging.getLogger(__name__)

def parse(p):
    res = []
    re = regex.compile('^(?P<S>[\w-,]+)=?(\(?((?P<F>[\w,]+)\|?)+\)?)?(=[\w,]+)*$')
    for f in [join(p, f) for f in ls(p) if isfile(join(p, f))]:
        for s in ET.parse(f).getroot().iter('se'):
            tokens = []
            for wordform in s.iter('ana'):
                lexem = wordform.attrib['lex']
                match = re.match(wordform.attrib['gr'])
                if not match:
                    log.warn('Match failed for "%s"', gr)
                else:
                    # НКРЯ размечен людьми. Разметка всегда однозначна.
                    grammems = grmodel.get_grammems(match)[0]
                    tokens.append(grammems)
            res.append(tokens)
    return res
