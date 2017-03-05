#!/usr/bin/python
# -*- coding: utf8 -*-

from pymystem3 import Mystem
from enum import Enum

__m = Mystem()

def get_features(token):
    res = {}
    gr = __m.analyze(token)[0]["analysis"][0]["gr"]
    for g in gr.replace('=',',').split(','):
        if g in __mapping_ru:
            val = __mapping_ru[g]
            res[val.__class__] = val
    return res, gr


class PoS(Enum):
    A      = 1  # прилагательное
    ADV	   = 2  # наречие
    ADVPRO = 3  # местоименное наречие
    ANUM   = 4  # числительное-прилагательное
    APRO   = 5  # местоимение-прилагательное
    COM    = 6  # часть композита - сложного слова
    CONJ   = 7  # союз
    INTJ   = 8  # междометие
    NUM    = 9  # числительное
    PART   = 10 # частица
    PR     = 11 # предлог
    S      = 12 # существительное
    SPRO   = 13 # местоимение-существительное
    V      = 14 # глагол

class Tense(Enum):
    PRAES   = 1 # настоящее
    INPRAES = 2 # непрошедшее
    PRAET   = 3 # прошедшее

class Case(Enum):
    NOM  = 1 # именительный
    GEN  = 2 # родительный
    DAT  = 3 # дательный
    ACC  = 4 # винительный
    INS  = 5 # творительный
    ABL  = 6 # предложный
    PART = 7 # партитив (второй родительный)
    LOC  = 8 # местный (второй предложный)
    VOC  = 9 # звательный

class Quantity(Enum):
     SG = 1 # единственное число
     PL = 2 # множественное число

__mapping_ru = {
    # PoS mapping
    'A'      : PoS.A,
    'ADV'    : PoS.ADV,
    'ADVPRO' : PoS.ADVPRO,
    'ANUM'   : PoS.ANUM,
    'APRO'   : PoS.APRO,
    'COM'    : PoS.COM,
    'CONJ'   : PoS.CONJ,
    'INTJ'   : PoS.INTJ,
    'NUM'    : PoS.NUM,
    'PART'   : PoS.PART,
    'PR'     : PoS.PR,
    'S'      : PoS.S,
    'SPRO'   : PoS.SPRO,
    'V'      : PoS.V,
    # Case mapping
    'им'    : Case.NOM,
    'род'   : Case.GEN,
    'дат'   : Case.DAT,
    'вин'   : Case.ACC,
    'твор'  : Case.INS,
    'пр'    : Case.ABL,
    'парт'  : Case.PART,
    'местн' : Case.LOC,
    'зват'  : Case.VOC,
    # Quantity mapping
    'ед'    : Quantity.SG,
    'мн'    : Quantity.PL,
    # Tense mapping
    'наст'   : Tense.PRAES,
    'непрош' : Tense.INPRAES,
    'прош'   : Tense.PRAET
}
