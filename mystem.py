import pymystem3
import regex
import grmodel

__m = pymystem3.Mystem()

def get_grammems(token):
    analysis = __m.analyze(token)[0]["analysis"]
    if not analysis:
        return None
    gr = analysis[0]["gr"]
    r = regex.match('(?P<S>[\w,]+)=(\(?((?P<F>[\w,]+)\|?)+\)?)?', gr)
    return grmodel.get_grammems(r)
