#coding: utf-8
import numpy as np
from public import utils

from data_utils import FEATURES, STATEVEC, CARDLIST, N_STATE, N_CARD

class DominionCardBase(object):
  def __init__(self, data):
    self.jname = data['名前']
    self.ename = data['Name']
    #self.set = data['']
    self.cost = int(data['コスト'])
    self.vp = int(data['勝利点'])
    self.coin = int(data['財宝'])
    self.card_type = data['種類']
    self.p_card = int(data['+card'])
    self.p_action = int(data['+action'])
    self.p_buy = int(data['+buy'])
    self.p_coin = int(data['+coin'])
    self.p_VPtoken = int(data['+VPtoken'])
  def __str__(self):
    self_str = ""
    d = self.__dict__
    for k in d:
      self_str += "%s: %s\n" % (k, str(d[k]))
    return self_str

def setup_card(card_id):
  card = DominionCardBase(CARDLIST[card_id])
  return card

print setup_card(1)
