# coding: utf-8
import numpy as np
import xml.etree.ElementTree as ET
from public import utils
from public import config
from public.config import RED, YELLOW, BLUE, BOLD
import collections 

# 入力特徴量はFEATURES + stateVec
FEATURES = ['coin', 'buy', 'minusCost', 'turn']
#deck,hand,discard,field,enemy,supply,isIncludedSupply (cardmax*7次元)
STATEVEC = ['deck', 'hand', 'discard', 
            'field', 'enemy', 'supply', 'isIncludedSupply']
N_STATE = len(STATEVEC) 
N_CARD = 58 #58(陰謀), 84(海辺) 

def read_cardlist(file_path="dataset/cardData.csv"):
  res = []
  with open(file_path) as f:
    delim = ','
    header = f.readline().replace('\n', '').split(delim)
    for l in f:
      record = {}
      l = l.split(delim)
      for i in xrange(1, len(header)):
        record[header[i]] = l[i]
      res.append(record)
  return res

CARDLIST = read_cardlist()

def get_feature_start_idx():
  starts = {}
  for i, f in enumerate(FEATURES):
    starts[f] = i

  for state in STATEVEC:
    starts[state] = len(FEATURES) + STATEVEC.index(state) * N_CARD 
  return starts

#サプライに登場済みのカードだけの情報
def get_game_info(input_data, supply_ids):
  
  d = {}
  starts = get_feature_start_idx()
  for feature in FEATURES:
    d[feature] = input_data[starts[feature]]
  for state in STATEVEC:
    d[state] = np.array([input_data[starts[state] + _id] for _id in supply_ids])
  d['cost'] = np.array([int(CARDLIST[_id]['コスト']) for _id in supply_ids])
  d['name'] = [CARDLIST[_id]['名前'] for _id in supply_ids]
  return d

def included_supplies(input_data):
  starts = get_feature_start_idx()
  idx = starts['isIncludedSupply']
  supply_ids = [i for i, val in enumerate(input_data[idx:idx+N_CARD]) if val == 1]
  supply_ids = [0] + supply_ids
  return supply_ids

def explain_state(input_data, answer=None, correct_answer=None):
  coin = input_data[FEATURES.index('coin')]
  buy = input_data[FEATURES.index('buy')]
  turn = input_data[FEATURES.index('turn')]
  minusCost = input_data[FEATURES.index('minusCost')]

  supply_ids = included_supplies(input_data)
  d = get_game_info(input_data, supply_ids)

  playercards = d['deck'] + d['hand'] + d['discard'] + d['field']
  enemycards = d['enemy']
  supply = d['supply']
  txt = ""
  txt += "<Supply>"
  txt += "\n" +  ",  ".join(["%s(%s)[%d]" % (CARDLIST[_id]["名前"], CARDLIST[_id]["コスト"], supply[i]) for i, _id in enumerate(supply_ids[:7])])
  txt += "\n" +  ",  ".join(["%s(%s)[%d]" % (CARDLIST[_id]["名前"], CARDLIST[_id]["コスト"], supply[i+7]) for i, _id in enumerate(supply_ids[7:])])
  
  txt += "\n" +  "<Player>   " + config.color("[coin:%d, buy:%d, turn:%d, minusCost:%d]" % (coin, buy, turn, minusCost), BOLD + RED)
  txt += "\n" +  ",  ".join(["%s[%d]" % (CARDLIST[_id]["名前"], playercards[i]) for i, _id in enumerate(supply_ids) if playercards[i] != 0 ])

  txt += "\n" +  "<Enemy>"
  txt += "\n" +  ",  ".join(["%s[%d]" % (CARDLIST[_id]["名前"], enemycards[i]) for i, _id in enumerate(supply_ids) if enemycards[i] != 0 ])
  if answer:
    txt += "\n" +  "<Buy>"
    txt += "\n" +  ", ".join(["%s" % CARDLIST[_id]["名前"] for _id in answer])
  if correct_answer:
    txt += "\n" +  "<Correct Answer> "
    txt += "\n" +  ", ".join(["%s" % CARDLIST[_id]["名前"] for _id in correct_answer])
  txt += "\n"
  return txt
  

def read_log(file_path):
  """
  gainの場合
  <stateVec> : deck,hand,discard,field,enemy,supply,isIncludedSupply (cardmax*7次元)
  <turn> : 今のターン数
  <isSente> : 先手:1, 後手:0
  <filename> : データの元になったログ名(option)
  <candidates> : サプライ残存枚数 (cardmax次元)
  <answer> : 獲得カードIDリスト
  """
  with open(file_path) as f:
    data_str = f.read()
    data_str = "<root>" + data_str + "</root>"
  root = ET.fromstring(data_str)

  data = []
  task_name = None
  LIST_FEATURES = ["stateVec", "candidates", "answer", "embargoTokenSupply"]
  EXCEPT_FEATURES = ["filename"]
  for line in root:
    record = {}
    for child in line.iter():
      if child.text:
        if child.tag in EXCEPT_FEATURES:
          pass
        elif child.tag in LIST_FEATURES:
          record[child.tag] = [int(x) for x in child.text.split(",")]
        else:
            record[child.tag] = int(child.text)
      elif task_name == None:
        task_name = child.tag
    if len(record['stateVec']) < N_STATE * N_CARD:
      # 以前の拡張のみで構成される場合データを0埋め
      zeros = [0 for _ in xrange(N_STATE * N_CARD - len(d['stateVec']))]
      d['stateVec'].extend(zeros)
    elif len(record['stateVec']) > N_STATE * N_CARD:
      # 対象外の拡張の場合除外
        continue
    data.append(record)
  return task_name, data



class BatchManager():
  def __init__(self, data, task_name=None):
    self.idx = 0
    self.task_name = task_name
    self.data = self._batching(data)
    self.size = len(self.data)
    
  def get(self, size):
    input_data = []
    targets = []
    d = self.data
    for i in xrange(size):
      input_data.append(d[self.idx][0])
      targets.append(d[self.idx][1])
      self.idx = (self.idx+1) % len(d)
    return input_data, targets

  def _batching(self, data):
    """
    self.n_feature = len(data[0]['stateVec'])
    if data[0].has_key('candidates'):
      self.n_target = len(data[0]['candidates'])
    else:
      self.n_target = 2 #2以外は保留
    """
    batched_data = []
    self.n_feature = N_STATE * N_CARD + len(FEATURES)
    self.n_target = N_CARD
    for d in data:
      """
      if len(d['stateVec']) < N_STATE * N_CARD:
        # 以前の拡張のみで構成される場合データを0埋め
        zeros = [0 for _ in xrange(N_STATE * N_CARD - len(d['stateVec']))]
        d['stateVec'].extend(zeros)
        #zeros = [0 for _ in xrange(N_CARD - len(d['candidates']))]
        #d['candidates'].extend(zeros)
      elif len(d['stateVec']) > N_STATE * N_CARD:
        continue
      """
      _input = [d[f] for f in FEATURES] + d['stateVec']
      # とりあえず2枚以上購入の場合は同確率(1/n)を割り当て
      if not 'answer' in d:
        d['answer'] = [0]
      _target = utils.zero_one_vector(d['answer'], self.n_target)
      _target = np.array(_target) #/ sum(_target)
      batched_data.append((_input, _target))
    return batched_data
    

