# coding: utf-8
import numpy as np
import xml.etree.ElementTree as ET
from public import utils

N_STATE= 7 #deck,hand,discard,field,enemy,supply,isIncludedSupply (cardmax*7次元)
N_CARD = 84 #58(陰謀), 84(海辺) 

def explain_state(record):
  n_card = len(record["stateVec"]) / N_STATE
  print n_card

def read_cardlist(file_path="dataset/cardData.csv"):
  res = []
  with open(file_path) as f:
    delim = ','
    header = f.readline().split(delim)
    for l in f:
      record = {}
      l = l.split(delim)
      for i in xrange(1, len(header)):
        record[header[i]] = l[i]
        res.append(record)
  return res

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
  for line in root:
    record = {}
    for child in line.iter():
      if child.text:
        #print child.tag
        #print child.text
        if child.tag in LIST_FEATURES:
          record[child.tag] = np.array(child.text.split(","), 
                                       dtype=int)
        else:
          record[child.tag] = int(child.text)
      elif task_name == None:
        task_name = child.tag
        
    data.append(record)
  return task_name, data



class BatchManager():
  def __init__(self, task_name, data):
    self.idx = 0
    self.n_feature = 0
    self.n_target = 0
    self.task_name = task_name
    self.data = self._batching(data)
    self.size = len(self.data)

  def get(self, size):
    l = []
    d = self.data
    for i in xrange(size):
      l.append(d[self.idx])
      self.idx = (self.idx+1) % len(d)
    return l

  def _batching(self, data):
    self.n_feature = len(data[0]['stateVec'])
    if data[0].has_key('candidates'):
      self.n_target = len(data[0]['candidates'])
    else:
      self.n_target = 2 #2以外は保留

    batched_data = []
    for record in data:
      _input = record['stateVec']

      # とりあえず2枚以上購入の場合は同確率を割り当て
      _target = utils.zero_one_vector(record['answer'], self.n_target)
      _target = np.array(_target) / sum(_target)
      batched_data.append((_input, _target))
    return batched_data
    

