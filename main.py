from run_ner import readfile
import torch as t
import torch as t
import numpy as np
nn = t.nn
F = t.nn.functional
from transformers import BertJapaneseTokenizer, BertModel
import datetime
from itertools import chain

def label_to_number(case):
    tokens, labels = case
    # tokens = ''.join(tokens)
    numbers = [1 if label != 'O' else 0 for label in labels]
    return tokens, numbers

def create_model_with_seed(seed):
    t.manual_seed(seed)
    np.random.seed(seed)
    m = Sector_2022()
    time_string = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'created model with seed {seed} at time {time_string}')
    return m

def read_test(name = 'data/data_five/1/test.txt'):
    data = readfile(name)
    data = [label_to_number(case) for case in data]
    return data

def read_train(name = 'data/data_five/1/train.txt'):
    data = readfile(name)
    data = [label_to_number(case) for case in data]
    return data

def read_tests():
    datas = []
    for i in range(1, 6):
        data = readfile(f'data/data_five/{i}/test.txt')
        data = [label_to_number(case) for case in data]
        datas.append(data)
    return datas

def read_trains():
    datas = []
    for i in range(1, 6):
        data = readfile(f'data/data_five/{i}/train.txt')
        data = [label_to_number(case) for case in data]
        datas.append(data)
    return datas


def flatten(l):
    return [item for sublist in l for item in sublist]

class Sector_2022(nn.Module):
  def __init__(self, learning_rate = 2e-5):
    super().__init__()
    self.learning_rate = learning_rate
    self.bert_size = 768
    self.verbose = False
    self.init_bert()
    self.init_hook()
    self.opter = t.optim.AdamW(self.get_should_update(), self.learning_rate)
    # self.cuda()

  def init_bert(self):
    self.bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-char')
    self.bert.train()
    self.toker = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-char')

  def get_should_update(self):
    return chain(self.bert.parameters(), self.classifier.parameters())

  def init_hook(self): 
    self.classifier = nn.Sequential( # 因为要同时判断多种1[sep]3, 2[sep]2, 3[sep]1, 所以多加一点复杂度
      nn.Linear(self.bert_size, 20),
      nn.LeakyReLU(0.1),
      nn.Linear(20, 20),
      nn.LeakyReLU(0.1),
      nn.Linear(20, 1),
      nn.Sigmoid()
    )

def encode(text, toker):
    return t.LongTensor(toker.encode(text))

def focal_loss(o, l, fl_rate = 0):
    assert len(l.shape) == 0
    assert len(o.shape) == 0
    pt = o if (l == 1) else (1 - o)
    loss = (-1) * t.log(pt) * t.pow((1 - pt), fl_rate)
    return loss

def test(m, ds_test):
    toker = m.toker
    bert = m.bert
    target_all = []
    result_all = []
    for text, labels in ds_test:
        ids = encode(text, toker)
        # out_bert = bert(ids.unsqueeze(0).cuda()).last_hidden_state # (1, seq_len + 2, 768)
        out_bert = bert(ids.unsqueeze(0)).last_hidden_state # (1, seq_len + 2, 768)
        out_bert = out_bert[:, 1:-1, :] # (1, seq_len, 768)
        out_mlp = m.classifier(out_bert) # (1, seq_len, 1)
        out_mlp = out_mlp.squeeze() # (seq_len)
        target_all.append(labels)
        result_all.append(out_mlp.tolist())
    return result_all, target_all

    

def train(m, ds_train, epoch = 1, batch = 16, iteration_callback = None, random_seed = True):
    first_time = datetime.datetime.now()
    toker = m.toker
    bert = m.bert
    opter = m.opter
    BCE = t.nn.BCELoss()
    for epoch_idx in range(epoch):
        print(f'Train epoch {epoch_idx}')
        ds = None
        if random_seed:
            ds = np.random.permutation(ds_train)
        else:
            ds = ds_train
        for row_idx, (text, labels) in enumerate(ds):
            if row_idx % 1000 == 0:
                print(f'finished: {row_idx}/{len(ds_train)}')
                pass
            ids = encode(text, toker)
            # out_bert = bert(ids.unsqueeze(0).cuda()).last_hidden_state # (1, seq_len + 2, 768)
            out_bert = bert(ids.unsqueeze(0)).last_hidden_state # (1, seq_len + 2, 768)
            out_bert = out_bert[:, 1:-1, :] # (1, seq_len, 768)
            out_mlp = m.classifier(out_bert) # (1, seq_len, 1)
            out_mlp = out_mlp[0, :, 0] # (seq_len)
            # labels = t.FloatTensor(labels).cuda() # (seq_len)
            labels = t.FloatTensor(labels) # (seq_len)
            if labels.shape != out_mlp.shape:
                print(row_idx)
                print(ids)
                print(text)
                print(labels)
            loss = BCE(out_mlp, labels)
            loss.backward()
            # backward
            if (row_idx + 1) % batch == 0:
                if iteration_callback is not None:
                    iteration_callback()
                opter.step()
                opter.zero_grad()
    opter.step()
    opter.zero_grad()
    last_time = datetime.datetime.now()
    delta = last_time - first_time
    print(delta.seconds)
    return delta.seconds

def cal_prec_rec_f1_v2(results, targets):
  TP = 0
  FP = 0
  FN = 0
  TN = 0
  for guess, target in zip(results, targets):
    if guess == 1:
      if target == 1:
        TP += 1
      elif target == 0:
        FP += 1
    elif guess == 0:
      if target == 1:
        FN += 1
      elif target == 0:
        TN += 1
  prec = TP / (TP + FP) if (TP + FP) > 0 else 0
  rec = TP / (TP + FN) if (TP + FN) > 0 else 0
  f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) != 0 else 0
  balanced_acc_factor1 = TP / (TP + FN) if (TP + FN) > 0 else 0
  balanced_acc_factor2 = TN / (FP + TN) if (FP + TN) > 0 else 0
  balanced_acc = (balanced_acc_factor1 + balanced_acc_factor2) / 2
  return prec, rec, f1, balanced_acc

def get_results_and_targets(m, ds_test):
    result_all, target_all = test(m, ds_test)
    results = flatten(result_all)
    results = [1 if res > 0.5 else 0 for res in results]
    targets = flatten(target_all)
    return results, targets

def test_chain(m, ds_test):
    results, targets = get_results_and_targets(m, ds_test)
    return cal_prec_rec_f1_v2(results, targets)

def get_m():
    RANDOM_SEED = 2000
    m = create_model_with_seed(RANDOM_SEED)
    return m

def run(m):
    ds_train = read_train()
    ds_test = read_test()
    results = []
    for _ in range(5):
        train(m, ds_train, epoch = 1, batch = 16, iteration_callback = None, random_seed = True)
        result = test_chain(m, ds_test)
        print(result)
        results.append(result)
    return results

RANDOM_SEEDs = [20, 22, 8, 29, 1648, 1,2]
    
def experiment():
    results_5X5 = []
    train_dss = read_trains()
    test_dss = read_tests()
    for _, (ds_train, ds_test) in enumerate(zip(train_dss, test_dss)):
        results = []
        for idx in range(5):
            m = create_model_with_seed(RANDOM_SEEDs[idx])
            for _ in range(4):
                train(m, ds_train, epoch = 1, batch = 16, iteration_callback = None, random_seed = True)
            result = test_chain(m, ds_test)
            print(result)
            results.append(result)
        results_5X5.append(results)
    return results_5X5





