from run_ner import readfile
import torch as t
import torch as t
import numpy as np
nn = t.nn
F = t.nn.functional
from transformers import BertJapaneseTokenizer, BertModel
import datetime
from itertools import chain
from torchcrf import CRF
from main import read_train, read_test, encode, flatten, cal_prec_rec_f1_v2, read_trains, read_tests, DATASET_ORDER_SEED

def create_model_with_seed(seed, cuda, wholeword):
    t.manual_seed(seed)
    np.random.seed(seed)
    m = Sector_2022_CRF(cuda = cuda, wholeword = wholeword)
    time_string = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'created model with seed {seed} at time {time_string}')
    return m

class Sector_2022_CRF(nn.Module):
  def __init__(self, learning_rate = 2e-5, cuda = False, wholeword = True):
    super().__init__()
    self.learning_rate = learning_rate
    self.bert_size = 768
    self.verbose = False
    self.init_bert(wholeword = wholeword)
    self.classifier = nn.Sequential( # 输出2个标签以结合crf
      nn.Linear(self.bert_size, 384),
      nn.LeakyReLU(0.1),
      nn.Linear(384, 2),
    )
    self.crf = CRF(2, batch_first=True)
    if cuda:
        self.cuda()
    self.is_cuda = cuda

  def init_bert(self, wholeword = True):
    if wholeword:
      self.bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
      self.bert.train()
      self.toker = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    else:
      self.bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-char')
      self.bert.train()
      self.toker = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-char')

def test(m, ds_test_org):
    ds_test = ds_test_org.copy()
    first_time = datetime.datetime.now()
    toker = m.toker
    bert = m.bert
    crf = m.crf
    target_all = []
    result_all = []
    for text, labels in ds_test:
        ids = encode(text, toker)
        assert ids.shape[0] == len(text) + 2
        if m.is_cuda:
            out_bert = bert(ids.unsqueeze(0).cuda()).last_hidden_state # (1, seq_len + 2, 768)
            # labels = t.FloatTensor(labels).cuda() # (seq_len)
            tags = t.LongTensor([labels]).cuda() # (1, seq_len)
        else:
            out_bert = bert(ids.unsqueeze(0)).last_hidden_state # (1, seq_len + 2, 768)
            # labels = t.FloatTensor(labels) # (seq_len)
            tags = t.LongTensor([labels]) # (1, seq_len)
        out_bert = out_bert[:, 1:-1, :] # (1, seq_len, 768)
        out_mlp = m.classifier(out_bert) # (1, seq_len, 2)
        results = m.crf.decode(out_mlp)
        result_all.append(results[0])
        target_all.append(tags.tolist()[0])
    return result_all, target_all


def train(m, ds_train_org, epoch = 1, batch = 16, iteration_callback = None, random_seed = True):
    ds_train = ds_train_org.copy()
    first_time = datetime.datetime.now()
    toker = m.toker
    bert = m.bert
    crf = m.crf
    opter = t.optim.AdamW(m.parameters(), m.learning_rate)
    for epoch_idx in range(epoch):
        print(f'Train epoch {epoch_idx}')
        ds = None
        if random_seed:
            numpy.random.seed(DATASET_ORDER_SEED) # 固定训练顺序
            ds = np.random.permutation(ds_train)
        else:
            ds = ds_train
        for row_idx, (text, labels) in enumerate(ds):
            if row_idx % 1000 == 0:
                print(f'finished: {row_idx}/{len(ds_train)}')
                pass
            ids = encode(text, toker)
            assert ids.shape[0] == len(text) + 2
            if m.is_cuda:
                out_bert = bert(ids.unsqueeze(0).cuda()).last_hidden_state # (1, seq_len + 2, 768)
                # labels = t.FloatTensor(labels).cuda() # (seq_len)
                tags = t.LongTensor([labels]).cuda() # (1, seq_len)
            else:
                out_bert = bert(ids.unsqueeze(0)).last_hidden_state # (1, seq_len + 2, 768)
                # labels = t.FloatTensor(labels) # (seq_len)
                tags = t.LongTensor([labels]) # (1, seq_len)
            out_bert = out_bert[:, 1:-1, :] # (1, seq_len, 768)
            out_mlp = m.classifier(out_bert) # (1, seq_len, 2)
            loss = -crf(out_mlp, tags)
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

def test_no_crf(m, ds_test_org):
    ds_test = ds_test_org.copy()
    first_time = datetime.datetime.now()
    toker = m.toker
    bert = m.bert
    crf = m.crf
    target_all = []
    result_all = []
    for text, labels in ds_test:
        ids = encode(text, toker)
        SEQ_LEN = len(text)
        assert ids.shape[0] == SEQ_LEN + 2
        if m.is_cuda:
            out_bert = bert(ids.unsqueeze(0).cuda()).last_hidden_state # (1, seq_len + 2, 768)
            # labels = t.FloatTensor(labels).cuda() # (seq_len)
            tags = t.LongTensor([labels]).cuda() # (1, seq_len)
        else:
            out_bert = bert(ids.unsqueeze(0)).last_hidden_state # (1, seq_len + 2, 768)
            # labels = t.FloatTensor(labels) # (seq_len)
            tags = t.LongTensor([labels]) # (1, seq_len)
        out_bert = out_bert[:, 1:-1, :] # (1, seq_len, 768)
        out_mlp = m.classifier(out_bert) # (1, seq_len, 2)
        out_mlp = out_mlp.view(SEQ_LEN, 2)
        results = out_mlp.argmax(1) # (seq_len)
        result_all.append(results)
        target_all.append(tags.tolist()[0])
    return result_all, target_all

def train_no_crf(m, ds_train_org, epoch = 1, batch = 16, iteration_callback = None, random_seed = True):
    ds_train = ds_train_org.copy()
    first_time = datetime.datetime.now()
    toker = m.toker
    bert = m.bert
    # crf = m.crf
    CEL = nn.CrossEntropyLoss()
    opter = t.optim.AdamW(m.parameters(), m.learning_rate)
    for epoch_idx in range(epoch):
        print(f'Train epoch {epoch_idx}')
        ds = None
        if random_seed:
            numpy.random.seed(DATASET_ORDER_SEED) # 固定训练顺序
            ds = np.random.permutation(ds_train)
        else:
            ds = ds_train
        for row_idx, (text, labels) in enumerate(ds):
            if row_idx % 1000 == 0:
                print(f'finished: {row_idx}/{len(ds_train)}')
                pass
            SEQ_LEN = len(text)
            ids = encode(text, toker)
            assert ids.shape[0] == SEQ_LEN + 2
            if m.is_cuda:
                out_bert = bert(ids.unsqueeze(0).cuda()).last_hidden_state # (1, seq_len + 2, 768)
                # labels = t.FloatTensor(labels).cuda() # (seq_len)
                tags = t.LongTensor([labels]).cuda() # (1, seq_len)
            else:
                out_bert = bert(ids.unsqueeze(0)).last_hidden_state # (1, seq_len + 2, 768)
                # labels = t.FloatTensor(labels) # (seq_len)
                tags = t.LongTensor([labels]) # (1, seq_len)
            out_bert = out_bert[:, 1:-1, :] # (1, seq_len, 768)
            out_mlp = m.classifier(out_bert) # (1, seq_len, 2)
            # Cross entropy loss
            out_mlp = out_mlp.view(SEQ_LEN, 2) # (seq_len, 2)
            tags = tags.view(SEQ_LEN) # (seq)
            loss = CEL(out_mlp, tags)
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

def get_results_and_targets(m, ds_test):
    result_all, target_all = test(m, ds_test)
    results = flatten(result_all)
    targets = flatten(target_all)
    return results, targets

def test_chain(m, ds_test):
    results, targets = get_results_and_targets(m, ds_test)
    return cal_prec_rec_f1_v2(results, targets)

def test_chain_no_crf(m, ds_test):
    result_all, target_all = test_no_crf(m, ds_test)
    results = flatten(result_all)
    targets = flatten(target_all)
    return cal_prec_rec_f1_v2(results, targets)

# m = create_model_with_seed(20, cuda = True, wholeword = True)
def run(m):
    ds_train = read_train('data_five/1/train.txt')
    ds_test = read_test('data_five/1/test.txt')
    results = []
    # m = create_model_with_seed(20, cuda = True, wholeword = True)
    for _ in range(5):
        train(m, ds_train, epoch = 1, batch = 16, iteration_callback = None, random_seed = True)
        result = test_chain(m, ds_test)
        print(result)
        results.append(result)
    return results


RANDOM_SEEDs = [21, 22, 8, 29, 1648, 1,2]

def write_result(path, result):
    f = open(path)
    f.write(str(result))
    f.close()
    print('RESULT WRITTEN')
    
def experiment(epoch = 5, cuda = True, wholeword = True):
    results_5X5X5 = []
    train_dss = read_trains()
    test_dss = read_tests()
    for _, (ds_train, ds_test) in enumerate(zip(train_dss, test_dss)):
        fs_by_model = []
        for idx in range(5):
            m = create_model_with_seed(RANDOM_SEEDs[idx], cuda, wholeword)
            fs = []
            for e in range(epoch):
                train(m, ds_train, epoch = 1, batch = 16, iteration_callback = None, random_seed = True)
                result = test_chain(m, ds_test)
                print(result)
                _,_,f,_ = result
                fs.append(f)
            fs_by_model.append(fs)
        results_5X5X5.append(fs_by_model)
        print('results_5X5X5:')
        print(results_5X5X5)
    write_result('with_crf.txt', results_5X5X5)
    return results_5X5X5

def experiment_no_crf(epoch = 5, cuda = True, wholeword = True):
    results_5X5X5 = []
    train_dss = read_trains()
    test_dss = read_tests()
    for _, (ds_train, ds_test) in enumerate(zip(train_dss, test_dss)):
        fs_by_model = []
        for idx in range(5):
            m = create_model_with_seed(RANDOM_SEEDs[idx], cuda, wholeword)
            fs = []
            for e in range(epoch):
                train_no_crf(m, ds_train, epoch = 1, batch = 16, iteration_callback = None, random_seed = True)
                result = test_chain_no_crf(m, ds_test)
                print(result)
                _,_,f,_ = result
                # fs.append(f)
                fs.append(result)
            fs_by_model.append(fs)
        results_5X5X5.append(fs_by_model)
        print('results_5X5X5:')
        print(results_5X5X5)
    write_result('without_crf.txt', results_5X5X5)
    return results_5X5X5
