# 使い方

```py
from main_crf import *
import numpy as np

result_no_crf, result_with_crf = run_experiment()
result_no_crf = np.array(result_no_crf)
result_with_crf = np.array(result_with_crf)
```
## 結果の並べ方

|      |  model1  |  model2  |  model3  |  model4  |  model5  |
| ---- | ----     | ----     | ----     | ----     | ----     |
|  dataset1  |  (precsion, recall, f-score, balanced-accuracy)  | ... | ... | ... | ... |
|  dataset2  |  (precsion, recall, f-score, balanced-accuracy)  | ... | ... | ... | ... |
|  dataset3  |  (precsion, recall, f-score, balanced-accuracy)  | ... | ... | ... | ... |
|  dataset4  |  (precsion, recall, f-score, balanced-accuracy)  | ... | ... | ... | ... |
|  dataset5  |  (precsion, recall, f-score, balanced-accuracy)  | ... | ... | ... | ... |

## プログラムの仕組み

このプログラムには 3 つの重要なファイルしかありません。1 つは CRF を使用しない BERT プログラムで、main.py にコードが含まれています。2 番目は CRF を使用する BERT プログラムで、ファイルは main\_crf.py です。 3 番目はrun\_ner.pyで、データセットを読み取るファイルです。

main.py の Sector\_2022 クラスはモデルの本体であり、モデルのすべての情報とパラメーターが含まれています。train はモデルのトレーニングに使用される関数です。test はテストに使用される関数です。 run 関数は単純な例であり、experiment 関数は特定の実験のコードです。

## パラメータ設定

|   name   |  value |
| ---- | ---- |
|  optimizer  |  AdamW |
|  learning rate  |  2e-5 |
|  BERT(for word based dataset)  | cl-tohoku/bert-base-japanese-whole-word-masking |
|  BERT(for character based dataset)  | cl-tohoku/bert-base-japanese-char |


## 初期値設定方法

`main_crf.py`に入って、`RANDOM_SEEDs`というglobal variableがある、５つのモデルは順番でこの値で初期化されている。

`DATASET_ORDER_SEED`はtraining setをshuffleする初期値である。

## Training setのラベル1の割合を計算する方法

```py
from main_crf import *
label_count, label_one = cal_label_percent()
print(label_one / label_count)
```

## スコアだけでなく可能性も取る方法

結果をリストからdictに変換した。
`dic.keys()`を使ってdicのすべてのkeywordを見れる。

```py
result_no_crf_raw, result_with_crf_raw = run_experiment_get_raw_output()
case0_pred = result_no_crf_raw['dataset0']['model0']['y_pred'][0]
print(case0_pred)
# 出力はこんな感じ: [0.11263526231050491, 0.12720605731010437, 0.12502093613147736, 0.11578959226608276, 0.12038103491067886, 0.11955509334802628, 0.10841980576515198, 0.10992693901062012, 0.0982583612203598, 0.13114511966705322, 0.13183309137821198, 0.11970293521881104, 0.080902598798275, 0.05768054723739624, 0.06931666284799576, 0.06536028534173965, 0.05445347726345062, 0.05253376439213753, 0.07992718368768692, 0.13279256224632263, 0.1735897660255432, 0.1821373701095581, 0.11884185671806335, 0.1317211091518402, 0.1243244856595993, 0.12247199565172195, 0.18926098942756653, 0.18819020688533783, 0.1932806372642517, 0.23031292855739594, 0.23812806606292725, 0.17765586078166962, 0.028003821149468422]
case0_true = result_no_crf_raw['dataset0']['model0']['y_true'][0]
```



