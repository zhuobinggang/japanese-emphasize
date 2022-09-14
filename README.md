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



