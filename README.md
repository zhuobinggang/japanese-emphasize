# 使い方

```py
from main import *
import numpy as np

result_5X5 = experiment()
result_5X5 = np.array(result_5X5)
print(result_5X5.shape) # (5, 5, 4)
```
## 結果の並べ方

|      |  model1  |  model2  |  model3  |  model4  |  model5  |
| ---- | ----     | ----     | ----     | ----     | ----     |
|  dataset1  |  (precsion, recall, f-score, balanced-accuracy)  | ... | ... | ... | ... |
|  dataset2  |  (precsion, recall, f-score, balanced-accuracy)  | ... | ... | ... | ... |
|  dataset3  |  (precsion, recall, f-score, balanced-accuracy)  | ... | ... | ... | ... |
|  dataset4  |  (precsion, recall, f-score, balanced-accuracy)  | ... | ... | ... | ... |
|  dataset5  |  (precsion, recall, f-score, balanced-accuracy)  | ... | ... | ... | ... |

