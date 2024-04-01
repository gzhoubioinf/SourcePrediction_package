
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ai4water import Model
from ai4water.utils.utils import TrainTestSplit

from ai4water.postprocessing import ProcessPredictions

from SeqMetrics import ClassificationMetrics

from utils import get_data

# %%

def f1_score_macro(t,p)->float:
    if (p == p[0]).all():
        return 0.1
    return ClassificationMetrics(t, p).f1_score(average="macro")

# %%

# reading data

X, y, _ = get_data(return_bf_only=False)

X_array = X.toarray()

# %%

TrainX, TestX, TrainY, TestY = TrainTestSplit(seed=313).split_by_random(X_array, y)


# %%

model = 'XGBClassifier'

input_features = [f'input_{n}' for n in range(20179)]


outputs = ['target']

model = Model(
    model=model,
    input_features=input_features,
    loss='binary_crossentropy',
    output_features=outputs,
    verbosity=-1,
)
model.fit(x=TrainX, y=TrainY.values)

# %%

test_p = model.predict(TestX)

# %%

print(f'F1 score: {f1_score_macro(TestY.values, test_p)}')

# %%

# confusion matrix

X = np.concatenate([TrainX, TestX])
y = np.concatenate([TrainY, TestY])

pred = model.predict(X)

processor = ProcessPredictions('classification',
                               show=False,
                               save=False)

im = processor.confusion_matrix(
    y, pred,
    cbar_params = {"border": False},
    annotate_kws = {'fontsize': 20, "fmt": '%.f', 'ha':"center"})
ax_ = im.axes

#ax_.set_title(target)
plt.show()