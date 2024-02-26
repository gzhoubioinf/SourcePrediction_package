"""
================================
4. Hyper-parameter Optimization
================================
"""

import os
import numpy as np

import matplotlib.pyplot as plt

from ai4water import Model
from ai4water.utils.utils import TrainTestSplit
from ai4water.utils.utils import jsonize, dateandtime_now
from ai4water.hyperopt import HyperOpt, Integer, Real, Categorical

from SeqMetrics import ClassificationMetrics

from utils import load_data_from_pickle, plot_confidence_interval, get_data

# %%

def f1_score_(t,p)->float:
    if (p == p[0]).all():
        return 0.1
    return ClassificationMetrics(t, p).f1_score(average="macro")

# %%

X, y, inputs = get_data(return_bf_only=False)

X = X.toarray()

TrainX, TestX, TrainY, TestY = TrainTestSplit(seed=313).split_by_random(X, y)

print(TrainX.shape, TrainY.shape, TestX.shape, TestY.shape)

# %%

model = Model(
    model="XGBClassifier",
    input_features=list(inputs),
    cross_validator={"KFold": {"n_splits": 10}}
)

model.reset_global_seed(313)

model.fit(x=TrainX, y=TrainY.values)

def eval_on_test(_model):

    test_true, test_pred = _model.predict(
        x=TestX, y=TestY.values,
        return_true=True,
        process_results=False
    )

    f1 = ClassificationMetrics(test_true.reshape(-1,), test_pred).f1_score(
        average="weighted")

    acc = ClassificationMetrics(test_true.reshape(-1,), test_pred).accuracy()
    print(f"f1_score : {f1}  acc: {acc}")
    return

#eval_on_test(model)

# %%

# hpo

SEP= os.sep
PREFIX = f"hpo_{dateandtime_now()}"  # folder name where to save the results
num_iterations = 200

ITER = 0
VAL_SCORES = []
CV = False

def objective_fn(
        prefix=None,
        return_model=False,
        fit_on_all_data=False,
        **suggestions
):
    global ITER

    suggestions = jsonize(suggestions)

    #suggestions['early_stopping'] = True
    suggestions['validation_fraction'] = 0.2

    _model = Model(
        model = {"XGBClassifier": suggestions},
        #                   ],
        cross_validator={"KFold": {"n_splits": 10}},
        prefix=prefix or PREFIX,
        input_features=list(inputs),
        verbosity=0
    )

    _model.reset_global_seed(313)

    if return_model:
        _model.fit(x=TrainX, y=TrainY.values)
        eval_on_test(_model)
        return _model

    if CV:
        val_score_ = _model.cross_val_score(
            x=TrainX, y=TrainY.values,
            scoring=f1_score_
        )[0]
    else:
        _model.fit(x=TrainX, y=TrainY.values)
        val_true, val_pred = _model.predict(
            x=TestX, y=TestY.values, return_true=True, process_results=False)

        val_score_ = ClassificationMetrics(val_true.reshape(-1,), val_pred).f1_score(
            average="weighted")

    val_score = 1 - val_score_

    VAL_SCORES.append(val_score)
    best_score = round(np.nanmin(VAL_SCORES).item(), 2)
    bst_iter = np.argmin(VAL_SCORES)

    print(f"{ITER} {round(val_score, 2)} {round(val_score_, 2)}. Best was {best_score} at {bst_iter}")

    ITER += 1

    if return_model:
        return _model
    return val_score

# %%

param_space = [Categorical(categories=['gbdt', 'dart', 'goss'], name='boosting_type'),
                  Integer(low=5, high=500, name='num_leaves'),
                  Real(low=0.04, high=0.1, name='learning_rate', prior='log-uniform'),
                  Integer(low=20, high=150, name='n_estimators'),
               ]

x0 = ['dart',5, 0.04, 100]

# %%

optimizer = HyperOpt(
    algorithm="tpe",
    objective_fn=objective_fn,
    param_space=param_space,
    x0=x0,
    num_iterations=num_iterations,
    process_results=False,  # we can turn it False if we want post-processing of results
    opt_path=f"results{SEP}{PREFIX}"
)

# results = optimizer.fit()

# %%

# best_iteration = optimizer.best_iter()
#
# print(f"optimized parameters are \n{optimizer.best_paras()} at {best_iteration}")


# %%

# optimizer._plot_convergence()
# plt.show()

# %%

# model = objective_fn(prefix=f"{PREFIX}{SEP}best",
#                      return_model=True,
#                      fit_on_all_data=True,
#                      **optimizer.best_paras())