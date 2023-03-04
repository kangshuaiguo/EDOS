#!/usr/bin/env python

# load required packages
import sys
import os
import pandas as pd
from sklearn.metrics import f1_score


# load submission

submission_df = pd.read_csv('test_out.csv') # the first file in the submission zip is expected to be the submission csv


# load gold standard data
gold_df = pd.read_csv('test_label .csv') # the first file in the gold standard zip is expected to be the gold standard csv


submission_df = submission_df.sort_values("rewire_id")
gold_df = gold_df.sort_values("rewire_id")


# calculate macro F1 score for the submission relative to the gold standard data
if len(pd.unique(gold_df.label))>2:
    f1 = f1_score(y_true = gold_df["label"], y_pred = submission_df["label_pred"], average="macro")
elif len(pd.unique(gold_df.label))==2:
    f1 = f1_score(y_true = gold_df["label"], y_pred = submission_df["label_pred"], pos_label=None, average="macro")
    # need to set pos_label to none because of quirk in sklearn 0.17, which is the version running on CodaLab..
    # https://scikit-learn.org/0.17/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score


print("MacroF1: {}".format(f1))