!git clone https://github.com/SAP/data-synthesis-for-machine-learning.git

!pip install ds4ml

import pandas as pd

%load_ext autoreload
%autoreload 2

%run ./data-synthesis-for-machine-learning/ds4ml/__init__.py
%run ./data-synthesis-for-machine-learning/ds4ml/attribute.py
%run ./data-synthesis-for-machine-learning/ds4ml/dataset.py
%run ./data-synthesis-for-machine-learning/ds4ml/evaluator.py
%run ./data-synthesis-for-machine-learning/ds4ml/metrics.py
%run ./data-synthesis-for-machine-learning/ds4ml/synthesizer.py
%run ./data-synthesis-for-machine-learning/ds4ml/utils.py

df = pd.read_csv('./data-synthesis-for-machine-learning/example/adult.csv')

from ds4ml import synthesizer
greedy_bayes(dataset=df, epsilon=0.1)