# Data Synthesis for Machine Learning

## Overview
With the enforcement of a series of data privacy protection regulations such as GDPR, data sharing has become a tricky problem. This tool intends to facilitate data sharing from the customer by synthesizing a dataset based on the original dataset for later machine learning. 
Two parts are included in this tool:
+ Data synthesizer  
  Synthesize a dataset based on the original dataset. It accepts CSV data as input, and output a synthesized dataset based on Differential Privacy. The algorithm in the data synthesizer reference to the paper - [PrivBayes](http://dimacs.rutgers.edu/~graham/pubs/papers/PrivBayes.pdf).
+ Data utility evaluation  
  Evaluate the data utility for the synthesized dataset. The original dataset and the synthesized dataset as the input, one utility evaluation report will be generated with several indicators.

## Quickstart
One demo dataset (part of public dataset from [Adult](https://archive.ics.uci.edu/ml/datasets/Adult)) and its synthesized dataset and the utility evaluation report can be got in the folder *example* for your reference.
### Prerequisites
+ Install python >= 3.6.0 from [python.org](https://www.python.org/)
+ Install project: run `pip install ds4ml`;
+ Download demo data from *example/adult.csv*
### Procedure
  There are two parts in the project, data synthesizer and data utility evaluation.
+ Synthesizer:  
  `data-synthesize <original-dataset> <-o> <synthesized-dataset-path> `  
  Use *adult.csv* as *original-dataset*, the synthesizer dataset *adult_a.csv* is generated under current folder by default.
+ Evaluation:  
  `data-evaluate <original-dataset> <synthesized-dataset> --class-label <attribute1,attribute...>`  
  Use *adult.csv* as *original-dataset*, *adult_a.csv* as *synthesized-dataset*, *sex,salary* as *attribute..* in "--class-label", one *report.html* is generated under current folder by default.

## Download and Installation
+ Install Python 
  Ensure your python >=3.6.0, you can download it from [python.org](https://www.python.org/)
+ After clone the project, install it as follows
  ```
  python setup.py install
  ```
  The project provides two commands: `data-synthesize` and `data-evaluate`. Run `data-synthesize -h` and `data-evaluate -h` for details.

+ Help of `data-synthesize`
  Run `data-synthesize -h`.
  ```
  usage: data-synthesize [-h] [--pseudonym LIST] [--delete LIST]
                       [--na-values LIST] [-o FILE] [--no-header] [-e FLOAT]
                       [--category LIST] [--retain LIST]
                       file

  Synthesize one dataset by Differential Privacy

  positional arguments:
    file                 set path of the CSV to be synthesized

  general arguments:
    -h, --help           show this help message and exit
    --pseudonym LIST     set candidate columns separated by a comma, which will
                        be replaced with a pseudonym. It only works on the
                        string column.
    --delete LIST        set columns separated by a comma, which will be deleted
                        when synthesis.
    --na-values LIST     set additional values to recognize as NA/NaN; (default
                        null values are from pandas.read_csv)
    -o, --output FILE    set the file name of output synthesized dataset
                        (default is input file name with suffix '_a')
    --no-header          indicate there is no header in a CSV file, and will
                        take [#0, #1, #2, ...] as header. (default: the tool
                        will try to detect and take actions)

  advanced arguments:
    -e, --epsilon FLOAT  set epsilon for differential privacy (default 0.1)
    --category LIST      set categorical columns separated by a comma.
    --retain LIST        set columns to retain the values
   
  ```


+ Help of `data-evaluate`
  Run `data-evaluate -h`.
  ```
  usage: data-evaluate [-h] [--na-values LIST] [-o FILE] [-t TEST]
                      [--class-label LIST]
                      source target

  Evaluate the utility of the synthesized dataset compared with the source dataset.

  positional arguments:
    source              set file path of source (raw) dataset to be compared with
                        synthesized dataset, only support CSV files
    target              set file path of target (synthesized) dataset to evaluate

  general arguments:
    -h, --help          show this help message and exit
    --na-values LIST    set additional values to recognize as NA/NaN; (default
                        null values are from pandas.read_csv)
    -o, --output FILE   set output path for evaluation report; (default is
                        "report.html" under current work directory)

  advanced arguments:
    -t, --test TEST     set test dataset for classification or regression task;
                        (default take 20 percent from source dataset)
    --class-label LIST  set column name as a class label for classification or
                        regression task; supports one or multiple columns
                        (separated by comma)
  ```
  
## How to obtain support
If you encounter an issue, you can open an issue in GitHub

## Contribute
Please check our [Contribution Guidelines](https://github.com/SAP/data-synthesis-for-machine-learning/blob/master/CONTRIBUTING.md)

## License
Copyright (c) 2019 SAP SE or an SAP affiliate company. All rights reserved. This file is licensed under the Apache Software License, v.2 except as noted otherwise in the [LICENSE file](https://github.com/SAP/data-synthesis-for-machine-learning/blob/master/LICENSE).

