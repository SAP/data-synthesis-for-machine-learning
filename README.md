# Data Synthesis for Machine Learning

[![REUSE status](https://api.reuse.software/badge/github.com/SAP/data-synthesis-for-machine-learning)](https://api.reuse.software/info/github.com/SAP/data-synthesis-for-machine-learning)

Note: fork of https://github.com/SAP/data-synthesis-for-machine-learning

## Overview
The recent enforcement of data privacy protection regulations, such as GDPR, has made data sharing more difficult. This tool intends to facilitate data sharing from a customer by synthesizing a dataset based on the original dataset for later machine learning. 
There are two parts to this tool:
+ Data synthesizer  
  Synthesize a dataset based on the original dataset. It accepts CSV data as input, and output a synthesized dataset based on Differential Privacy. The algorithm in the data synthesizer reference to the paper - [PrivBayes 2017](http://dimacs.rutgers.edu/~graham/pubs/papers/privbayes-tods.pdf).
+ Data utility evaluation  
  Evaluate the data utility for the synthesized dataset. The original dataset and the synthesized dataset as the input, one utility evaluation report will be generated with several indicators.
### Positioning
Our project is independent of any DB and we are focus on the later machine learning. There is also one data anonymization feature in the HANA (not open sourced). Of a customer has HANA, please use this feature within HANA. If a customer does not have HANA and it's about a machine learning use case, please use our tool. Exception: If such a customer wants to try out the algorithm implemented in our tool to figure out if it provides better results than HANA, please use our tool.

## Quickstart
There are one demo dataset (part of the public dataset from [Adult](https://archive.ics.uci.edu/ml/datasets/Adult)), the synthesized dataset and the utility evaluation report in the folder *example* for your reference.
### Prerequisites
+ Install python >= 3.6.0 from [python.org](https://www.python.org/)
+ Install project: run `pip install ds4ml`;
+ Download demo data from [example/adult.csv](https://github.com/SAP/data-synthesis-for-machine-learning/blob/master/example/adult.csv)
### Procedure
  There are two parts in the project, data synthesizer and data utility evaluation.
+ Synthesizer:  
  ```
  data-synthesize <original-dataset> <-o> <synthesized-dataset-path>
  ```  
  Use *adult.csv* as *original-dataset*, the synthesizer dataset *adult_a.csv* is generated under current folder by default.
  
  - To save the cost of (synthesized) data transfer, command `data-pattern` can help, which serialize anonymous patterns of a dataset. And then use `data-synthesize` to generate dataset from the pattern.
  ```
  data-pattern <original-dataset> <-o> <anonymous-pattern-path>
  data-synthesize <anonymous-pattern-path> <-o> <synthesized-dataset-path>
  ```
+ Evaluation:  
  ```
  data-evaluate <original-dataset> <synthesized-dataset> --class-label <attribute1,attribute...>
  ```  
  Use *adult.csv* as *original-dataset*, *adult-a.csv* as *synthesized-dataset*, *sex,salary* as *attribute* in "--class-label", one *report.html* is generated under current folder by default.

## Download and Installation
+ Install Python 
  Ensure your python >=3.6.0, you can download it from [python.org](https://www.python.org/)
+ After clone the project, install it as: `python setup.py install`.
  The project provides three commands: `data-synthesize`, `data-pattern` and `data-evaluate`. Run them with option `-h` for details.

+ Help of `data-synthesize`
  Run `data-synthesize -h`.
  ```
  usage: data-synthesize [-h] [--pseudonym LIST] [--delete LIST]
                       [--na-values LIST] [-o FILE] [--no-header] [-e FLOAT]
                       [--category LIST] [--retain LIST]
                       file

  Synthesize one dataset by Differential Privacy

  positional arguments:
    file                set path of a csv file to be synthesized or path of a 
                        pattern file to be generated

  general arguments:
    -h, --help          show this help message and exit
    --pseudonym LIST    set candidate columns separated by a comma, which will
                        be replaced with a pseudonym. It only works on the
                        string column.
    --delete LIST       set columns separated by a comma, which will be deleted
                        when synthesis.
    --na-values LIST    set additional values to recognize as NA/NaN; (default
                        null values are from pandas.read_csv)
    -o, --output FILE   set the file name of output synthesized dataset
                        (default is input file name with suffix '-a.csv')
    --no-header         indicate there is no header in a CSV file, and will
                        take [#0, #1, #2, ...] as header. (default: the tool
                        will try to detect and take actions)
    --records INT       specify the records you want to generate; default is the
                        same records with the original dataset
    --sep STRING        specify the delimiter of the input file

  advanced arguments:
    -e, --epsilon FLOAT  set epsilon for differential privacy (default 0.1)
    --category LIST      set categorical columns separated by a comma.
    --retain LIST        set columns to retain the values
  ```

  - Note: argument **epsilon** (ε) defines the privacy guarantee, which depends on the size and features of the dataset to be anonymized. The lower the value of epsilon, the more noise is applied, and the less the utility. 


+ Help of `data-pattern`
  Run `data-pattern -h`.
  ```
  usage: data-pattern [-h] [--pseudonym LIST] [--delete LIST] [--na-values LIST]
                    [-o FILE] [--no-header] [--sep STRING] [-e FLOAT]
                    [--category LIST] [--retain LIST]
                    file

  Serialize patterns of a dataset anonymously

  positional arguments:
    file                 set path of a csv file to be patterned anonymously

  general arguments:
    -h, --help           show this help message and exit
    --pseudonym LIST     set candidate columns separated by a comma, which will
                         be replaced with a pseudonym. It only works on the
                         string column.
    --delete LIST        set columns separated by a comma, which will be deleted
                         when synthesis.
    --na-values LIST     set additional values to recognize as NA/NaN; (default
                         null values are from pandas.read_csv)
    -o, --output FILE    set the file name of anonymous patterns (default is
                         input file name with a suffix '-pattern.json')
    --no-header          indicate there is no header in a CSV file, and will
                         take [#0, #1, #2, ...] as header. (default: the tool
                         will try to detect and take actions)
    --sep STRING         specify the delimiter of the input file

  advanced arguments:
    -e, --epsilon FLOAT  set epsilon for differential privacy (default 0.1)
    --category LIST      set categorical columns separated by a comma.
  ```

  - Note: after the pattern (.json) file is generated by `data-pattern` anonymously, then run command `data-synthesize *-pattern.json` to synthesize the data.


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
    --category LIST     set categorical columns separated by a comma.
    --class-label LIST  set column name as a class label for classification or
                        regression task; supports one or multiple columns
                        (separated by comma)
  ```
  
## How to obtain support
If you encounter an issue, you can open an issue in GitHub.

## Contribute
Please check our [Contribution Guidelines](/CONTRIBUTING.md).

## Licensing

Copyright 2019-2021 SAP SE or an SAP affiliate company and data-synthesis-for-machine-learning contributors. Please see our [LICENSE](LICENSE) for copyright and license information. Detailed information including third-party components and their licensing/copyright information is available [via the REUSE tool](https://api.reuse.software/info/github.com/SAP/data-synthesis-for-machine-learning).
