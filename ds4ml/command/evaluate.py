"""
evaluate.py

"""

import argparse
import os
import time

from ds4ml.evaluator import BiFrame
from ds4ml.utils import read_data_from_csv, CustomFormatter, str_to_list


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate the utility of synthesized dataset compared with '
                    'the source dataset.',
        formatter_class=CustomFormatter,
        add_help=False)
    # positional arguments
    parser.add_argument('source',
                        help='set file path of source (raw) dataset to be '
                             'compared with synthesized dataset, only support '
                             'CSV files')
    parser.add_argument('target',
                        help='set file path of target (synthesized) dataset to '
                             'evaluate')

    # optional arguments
    group = parser.add_argument_group('general arguments')
    group.add_argument("-h", "--help", action="help",
                       help="show this help message and exit")
    group.add_argument('--na-values', metavar='LIST',
                       help='set additional values to recognize as NA/NaN; ('
                            'default null values are from pandas.read_csv)')
    group.add_argument('-o', '--output', metavar='FILE', default='report.html',
                       help='set output path for evaluation report; (default '
                            'is "report.html" under current work directory)')

    group = parser.add_argument_group('advanced arguments')
    group.add_argument('--category', metavar='LIST',
                       help='set categorical columns separated by a comma.')
    group.add_argument('-t', '--test',
                       help='set test dataset for classification or regression '
                            'task; (default take 20%% from source dataset)')
    group.add_argument('--class-label', metavar='LIST',
                       help='set column name as class label for classification '
                            'or regression task; supports one or multiple '
                            'columns (separated by comma)')

    args = parser.parse_args()
    start = time.time()

    na_values = str_to_list(args.na_values)
    class_labels = str_to_list(args.class_label)
    categories = str_to_list(args.category)

    # check kinds of parameters
    args.output = os.path.join(os.getcwd(), args.output)
    # if output folder not exists, then create it.
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))

    def complement(attrs, full):
        return set(attrs or []) - set(full)

    # Initialization task:
    source = read_data_from_csv(args.source, na_values=na_values, header='infer')
    target = read_data_from_csv(args.target, na_values=na_values, header='infer')
    test = read_data_from_csv(args.test) if args.test is not None else None

    comp = complement(class_labels, source.columns)
    if comp:
        parser.exit(message=f'--class-label(s): {comp} are not in source file.')
    comp = complement(class_labels, target.columns)
    if comp:
        parser.exit(message=f'--class-label(s): {comp} are not in target file.')

    frame = BiFrame(source, target, categories=categories)
    frame.to_html(buffer=args.output, title='Data Utility Evaluation Report',
                  labels=class_labels, test=test)

    duration = time.time() - start
    print(f'Evaluate dataset {args.source} and {args.target} and generate '
          f'report at {args.output} in {round(duration, 2)} seconds.')


if __name__ == '__main__':
    # For Testing
    main()
