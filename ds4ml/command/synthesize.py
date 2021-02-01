"""ds4ml.command.synthesize

Provides the command line to synthesize a dataset directly (e.g. csv file) or
from a pattern file which describes the dataset anonymously (and which can be
got through command `data-pattern`).

"""

from ds4ml.dataset import DataSet
from ds4ml.utils import (CustomFormatter, read_data_from_csv, file_name,
                         str_to_list, ends_with_json)

import argparse
import time


def main():
    parser = argparse.ArgumentParser(
        description='Synthesize one dataset by differential privacy',
        formatter_class=CustomFormatter,
        add_help=False)
    parser.add_argument('file', help='set path of a csv file to be synthesized '
                                     'or path of a pattern file to be generated')

    # optional arguments
    group = parser.add_argument_group('general arguments')
    group.add_argument("-h", "--help", action="help",
                       help="show this help message and exit")
    group.add_argument('--pseudonym', metavar='LIST',
                       help='set candidate columns separated by a comma, which '
                            'will be replaced with a pseudonym. It only works '
                            'on the string column.')
    group.add_argument('--delete', metavar='LIST',
                       help='set columns separated by a comma, which will be '
                            'deleted when synthesis.')
    group.add_argument('--na-values', metavar='LIST',
                       help='set additional values to recognize as NA/NaN; '
                            '(default null values are from pandas.read_csv)')
    group.add_argument('-o', '--output', metavar='FILE',
                       help="set the file name of output synthesized dataset ("
                            "default is input file name with suffix '-a.csv')")
    group.add_argument('--no-header', action='store_true',
                       help='indicate there is no header in a CSV file, and '
                            'will take [#0, #1, #2, ...] as header. (default: '
                            'the tool will try to detect and take actions)')
    group.add_argument('--records', metavar='INT', type=int,
                       help='specify the records you want to generate; default '
                            'is the same records with the original dataset')
    group.add_argument('--sep', metavar='STRING', default=',',
                       help='specify the delimiter of the input file')

    group = parser.add_argument_group('advanced arguments')
    group.add_argument('-e', '--epsilon', metavar='FLOAT', type=float,
                       help='set epsilon for differential privacy (default 0.1)',
                       default=0.1)
    group.add_argument('--category', metavar='LIST',
                       help='set categorical columns separated by a comma.')
    group.add_argument('--retain', metavar='LIST',
                       help='set columns to retain the values')

    args = parser.parse_args()
    start = time.time()

    pseudonyms = str_to_list(args.pseudonym)
    deletes = str_to_list(args.delete)
    categories = str_to_list(args.category)
    na_values = str_to_list(args.na_values)
    retains = str_to_list(args.retain)
    header = None if args.no_header else 'infer'

    # check the file type from its extension
    is_pattern = ends_with_json(args.file)
    if is_pattern:
        if retains is not None and len(retains) != 0:
            parser.exit(message='Do not support --retain option when '
                                'synthesize from pattern file.')
        # construct DataSet from pattern file
        dataset = DataSet.from_pattern(args.file)
    else:
        data = read_data_from_csv(args.file, na_values=na_values, header=header,
                                  sep=args.sep)

        def complement(attrs, full):
            return set(attrs or []) - set(full)

        # check parameters: pseudonyms, deletes, categories
        comp = complement(pseudonyms, data.columns)
        if comp:
            parser.exit(
                message=f'--pseudonym columns: {comp} are not in csv file.')
        comp = complement(deletes, data.columns)
        if comp:
            parser.exit(
                message=f'--delete columns: {comp} are not in csv file.')
        comp = complement(categories, data.columns)
        if comp:
            parser.exit(
                message=f'--category columns: {comp} are not in csv file.')

        dataset = DataSet(data, categories=categories)

    synthesized = dataset.synthesize(epsilon=args.epsilon,
                                     pseudonyms=pseudonyms, deletes=deletes,
                                     retains=retains, records=args.records)
    if args.output is None:
        name = file_name(args.file)
        args.output = f'{name}-a.csv'
    synthesized.to_csv(args.output, index=False, sep=args.sep)

    duration = time.time() - start
    print(f'Synthesize from {args.file} to file {args.output} in '
          f'{round(duration, 2)} seconds.')


if __name__ == '__main__':
    main()
