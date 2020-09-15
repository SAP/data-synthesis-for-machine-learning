#!/usr/bin/env python

from setuptools import setup

INSTALL_REQUIRES = [
    'numpy >= 1.14.3',
    'matplotlib >= 2.2.2',
    'mako ==1.0.12',
    'pandas >= 0.24.2',
    'scikit-learn >= 0.20.2',
    'pytest >= 4.6.2',
    'python-dateutil >= 2.7.3',
    'setuptools >= 39.1.0'
]


def main():
    setup(name='ds4ml',
          description='A python library for data synthesis and evaluation',
          version='0.1.0',
          packages=['ds4ml', 'ds4ml.command'],
          package_data={
              '': ['template/*.html']
          },
          entry_points={
              'console_scripts': [
                  'data-synthesize = ds4ml.command.synthesize:main',
                  'data-evaluate = ds4ml.command.evaluate:main'
              ]
          },
          author=['Rongjun', 'Yan', 'David'],
          author_email=['rongjun.gao@sap.com', 'yan.zhao01@sap.com', 'd.xia@sap.com'],
          install_requires=INSTALL_REQUIRES,
          platform='any')


if __name__ == '__main__':
    main()
