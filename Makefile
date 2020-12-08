.PHONY: help

NOCOLOR=\033[0m
YELLOW=\033[1;33m

help:
	@echo "${YELLOW}make help${NOCOLOR}: show help document"
	@echo "${YELLOW}make init${NOCOLOR}: setup environment"
	@echo "${YELLOW}make it${NOCOLOR}: run integration tests"
	@echo "${YELLOW}make ut${NOCOLOR}: run unit tests"
	@echo "${YELLOW}make publish${NOCOLOR}: publish to PyPi"

init:
	pip3 install -r requirements.txt

it:
	PYTHONPATH=. python3 ./ds4ml/command/synthesize.py example/adult.csv -o adult-a.csv
	PYTHONPATH=. python3 ./ds4ml/command/pattern.py example/adult.csv -o adult-pattern.json
	PYTHONPATH=. python3 ./ds4ml/command/synthesize.py adult-pattern.json --records 8141 -o adult-pattern.csv
	PYTHONPATH=. python3 ./ds4ml/command/evaluate.py adult-a.csv adult-pattern.csv -o report.html

ut:
	pytest

publish:
	pip3 install 'twine>=3.0.0'
	python3 setup.py sdist bdist_wheel
	python3 -m twine upload dist/*
	rm -fr build dist .egg ds4ml.egg-info
