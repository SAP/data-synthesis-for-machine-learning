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
	@echo "==> Integration testing on parameters of command/synthesize.py"
	PYTHONPATH=. python3 ./ds4ml/command/synthesize.py example/adult.csv -e 0.01 --category workclass --pseudonym marital-status --delete native-country --records 8145 -o adult-a1.csv
	PYTHONPATH=. python3 ./ds4ml/command/pattern.py example/adult.csv -o adult-p1.json
	PYTHONPATH=. python3 ./ds4ml/command/synthesize.py adult-p1.json -o adult-b1.csv
	PYTHONPATH=. python3 ./ds4ml/command/evaluate.py adult-a1.csv adult-b1.csv --class-label salary -o report-1.html
	@echo "==> Integration testing on parameters of command/pattern.py"
	PYTHONPATH=. python3 ./ds4ml/command/synthesize.py example/adult.csv -o adult-a2.csv
	PYTHONPATH=. python3 ./ds4ml/command/pattern.py example/adult.csv -e 0.01 --category workclass --pseudonym marital-status --delete native-country -o adult-p2.json
	PYTHONPATH=. python3 ./ds4ml/command/synthesize.py adult-p2.json --records 8145 -o adult-b2.csv
	PYTHONPATH=. python3 ./ds4ml/command/evaluate.py adult-a2.csv adult-b2.csv --class-label salary -o report-2.html

ut:
	pytest

publish:
	pip3 install 'twine>=3.0.0'
	python3 setup.py sdist bdist_wheel
	python3 -m twine upload dist/*
	rm -fr build dist .egg ds4ml.egg-info
