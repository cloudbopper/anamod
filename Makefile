.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test clean-docs ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-docs:
	rm -f docs/anamod.rst
	rm -f docs/modules.rst
	$(MAKE) -C docs clean

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -fr prof/

lint: ## check style with pylint/flake8
	flake8 anamod tests
	pylint anamod tests
	doc8 docs README.rst

test: ## run tests quickly with the default Python
	pytest -rA tests

test-update-golds: ## run tests and update gold files (two passes required for hierarchical analysis tests)
	-pytest -rA tests --force-regen
	-pytest -rA tests --force-regen
	python -m tests.gen_condor_tests -type hierarchical -overwrite_golds 1
	python -m tests.gen_condor_tests -type temporal -overwrite_golds 1
	python -m tests.gen_condor_tests -type baselines -overwrite_golds 1

test-condor: ## run tests in parallel over condor with non-shared filesystem
	pytest -rA tests/condor_tests/ -n 20

test-condor-sharedfs: ## run tests in parallel over condor with shared filesystem; need to specify shared working directory
	pytest -rA tests/condor_tests/ -n 20 --shared-fs --basetemp=condor_test_runs

test-all: ## run tests on every Python version with tox
	tox

coverage: ## check code coverage quickly with the default Python
	pytest --cov=anamod tests
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

profile:  # profile the code
	pytest --profile-svg tests
	open prof/combined.svg

docs: clean-docs ## generate Sphinx HTML documentation, including API docs
	# sphinx-apidoc -o docs/ anamod
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release: dist ## package and upload a release
	twine upload dist/*

dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	python setup.py install
