PYTHON ?= python

.PHONY: install lint test train evaluate clean

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

lint:
	$(PYTHON) -m flake8 src/ tests/ --max-line-length=100

test:
	$(PYTHON) -m pytest -v tests/

train:
	$(PYTHON) -m src.train

evaluate:
	$(PYTHON) -m src.evaluate

clean:
	$(PYTHON) -c "import pathlib, shutil; [p.unlink() for d in ('src', 'tests') for p in pathlib.Path(d).rglob('*.pyc')]; [shutil.rmtree(p) for d in ('src', 'tests') for p in list(pathlib.Path(d).rglob('__pycache__'))]; shutil.rmtree('.pytest_cache', ignore_errors=True)"
