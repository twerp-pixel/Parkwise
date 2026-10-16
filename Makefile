.PHONY: install build serve test evaluate
install:
	pip install -r requirements-dev.txt
build:      ## simulate data, train, evaluate, write artifacts/ (about 2 minutes)
	python -m parkwise build
serve:      ## dashboard + API on http://127.0.0.1:8000
	python -m parkwise serve
test:
	python -m unittest discover -s tests -t . -v
evaluate:   ## print headline numbers from artifacts/report.json
	python -m parkwise evaluate
