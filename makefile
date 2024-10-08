.PHONY: clean test pyright

flist = $(wildcard cellcommunicationpf2/figures/figure*.py)
allOutput = $(patsubst cellcommunicationpf2/figures/figure%.py, output/figure%.svg, $(flist))

all: $(allOutput)

output/figure%.svg: cellcommunicationpf2/figures/figure%.py
	@ mkdir -p ./output
	rye run fbuild $*

test: .venv
	rye run pytest -s -v -x

.venv:
	rye sync

coverage.xml: .venv
	rye run pytest --junitxml=junit.xml --cov=cellcommunicationpf2 --cov-report xml:coverage.xml

pyright: .venv
	rye run pyright cellcommunicationpf2

clean:
	rm -rf output profile profile.svg
	rm -rf factor_cache
