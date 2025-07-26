.PHONY: clean test pyright profile_cpu

flist = $(wildcard cellcommunicationpf2/figures/figure*.py)
allOutput = $(patsubst cellcommunicationpf2/figures/figure%.py, output/figure%.svg, $(flist))

all: $(allOutput)

output/figure%.svg: cellcommunicationpf2/figures/figure%.py
	@ mkdir -p ./output
	rye run fbuild $*

profile-cpu: .venv
	@ echo "Profiling figure $(F)..."
	@ # Step 1: Run the profiler and save stats to a file. This will still show errors if the script crashes.
	rye run python -m cProfile -o ./output/profile_cpu_$(F).prof .venv/bin/fbuild $(F)
	@ echo "\n--- Profiler Results ---"
	@ # Step 2: Print the saved stats, sorted by total time.
	rye run python -c "import pstats; p = pstats.Stats('./output/profile_cpu_$(F).prof'); p.sort_stats('tottime').print_stats(100)"

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
