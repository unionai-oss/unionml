.PHONY: docs

clear-cache:
	pyflyte local-cache clear

docs:
	rm -rf docs/source/generated_api_reference && \
	$(MAKE) -C docs clean html SPHINXOPTS='-j 4 -v -W'

quick-docs:
	NB_EXECUTION_MODE=off $(MAKE) docs

setup:
	pip install -r requirements-dev.txt
