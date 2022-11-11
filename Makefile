.PHONY: docs

clear-cache:
	pyflyte local-cache clear

docs:
	$(MAKE) -C docs clean html SPHINXOPTS='-j 4 -v -W'
