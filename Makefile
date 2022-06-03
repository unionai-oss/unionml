.PHONY: docs

docs:
	$(MAKE) -C docs clean html SPHINXOPTS='-j 4 -W'
