.PHONY: docs

clear-cache:
	pyflyte local-cache clear

docs:
	rm -rf docs/source/generated_api_reference && \
	$(MAKE) -C docs clean html SPHINXOPTS='-j 4 -v -W'
