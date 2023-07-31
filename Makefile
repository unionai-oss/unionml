.PHONY: docs

define PIP_COMPILE
pip-compile $(1) ${PIP_ARGS} --upgrade --verbose --resolver=backtracking
endef

install-piptools:
	pip install pip-tools

requirements-dev.txt: requirements-dev.in install-piptools
	$(call PIP_COMPILE,requirements-dev.in)

requirements-docs.txt: requirements-docs.in install-piptools
	$(call PIP_COMPILE,requirements-docs.in)

requirements-ci.txt: requirements-ci.in install-piptools
	$(call PIP_COMPILE,requirements-ci.in)

clear-cache:
	pyflyte local-cache clear

docs:
	rm -rf docs/source/generated_api_reference && \
	$(MAKE) -C docs clean html SPHINXOPTS='-j 4 -v -W'

quick-docs:
	NB_EXECUTION_MODE=off $(MAKE) docs

setup:
	pip install -r requirements-dev.txt
