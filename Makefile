PIP_COMPILE = pip-compile --upgrade --verbose

.PHONY: install-piptools
install-piptools:
	pip install -U pip-tools setuptools wheel "pip>=22.0.3"

requirements.txt: requirements.in install-piptools
	$(PIP_COMPILE) $<

requirements-dev.txt: requirements-dev.in requirements.txt install-piptools
	$(PIP_COMPILE) $<

requirements-docs.txt: requirements-docs.in install-piptools
	$(PIP_COMPILE) $<
