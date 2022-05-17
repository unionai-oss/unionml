(contributing)=

# Contributing

We are striving to build an open and inclusive community around UnionML. Whether you are a novice or
experienced software developer, data scientist, or machine learning practitioner, all contributions
and suggestions are welcome!

## Ways to Contribute

There are many ways to contribute to this project:

- 📖 **Improve the documentation**: See typos and grammatical errors? Open up a PR to fix them!
- 🐞 **Report Bugfixes**: Coming up against unexpected behavior? [Open up an issue](https://github.com/unionai-oss/unionml/issues/new)!
- 🙏 **Feature Requests**: Have ideas on new features or improvements? [Open up an issue](https://github.com/unionai-oss/unionml/issues/new)!
- 🔧 **Pull Requests**: Pick up one of the issues in the [issue log](https://github.com/unionai-oss/unionml/issues) and submit a PR!

## Environment Setup

Create a virtual environment:

```
python -m venv ~/venvs/unionml
source activate ~/venvs/unionml/bin/activate
```

Install dependencies:

```
pip install -r requirements-dev.txt
```

### `pre-commit` Setup

This project uses [pre-commit](https://pre-commit.com/) to ensure code standards. Follow the
[installation](https://pre-commit.com/#installation) instructions, then setup the `pre-commit` hooks:

```
pre-commit install
```

Make sure everything is working correctly by running:

```
pre-commit run --all
```

###  Run Unit Tests

Finally, to make sure your development environment is ready to go, run the unit tests:

```
pytest tests/unit
```

### Run Integration Tests [Optional]

Optionally, you can run integration tests locally. First install [flytectl](https://docs.flyte.org/projects/flytectl/en/latest/#installation), then:

```
python tests/integration
```
