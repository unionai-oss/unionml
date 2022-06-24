(contributing_guide)=

# Contributing Guide

We are striving to build an open and inclusive community around UnionML. Whether you are a novice or
experienced software developer, data scientist, or machine learning practitioner, all contributions
and suggestions are welcome!

## Ways to Contribute

There are many ways to contribute to this project:

- 📖 **Improve the documentation**: See typos and grammatical errors? Open up a PR to fix them!
- 🐞 **Report Bugfixes**: Coming up against unexpected behavior? [Open up an issue](https://github.com/unionai-oss/unionml/issues/new)!
- 🙏 **Feature Requests**: Have ideas on new features or improvements? [Open up an issue](https://github.com/unionai-oss/unionml/issues/new)!
- 🔧 **Pull Requests**: Pick up one of the issues in the [issue log](https://github.com/unionai-oss/unionml/issues) and submit a PR!

## Roadmap

[![Roadmap](https://img.shields.io/badge/Project-Roadmap-blueviolet?style=for-the-badge)](https://github.com/orgs/unionai-oss/projects/1/views/4)
[![OSS Planning](https://img.shields.io/badge/Event-OSS_Planning-yellow?style=for-the-badge)](https://app.addevent.com/event/tj14110550/)

The release roadmap is a living being 🌳, so you can also participate in roadmapping and prioritization!

Each milestone in the roadmap is made up of multiple issues, and you can upvote roadmap items by going
to the issue page and giving it a thumbs up 👍.

The UnionML maintainers hold a bi-weekly open source sync up and re-prioritize the roadmap
based on community interest, so subscribe to the google calendar event by clicking on the
"OSS Planning" badge above if you're interested in joining.

## Development Environment Setup

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
