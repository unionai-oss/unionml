import subprocess
import venv

import git

app_name = "{{ cookiecutter.app_name }}"
create_venv = "{{ cookiecutter.venv }}" == "True"


# initialize app directory as a git repo
repo = git.Repo.init(".")


if create_venv:
    print(f"[unionml] creating virtual environment in {app_name}/venv")
    venv.create("./venv", with_pip=True)
    print("[unionml] install dependencies")
    subprocess.run(
        [
            "./venv/bin/pip",
            "install",
            "-r",
            "requirements.txt",
        ],
    )

repo.git.add(all=True)
repo.git.commit("-m", f"setup {app_name}")
