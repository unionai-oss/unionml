import git

app_name = "{{ cookiecutter.app_name }}"


# initialize app directory as a git repo
repo = git.Repo.init(".")
repo.git.add(all=True)
repo.git.commit("-m", f"initial commit for {app_name}")
