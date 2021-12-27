# Contributing

Simple guide on how to contribute.

## Develop flow

The feature requests forms into TODO list. Then a new branch with feature name is created. When it's finished, a pull request is created on the new branch.

Repository uses semantic versioning. 

Some maintainer of repo will push the pull request to the master branch when Bugfix or when it's backward compatible feature. Usually, more pull requests will be joined for a new release. If changes are not backward compatible, it's merged to develop branch, which is always a new major version.


## Ask for feature - report bug

If someone wants a new feature or found a bug, just fill a new issue on GitHub page. Use appropriate tag. It would be nice that if it's a bug, code is attached, so the problem is replicable.


## TODO list

A necessary part of software development is a plan of what will be done in future. There is a TODO.md file where is a simple list of what will be done, typically sorted by priority and tagged with complexity.

This list is curated by maintainers and inspired by feature requests.


## Pull Requests, merging and releases

There are no special needs or recommendations about pull requests from any contributor.

Pull requests are checked and commented and if OK, it's merged locally by maintainers and then pushed to master with pipeline.

It uses no public CI/CD (Travis was historically used, but did not meet expectations). Custom CI/CD *mypythontools* is used, and it runs locally. It creates venv if not exists, update libraries to defined versions from requirements, reformat with `black`, run tests and if tests passes, and it's master branch, version is auto incremented, tag is created, merged branch is pushed, and it's released to PyPi.


## Code of Conduct

Be kind to each other. I think that code subculture is cultivated enough, and people here do what they love, so I don't expect any kind of nasty behavior here.

If you criticize something, explain why and don't pretend that you are better than someone else.

It's not necessary to be way formal here...

Be brief, but don't forget that hello, please and thanks want to be read...


## History notes - messy mode explanation

When there is only one contributor and especially when project is in early phase some steps are not necessary, also git messages can get dummy.

It's another story, when other contributors appears, then code must be cleaner, git messages needs to make sense etc.