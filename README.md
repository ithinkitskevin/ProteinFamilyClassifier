# instadeep-ml-pipeline


## Find stronger model


## Build a local environment using docker


## Clean & Refactor the code into Python Scripts
The code was refactored into Python scirpts by separating the notebook into separate codes in the src folder. 

## Add command line to predict and/or evaluate a trained model
### Train Model
Created cause I thought it would be cool. 

## Ensure code quality & consistency. Add tests
To ensure code quality & consistency, I added in PyLint. I first added in .github\workflows\python-lint.yml. This file will  Follow Style Guides: Adhere to PEP 8 for Python coding standards. Tools like flake8 or pylint can automatically check your code for style issues.  You can also integrate flake8 and pylint in your CI pipeline (like GitHub Actions, GitLab CI/CD) to automatically check code quality in merge requests or commits.

The python-lint.yml file is a configuration file for GitHub Actions, a CI/CD (Continuous Integration/Continuous Deployment) service provided by GitHub. This specific file is set up to run a linter on your Python code. A linter is a tool that analyzes source code to flag programming errors, bugs, stylistic errors, and suspicious constructs.  When you push changes to your repository or make a pull request, GitHub Actions will automatically run Pylint on your code. It will then report any issues it finds, helping to ensure that all code in the repository maintains a certain level of quality and consistency.

This configuration file also contains a way to run PyTest. This code will ensure that any codes that gets pushed into the repo will be correct. There are several tests.

### test_dataset


### test_infer


### test_model

### test_performance

### test_training


This configuration file also ensured the code is well formatted using Black and also cleaned up the Python library imports using isort. 

## Document the repository
Here it is! The README.md.