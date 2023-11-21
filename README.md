# InstaDeep Machine Learning Pipeline
This repository hosts the code for the Protein Classifier project, designed to categorize protein domains based on their amino acid sequences. It features an enhanced Bidirectional LSTM model, outperforming the former CNN model in terms of accuracy and F1-Score. The repository includes  the restructured code as well as comprehensive test scripts. Additionally, it incorporates a workflow for continuous integration, ensuring that new updates undergo unit testing, code style checks, and more. 

## Find a Stronger Model

To build a stronger model, I made some changes to the model architecture and training process. Here are the key changes:

### Model Architecture

#### Previous Model (CNN with One-Hot Encoding)

- The previous model used a Convolutional Neural Network (CNN) with one-hot encoding of amino acids.
- It captured local patterns within the sequence but didn't consider broader context information between amino acids in different positions.

#### Improved Model (Bidirectional LSTM with Embedding Layer)

- I switched to a Bidirectional Long Short-Term Memory (BiLSTM) model with an embedding layer.
- The embedding layer mapped categorical amino acids into continuous vectors, allowing the model to capture richer context information.
- Bidirectional LSTMs were used to learn both local and global sequence patterns.
![bilstm](docs/images/bilstm_arch.png)

### Early Stopping

- To prevent overfitting, I implemented early stopping for both the previous and improved models.
- Early stopping monitors the validation loss and stops training when the loss starts to increase, indicating overfitting.

### Hyperparameter Tuning

- While I haven't performed extensive hyperparameter tuning, I believe increasing the LSTM/Embedding dimensions in the BiLSTM model will lead to better results.
- Hyperparameter optimization, such as Grid Search, Random Search, could be implemented. 

### Evaluation Metrics

- I used the F1-score as one of the evaluation metrics, especially important for imbalanced datasets.
- The F1-score provides a balanced measure of precision and recall and is useful when dealing with uneven class distributions.

### Model Comparison

- The improved BiLSTM model showed an increase in accuracy and F1-score compared to the previous CNN model.

CNN Score
- Test Accuracy: 94.56%
- Test F1 Score: 94.51%

Bi-LSTM Score
- Test Accuracy: 95.60%
- Test F1 Score: 95.36%

## Build a Local Environment Using Docker

To build a local environment, I created a Docker image named `instadeep` and provided the following commands for evaluation and prediction:
- --type : Type of run
- --model_type : Type of models - Can be expanded to not just ProtCNN later on
- --model_dir : Path to model directory
- --seq_max_len : Length of protein sequence
- --epochs : Number of epochs to train for
- --batch_size : Batch size for training
- --data_dir : Path to data directory
- --predict_dir : Path to predict directory
- --data_workers : Number of workers for data loading

We can expand the arguments to have more features for different hyperparameter tuning. This refactored codebase gives the baseline to do so.

We first build the docker by
`docker build -t instadeep`

### Train Model Example
```bash
docker run --gpus all -it --rm instadeep --model_type cnn --type train --data_workers 2 --seq_max_len 120 --batch_size 256 
```

### Evaluate Model Example
```bash
docker run --gpus all -it --rm instadeep --model_type cnn --type evaluate --data_workers 5 --model_dir models/cnn_2023-11-19_16-07-08.ckpt --seq_max_len 120 --batch_size 256
```

### Predict Model Example
```bash
docker run --gpus all -it --rm instadeep --model_type cnn --type predict --data_workers 5 --predict_dir data/sample/test --model_dir models/cnn_2023-11-19_16-07-08.ckpt --seq_max_len 120 --batch_size 256
```

## Python Script Refactoring and Code Cleanup
I restructured the project by refactoring the existing notebook into distinct Python scripts, organizing them in the `src` folder. This separation led to the creation of several focused modules: `dataset`, `evaluate`, `model`, `predict`, `train`, and `main`.

### main.py
`main.py` serves as the central module, coordinating the use of other modules based on the required tasks.

### dataset.py
`dataset.py` handles data retrieval and preprocessing. It sources data from the `data` folder and formats it into the `SequenceDataset` structure. This folder includes three subfolders: raw for raw input data, sample for testing out the prediction, and processed for processed data to expedite training process - unfortunately never used :(.

### evaluate.py
`evaluate.py` is responsible for model evaluation. It takes a specified model (through the `model_dir` argument) and evaluates its performance using the test dataset provided.

### predict.py
`predict.py` manages model predictions. It uses the model specified in the `model_dir` argument and processes data from the `predict_dir` argument. The results are then outputted to a newly generated file.

### train.py
`train.py` focuses on model training. It utilizes the model defined in the `model_dir` argument, executing the training process and refining the model's performance.

## Code Quality Assurance and Consistency with Automated Tests

I have enhanced the project's code quality and consistency by integrating PyLint and setting up automated testing frameworks. Key components of this process include:

- **GitHub Workflow Integration**: I created `.github/workflows/python-lint.yml`. This integration ensures that each commit and pull request adheres to established Python coding guidelines. Tools like `flake8` and `pylint` are used within this workflow to automatically review the code for style and consistency issues, and enforces PEP 8 coding standards. This workflow will also flag any potential error and bugs in the code. This ensures ongoing quality and consistency throughout the development process.

- **Code Formatting with Black**: To maintain a uniform code style, the configuration file includes Black, a Python code formatter. This tool standardizes the formatting across all Python files, enhancing readability and maintainability.

- **Optimized Import Management with isort**: isort is integrated for organizing and cleaning up the Python library imports, ensuring an orderly and efficient code structure.

- **Automated Testing with PyTest**: Additionally, this configuration incorporates PyTest to validate the correctness of the code. It automatically executes tests on new submissions to the repository, verifying that all changes meet the required standards.

### Testing Modules

#### test_dataset.py
`test_dataset.py` focuses on validating the data handling functionalities, ensuring accurate data loading and preprocessing.

#### test_evaluate.py
`test_evaluate.py` tests the evaluation module, verifying the accuracy and reliability of the model evaluation process.

#### test_model.py
`test_model.py` tests the model's architecture and functions, ensuring its able to be initalized.

#### test_performance.py
`test_performance.py` assesses the inference time of the model.

#### test_predict.py
`test_predict.py` evaluates the prediction capabilities of the model, ensuring it accurately predicts based on given inputs.

#### test_utils.py
`test_utils.py` tests utility functions, confirming their reliability and correctness in supporting the main functionalities of the project.
