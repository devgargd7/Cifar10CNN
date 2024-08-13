# CIFAR-10 Benchmark CNN
Re-implementation of [94% on CIFAR-10 in 3.29 Seconds on a Single GPU](https://arxiv.org/pdf/2404.00498) for TAMU's CSCE-636: Deep Learning project.

Report: [report.pdf](https://github.com/devgargd7/Cifar10CNN/blob/main/report.pdf)

## Steps to run code
Given the data is in 'data' directly, outside of 'code' directory.
### Training
```cd code
python3 main.py train ../data .
```
### Testing
```
cd code
python3 main.py test ../data .
```
### Predicting
```
cd code
python3 main.py predict ../data ../predictions
```
This creates a predictions.npy file outside of 'code' directory.


## Directory Structure
|- <b>code</b> (directory containing all the python code files) <br>
|- <b>data</b> (directory containing all the training and testing CIFAR-10 data) <br>
|- <b>saved_models</b> (directory containing the saved model which can be used for testing/predicting)<br>
|- <b>logs</b> (directory containing logs for all experimentation runs)
|- <b>README</b> <br>
