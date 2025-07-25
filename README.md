# DeepVaris

## Overview

DeepVaris is a deep learning-based approach for feature (variable) selection, with an emphasis on interpretability. It leverages convolutional neural networks (CNNs) to predict outcomes and utilizes surrogate modeling for variable importance score (FIS) estimation while ensuring feature stability across different data batches. This method provides enhanced control over the false discovery rate (FDR) and improves the identification of true features compared to other state-of-the-art methods.

## Requirements

This project is developed in Python 3.6.5. To set up the environment, you can use the provided `environment.yml` file. It includes the necessary dependencies for running the code.

To install the required packages, use the following command:

```bash
conda env create -f environment.yml
conda activate DeepVaris
```

## Experiment Execution

### Run DeepVaris

Under the Example folder, there is a sample result using dataset3, with a sample size of 1000 and a feature dimension of 784. You can run this example as follows :

```bash
python simulation/dataset3.py
```

### Results

After the experiment is completed, the system will generate a `results` folder, along with a subfolder named after the corresponding dataset. Inside the subfolder, multiple result files will be generated. First, the folder will contain text files that record the model evaluation metrics for each run, including accuracy, loss, AUC, F1 score, and other relevant data representing the results of individual experiments. Next, a final result file, which records the outcome after the intersection of features, will be saved in a text file. Another key result is the process of how the number of features and important features changes with the number of intersection iterations, which will be saved in a CSV file. 

## Citation
Hu, X., Ma, Y., Jiang H. Variable Selection by Interpreting Deep Learning Predictors with Surrogate Modeling. (Submitted)

## Contact
If you have any questions, please feel free to contact *xiaoyuehu@zju.edu.cn*.