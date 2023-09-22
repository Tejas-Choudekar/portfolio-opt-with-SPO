# Portfolio Optimization using Smart "Predict, then Optimize" Framework with Machine Learning Techniques
This repository experiments with Smart "Predict, then Optimize" framework in the context of portfolio optimization. This project was created as a part of Master Thesis for a program named Data analytics and Decision science in RWTH Aachen University.

## Motivation for this Project
Portfolio optimization is an essential part of modern finance, aimed at maximizing returns while minimizing risk. Traditional approaches often treat the prediction and optimization processes as sequential steps, lacking a coherent methodology to integrate them. This project uses the Smart "Predict, then Optimize" (SPO) framework to tackle both steps jointly, thereby improving the effectiveness of portfolio optimization strategies.

## Approach
The research aims to use machine learning techniques to predict stock returns and optimize an investment portfolio based on these predictions. Unlike traditional approaches that handle prediction and optimization as separate stages, the SPO framework leverages the optimization problem structure—objectives and constraints—to guide prediction. Given the computational expense of using SPO loss, a convex surrogate loss function, SPO+ loss, is utilized in model training. The study also delves into feature importance in predictive modeling.

## Results
- The model trained on the SPO framework showed comparable performance to models trained on Huber loss and MSE loss but did not outperform them.
- Interpretability experiments using Shapley values illustrated a marked difference in decision-making between the SPO model and those trained on Huber loss, MSE loss, and MAE loss.

## Project Structure
The project is organized into the following folder structure:

- data: This folder contains all data sets required for the experiments.
- src: Houses essential files for data evaluation and model training based on the SPO framework.
- entrypoint: Includes individual Jupyter notebooks detailing each experiment conducted.
- models: Contains the final saved models.
- config: Consists of a YAML file containing all static data needed for the project.


#### To run this project several python packages needs to be installed which are mentioned in the requirements.txt file
To install these packages simply write below command in by opening the command prompt in the directory containing requirements.txt file:
```pip install -r requirements.txt```

#### The complete data to run this repo can be downloaded from [Zhong and Hitchcock (2021)](https://github.com/Shanlearning/SP-500-Stock-Prediction/tree/master)