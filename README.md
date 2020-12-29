# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
We are going to work with a dataset from a banking institution. It will be a classification problem.
We have 20 features and the target is binary (Has the client subscribed a term deposit? Yes / No)

More information about the dataset can be found here: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

The best performing model was found by autoML with an accuracy of **0.9167** versus **0.9104** for a hyperdrive run of a LogisticRegression model.

The model from autoML is a votingEnsemble: prefittedsoftvotingclassifier.


## Scikit-learn Pipeline

### Compute

Compute Instances for the notebook: STANDARD_DS3_V2 (CPU Only)
Compute cluster for the training: STANDARD_D2_V2 (CPU Only) [4 nodes]
The compute cluster is created in the notebook. It’s a cluster of 4 nodes with a low priority setting and a min_nodes of 0. With this configuration, when there is no job, the cluster cost nothing and is only resizing on demand.

### Data Preparation

First, we want to create a TabularDataset object from a csv file available at a url.
We use TabularDatasetFactory.from_delimited_files() method to create the TabularDataset. 
Then we use the clean_data() method to prepare the data for the training process.
-	jobs, contact and education features are transformed from categorical variable to dummies variable.
-	days_of_week and month are mapped to a dict value.
-	Binary features like loan are convert into Boolean (1 if loan else 0)

Once the data is cleaned, we split the dataset into 4 with the train_test_split() method.

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

70% of the dataset will be use for training and 30% for the validation.

### Training configuration

The algorithm used to train the model is a LogisticRegression from Scikit-Learn.
The solver used is liblinear and the penalization is the L2 regularisation.

We pass two parameters to the classifier: C and max_iter.

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")




### Hyper parameters tuning

Hyperparameters are adjustable parameters that let you control the model training process. Model performance depends heavily on hyperparameters.
Hyperparameter tuning is the process of finding the configuration of hyperparameters that results in the best performance. The process is typically computationally expensive and manual.

Azure Machine Learning lets you automate hyperparameter tuning and run experiments in parallel to efficiently optimize hyperparameters.
In this case, there is two hyperparameters to optimize: C (Inverse of regularization strength. Smaller values cause stronger regularization) and max_iter (Maximum number of iterations to converge).

Hyperdrive is a tool that allow to specify the parameters we want to tune and the way to do it. Let’s look at the code.

First, we create an SKLearn estimator where we can specify the training script to use and the compute target.
Note: Estimator are now deprecated, use ScriptRunConfig instead.

    est = SKLearn(source_directory = '.', compute_target = cpu_cluster_name  , entry_script='train.py')   

Then we specify a parameter sampling method:

    # Specify parameter sampler
    ps = RandomParameterSampling(
        {
            "C": uniform(0.0001,0.1),
            "max_iter": choice(100,500,1000)
        }
    )

The RandomParameterSampling method is choosing randomly from the search space define for C and max_iter for each run. It’s a good way to start the tuning process because you can see trend in the search space and switch to another method later by reducing the search space.
C has a default value of 1.0 but during the exploration of the model, I found that smaller value performed better than value > 1.0. That’s why I chose to setup the hyperdrive tuning between 0.0001 and 0.1.

    "C": uniform(0.0001,0.1),

Uniform() means that hyperdrive will chose a value from the uniform distribution between 0.0001 and 0.1. Uniform means that all values between the range are equally likely to occur.

We also use a more standard parameter expression for discrete value like max_iter choice() allow us to list the value we want to test.

Ressource: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters

Once the space is defined, we can apply a policy to stop a run early if it don’t meet specific requirements. This way we can win time when a run is not promising.
### Specify a Policy

    policy = BanditPolicy(slack_factor= 0.1, evaluation_interval = 1, delay_evaluation=5)

From the documentaion:

slack_factor  
> The ratio used to calculate the allowed distance from the best performing experiment run. 

evaluation_interval:
>The frequency for applying the policy.

delay_evaluation
> The number of intervals for which to delay the first policy evaluation. If specified, the policy applies every multiple of  evaluation_interval that is greater than or equal to delay_evaluation.

**BanditPolicy** will compare each run to the best run to date and stop any run that is not approaching the best accuracy. It’s a good policy in our case because all runs are in a similar range of accuracy and the median policy would not be suited in this case.

Now let’s create the object to wrap the run configuration:

    # Create a HyperDriveConfig using the estimator, hyperparameter sampler, and policy.
    hyperdrive_config = HyperDriveConfig(estimator=est,
                                         hyperparameter_sampling=ps,
                                         policy=policy,
                                         primary_metric_name='Accuracy',
                                         primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                                         max_total_runs=20,
                                         max_concurrent_runs=4)

The goal is to use hyperdrive to maximise the model accuracy so we set the primary_metric_name to ‘Accuracy’ (named that way in our train.py script) and primary_metric_goal to MAXIMIZE.

### Results
After 20 runs, let’s have look at the best run:

    # Get your best run and save the model from that run.
    
    best_run_hdr = hyperdrive_run.get_best_run_by_primary_metric()
    print(best_run_hdr.get_metrics())
    

> {'Regularization Strength:': 0.07089813510465263, 'Max iterations:':
> 1000, 'Accuracy': 0.9104704097116844}

Our best Accuracy (0.9104) was obtained with theses parameters values:
-	C = 0.07089813510465263
-	Max_iter = 1000


## AutoML

With autoML, we used the same clean dataset and train it with a classification task.

The compute cluster is the same (4 nodes, CPU only) and we put an experiment timeout of 30 minutes.

In 30 minutes, the autoML had the time to do 37 runs testing different combinations of scaling + algorithm (MaxAbsScaler + LightGBM, RobustScaler + LightGBM, …) and the best accuracy was obtained with a VotingEnsemble Model with an accuracy of 0.9167.

A VotingEnsemble is a soft Voting/Majority Rule classifier. It combines the predictions from multiple other models.

## Pipeline comparison

The difference in accuracy is small but AutoML won the match. It can be explained by the fact that autoML used several algorithms versus only one for the hyperdrive run.

> Accuracy of best autoML model vs best HDR model is 0.9167223065250379 vs 0.9104704097116844


Both have the same data preparation steps. The only difference is the approach to solve the problem (predict the value of y). Hyperdrive could be use to test different model too, we could pass a parameter with a list of model names that would be defined in the train.py script but it’s so much easier to do an autoML run in comparison. No training script is required.

AutoML is a good tool to use at the beginning of a project. It could be included in the EDA steps. It allows us to find a base accuracy and the explanation tab can be used to see the most important features.
Hyperdrive is a more precise tool that would be use later in the process to find the best hyperparameters values before going to production.


## Future work

AutoML found an algorithm that performed better than a fine tune LogisticRegression for our use case. It could be a good idea to select this algorithm and use hyperdrive to fine tune the hyperparameters and maybe improve the accuracy.
Another information we learn during the autoML analysis is that the dataset has a class balancing problem due too the small sample of positive target.

![class balancing detection](/img/class_balance_issue.png)

 
This problem can be visualized in the raw confusion matrix:

![raw confusion matrix](/img/raw_confusion_matrix.png)

The amount of class 0 (28240 + 1018) is way superior of the amount of class 1 ( 1966 + 1726).
We can also see that even if we obtain a good accuracy, the fact that we have a class imbalance issue hide the poor performance of the model.
If we look at the normalized confusion matrix, we can see that the model doesn’t have a good prediction results for the positive class (Only 0.5325 accuracy of True Positive).

![normalized confusion matrix](/img/normalized_confusion_matrix.png)
 
We should probably use other metrics to find a better model, maybe using Recall instead of accuracy would be better to improve the True Positive rate.
