# Credit_Card_Fraud_Detection

![](/images/fraud_detection.png)

## **Project Overview**
In this project, I useed a credit card fraud detection dataset, and build a binary classification model that can identify transactions as either fraudulent or valid, based on provided, historical data.
The payment fraud data set (Dal Pozzolo et al. 2015) was downloaded from Kaggle. This has features and labels for thousands of credit card transactions, each of which is labeled as fraudulent or valid. In this notebook, we'd like to train a model based on the features of these transactions so that we can predict risky or fraudulent transactions in the future.

## **EDA**
* The dataset I used to train the model was downloaded using this [link](https://s3.amazonaws.com/video.udacity-data.com/topher/2019/January/5c534768_creditcardfraud/creditcardfraud.zip).
* I loaded and explored the dataset.
* Dataset was split into training and testing set.

## **Modelling**
* In this notebook, I defined and train the SageMaker, built-in algorithm, [LinearLearner](https://sagemaker.readthedocs.io/en/stable/linear_learner.html).
Since we are predicting a credit card transaction as either valid or frudulent, the problem requires a binary classification algorithm, in which a line is separating two classes of data and effectively outputs labels; either 1 for data that falls above the line or 0 for points that fall on or below the line.
![](https://github.com/SamyySwift/Credit_Card_Fraud_Detection/blob/master/images/linear_separator.png)

* I instantiated a LinearLearner Estimator passing it important arguments.
* Finally, I converted the data into Record_set format and fitted it to the estimator.

## **Deployment**
In this step, I used Amason's inbuilt deployment algorithm to deploy the trained moddel to an endpoint. I then used this to make predictions on the test data and to evaluate the model.

## **Model Evaluation**
After the model is deployed, I tested and evaluated the model with the test data and got the following results:
Metrics for simple, LinearLearner.

prediction (col)    0.0  1.0
actual (row)                
0.0               85269   33
1.0                  32  109

Recall:     0.773
Precision:  0.768
Accuracy:   0.999

As we can see,the default LinearLearner got a high accuracy, but still classified fraudulent and valid data points incorrectly. Specifically classifying more than 30 points as false negatives (incorrectly labeled, fraudulent transactions), and a little over 30 points as false positives (incorrectly labeled, valid transactions).

## **Model Tuning**
To improve the model's recall and precision, I tuned some hyperparameters of the estimator as showned below.To aim for a specific metric, LinearLearner offers the hyperparameter binary_classifier_model_selection_criteria, which is the model evaluation criteria for the training dataset.

linear_recall = LinearLearner(role=role,
                              train_instance_count=1, 
                              train_instance_type='ml.c4.xlarge',
                              predictor_type='binary_classifier',
                              output_path=output_path,
                              sagemaker_session=sagemaker_session,
                              epochs=15,
                              binary_classifier_model_selection_criteria='precision_at_target_recall', # target recall
                              target_recall=0.9) # 90% recall
                              
## **Dependencies**
* Amason SageMaker.
* Amason s3 console.


