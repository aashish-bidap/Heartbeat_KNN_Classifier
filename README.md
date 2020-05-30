# Heartbeats-Classification-KNN_Classifier

**Problem Statement:**

Using KNN classifier make predictions for the time series data.

**Dataset :**

Data which was formatted as part of a thesis “Generalized feature extraction for structural pattern recognition in time-series data” at Carnegie Mellon University, 2001.The dataset includes the electrical activity reported during a heartbeat (Olszewski, Maxion and Siewiorek D,2001). 

The 2 classes in the data include a Normal Heartbeat and Myocardial infarction which is a medical terminology for a heart attack.

**K Nearest Neighbor (KNN) :**
KNN is an algorithm which is used in classification and regression problems. Classification problems are the one in which the target variable is discrete while the regression problems are the one with continuous target variable. KNN stores all available cases and classifies a new variable based on the distance function. KNN calculates the output of the new example depending on the k closest training examples in the feature space.


**Distance Metric :**
Used Minkowski distance in order to find the distance between the new test data and the already classified training data. The best part about the Minkowski distance, it is a metric which can be considered as generalization for the Euclidean distance and the Manhattan distance


Execution for different P and K with different values of P and K. P = [0.5, 1, 2 ,4] & K = [3, 5, 11].


**Execution Results:**

With careful validation of the classification accuracy of the model it can be observed that the Experiment 4 with K= 3 and P = 4 yields the best accuracy of 93% and the lowest RMSE of 0.52.



Source:
1. 2001.Olszewski R., Maxion R. and Siewiorek D. Generalized feature extraction for structural pattern recognition in time-series data. Dissertation. Carnegie Mellon University, USA.ISBN 978-0-493-53871-6 Order Number: AAI3040489.
