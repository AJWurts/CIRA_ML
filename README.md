# CIRA_ML
CIRA (Chromosomal Image Recognition Algorithm) Image Processing and ML

# Uses Machine Learning to label cells as unhealthy or healthy.
Runs on red cell images. 

# Machine Learning Algorithms
## Clustering Algorithms
The clustering algorithms were used as classifying algorithms by clustering the data with varying values for K then labeling each cluster with the class that occured the most. When a new cell needed to be classified it was placed in the closest cluster and given the same label as that cluster. Accuracy was calculated using this method and the original class labels. 
### K-Means
Accuracy: 0.791
### Agglomerative Clustering and Ward
Accuracy: 0.610
### Agglomerative Clustering with Complete Link
Accuracy: 0.712
## Classification Algorithms
### Support Vector Machine
Accuracy 0.4975
### Logistic Regression
Accuracy: 0.87
### Multi-Level Perceptron
Accuracy: 0.86025
