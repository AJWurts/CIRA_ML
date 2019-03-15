# CIRA_ML
CIRA (Chromosomal Image Recognition Algorithm) Image Processing and ML

# Uses Machine Learning to label cells as unhealthy or healthy.
Runs on red cell images.   
Example Data:  
![Image1](https://github.com/AJWurts/CIRA_ML/blob/master/Image%20Results/406_healthy-22_cancerous-run2.jpg "K-Means Visualization")    
More examples and results can be viewed in the Image Results folder.


# Machine Learning Algorithms
## Clustering Algorithms
The clustering algorithms were used as classifying algorithms by clustering the data with varying values for K then labeling each cluster with the class that occured the most. When a new cell needed to be classified it was placed in the closest cluster and given the same label as that cluster. Accuracy was calculated using this method and the original class labels. 
### K-Means
Accuracy: 0.791

![Image1](https://lh3.googleusercontent.com/gdN7ptckeO6W30wd1VehuMuQ6RURhS9Mjtx3ZkyKPZ__JH-ViiQshJmcACokZMWoXkD4vK42H4_aU7IFXZePBFRd1XB-J3B2gCPNASh4ety8O_xN8hp0XF73rR-_DgOciRlVLMJ2 "K-Means Visualization")  
Image 1. K-Means K=2 Graph: This shows the best clustering result among the 3 chosen methods K-Means, with K=2. The left side would be the unhealthy cluster, and the right side would be the healthy cluster. 

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
