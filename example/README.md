# Example

## Overview
This folder contains one demo dataset(adult.csv), its synthesized dataset(adult_new.csv), and the utility evaluation report(report.html).

## Report
+ Basic Information

    ![basic](https://user-images.githubusercontent.com/55168900/66983937-4dd0fa00-f0ec-11e9-926d-266a207fef09.png)

    Two indicators are included to display the distribution similarity.
    - *err*, is the relative error of two values. Here it measures the discrepancy of each column between the original data 
      (e.g. *adult.csv*) and synthesized data (e.g. *adult_new.csv*).
    - *jsd*, is Jensen-Shannon divergence. It is a measure of the similarity between the probability distribution of the original and synthesized data.

+ Attribute Distribution

    ![attribute](https://user-images.githubusercontent.com/55168900/66983935-4d386380-f0ec-11e9-8e53-e12610cf9542.png)


    It shows the count of values for each column.
    If the column is numerical, its values are split to an array of bins with the same step (e.g. above 
    distribution of *age*); If the column is categorical, its values are unique, just like column *race*:

    ![basic_race](https://user-images.githubusercontent.com/55168900/66983938-4e699080-f0ec-11e9-9628-5d558b79c9b7.png)


 + Pair-wise Correlation

   ![pair_wise](https://user-images.githubusercontent.com/55168900/66983939-4f022700-f0ec-11e9-8be5-4457f43e9eb2.png)


   It show a matrix of [mutual information](https://en.wikipedia.org/wiki/Mutual_information) for pair-wise 
   column. Its value ranges from 0 (no mutual information) to 1 (perfect correlation).

 + Do Classification Tasks
   
   Besides the above indicators, you can evaluate them by doing some machine learning tasks, e.g. classification. 
   For example, run command `data-evaulate adult.csv adult_new.cv --class-label sex,salary`. It will 
   generate the fourth section in the *report.html*, like:

   ![svm](https://user-images.githubusercontent.com/55168900/66984062-8f61a500-f0ec-11e9-9a9e-c9a56c5754c8.png)

   - [*Confusion Matrix*](https://en.wikipedia.org/wiki/Confusion_matrix) can evaluate the accuracy of a 
     classification algorithm, e.g. The value *0.18* shows that there are 180 persons whose salary >50K, but classified as <=50K
   - *Misclassification Error Rate* is calculated based on the confusion matrix
