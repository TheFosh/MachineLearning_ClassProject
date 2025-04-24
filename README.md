# MachineLearning_ClassProject

### Task: "Do something interesting with neural networks"

## My Project:
### [Dataset Used](https://www.kaggle.com/datasets/kane6543/sub-sandwich-customer-satisfaction-score)
<p>For this project I wanted to test the classification ability of a neural network. I did this by using the data set listed above.</p>
<p>In it contains data on different types of satisfaction scores for a brand that makes subway sandwiches. All were from 0-7 quantifiable values. All 41 columns were used as features. The column of the brand data was used as the labels. There were 7 possible brands for it to choose.</p>
<p>For using this data, some cleanup was needed since it was found to have empty cells in the original file. I cleaned this up by removing records that contain a Nan cell.</p>
<p>It was found that using 2 hidden layers with cross entropy loss function worked well.</p>

### Results:
<p>After being trained for 100 epochs, the neural network did better than a 1 out of 7 guess (~14%). It averaged ~35% accuracy on the validation sets.</p>
<p>This can be seen in the graph below. It shows the increase in accuracy over time.</p>

![AccuracyVTime.png](https://github.com/TheFosh/MachineLearning_ClassProject/blob/main/Images/AccuracyOverTime.png)

<p>As well, these guesses aren't too far off the real results when viewing the confusion matrix of the neural network below.</p>

![ConfusionMatrix](https://github.com/TheFosh/MachineLearning_ClassProject/blob/main/Images/foodNN_CM.png)

### Take Aways:
<p>It seems that the network produced a better than random guess. It even showed a trend of continuing to increase in accuracy as it was trained more. However, this can likely be assumed to be due to overfitting. If this is the case, we can use the elbow method for finding a best guess for a more likely accuracy. Looking at the graph of accuracy over time, there appears to be an elbow at about 25-50 epoch's with accuracy 27%-30%.</p>
<p>As well, looking at the confusion matrix, the network is likely to predict the correct brand relative to the others. However, some categories have an accuracy of less than 50%. Meaning it is more likely to be wrong than correct for its guesses. It is also odd that "Firehouse Subs" seems dominant in the guesses. Perhaps there is an unbalanced amount data for this brand compared to the others. A solution to this is using inserting a bias to more evenly predict brands.</p>
