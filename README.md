# MachineLearning_ClassProject

### Task: "Do something interesting with neural networks"

## My Project:
### [Dataset Used](https://www.kaggle.com/datasets/kane6543/sub-sandwich-customer-satisfaction-score)
<p>For this project I wanted to test the classification ability of a neural network. I did this by using the dataset listed above.</p>
<p>In it contains data on different types of satisfaction scores for a brand that makes subway sandwiches. All were from 0-7 quantifilable values. All 41 columns were used as features. The column of the brand data was used as the labels. There were 7 possible brands for it to choose.</p>
<p>It was found that using 2 hidden layers with cross entryopy loss function worked well.</p>

### Results:
<p>After being trained for 100 epochs, the neural network did better then a 1 out of 7 guess (~14%). It averaged ~35% accuracy on the validation sets.</p>
<p>This can be seen in graph below. It shows the increase in accuracy over time.</p>

![AccuracyVTime.png](https://github.com/TheFosh/MachineLearning_ClassProject/blob/main/Images/AccuracyOverTime.png)

<p>As well, these guesses aren't two far off the real reaults when viewing the confusion matrix of the neural network below.</p>

![ConfusionMatrix](https://github.com/TheFosh/MachineLearning_ClassProject/blob/main/Images/foodNN_CM.png)

### Take Aways:
