* Here Pandas will be dealing with the big datasets, This can be also used in deleteing or adding any rows in the datasets.

* Pylab will used to import the function of numpy

* numpy will help us with dealing with numbers and array.

* scipy.optimize is used here for optimizing the algorithm which we are choosing.

* mathplotlib is used for visualizing the data in bar garph , pie charts and confusion matrix.

* Seaborn is used to understand the data.


Data Prepration:-

* In this first we need to upload the datasets and the manipluate it according to our needs suppose if we don't need any kind of coloumn we can delete it.

* Here axis = 1 means we are manipluating data in column level, if axis = 0 then we are manipluating with the row level.



# handling missing values.


# splitting the datasets into Test and Train Sets.

1.we always have to train and test the data in machine learning.

2.First we have to train the dataset which we have and then we will test the data whatever we have trainned the data it is working or not.

3.Here in this project we have stored some columns is the x axis and some values in Y axis, where x is the label value and Y is the Target value.

Here we have done normalization which i need more dig into it.

suppose here this dataset we will be doing mean median mode.

Sklearn module will helps us to procide tools to us which is used train and test the datasets.


# Exploratory Data Analysis of Heart Disease Dataset.

Here we are passing manipluated data to visualize the data with the help of mathlib, where length and width is provided.


# Final outcome where we will perform actual algorithm that is logistic regression Model.

* To reach here we have removed unwanted coloumn and we have rename it and we have trained the test the data as well.
       * remove unwanted coloumn
       * rename the coloumn as per our need
       *  Train and Test the dataset
       *  seen visual aspect of the data as well

Now we will make use of sklearn which will provide tools and from there we are importing LogisticRegression.
       Here x_train will acts as input and y_train will acts as output and we are passing y_train into the logistic_regression algorithm.
       From here we have got x_test value as a final target value and passing it to the y_pred.

# Evaluation and accuracy









