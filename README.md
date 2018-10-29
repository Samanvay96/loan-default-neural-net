# Loan Default Neural Network
## Aim
To demonstrate the implementation of a multi-layer perceptron (MLP) neural classifier model with one hidden layer. 

## Dataset 
The data was sourced from www.lendingclub.com. The complete loan dataset was used from 2007-2011. 

## Methodology 
1. The data was cleaned to exclude all columns that were deemed unnecessary for the purpose of the build, for example the notes/comments column. Primarily numerical and categorical data was used for the prediction.
2. Convert all categorical variables to boolean ones using a pandas function called `get_dummies`
3. Split the dataset into test and train 
4. Normalise all features as the MLP model used is particularly sensitive to this. 
5. Run the model. A MLP classifier is used with default values. 
6. Run model over test dataset
7. Test the accuracy of the model 

## Outcome
Simply using a single hidden layer of 3 perceptrons, a 99% accuracy was achieved. 
