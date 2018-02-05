# data-incubator-hengjia-wang
Use machine learning to spot abuse and harassment online.

# About
Discussing things you care about can be difficult. The threat of abuse and harassment online means that many people stop expressing themselves and give up on seeking different opinions. This project is to build a machine learning predictor that tells if a comment is toxic or not and also breakdowns 6 different types of toxicity, such as threats, obscenity, insults and *etc*. 

The predicting model is then deployed onto Heroku. Here is the link to the Webapp: <a href="http://data-incubator-hengjia-wang.herokuapp.com">
  http://data-incubator-hengjia-wang.herokuapp.com </a>
  
Data Source: <a href="https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data">
 Toxic Comment Classification Challenge on Kaggle </a>
 
# Steps

**train_model.py**
Read data in csv format, train an *ensemble.ExtraTreesClassifier* model (Achieved 0.982 for the mean column-wise ROC AUC.), and then save the model.

**predict.py**
contain a function *predict* that returns a list of 6 numeric values in range of (0, 1), each for one type of toxicity. The function is callable by other programs.

**app.py**
This program read the input comment and return the prediction on the Webapp.

# Contact
For any questions or advices, please email <hengjiawang1990@gmail.com>
