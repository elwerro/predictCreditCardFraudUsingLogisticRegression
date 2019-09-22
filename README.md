# predictCreditCardFraudUsingLogisticRegression
This is the code from https://www.data-blogger.com/2017/06/15/fraud-detection-a-simple-machine-learning-approach/
I implemented it in Jupyter notebook and also uploaded a .py version of it but you might need to add plt.show() for the plots to show.
I used seaborn to visualize which variable had the most and least amount of credit card fraud cases associated with it, then I tried to get rid of the transactions(instances) that were not relevent in the data to try and increase the ratio between number of cases that are fraud to the number of cases that are not fraud. 
Getting rid of cases with an Amount greater than 3000 and getting rid of cases in variable V2 that were below -25 increased the accuracy by 1% from 88% to 89%.
