from classificator.naive_bayesian import NaiveBayesianClassificator
from dataset.dataset import BeanDataset

D = BeanDataset("data/bean_train.csv")

NBC = NaiveBayesianClassificator(D)
NBC.fit()
for i in D.get_in_val:
    print(NBC.predict(i))
