from classificator.naive_bayesian import NaiveBayesianClassificator
from dataset.dataset import BeanDataset

D = BeanDataset("data/bean_train.csv")

NBC = NaiveBayesianClassificator(D)
NBC.fit()
print(NBC.predict([0, 0, 0, 1, 0, 0, 0, 0, 1, 2, 2, 2, 1, 2, 2, 2]))
