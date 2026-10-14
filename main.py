from classificator.naive_bayesian import NaiveBayesianClassificator
from dataset.dataset import BeanDataset

D = BeanDataset("data/bean_train.csv")
V = BeanDataset("data/bean_test.csv")

NBC = NaiveBayesianClassificator(D, False)
NBC.fit()

correction = 0
for i in V.val_set:
    if NBC.predict(V.get_in_val(i)[0])[0] == V.get_in_val(i)[1]:
        correction += 1
print(correction/V.length()[1])
