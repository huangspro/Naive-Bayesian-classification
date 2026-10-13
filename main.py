from classificator.naive_bayesian import NaiveBayesianClassificator
from dataset.dataset import BeanDataset

D = BeanDataset("data/bean_train_1000.csv")

NBC = NaiveBayesianClassificator(D)
NBC.fit()

print(NBC.predict([1, 1, 1, 2, 0, 0, 1, 1, 2, 2, 2, 2, 0, 2, 2, 2]))

correction = 0
for i in D.val_set:
    if NBC.predict(D.get_in_val(i)[0])[0] == D.get_in_val(i)[1]:
        correction += 1
        print("true")
print(correction/len(D)[1])
