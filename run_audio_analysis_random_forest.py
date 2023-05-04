import pandas
import kfold_template

from sklearn.preprocessing import StandardScaler 
import numpy

from sklearn.ensemble import RandomForestClassifier

dataset = pandas.read_csv("dataset.csv")
dataset = dataset.drop(['filename'], axis=1)


dataset = dataset.sample(frac=1).reset_index(drop=True)


target = dataset.iloc[:,-1]
target, key = pandas.factorize(target)
print(key)

data = dataset.iloc[:,:-1]
scaler = StandardScaler()
data = scaler.fit_transform(numpy.array(data, dtype=float))

machine = RandomForestClassifier(criterion="gini", max_depth=100, n_estimators = 100, bootstrap=True)


results = kfold_template.run_kfold(data, target, machine, 4, use_r2=False, use_accuracy=True, use_confusion=True)

print(results)

