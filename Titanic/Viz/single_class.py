# survival rate concerning class, sex and marriage status.

from pandas import Series, DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import pylab

df = pd.read_csv('../Data/train.csv')

# slice
idx_male = df.Sex[df.Sex == 'male'].index
idx_female = df.index.diff(idx_male)

idx_single = df.SibSp[df.SibSp == 0].index
idx_not_single = df.index.diff(idx_single)

idx_survived = df.Survived[df.Survived == 1].index
idx_died = df.index.diff(idx_survived)

idx_class1 = df.Pclass[df.Pclass == 1].index
idx_class2 = df.Pclass[df.Pclass == 2].index
idx_class3 = df.Pclass[df.Pclass == 3].index

idx_f1 = idx_female.intersection(idx_class1)  # class-1 female
idx_f2 = idx_female.intersection(idx_class2)  # class-2 female
idx_f3 = idx_female.intersection(idx_class3)  # class-3 female

idx_sf1s = idx_f1.intersection(idx_single).intersection(idx_survived)
idx_nsf1s = idx_f1.intersection(idx_not_single).intersection(idx_survived)
idx_sf2s = idx_f2.intersection(idx_single).intersection(idx_survived)
idx_nsf2s = idx_f2.intersection(idx_not_single).intersection(idx_survived)
idx_sf3s = idx_f3.intersection(idx_single).intersection(idx_survived)
idx_nsf3s = idx_f3.intersection(idx_not_single).intersection(idx_survived)

xlabels = ['Single', 'Company']

ax = plt.subplot(1, 3, 1)
single_female_class1 = [
    float(len(idx_sf1s))/len(idx_f1.intersection(idx_survived)),
    float(len(idx_nsf1s))/len(idx_f1.intersection(idx_survived))]
ax.bar(range(2), single_female_class1, align='center', alpha=0.4)
ax.set_xlabel("high class")
pylab.xticks(range(2), xlabels)

ax = plt.subplot(1, 3, 2)
single_female_class2 = [
    float(len(idx_sf2s))/len(idx_f2.intersection(idx_survived)),
    float(len(idx_nsf2s))/len(idx_f2.intersection(idx_survived))]
ax.bar(range(2), single_female_class2, align='center', alpha=0.4)
ax.set_xlabel("middle class")
pylab.xticks(range(2), xlabels)

ax = plt.subplot(1, 3, 3)
single_female_class3 = [
    float(len(idx_sf3s))/len(idx_f3.intersection(idx_survived)),
    float(len(idx_nsf3s))/len(idx_f3.intersection(idx_survived))]
ax.bar(range(2), single_female_class3, align='center', alpha=0.4)
ax.set_xlabel("low class")
pylab.xticks(range(2), xlabels)

plt.suptitle("Survival rate for female of different social classes.")

plt.show()

# idx_sm = idx_male.intersection(idx_single)  # single male
# idx_sf = idx_female.intersection(idx_single)  # single female
# idx_nsm = idx_male.intersection(idx_not_single)
# idx_nsf = idx_female.intersection(idx_not_single)

# def main():
# if __name__ == '__main__':
#     main()
