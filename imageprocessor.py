import pandas as pd
import numpy as np
import sys
from os import listdir
from os.path import isfile, join
from PIL import Image
from sklearn.svm import SVC, LinearSVC
from random import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans

def processArray(array):
    X = []
    for i in range(len(df)):
        c = array[i]
        for char in '[]\n\r,':
            c = c.replace(char, '')
        
        result = np.array(c.split(' '), dtype=str)
        result = result[result != '']
        result = list(map(lambda x: float(x), result))
        X.append(result)

    return X

def remakeImage(array):
    

# CANCER_FOLDER_NAME = '.\\sendzip\\unhealthyCells'
# HEALTHY_FOLDER_NAME = '.\\sendzip\\healthyCells'

# cancer_images = []
# healthy_images = []
# # cancer_images = [join(CANCER_FOLDER_NAME, f) for f in listdir(CANCER_FOLDER_NAME) if isfile(join(CANCER_FOLDER_NAME, f))]
# # healthy_images = [join(HEALTHY_FOLDER_NAME, f) for f in listdir(HEALTHY_FOLDER_NAME) if isfile(join(HEALTHY_FOLDER_NAME, f))]
# base = '.\\sendzip\\'
# for i in [1,2,3,4,5,6,8,9,10]:
#     baseName = base + str(i) + '\\healthyCells\\'
#     healthy_images += [join(baseName, f) for f in listdir(baseName) if isfile(join(baseName, f))]
#     baseName2 = base + str(i) + '\\unhealthyCells\\'
#     cancer_images += [join(baseName2, f) for f in listdir(baseName2) if isfile(join(baseName2, f))]

# df = pd.DataFrame(columns=['data', 'class'])
# both = cancer_images + healthy_images

# shuffle(both)
# for img in both:
#     image = Image.open(img)
#     image = image.resize((50, 50))
#     R = []
#     for x in range(image.size[0]):
#         for y in range(image.size[0]):
#             pixel = image.getpixel((x, y))
      
#             R.append(pixel[0])
#     if 'unhealthy' in img:
#         df = df.append({'data': R, "class": 'cancer'}, ignore_index=True)
#     else:
#         df = df.append({'data': R, "class": 'healthy'}, ignore_index=True)


# df.to_csv("data_3030.csv")

df = pd.read_csv('data_5050.csv')

df['data'] = processArray(df['data'])

def train_test_split(X, where=0.8):
  chop = int(len(X) * where)
  train = X[:chop]

  test = X[chop:]

  return train, test

print("Started Training")


clustering = KMeans(n_clusters=2).fit(list(df['data']))
print(clustering.labels_)
negative = clustering.labels_[clustering.labels_ == 0]
positive = clustering.labels_[clustering.labels_ == 1]

negativeClass = df['class'][clustering.labels_ == 0].value_counts()
negativeAntiClass = df['class'][clustering.labels_ == 1].value_counts()

print(negativeClass, negativeAntiClass)

print(len(positive))
print(len(negative))

# train, test = train_test_split(df)
# clf = SVC(gamma=2, C=1)
# # clf = DecisionTreeClassifier()
# clf.fit(list(train['data'].values), train['class'].values)

# score = clf.score(list(test['data'].values), test['class'].values)

# print(score)

# prediction = clf.predict(list(test['data']))
# print(len(prediction[(prediction == 'healthy') & (test['class'] == 'cancer')]))
# print(len(prediction[(prediction == 'cancer') & (test['class'] == 'healthy')]))
# # print(clf.predict(list(train['data'])))
