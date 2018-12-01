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

CANCER_FOLDER_NAME = '.\\sendzip\\unhealthyCells'
HEALTHY_FOLDER_NAME = '.\\sendzip\\healthyCells'

IMAGE_SIZE = 50

def processArray(array):
    X = []
    for i in range(len(array)):
        c = array[i]
        for char in '[]\n\r,':
            c = c.replace(char, '')
        
        result = np.array(c.split(' '), dtype=str)
        result = result[result != '']
        result = list(map(lambda x: float(x), result))
        X.append(result)

    return X


def remakeImage(array, size=IMAGE_SIZE, isCancer=False):
    if type(size) == int:
        size = (size, size)
    image = Image.new('RGB', (size[0], size[1]))
    output = np.zeros((size[0], size[1], 3))
    for x in range(size[0]):
        for y in range(size[1]):
            red_val = int(array[x * size[1] + y])
            if isCancer:
                image.putpixel((x, y), (0, red_val, 0))
                output[x,y] = (0, red_val, 0)
            else:
                image.putpixel((x, y), (red_val, 0, 0))
                output[x,y] = (red_val, 0, 0)
            
    
    return image, output


def remakeColoredImage(array, size=IMAGE_SIZE):
    if type(size) == int:
        size = (size, size)
    image = Image.new('RGB', (size[0], size[1]))
    for x in range(size[0]):
        for y in range(size[1]):
            pixel = array[y, x]
            image.putpixel((x, y), (int(pixel[0]), int(pixel[1]), int(pixel[2])))
    
    return image




def combineImages(image_arrays, size=IMAGE_SIZE):
    # Width is constant, Height is defined by how many images it can fit.
    #
    width = 1000
    numRows = int(len(image_arrays) / (width / size)) + 1
    height =  numRows * size

    img_arr = np.zeros((height, width, 3))
    # print(img_arr.size)
    x = 0
    y = 0
    x_incr = size
    y_incr = size

    dictionary  = image_arrays.to_dict('records')
    for img in dictionary:
        if x >= width:
            x = 0
            y += y_incr
        _, array = remakeImage(img['data'], isCancer=img['class'] == 'cancer')
        img_arr[y:y+size, x:x+size] = array
        x += x_incr

    image = remakeColoredImage(img_arr, size=(width, height))
    return image
        
    


def processImages(size=IMAGE_SIZE):

    cancer_images = []
    healthy_images = []
    # cancer_images = [join(CANCER_FOLDER_NAME, f) for f in listdir(CANCER_FOLDER_NAME) if isfile(join(CANCER_FOLDER_NAME, f))]
    # healthy_images = [join(HEALTHY_FOLDER_NAME, f) for f in listdir(HEALTHY_FOLDER_NAME) if isfile(join(HEALTHY_FOLDER_NAME, f))]
    base = '.\\sendzip\\'
    for i in [1,2,3,4,5,6,8,9,10]:
        baseName = base + str(i) + '\\healthyCells\\'
        healthy_images += [join(baseName, f) for f in listdir(baseName) if isfile(join(baseName, f))]
        baseName2 = base + str(i) + '\\unhealthyCells\\'
        cancer_images += [join(baseName2, f) for f in listdir(baseName2) if isfile(join(baseName2, f))]

    df = pd.DataFrame(columns=['data', 'class'])
    both = cancer_images + healthy_images

    shuffle(both)
    for img in both:
        image = Image.open(img)
        image = image.resize((size, size))
        # print(image.size)
        R = []
        for x in range(image.size[0]):
            for y in range(image.size[1]):
                pixel = image.getpixel((x, y))
        
                R.append(pixel[0])
        if 'unhealthy' in img:
            df = df.append({'data': R, "class": 'cancer'}, ignore_index=True)
        else:
            df = df.append({'data': R, "class": 'healthy'}, ignore_index=True)


    df.to_csv('data_' + (2 * str(size)) + '.csv')
    return df


def train_test_split(X, where=0.8):
    chop = int(len(X) * where)
    train = X[:chop]

    test = X[chop:]

    return train, test


def loadData(size=IMAGE_SIZE):

    df = pd.read_csv('data_' + (2 * str(IMAGE_SIZE)) + '.csv')

    df['data'] = processArray(df['data'])

    return df


def kmeans(df, clusters=5):

    clustering = KMeans(n_clusters=clusters).fit(list(df['data']))
    print(clustering.labels_)
    negative = clustering.labels_[clustering.labels_ == 0]
    positive = clustering.labels_[clustering.labels_ == 1]

    classes = []
    for i in range(clusters):
        classes.append(df[clustering.labels_ == i])
        print("Class", str(i), "count:\n", classes[-1]['class'].value_counts())
    # c1 = df[clustering.labels_ == 0]
    # c2 = df[clustering.labels_ == 1]
    # c3 = df[clustering.labels_ == 2]

    # print("Class 0 Count:", c1['class'].value_counts())
    # print("Class 1 count:", c2['class'].value_counts())
    # print("Class 2 count:", c3['class'].value_counts())


    return classes

def svm(df):
    train, test = train_test_split(df)
    clf = SVC(gamma=2, C=1)
    # clf = DecisionTreeClassifier()
    clf.fit(list(train['data'].values), train['class'].values)

    score = clf.score(list(test['data'].values), test['class'].values)

    print(score)

    prediction = clf.predict(list(test['data']))
    print(len(prediction[(prediction == 'healthy') & (test['class'] == 'cancer')]))
    print(len(prediction[(prediction == 'cancer') & (test['class'] == 'healthy')]))
    # print(clf.predict(list(train['data'])))

print("Loading Data...")
# df = processImages()
df = loadData()
print("Started Training...")
classes = kmeans(df, 10)
print("Creating results...")
for i, c in enumerate(classes):
    result = combineImages(c)
    result.save("C" + str(i) + '.jpg')
print("Finished")