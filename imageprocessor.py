import pandas as pd
import numpy as np
import sys
from os import listdir
from os.path import isfile, join
from PIL import Image
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import homogeneity_score, v_measure_score
from random import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import timeit
from graphing import graphThings

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


def remakeImage(df, size=IMAGE_SIZE, isCancer=False):
    if type(size) == int:
        size = (size, size)
    image = Image.new('RGB', (size[0], size[1]))
    output = np.zeros((size[0], size[1], 3))
    for x in range(size[0]):
        for y in range(size[1]):
            red_val = int(df['red'][x * size[1] + y])
            green_val = int(df['green'][x * size[1] + y])
            if isCancer:
                image.putpixel((x, y), (0, green_val , 0))
                output[x,y] = (0, green_val , 0)
            else:
                image.putpixel((x, y), (red_val , 0, 0))
                output[x,y] = (red_val , 0, 0)
            
    
    return image, output


def remakeColoredImage(array, size=IMAGE_SIZE):
    if type(size) == int:
        size = (size, size)
    image = Image.new('RGB', (size[0], size[1]))
    for x in range(size[0]):
        for y in range(size[1]):
            pixel = array[y, x]
            image.putpixel((x, y), (int(pixel[0] ), int(pixel[1] ), int(pixel[2] )))
    
    return image




def combineImages(image_arrays, size=IMAGE_SIZE):
    # Width is constant, Height is defined by how many images it can fit.
    #
    # Num per row = 
    width = 800
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
        _, array = remakeImage(img, isCancer=img['class'] == 'cancer')
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

    df = pd.DataFrame(columns=['red', 'class'])
    both = cancer_images + healthy_images

    shuffle(both)
    for img in both:
        image = Image.open(img)
        image = image.resize((size, size))
        # print(image.size)
        R = []
        G = []
        for x in range(image.size[0]):
            for y in range(image.size[1]):
                pixel = image.getpixel((x, y))
        
                R.append(pixel[0])
                G.append(pixel[1])
        if 'unhealthy' in img:
            df = df.append({'red': R, 'green': G, "class": 'cancer'}, ignore_index=True)
        else:
            df = df.append({'red': R, 'green': G, "class": 'healthy'}, ignore_index=True)

    df.to_csv('data_' + (2 * str(size)) + '.csv')
    return df


def train_test_split(X, where=0.8):
    chop = int(len(X) * where)
    train = X[:chop]

    test = X[chop:]

    return train, test


def loadData(size=IMAGE_SIZE):

    df = pd.read_csv('data_' + (2 * str(IMAGE_SIZE)) + '.csv')

    df['red'] = processArray(df['red'])
    df['green'] = processArray(df['green'])

    return df


def kmeans(df, clusters=2):
    train, test = train_test_split(df)

    clustering = KMeans(n_clusters=clusters).fit(list(train['red']))
    print(clustering.labels_)
  

    classes = []
    cluster_labels = {}
    for i in range(clusters):
        
        # print(counts)
        # classes.append((train[clustering.labels_ == i] ))
        cluster = train[clustering.labels_ == i]

        counts = cluster['class'].value_counts()

        if 'healthy' not in count:
            cluster_label[i] = 'cancer'
            continue
        elif 'cancer' not in count:
            cluster_label[i] = 'healthy'
            continue
        if counts['healthy'] > counts['cancer']:
            cluster_labels[i] = 'healthy'
        else:
            cluster_labels[i] = 'cancer'
        classes.append(cluster)
        # print("Class", str(i), "count:\n", counts)

    print(homogeneity_score(test['class'], clustering.predict(list(test['red']))))
    print(v_measure_score(test['class'], clustering.predict(list(test['red']))))

    test_predictions = clustering.fit_predict(list(test['red']))
    
    result = np.array(list(map(lambda x: cluster_labels[x], test_predictions)))

    correct = len(test[result == test['class']])

    print("ACcuracy: ", correct / len(test))



    return classes


def svm(df):
    train, test = train_test_split(df)
    clf = SVC(gamma=2, C=1)
    # clf = DecisionTreeClassifier()
    clf.fit(list(train['red'].values), train['class'].values)

    score = clf.score(list(test['red'].values), test['class'].values)

    print("SVM Score: ", score)

    prediction = clf.predict(list(test['red'].values))
    # print(len(prediction[(prediction == 'healthy') & (test['class'] == 'cancer')]))
    # print(len(prediction[(prediction == 'cancer') & (test['class'] == 'healthy')]))
    print(prediction)
    healthy = test[prediction == 'healthy']
    cancer = test[prediction == 'cancer']


    result = combineImages(healthy)
    result.save("healthy_svm" + str(IMAGE_SIZE) + ".jpg")
    result = combineImages(cancer)
    result.save("cancer_svm" + str(IMAGE_SIZE) + ".jpg")
    # print(clf.predict(list(train['red'])))

    return clf


def logistic(df):
    train, test = train_test_split(df)
    clf = LogisticRegression()
    clf.fit(list(train['red'].values), train['class'].values)

    score = clf.score(list(test['red'].values), test['class'].values)
    print("Logistic Score: ", score)

    prediction = clf.predict(list(test['red'].values))

    # TP = test[(prediction == 'healthy') & (train['class'] == 'healthy')]
    # # TP['color'] = 'green'
    # TN = test[(prediction == 'cancer') & (train['class'] == 'cancer')]
    # # TN['color'] = 'red'
    # FP = test[(prediction == 'healthy') & (train['class'] == 'cancer')]
    # # FP['color'] = 'yellow'
    # FN = test[(prediction == 'cancer') & (train['class'] == 'healthy')]
    # # FN['color'] = 'blue'


    healthy = test[prediction == 'healthy']
    cancer = test[prediction == 'cancer']


    result = combineImages(healthy)
    result.save("healthy_log" + str(IMAGE_SIZE) + ".jpg")
    result = combineImages(cancer)
    result.save("cancer_log" + str(IMAGE_SIZE) + ".jpg")


    print(score)

    return clf


def agglomerative(df, clusters=2):
    train, test = train_test_split(df)

    clustering = AgglomerativeClustering(n_clusters=clusters, linkage='ward').fit(list(train['red']))
    print(clustering.labels_)
  

    classes = []
    for i in range(clusters):
        classes.append(train[clustering.labels_ == i])
        print("Class", str(i), "count:\n", classes[-1]['class'].value_counts())

    print(homogeneity_score(test['class'], clustering.fit_predict(list(test['red']))))
    print(v_measure_score(test['class'], clustering.fit_predict(list(test['red']))))
    
    return classes


def spectral(df, clusters=2):
    train, test = train_test_split(df)

    clustering = SpectralClustering(n_clusters=clusters).fit(list(train['red']))
    print(clustering.labels_)
  

    classes = []
    for i in range(clusters):
        classes.append(train[clustering.labels_ == i])
        print("Class", str(i), "count:\n", classes[-1]['class'].value_counts())

    print(homogeneity_score(test['class'], clustering.fit_predict(list(test['red']))))
    print(v_measure_score(test['class'], clustering.fit_predict(list(test['red']))))
    
    return classes


def neuralnetwork(df):
    train, test = train_test_split(df)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2500,1000,500,50,2), random_state=1)
    clf.fit(list(train['red'].values), train['class'].values)


    prediction = clf.predict(list(test['red'].values))
    score = clf.score(list(test['red'].values), test['class'].values)
    print(score)

    
    healthy = test[prediction == 'healthy']
    cancer = test[prediction == 'cancer']


    result = combineImages(healthy)
    result.save("healthy_nn" + str(IMAGE_SIZE) + ".jpg")
    result = combineImages(cancer)
    result.save("cancer_nn" + str(IMAGE_SIZE) + ".jpg")

print("Loading Data...")
# df = processImages()
df = loadData()
print("Started Training...")
classes = kmeans(df, 20)
# print(timeit.timeit("logistic(df)", globals=globals(), number=1))
# logistic(df)
# classes = dbscan(df)
# svm(df)
# logistic(df)
# neuralnetwork(df)

# graphThings(classes, ratio=False)


# print("Creating results...")
# for i, c in enumerate(classes):
#     result = combineImages(c)
#     result.save("C" + str(i) + '.jpg')
print("Finished")





