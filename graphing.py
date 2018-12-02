import matplotlib.pyplot as plt
import numpy as np


def graphThings(classes):
    count = []

    for c in classes:
        counts = c['class'].value_counts()
        total = np.sum(counts)
        if 'cancer' in counts.index:
            count.append([counts['healthy'] / total, counts['cancer'] / total])
        else:
            arr = [counts['healthy'] / total, 0]
            count.append(arr)

    count = sorted(count, key=lambda x: x[1])    
    index = np.arange(len(classes))
    bar_width = 0.35
    _, ax = plt.subplots()
    ax.bar(index, list(map(lambda x: x[0], count)), bar_width, color='g')
    ax.bar(index + bar_width, list(map(lambda x: x[1], count)), bar_width, color='r')

    plt.show()
    



# if __name__ == "__main__":
