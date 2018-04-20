import arff
import numpy as np
import pandas as pd
import math
import operator
from classifier import classifier

class knn(classifier):
    def __init__(self, k = 2):
        self.k = k
        self.train_data = None
        self.test_data = None
        self.train_labels = None

    def fit(self, X, Y):
        self.train_data = X
        self.train_labels = Y

    def predict(self, X):
        self.test_data = X
        predictions=[]
        for x in range(len(X)):
            neighbors = self.findNeighbors(self.train_data, self.test_data[x], self.k, self.train_labels)
            result = self.decision(neighbors)
            predictions.append(result)
        return predictions
    
    def euclidDist(self, inst1, inst2, length):
        dist = 0
        for x in range(length):
            dist += pow((inst1[x] - inst2[x]), 2)
        finalDist = math.sqrt(dist)
        return finalDist

    def findNeighbors(self, trainingSet, testInst, k, train_labels):
        dist = []
        length = len(testInst) - 1
        for x in range(len(trainingSet)):
            finalDist = self.euclidDist(testInst, trainingSet[x], length)
            dist.append((trainingSet[x], train_labels[x], finalDist))
        dist.sort(key=operator.itemgetter(2))
        neighbors = []
        for x in range(k):
            neighbors.append(dist[x][:2])
        return neighbors

    def decision(self, neighbors):
        classVotes = {}
        for x in range(len(neighbors)):
            response = neighbors[x][-1][0]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
        return sortedVotes[0][0]

    def findAccuracy(self, test_labels, predictions):
        correctPred = 0
        for x in range(len(test_labels)):
            if test_labels[x][0] == predictions[x]:
                correctPred += 1
        accuracy = (correctPred/float(len(test_labels))) * 100.0
        return accuracy
    
if __name__ == '__main__':
    # Loading data into DataFrame
    data = arff.load('PhishingData.arff')
    df = pd.DataFrame(data)

    PhishingData_data = df.iloc[:,:9]
    PhishingData_labels = df.iloc[:,9:]

    # Spliting into training and testing set (80/20)
    split = int(len(PhishingData_data) * 0.8)
    train_data = PhishingData_data[:split]
    train_labels = PhishingData_labels[:split]
    test_data = PhishingData_data[split:]
    test_labels = PhishingData_labels[split:]

    # Converting training data and labels to array and float type
    train_data = np.asarray(train_data)
    train_data = train_data.astype(np.float)
    train_labels = np.asarray(train_labels)
    train_labels = train_labels.astype(np.float)

    # Converting testing data and labels to array and float type
    test_data = np.asarray(test_data)
    test_data = test_data.astype(np.float)
    test_labels = np.asarray(test_labels)
    test_labels = test_labels.astype(np.float)

    # Main logic execution
    for i in range(2,13):
        knnImpl = knn(k = i)
        knnImpl.fit(train_data, train_labels)
        predictions = knnImpl.predict(test_data)
        # Calculating Accuracy
        accuracy = knnImpl.findAccuracy(test_labels, predictions)
        print('Accuracy: ' + repr(accuracy) + '%', 'when k = ' + repr(i))