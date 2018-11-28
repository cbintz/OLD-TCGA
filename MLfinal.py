# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 15:02:56 2018
final project machine learning
@author: corinnebintz and lillieatkins
"""
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.feature_extraction import DictVectorizer
import csv
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from numpy import array

def main():
    """ Runs program """
    geneData = readGeneFile()
    peopleData = readPatientFile()
    dv = DictVectorizer()
    dv.fit(geneData[3])
    X_vec = dv.transform(geneData[3])

    # rand = random.sample(range(0, 2500), 2500) # randomly select 2500 genes to use for clustering patients
    #
    # randomGenes = limitGenes(rand, geneData[2]) # get patient gene data for these randomly selected genes
    #
    # randCluster = makeClusters(randomGenes) # make clusters from random genes
    #
    # makeFirstDendrogram(randomGenes) # make dendrogram from random genes
    #
    # survival0 = findSurvivalRatio(randCluster, geneData[1], peopleData[2]) # get the survival stats for clusters
    #
    # print(len(survival0[0])) # number of patients in cluster0
    # print(len(survival0[1])) # number of patients in cluster1
    # print(survival0[2]) # survival rate of patients in cluster 0
    # print(survival0[3]) # survival rate of patients in cluster 1

    # selectedFeatures = selectFeatures(dv, X_vec, peopleData[3]) # select top 50 genes using f_classif
    # print("selected features")
    #
    # geneIndices = findGeneIndex(selectedFeatures[0], geneData[0]) # gene indices of selected genes
    # print("found gene indices")
    #
    # newGenes = limitGenes(geneIndices, geneData[2]) # patient gene data for selected genes
    # print(" got new gene data")
    #
    # newCluster = makeClusters(newGenes) # make clusters from selected genes
    # print("made new clusters")
    #
    # makeNewDendrogram(newGenes) # make new dendrogram from selected genes
    # print("made dendrogram")
    #
    # survival = findSurvivalRatio(newCluster, geneData[1], peopleData[2])# get the survival stats for  clusters
    # print("got survival stats")
    # print(len(survival[0])) # number of patients in cluster0
    # print(len(survival[1])) # number of patients in cluster1
    # print(survival[2]) # survival rate of patients in cluster 0
    # print(survival[3])  # survival rate of patients in cluster 1

    x = geneData[2]
    y = peopleData[3]
    [training_x, training_y, csv_x, csv_y, test_x, test_y, training_data_labeled, csv_data_labeled] = splitData(x, y)
    runNN(training_x, training_y, 0.01, 100)
    testNN(test_x, test_y, 0.01, 100)

def runNN(x_data_set, y_data_set, alpha, num_epochs):
    #NEED TO ADD TEST AND TRAIN ACCURARCY AND KEEP TRACK OF THE LOSS THEN WE CAN GRAPH THAT TO SEE THE CHANGE WITH ITERATIONS
    train_accurary = []

    model = Neural_Network(len(x_data_set[0]), 50)

    model.train() #set model to training mode

    criterion = torch.nn.BCELoss(size_average=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=alpha)

    for epoch in range(num_epochs): #change range
        y_pred = model(x_data_set)

        loss = criterion(y_pred, y_data_set)

        print(epoch, loss.data[0])

        optimizer.zero_grad()

        loss.backward() #calculates the gradients

        optimizer.step() #institutes gradient descent

    return train_accurary

def testNN(x_data_set, y_data_set, alpha, num_epochs):
    test_accuracy = []

    model = Neural_Network(len(x_data_set[0]), 50)

    model.eval() #set model to evaluation mode

    criterion = torch.nn.BCELoss(size_average=True)

    for epoch in range(num_epochs): #change range
        y_pred = model(x_data_set)

        loss = criterion(y_pred, y_data_set)

        print(epoch, loss.data[0])

    return test_accuracy

def limitGenes(geneIndices, patientData):
    newGeneData = []
    for patient in patientData:
        array = []
        for i in range(len(patient)):
            if i in geneIndices:
                array.append(patient[i])
        newGeneData.append(array)
    return newGeneData

def findGeneIndex(featureGenes, genes):
    indices = []
    for gene in featureGenes:
        indices.append(genes.index(gene))
    return indices

def findMissingPatients(genePatients, survivalPatients):
    missing = []
    for patient in survivalPatients:
        if patient not in genePatients:
            missing.append(patient)
    return missing

def selectFeatures(dv, X, Y):
    features = mutual_info_classif(X, Y)
    topFeatures = []
    topFeatureDict ={}
    for score, fname in sorted(zip(features, dv.get_feature_names()), reverse=True)[:50]:
        topFeatures.append(fname)
        topFeatureDict[fname] = score
    return [topFeatures, topFeatureDict]

def readGeneFile():
    """ Returns array of genes, patients, and patientData """
    patients = []
    patientData = []
    genes = []
    geneDict = []
    with open('transposed.csv') as geneFile:
        line_count = 0
        csv_reader = csv.reader(geneFile, delimiter=',')
        for row in csv_reader:
            if line_count == 0:
                genes.append(row)
                genes = genes[0][1:]
            else:
                intArray = []
                dict = {}
                patients.append(row[0])
                for i in range(1, len(row)):
                    dict[genes[i-1]] = float(row[i])
                    intArray.append(float(row[i]))
                patientData.append(intArray)
                geneDict.append(dict)
            line_count += 1
    return [genes, patients, patientData, geneDict]

def readPatientFile():
    demographics = []
    people = []
    survivalData = []
    survival = []
    """ Returns an array of people, demographics, and survivalData"""
    with open('TCGA_LUAD_survival.csv') as survivalFile:
        line_count = 0
        csv_reader = csv.reader(survivalFile, delimiter=',')
        for row in csv_reader:
            if line_count == 0:
                demographics.append(row)
            else:
                array = []
                people.append(row[0])
                survival.append(row[24])
                for i in range(1, len(row)):
                    array.append(row[i])
                array = np.array(array)
                survivalData.append(array)
            line_count += 1
    return [people, demographics, survivalData, survival]

# def makeClusters(patientData):
#     """ Returns clusters"""
#     cluster = AgglomerativeClustering().fit_predict(patientData)  # perform clustering on patientData and return cluster labels
#     return cluster
#
#
# def makeFirstDendrogram(patientData):
#     """ Creates initial dendrogram"""
#     linked = linkage(patientData, 'single')
#     plt.figure(figsize=(100, 100))
#     dendrogram(linked,  orientation='top',labels=None,distance_sort='descending',show_leaf_counts=True)
#     plt.title("Initial Gene Dendrograms")
#     plt.xlabel("Genes")
#     plt.ylabel("Euclidean Distance between points")
#     plt.show()
#
# def makeNewDendrogram(patientData):
#     """ Creates new dendrogram for smaller subset of genes"""
#     linked = linkage(patientData, 'single')
#     plt.figure(figsize=(100, 100))
#     dendrogram(linked,  orientation='top',labels=None,distance_sort='descending',show_leaf_counts=True)
#     plt.title("Selected Gene Dendrogram")
#     plt.xlabel("Genes")
#     plt.ylabel("Euclidean Distance between points")
#     plt.show()
#
#
# def findSurvivalRatio(cluster, patients, survivalData):
#     """ Returns an array containing each clusters survival rate ratio"""
#     print(cluster)
#     cluster0 = []
#     survival0 = []
#     cluster1 = []
#     survival1 = []
#     for i in range(len(cluster)):
#         if cluster[i] == 0:
#             cluster0.append(patients[i])
#             survival0.append(survivalData[i][23])
#         else:
#             cluster1.append(patients[i])
#             survival1.append(survivalData[i][23])
#
#     count0 = 0
#     count1 = 0
#     for i in range(len(survival0)):
#         if survival0[i] == "0":
#             count0 += 1
#     for i in range(len(survival1)):
#         if survival1[i] == "0":
#             count1 += 1
#     cluster0survivalratio = count0/len(survival0)
#     cluster1survivalratio = count1/len(survival1)
#     return [cluster0, cluster1, cluster0survivalratio, cluster1survivalratio]


def splitData(x_data, y_data):
    X = torch.tensor(([2, 9], [1, 5], [3, 6]), dtype=torch.float)  # 3 X 2 tensor
    y = torch.tensor(([92], [100], [89]), dtype=torch.float)
    training_x = []
    training_y = []
    training_data_labeled = []
    csv_x = []
    csv_y = []
    csv_data_labeled = []
    test_x = []
    test_y = []
    for i in range(len(x_data)):
        if i < 412:
            training_x.append(x_data[i])
            training_y.append(y_data[i])
        elif i < 463:
            csv_x.append(x_data[i])
            csv_y.append(y_data[i])
        else:
            test_x.append(x_data[i])
            test_y.append(y_data[i])

    csv_x_copy = csv_x
    training_x_copy = training_x

    for j in range(463):
        if j < 412:
            training_x_copy[j].append(training_y[j])
            training_data_labeled.append(training_x_copy[j])
        else:
            csv_x_copy[j - 412].append(csv_y[j - 412])
            csv_data_labeled.append(csv_x_copy[j-412])

    # convert to numpy arrays
    training_x = array(training_x, dtype=np.float32)
    training_y = array(training_y, dtype=np.float32)
    csv_x = array(csv_x, dtype=np.float32)
    csv_y = array(csv_y, dtype=np.float32)
    test_x = array(test_x, dtype=np.float32)
    test_y = array(test_y, dtype=np.float32)

    #convert to tensors
    training_x = Variable(torch.from_numpy(training_x))
    training_y = Variable(torch.from_numpy(training_y))
    csv_x = Variable(torch.from_numpy(csv_x))
    csv_y = Variable(torch.from_numpy(csv_y))
    test_x = Variable(torch.from_numpy(test_x))
    test_y = Variable(torch.from_numpy(test_y))
    return [training_x, training_y, csv_x, csv_y, test_x, test_y, training_data_labeled, csv_data_labeled]


class Neural_Network(nn.Module):
    def __init__(self, input_size, hidden_layers_size):
        super(Neural_Network, self).__init__()

        #right now I'm creating a nueral network with 2 hidden layers and an output layer
        self.l1 = nn.Linear(input_size, hidden_layers_size) #creates the hidden layer
        self.l2 = nn.Linear(hidden_layers_size, hidden_layers_size)
        self.l3 = nn.Linear(hidden_layers_size, 1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out1 = F.relu(self.l1(x))
        out2 = F.relu(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred

if __name__ == "__main__":
    main()

