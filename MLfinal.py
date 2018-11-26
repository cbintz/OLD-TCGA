# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 15:02:56 2018

@author: corinnebintz
"""
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram

import csv

def main():
    """ Runs program """
    geneData = readGeneFile()
    peopleData = readPatientFile()
    cluster = makeClusters(geneData[2])
    makeDendrogram(geneData[2], geneData[0])
    findSurvivalRatio(cluster, geneData[1])


def readGeneFile():
    """ Returns array of genes, patients, and patientData """
    patients = []
    patientData = []
    genes = []
    with open('transposed.csv') as geneFile:
        line_count = 0
        csv_reader = csv.reader(geneFile, delimiter=',')
        for row in csv_reader:
            if line_count == 0:
                genes.append(row)
            else:
                intArray = []
                patients.append(row[0])
                for i in range(1, len(row)):
                    intArray.append(float(row[i]))
                patientData.append(intArray)
            line_count += 1
    genes = genes[0][1:]
    print(genes)
    print(patients)
    print(patientData[0])
    return [genes, patients, patientData]

demographics = []
people = []
survivalData = []

def readPatientFile():
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
                for i in range(1, len(row)):
                    array.append(row[i])
                array = np.array(array)
                survivalData.append(array)
            line_count += 1
    return [people, demographics, survivalData]

def makeClusters(patientData):
    """ Returns clusters"""
    cluster = AgglomerativeClustering().fit(patientData)  # fit the hierarchical clustering to patientData
    points = cluster.fit_predict(patientData)  # perform clustering on patientData and return cluster labels
    return cluster


def makeDendrogram(patientData, genes):
    """ Creates dendrogram"""
    linked = linkage(patientData, 'single')
    plt.figure(figsize=(100, 100))
    dendrogram(linked,  orientation='top',labels=genes,distance_sort='descending',show_leaf_counts=True)
    plt.title("Initial Gene Dendrograms")
    plt.xlabel("Genes")
    plt.ylabel("Euclidean Distance between points")
    plt.show()


def findSurvivalRatio(cluster, patients):
    """ Returns an array containing each clusters survival rate ratio"""
    cluster0 = []
    survival0 = []
    cluster1 = []
    survival1 = []
    for i in range(len(cluster.labels_)):
        if cluster.labels_[i] == 0:
            cluster0.append(patients[i])
            survival0.append(survivalData[i][23])
        else:
            cluster1.append(patients[i])
            survival1.append(survivalData[i][23])

    count0 = 0
    count1 = 0
    for i in range(len(survival0)):
        if survival0[i] == "0":
            count0 += 1
        if survival1[i] == "0":
            count1 += 1
    cluster0survivalratio = count0/len(survival0)
    cluster1survivalratio = count1/len(survival1)
    return [cluster0survivalratio, cluster1survivalratio]

if __name__ == "__main__":
    main()

