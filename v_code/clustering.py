import numpy as np
from kruskal import kruskal

# Input:
# - dataP:          Data points (numCoordinates x numPoints)
#                   if data points is numPoints x numPoints, data points
#                   transposed symetric matrix
# - confLevel:      Confidence level for calculating weight threshold (0~1)
# - minRate:        Noise cluster(data) if cluster size smaller than minRate*numPoints (0~1)
# - lambda:         Regularization parameter of the SMCE optimization program
# - KMax:           Maximum neighborhood size to select the sparse neighbors from
# - verbose:        True if want to see the optimization information, else false
# - visualization:  Visualization clustering
# Output:
# - fClusterPCoordinate:    Cluster coordinates (numCorrdinates x numClusters)
# - fClsuterPVoting:        Number of datas for each cluster (numClusters)
# - fClusterPSumweight:     Sum of weights for each cluster (numClusters)
# - pointToClusterNo:       Cluster No. for each data (numPoints)

def clustering(dataP,confLevel=0.95,minRate=0.01,_lambda=10,KMax=50,verbose=False,visualization=False):
    numPoints = len(dataP)
    #print(numPoints)
    print("********************** Clustering **********************")

    # Complete Graph and Kruskal MST
    distMatrix = np.zeros((numPoints, numPoints))
    repEdges = np.zeros((numPoints*(numPoints-1),2))
    repEdgeWeights = np.zeros((numPoints*(numPoints-1),1))

    count = 0

    for k in range(numPoints):
        for l in range(k+1,numPoints):
            dist = np.linalg.norm(dataP[k][:]-dataP[l][:],2)

            distMatrix[k][l] = dist
            distMatrix[l][k] = dist

            repEdges[count][0] = k
            repEdges[count][1] = l
            repEdgeWeights[count][0] = dist

            count += 1

            repEdges[count][0] = l
            repEdges[count][1] = k
            repEdgeWeights[count][0] = dist

            count += 1

    totalWeight, ST, XSt = kruskal(repEdges, repEdgeWeights)


    #print(distMatrix)




    return 1
