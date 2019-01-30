from fetch_training_data import *
from initial_variables import *
from initialize_weights import *
from filter_dataset import *
from fetch_test import *
import math

# filter_data()

# pattern = fetch_data()
# (
#     MAX_CLUSTERS,
#     VEC_LEN,
#     INPUT_PATTERNS,
#     INPUT_TESTS,
#     MIN_ALPHA,
#     MAX_ITERATIONS,
#     SIGMA,
#     INITIAL_LEARNING_RATE,
#     INITIAL_RADIUS
# ) = initialize_variables(pattern)


# w = random_weights(pattern,MAX_CLUSTERS,VEC_LEN)
# tests = fetch_tests()
# print(len(tests[3]))
# print(MAX_CLUSTERS,VEC_LEN,INPUT_PATTERNS)
# print(training_labels)

class SOM_Class:
    def __init__(self, vectorLength, maxClusters, numPatterns, numTests, minimumAlpha, weightArray, maxIterations,
                 sigma, initialAlpha, initialSigma):
        """
        :param vectorLength:
        :param maxClusters:
        :param numPatterns:
        :param numTests:
        :param minimumAlpha:
        :param weightArray:
        :param maxIterations:
        :param sigma:
        :param initialAlpha:
        :param initialSigma:
        """
        self.mVectorLen = vectorLength
        self.mMaxClusters = maxClusters
        self.mNumPatterns = numPatterns
        self.mNumTests = numTests
        self.mMinAlpha = minimumAlpha
        self.mAlpha = initialAlpha
        self.d = []
        self.w = weightArray
        self.maxIterations = maxIterations
        self.sigma = SIGMA
        self.mInitialAlpha = initialAlpha
        self.mInitialSigma = sigma
        return




    def compute_input(self, vectorArray, vectorNumber):
        """
        :param vectorArray:
        :param vectorNumber:
        :return:
        """
        # print(len(w),len(vectorArray))
        self.d = [0.0] * self.mMaxClusters
        # print(self.mMaxClusters,self.mVectorLen)
        for i in range(self.mMaxClusters):
            for j in range(self.mVectorLen):
                self.d[i] = self.d[i] + math.pow((self.w[i][j] - vectorArray[vectorNumber][j]), 2)
            self.d[i]=math.sqrt(self.d[i])
                # print(vectorNumber,i,j)
                # print(self.d)
                # print(self.d)


        # print(self.d)
        return



    def get_minimum(self, nodeArray):  # NodeArray holding the distances of selected instance with each node
        minimum = 0
        foundNewMinimum = False
        done = False

        while not done:
            foundNewMinimum = False
            for i in range(self.mMaxClusters):
                if i != minimum:
                    if nodeArray[i] < nodeArray[minimum]:
                        minimum = i
                        foundNewMinimum = True

            if foundNewMinimum == False:
                done = True

        return minimum

    def update_weights(self, vectorNumber, dMin, patternArray):

        # Adjust weight of winning neuron
        for l in range(self.mVectorLen):
            self.w[dMin][l] = self.w[dMin][l] + (
                    self.mAlpha * 1 * (patternArray[vectorNumber][l] - self.w[dMin][l]))
        # Now search for neighbors
        dis = 0.00
        # print('MAX :' + str(self.mMaxClusters))
        for i in range(self.mMaxClusters):
            for j in range(self.mVectorLen):
                if (i != dMin):
                    dis = dis + math.pow((self.w[dMin][j] - self.w[i][j]), 2)
                else:
                    continue
            dis = math.sqrt(dis)
            # Consider as neighbor if distance is less than sigma

            if (dis < self.sigma):
                # Neighborhood function
                h = math.exp(-1 * (pow(dis, 2)) / (2 * (self.sigma ** 2)))

                # once accepted as neighbor update its weight
                for x in range(self.mVectorLen):
                    self.w[i][x] = self.w[i][x] + (self.mAlpha * h * (patternArray[vectorNumber][x] - self.w[i][x]))

        return

    def training(self, patternArray):
        print("Entered training")
        iterations = 0
        while (iterations != self.maxIterations):
            iterations = iterations + 1
            for i in range(self.mNumPatterns):
                self.compute_input(patternArray, i)
                dMin = self.get_minimum(self.d)
                self.update_weights(i, dMin, patternArray)

            if self.mAlpha > 0.01:
                self.mAlpha = self.mInitialAlpha /((1 + (iterations / self.maxIterations)))
            print("Learning Rate:"+str(self.mAlpha))

            if self.sigma > -1*(self.mInitialSigma):
                self.sigma = self.mInitialSigma / ((1 + (iterations / self.maxIterations)))
            print("Radius :" + str(self.sigma))
            print("Iterations" + str(iterations) + '\n')


        return

    def classify(self, tests, map_dict , train_lab_dict):
        threshold = 0.5
        sum_dict = dict()
        prototypeVector = list()
        numInstances = dict()  # Dictionary to hold number of instances mapped to the neuron
        for key in map_dict:
            c = 0
            for x in map_dict[key]:
                c = c + 1
            numInstances[key] = c
        print(numInstances)

        # Calculating averages of labels mapped to each neuron
        for key in train_lab_dict:
            sum_list = [sum(x) / numInstances[key] for x in zip(*train_lab_dict[key])]
            if key not in sum_dict:
                sum_dict[key] = []
                sum_dict[key].append(sum_list)
            else:
                sum_dict[key].append(sum_list)

        print(sum_dict)
        v = list()  # v vector
        for i in range(self.mMaxClusters):
            v.append([0.0] * len(training_labels[0]))
        print(v)
        # for deterministic prediction , set threshold = 0.5: if > 0.5 then 1 else 0
        for key in sum_dict:
            numNeuron = key
            for x in sum_dict[key]:
                v[numNeuron] = x
        print(v)

        for i in range(len(v)):
            for j in range(len(training_labels[0])):
                if (v[i][j] >= threshold):
                    v[i][j] = 1
                else:
                    v[i][j] = 0
        print(v)
        return v
        # In v vector the vector at position say(x) represents the threshold calculated average of labels of instances for that neuron number

        # pick up test instances
        # print("Classification results :")
        # for i in tests:  # i is test instance
        #     matchFlag = 0
        #     for x in v:
        #         if i == x:
        #             matchFlag = 1
        #             ind = v.index(x)  # ind is the neuron number to which the test should n=be mapped
        #             print(i, end=' ')
        #             print(": falls under category" + str(ind))
        #
        #     if (matchFlag == 0):
        #         print("Test Instance", end=' ')
        #         print(i, end=" ")
        #         print("has no match")

    def print_results(self, patternArray, testArray):
        # Printing the clusters created

        map_dict = dict()  # dictn to hold mapped vwctors along with respective neurons
        train_lab_dict = dict()
        print("Clusters for training input: \n")
        for i in range(self.mNumPatterns):
            map_list = list()  # list to hold mapped instances to a particular neuron
            train_lab_list = list()
            self.compute_input(patternArray, i)
            dMin = self.get_minimum(self.d)

            print("Vector (")
            for j in range(self.mVectorLen):
                map_list.append(patternArray[i][j])
                print(str(patternArray[i][j]) + ", ")
            for k in range(len(training_labels[0])):
                train_lab_list.append(training_labels[i][k])
            if dMin not in train_lab_dict:
                train_lab_dict[dMin] = []
                train_lab_dict[dMin].append(train_lab_list)
            else:
                train_lab_dict[dMin].append(train_lab_list)

            print(") fits into category " + str(dMin) + "\n")
            if dMin not in map_dict:
                map_dict[dMin] = []
                map_dict[dMin].append(map_list)
            else:
                map_dict[dMin].append(map_list)

        # Print weight matrix.

        print("------------------------------------------------------------------------\n")
        for i in range(self.mMaxClusters):
            print("Weights for Node " + str(i) + " connections:\n")
            print("     ")
            for j in range(self.mVectorLen):
                print("{:03.3f}".format(self.w[i][j]) + ", ")

            print("\n")

        print("Dictionary - Maping of instances to respective neurons :")
        print(map_dict)

        return map_dict, train_lab_dict

    def post_train(self,testArray,v):
        # Print post-training tests.
        post_test_labels = [[] for i in range(self.mNumTests)]
        print("------------------------------------------------------------------------\n")
        print("Categorized test input:\n")
        for i in range(self.mNumTests):
            self.compute_input(testArray, i)

            dMin = self.get_minimum(self.d)
            if dMin is not None:
                post_test_labels[i]=v[dMin]
            else:
                post_test_labels[i]=[]
            print("Vector (")
            for j in range(self.mVectorLen):
                print(str(testArray[i][j]) + ", ")

            print(") fits into category " + str(dMin) + "\n")
        return post_test_labels

    def evaluate(self, predicted_labels, testing_labels):
        p = 0
        #Testing precision
        for x in range(len(predicted_labels)):
            if predicted_labels[x] == testing_labels[x]:
                p+=1


        return float(float(p)/len(predicted_labels))




if __name__ == '__main__':
    # filter_data()
    pattern = fetch_data()
    (
        MAX_CLUSTERS,
        VEC_LEN,
        INPUT_PATTERNS,
        INPUT_TESTS,
        MIN_ALPHA,
        MAX_ITERATIONS,
        SIGMA,
        INITIAL_LEARNING_RATE,
        INITIAL_RADIUS
    ) = initialize_variables(pattern)
    w = random_weights(pattern, MAX_CLUSTERS, VEC_LEN)
    tests = fetch_tests()
    som = SOM_Class(VEC_LEN, MAX_CLUSTERS, INPUT_PATTERNS, INPUT_TESTS, MIN_ALPHA, w, MAX_ITERATIONS, SIGMA,
                     INITIAL_LEARNING_RATE, INITIAL_RADIUS)
    som.training(pattern)
    map_dict, train_lab_dict = som.print_results(pattern, tests)
    v = som.classify(tests, map_dict , train_lab_dict)
    predicted_labels = som.post_train(tests,v)
    print("Predicted labels : " + str(len(predicted_labels)))
    print(predicted_labels)

    for x in range(len(predicted_labels)):
        for y in range(LABEL_COUNT):
            predicted_labels[x][y] = float(predicted_labels[x][y])
    print("Predicted labels : " + str(len(predicted_labels)))
    print(predicted_labels)
    print("Testing actual labels :" + str(len(testing_labels)))
    print(testing_labels)
    precision = som.evaluate(predicted_labels,testing_labels)
    print(precision)




























































































