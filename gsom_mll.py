import csv
import random
import math
LABEL_COUNT = int(input("Enter the label count : "))
truncate_value = -1*(LABEL_COUNT)
training_set = input("Enter the complete data filename to be used for training GSOM :")
test_set = input("Enter the complete data filename to be used for testing GSOM :")
new_training_set = input("Enter the complete data filename to be made for training after filtering :")
new_testing_set = input("Enter the complete data filename to be made for testing after filtering :")
train_label = list()
new_train = list()
training_labels = list()
testing_labels = list()
new_test = list()
w = list()

def initialize_variables(pattern):
    MAX_CLUSTERS = 64
    VEC_LEN = len(random.choice(pattern))
    INPUT_PATTERNS = len(pattern)
    INPUT_TESTS = len(testing_labels)
    MIN_ALPHA = 0.01
    MAX_ITERATIONS = 500
    SIGMA = 2.0
    INITIAL_LEARNING_RATE = 0.10
    INITIAL_RADIUS = SIGMA
    
    return (
        MAX_CLUSTERS,
        VEC_LEN,
        INPUT_PATTERNS,
        INPUT_TESTS,
        MIN_ALPHA,
        MAX_ITERATIONS,
        SIGMA,
        INITIAL_LEARNING_RATE,
        INITIAL_RADIUS
    )

def random_weights(patterns,max_clusters,vector_length):
    flag = 0
    while(flag<max_clusters):
        random_instance = random.choice(patterns)
        if random_instance not in w:
            flag = flag + 1
            w.append(random_instance)
        else:
            continue

    return w


def filter_data():
    """
    This functions filters the dataset and separates labels and data (actual) for both
    training and testing into different files and labels are hold in different lists
    :return:
    """
    with open(training_set,"r+") as t:
        csv_reader = csv.reader(t)
        for line in csv_reader:
            train_label = [float(x) for x in line[truncate_value:]]
            data = line[:truncate_value]
            new_train.append(data)
            training_labels.append(train_label)

    with open(new_training_set,"w") as w1:
        csv_writer = csv.writer(w1)
        for line in new_train:
            line = [float(x) for x in line]
            csv_writer.writerow(line)

    with open(test_set,'r+') as t1:
        csv_reader1 = csv.reader(t1)
        for line in csv_reader1:
            test_label = [float(x) for x in line[truncate_value:]]
            test = line[:truncate_value]
            new_test.append(test)
            testing_labels.append(test_label)
    with open(new_testing_set,"w") as w2:
        csv_writer1 = csv.writer(w2)
        for y in  new_test:
            y = [float(k) for k in y]
            csv_writer1.writerow(y)

def fetch_data():
    '''
    This function fetches the data for training (instances)
    :return:
    '''
    filename = new_training_set
    patterns = list()

    with open(filename,"r+") as r:
        csv_reader3 = csv.reader(r)
        for line in csv_reader3:
            patterns.append([float(k) for k in line])

    patterns = [e for e in patterns if e]

    return patterns


class GSOM:
    def __init__(self, vectorLength, maxClusters, numPatterns, numTests, minimumAlpha, weightArray, maxIterations,
                 sigma, initialAlpha, initialSigma):
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

    def get_minimum(self, nodeArray):
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


    def compute_input(self, vectorArray, vectorNumber):
        self.d = [0.0] * self.mMaxClusters
        for i in range(self.mMaxClusters):
            for j in range(self.mVectorLen):
                self.d[i] = self.d[i] + math.pow((self.w[i][j] - vectorArray[vectorNumber][j]), 2)
            self.d[i]=math.sqrt(self.d[i])
        return


    def update_weights1(self, vectorNumber, dMin, patternArray):

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
        while(iterations != 1):
            iterations = iterations + 1
            for i in range(self.mNumPatterns):
                self.compute_input(patternArray, i)
                dMin = self.get_minimum(self.d)
                self.update_weights1(i, dMin, patternArray)

        if self.mAlpha > 0.01:
            self.mAlpha = self.mInitialAlpha*math.exp(-1*(iterations)/self.maxIterations)
        print("Learning Rate:" + str(self.mAlpha))
        t1 = self.maxIterations/math.log(self.mInitialSigma,10)
        self.sigma = self.mInitialSigma*math.exp(-1*(iterations)/t1)
        print("Radius :" + str(self.sigma))
        print("Iterations" + str(iterations) + '\n')


if __name__=='__main__':
    filter_data()
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
    #fetch the tests data here

    gsom = GSOM(VEC_LEN, MAX_CLUSTERS, INPUT_PATTERNS, INPUT_TESTS, MIN_ALPHA, w, MAX_ITERATIONS, SIGMA,
                    INITIAL_LEARNING_RATE, INITIAL_RADIUS)

    gsom.training(pattern)
