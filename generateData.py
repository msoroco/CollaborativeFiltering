import numpy as np


class Dataloader():
    def __init__(self, n:int, d:int, k_ans:int, addNoise=True) -> tuple:
        self.Y, self.Xtrain, self.Xvalidate, self.Xtest = self.__generateData(addNoise, n, d, k_ans)

        self.train_indices
        self.validation_indices
        self.test_indices


    def __constructRandomMatrix(self, rows, cols):
        return np.random.randint(0, 7, (rows, cols))  # this gives values in 0 - 100 approx with k = 5

    def __addNoise(self, Y):
        (n, d) = Y.shape
        n_samples = n * d
        samples = Y.flatten()
        sd = np.std(samples)/10     # arbitrarily set noise sd to be 1/10 the datasets'
        s = np.random.randint(0, n*d, size=int(n_samples * 0.1))  # add noise to ~10% of data randomly
        for i in s:
            samples[i] = samples[i] + np.random.normal(0, sd)
        
        return np.reshape(Y, (n, d))
            
        

    def __extractToEmptyMatrix(self, rows:int, cols:int, indices:int, origMatrix):
        matrix = np.empty(rows*cols, dtype=object)
        matrix[indices] = origMatrix.flatten()[indices]
        matrix = np.reshape(matrix, (rows, cols))
        return matrix


    def __splitData(self, percent_train, percent_test, n_samples):
        # shuffle the possible indices of entries in this matrix to divide the data.
        indices = list(range(n_samples))
        np.random.shuffle(indices)
        train_split = int(np.floor(percent_train/100 * n_samples))
        test_split = int(np.floor((1 - percent_test)/100 * n_samples))

        # determine which data will be used for which matrix:
        #  example:  Xtrain (85%), Xvalidate (15%), Xtest (15%)
        self.train_indices = indices[:train_split]
        self.validation_indices = indices[train_split:test_split]
        self.test_indices = indices[test_split:]


    def __generateData(self, addNoise, n = 10, d = 10, k_ans = 5):
        # generate random A and B to construct matrix Y of solutions
        A = self.__constructRandomMatrix(n, k_ans)
        B = self.__constructRandomMatrix(k_ans, d)

        Y = A@B

        if addNoise:
            Y = self.__addNoise(Y)

        # hide a random subset of values (test set)
        # hide a random subset of values (validation set)
        # remaining values are the training set
        # split the data as 70 % - 15 % - 15 % for training, validation, test respectively
        self.__splitData(60, 20, n*d)

        Xtrain = self.__extractToEmptyMatrix(n, d, self.train_indices, Y)
        Xvalidate = self.__extractToEmptyMatrix(n, d, self.validation_indices, Y)
        Xtest = self.__extractToEmptyMatrix(n, d, self.test_indices, Y)

        return Y, Xtrain, Xvalidate, Xtest


    def get_maxRating(self):
        return max(self.Y.flatten())

    def get_minRating(self):
        return min(self.Y.flatten())

    def get_train_sample_indices(self):
        return self.train_indices

    def get_validation_sample_indices(self):
        return self.validation_indices

    def get_test_sample_indices(self):
        return self.test_indices





# # examples of use:

# p1 = Dataloader(10, 10, 5)

# print(p1.get_maxRating())

# print(p1.Y)
# print(p1.Xtest)
# print(p1.Xvalidate)
# print(p1.Xtrain)
