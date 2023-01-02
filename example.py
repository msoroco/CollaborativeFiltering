from misc import apply
import regularizedLeastSquares
import LeastSquares
from generateData import Dataloader
import semidef_prog
import numpy as np

import xlwt
from time import gmtime, strftime

book = xlwt.Workbook()
# t = strftime("%Y-%m-%d__%H_%M_%S", gmtime())
sh = book.add_sheet("results")

col1_name = 'Least Squares'
col1_1_name = 'fixed k'

col2_name = 'Least Squares'
col2_1_name = 'best k'

col3_name = 'Least Squares Bias'
col3_1_name = 'best k'

col4_name = 'Regularized Least Squares'
col4_1_name = 'fixed k'
col4_2_name = 'fixed lambda'

col5_name = 'lambda Tuned Regularized Least Squares'
col5_1_name = 'k'
col5_2_name = 'best lambda'

col6_name = 'All Tuned Regularized Least Squares'
col6_1_name = 'best_k'
col6_2_name = 'best lambda'

col7_name = 'All Tuned Biased Regularized Least Squares'
col7_1_name = 'best k'
col7_2_name = 'best lambda'

col8_name = 'Nuclear Norm'

base_row = 5
row2 = sh.row(base_row)
row2.write(1, col1_name)
row2.write(2, col1_1_name)
row2.write(3, col2_name)
row2.write(4, col2_1_name)
row2.write(5, col3_name)
row2.write(6, col3_1_name)
row2.write(7, col4_name)
row2.write(8, col4_1_name)
row2.write(9, col4_2_name)
row2.write(10, col5_name)
row2.write(11, col5_1_name)
row2.write(12, col5_2_name)
row2.write(13, col6_name)
row2.write(14, col6_1_name)
row2.write(15, col6_2_name)
row2.write(16, col7_name)
row2.write(17, col7_1_name)
row2.write(18, col7_2_name)
row2.write(19, col8_name)

nIterations = 40
for i in range(base_row + 1, base_row + 1 + nIterations):

    row = sh.row(i)
    ##########################

    n = 50
    d = 30

    k_ans = 5       # <---- this should NOT be used in any way except to create Y

    dl = Dataloader(n, d, k_ans)
    Xtrain, Xtest, Xvalidate = dl.Xtrain, dl.Xtest, dl.Xvalidate

    train_indices = dl.get_train_sample_indices() # tells us which entries in Xtrain are filled
    validation_indices = dl.get_validation_sample_indices() # tells us which entries in Xvalidate are filled
    test_indices = dl.get_test_sample_indices() # tells us which entries in Xtest are filled

    # print(dl.get_maxRating())   # largest rating defined by dataset (not necessarily a 1 - 10 scale)
    # print(dl.get_minRating())   # largest rating defined by dataset (not necessarily a 1 - 10 scale)

    # print(dl.Y)             # solution matrix

    # print(Xtest)         # for testing final performance
    # print(Xvalidate)     # for fitting lambda
    # print(Xtrain)        # for finding W and Z


    ##########################
    ### Least Squares

    k = 3   # we need to hand pick what is the best k 
            # (pretend that you don't know k_ans, since with real data we don't know k_ans)

    W, Z = LeastSquares.LeastSquares(Xtrain, k, train_indices)

    m = apply(Z@W, Xtrain, train_indices)
    trainingError = np.mean(((Z@W - m).flatten() )**2)
    print("Training error w/ least squres, k = {0:2d}  is: {1:8.2f}".format(k, trainingError))


    m = apply(Z@W, Xtest, test_indices)
    testError = np.mean(((Z@W - m).flatten() )**2)
    print("Test error w/ least squres, k = {0:2d}  is: {1:8.2f}".format(k, testError))


    row.write(1, testError)
    row.write(2, k)

    ###################################
    ### Least Squares while training k:



    min_validationError = np.inf
    best_k = 1
    for k in range(3, 8, 1):       # [start, stop) with step size <- range(start, stop, step)

        W, Z = LeastSquares.LeastSquares(Xtrain, k, train_indices)          # train model with this k

        m = apply(Z@W, Xvalidate, validation_indices)                           
        validationError = np.mean(((Z@W - m).flatten() )**2)     # use validation error to evaluate performance of k
        # print("Validation error w/ least squares, k = " + str(k) + " is: " + str(validationError))

        # if validation error improves, keep this k
        if validationError < min_validationError:
            min_validationError = validationError
            best_k = k
            

    # final test of performance
    W, Z = LeastSquares.LeastSquares(Xtrain, best_k, train_indices)          # train model with best k
    m = apply(Z@W, Xtest, test_indices)
    testError = np.mean(((Z@W - m).flatten() )**2)
    print("Test error w/ least squres, best k = {0:2d}  is: {1:8.2f}".format(best_k,testError))

    row.write(3, testError)
    row.write(4, best_k)

    ###################################
    ### Least Squares with bias while training k:



    min_validationError = np.inf
    best_k = 3
    for k in range(3, 8, 1):       # [start, stop) with step size <- range(start, stop, step)

        W, Z, b = LeastSquares.LeastSquares_bias(Xtrain, k, train_indices)          # train model with this k

        predictions = Z@W + b
        m = apply(predictions, Xvalidate, validation_indices)                           
        validationError = np.mean(((predictions - m).flatten() )**2)                     # use validation error to evaluate performance of k
        # print("Validation error w/ biased least squres, k = " + str(k) + " is: " + str(validationError))

        # if validation error improves, keep this k
        if validationError < min_validationError:
            min_validationError = validationError
            best_k = k
            

    # final test of performance
    W, Z, b = LeastSquares.LeastSquares_bias(Xtrain, best_k, train_indices) # train with best k
    predictions = Z@W + b
    m = apply(predictions, Xtest, test_indices)
    testError = np.mean(((predictions - m).flatten() )**2)  # make predictions on unseen data and compute error
    print("Test error w/ biased least squres, best k = {0:2d}  is: {1:8.2f}\n".format(best_k,testError))

    row.write(5, testError)
    row.write(6, best_k)

    #############################
    ### Regularized Least Squares, while training lambda

    k = 5
    lambda1 = 1
    W, Z = regularizedLeastSquares.Reg_LeastSquares(Xtrain, k, lambda1, train_indices)

    m = apply(Z@W, Xtrain, train_indices)
    trainingError = np.mean(((Z@W - m).flatten() )**2)
    print("training error w/ regularized least squres, k = {0:2d}, lambda = {1:2d} is: {2:8.2f}".format(k, lambda1, trainingError))

    # early test of performance
    m = apply(Z@W, Xtest, test_indices)
    testError = np.mean(((Z@W - m).flatten() )**2)
    print("Test error w/ regularized least squres, k = {0:2d}, lambda = {1:2d} is: {2:8.2f}".format(k, lambda1, testError))

    row.write(7, testError)
    row.write(8, k)
    row.write(9, lambda1)

    min_validationError = np.inf
    best_lambda = lambda1
    for lambda1 in range(1, 10, 1):       # [start, stop) with step size <- range(start, stop, step)
        W, Z = regularizedLeastSquares.Reg_LeastSquares(Xtrain, k, lambda1, train_indices)          # train model with this lambda

        m = apply(Z@W, Xvalidate, validation_indices)                           
        validationError = np.mean(((Z@W - m).flatten() )**2)                                    # use validation error to evaluate performance of lambda
        # print("validation error w/ regularized least squres, k = {0:2d} lambda = {1:2d} is: {2:8.2f}".format(k, lambda1, validationError))
        # if validation error improves, keep this lambda
        if validationError < min_validationError:
            min_validationError = validationError
            best_lambda = lambda1

    # to compare training error after picking a better lambda  
    W, Z = regularizedLeastSquares.Reg_LeastSquares(Xtrain, k, best_lambda, train_indices)
    m = apply(Z@W, Xtrain, train_indices)
    trainingError = np.mean(((Z@W - m).flatten() )**2)
    print("training error w/ regularized least squres, k = {0:2d}, best lambda = {1:2d} is: {2:8.2f}".format(k, best_lambda, trainingError))

    # final test of performance
    m = apply(Z@W, Xtest, test_indices)
    testError = np.mean(((Z@W - m).flatten() )**2)
    print("Test error w/ regularized least squres, k = {0:2d}, best lambda = {1:2d} is: {2:8.2f}\n".format(k, best_lambda, testError))


    row.write(10, testError)
    row.write(11, k)
    row.write(12, best_lambda)


    ###################################
    ### Regularized Least Squares while training k and lambda :



    min_validationError = np.inf
    best_lambda = 1
    best_k = 3
    for lambda1 in range(1, 10, 1):       # [start, stop) with step size <- range(start, stop, step)
        for k in range(3, 8, 1):       # [start, stop) with step size <- range(start, stop, step)

            W, Z = regularizedLeastSquares.Reg_LeastSquares(Xtrain, k, lambda1, train_indices)          # train model with this lambda & k

            m = apply(Z@W, Xvalidate, validation_indices)                           
            validationError = np.mean(((Z@W - m).flatten() )**2)                                    # use validation error to evaluate performance of lambda
            # print("Validation error w/ regularized L2 error, k = " + str(k) + "  lambda = " + str(lambda1) + " is: " + str(validationError))

            # if validation error improves, keep this lambda
            if validationError < min_validationError:
                min_validationError = validationError
                best_lambda = lambda1
                best_k = k
            

    # final test of performance
    W, Z = regularizedLeastSquares.Reg_LeastSquares(Xtrain, best_k, best_lambda, train_indices)  # train model with best lambda & k
    m = apply(Z@W, Xtest, test_indices)
    testError = np.mean(((Z@W - m).flatten() )**2)
    print("Test error w/ regularized least squres, best k = {0:2d}, best lambda = {1:2d} is: {2:8.2f}".format(best_k, best_lambda, testError))

    row.write(13, testError)
    row.write(14, best_k)
    row.write(15, best_lambda)

    ###################################
    ### Regularized Least Squares with bias while training k and lambda :



    min_validationError = np.inf
    best_lambda = 1
    best_k = 3
    for lambda1 in range(1, 10, 1):       # [start, stop) with step size <- range(start, stop, step)
        for k in range(3, 8, 1):       # [start, stop) with step size <- range(start, stop, step)

            W, Z, b = regularizedLeastSquares.Reg_LeastSquares_bias(Xtrain, k, lambda1, train_indices)          # train model with this lambda & k

            m = apply(Z@W + b, Xvalidate, validation_indices)                           
            validationError = np.mean(((Z@W + b - m).flatten() )**2)                                    # use validation error to evaluate performance of lambda
            # print("Validation error w/ regularized L2 error, k = " + str(k) + "  lambda = " + str(lambda1) + " is: " + str(validationError))

            # if validation error improves, keep this lambda
            if validationError < min_validationError:
                min_validationError = validationError
                best_lambda = lambda1
                best_k = k
            

    # final test of performance
    W, Z, b = regularizedLeastSquares.Reg_LeastSquares_bias(Xtrain, best_k, best_lambda, train_indices)  # train model with best lambda & k
    m = apply(Z@W + b, Xtest, test_indices)
    testError = np.mean(((Z@W + b - m).flatten() )**2)
    print("Test error w/ regularized biased least squres, best k = {0:2d}, best lambda = {1:2d} is: {2:8.2f}".format(best_k, best_lambda, testError))

    row.write(16, testError)
    row.write(17, best_k)
    row.write(18, best_lambda)

    # print(Xtrain)
    # print((Z@W + b).round(1))


    ################
    ## Matrix completion via semi definite programming

    Y_model = semidef_prog.minimize_trace(Xtrain, train_indices)

    m = apply(Y_model, Xtest, test_indices)
    testError = np.mean(((Y_model - m).flatten() )**2)
    print("Test error from min nuclear norm is: {0:8.2f}".format(testError))

    row.write(19, testError)


sh.write(1,1,'n')
sh.write(1,2,'d')
sh.write(1,3,'k_ans')
sh.write(2,1,n)
sh.write(2,2,d)
sh.write(2,3, k_ans)

t = strftime("%Y-%m-%d__%H_%M_%S", gmtime())
book.save('results' + str(t) + '.xls')
