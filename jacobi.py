import numpy as np
# Returns normalized A and b so that |D| < 1
def normalize(A, b):
    normalizedMatrix = []
    normalizedVector = []
    for idx, row in enumerate(A):
        normalizedMatrix.append(row/np.max(row))
        normalizedVector.append(b[idx]/np.max(row))
    return np.array(normalizedMatrix), np.array(normalizedVector)


# performs jacobi
# A is the input matrix, b is the solution vector,
# x is the initial input, epsilon is the maximum mean squared error
def jacobi(A, b, x, epsilon = 0.01, return_iterations = False):
    normalizedA,normalizedb = normalize(A,b)
    I = np.identity(A.shape[0])
    # Finds D with normalized A so that |D| < 1
    D = I - normalizedA
    # Computes target solution
    x_real = np.linalg.solve(A, b)
    num_iterations = 0
    # Iterates until the mean squared error is more than epsilon
    while (np.square(x_real - x)).mean()> epsilon:
        num_iterations+=1
        x = D.dot(x) + normalizedb
    if return_iterations:
        return x, num_iterations
    else:
        return x


# Put input variables here
A = np.array([[7,1,2],[1,5,3],[2,3,8]])
b = np.array([30,10,12])

# Driver for part b)
x = np.array([0,0,0])
x, num_iterations = jacobi(A, b, x, epsilon = 0.0001, return_iterations = True)
print("PART b): x = ", x, f" in {num_iterations} iterations")

# Driver for part c)
x = [100,100,100]
x, num_iterations = jacobi(A, b, x, epsilon = 0.0001, return_iterations = True)
print("PART c): x = ", x, f" in {num_iterations} iterations")