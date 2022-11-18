import numpy as np




# parameter inicialization
def parameters_init(dims_layers):
    
    parameters = {}
    
    comp = len(dims_layers)

    for i in range(1, comp):
        
        # weights
        parameters["W" + str(i)] = np.random.randn(dims_layers[i], dims_layers[i - 1]) * 0.01
        
        # bias
        parameters["b" + str(i)] = np.zeros((dims_layers[i], 1))
    
    return parameters




# Sigmoid function
def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A, Z




# ReLu (Rectified Linear Unit)
def relu(Z):
    A = abs(Z * (Z > 0))
    return A, Z





# Activation
# A is input layer
# W is weights
# b is bias
def linear_activation(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache





# Forward 
def forward(A_prev, W, b, activation):
    
    if activation == "sigmoid":
        Z, linear_cache = linear_activation(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
        
    elif activation == "relu":
        Z, linear_cache = linear_activation(A_prev, W, b)
        A, activation_cache = relu(Z)
        
    cache = (linear_cache, activation_cache)
    
    return A, cache





# Forward Propragation
def forward_propagation(X, parameters):
    
    caches = []

    A = X
    
    L = len(parameters) // 2

    for i in range(1, L):
      
        A_prev = A
        
        A, cache = forward(A_prev, parameters["W" + str(i)], parameters["b" + str(i)], activation = "relu")
        
        caches.append(cache)
    
    # Output layer
    A_last, cache = forward(A, parameters["W" + str(L)], parameters["b" + str(L)], activation = "sigmoid")
    
    caches.append(cache)
    
    return(A_last, caches)





# Cross-entropy cost function
def cross_entropy(A_last, Y):
    
    m = Y.shape[1]
    
    cost = (-1 / m) * np.sum((Y * np.log(A_last)) + ((1 - Y) * np.log(1 - A_last)))
    
    cost = np.squeeze(cost)
    
    return(cost)




# Sigmoid Derivative - Backpropagation
def sigmoid_backward(da, Z):
    
    dg = (1 / (1 + np.exp(-Z))) * (1 - (1 / (1 + np.exp(-Z))))
    
    dz = da * dg
    return dz



# Relu Derivative - Backprogagation
def relu_backward(da, Z):
    
    dg = 1 * ( Z >= 0)
    dz = da * dg
    return dz




# Linear Backward Function
def linear_backward_function(dz, cache):
    
    A_prev, W, b = cache
    
    m = A_prev.shape[1]
    
    dW = (1 / m) * np.dot(dz, A_prev.T)
    
    db = (1 / m) * np.sum(dz, axis = 1, keepdims = True)
    
    dA_prev = np.dot(W.T, dz)
    
    return dA_prev, dW, db




# Linear Backward Activation
def linear_activation_backward(dA, cache, activation):
    
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward_function(dZ, linear_cache)

    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward_function(dZ, linear_cache)
        
    return dA_prev, dW, db





# Backpropagation Algorithm
# AL = predicted output
# Y = real value
def backward_propagation(AL, Y, caches):
    
    grads = {}
    
    L = len(caches)

    m = AL.shape[1]

    Y = Y.reshape(AL.shape)

    dAL = -((Y / AL) - ((1 - Y) / (1 - AL)))

    current_cache = caches[L - 1]
    
    # gradients
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")

    for l in reversed(range(L - 1)):

        current_cache = caches[l]

        dA_prev, dW, db = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")

        grads["dA" + str(l)] = dA_prev
        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db
        
    return grads





# Updating weights and biass
def weight_update(parameters, grads, learning_rate):
    
    L = len(parameters)//2

    for l in range(L):

        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - (learning_rate * grads["dW" + str(l + 1)])

        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - (learning_rate * grads["db" + str(l + 1)])
    
    return parameters




# Neural Network
def NNmodel(X, Y, dims_layers, learning_rate = 0.0075, num_iterations = 100):
    
    costs = []
    
    parameters = parameters_init(dims_layers)
    
    for i in range(num_iterations):
        
        # Forward Propagation
        AL, caches = forward_propagation(X, parameters)
        
        # Cost function
        cost = cross_entropy(AL, Y)
        
        # Backward Propagation
        gradients = backward_propagation(AL, Y, caches)
        
        # Updating weights
        parameters = weight_update(parameters, gradients, learning_rate)
        
        # Print for control
        if i % 10 == 0:
            print("Cost after " + str(i) + " iterations is " + str(cost))
            costs.append(cost)
            
    return parameters, costs 



# Prediction Function

def predict(X, parameters):
    AL, caches = forward_propagation(X, parameters)
    return AL

