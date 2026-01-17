import numpy as np

# ==================== Activation Functions ====================

def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(a):
    """Derivative of sigmoid"""
    return a * (1 - a)

def relu(z):
    """ReLU activation function"""
    return np.maximum(0, z)

def relu_derivative(z):
    """Derivative of ReLU"""
    return (z > 0).astype(float)

def softmax(z):
    """Softmax activation function"""
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def dropout(A, keep_prob, training=True):
    """Dropout regularization"""
    if not training or keep_prob == 1.0:
        return A, None
    D = np.random.rand(*A.shape) < keep_prob
    A = A * D / keep_prob  # Inverted dropout
    return A, D

# ==================== Model Initialization ====================

def initialize_mlp_classification_model(layer_dims):
    """
    Khởi tạo model MLP với tham số ngẫu nhiên
    """
    np.random.seed(42)
    model = {}
    num_layers = len(layer_dims)
    
    for l in range(1, num_layers):
        # He initialization cho ReLU
        model[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
        model[f'b{l}'] = np.zeros((layer_dims[l], 1))
    
    model['num_layers'] = num_layers - 1
    return model

# ==================== Forward Propagation ====================

def forward_propagation(model, X, keep_prob=1.0, training=True):
    cache = {'A0': X}
    A = X
    num_layers = model['num_layers']
    
    for l in range(1, num_layers):
        W = model[f'W{l}']
        b = model[f'b{l}']
        Z = np.dot(W, A) + b
        A = relu(Z)
        A, D = dropout(A, keep_prob, training)
        cache[f'Z{l}'] = Z
        cache[f'A{l}'] = A
        cache[f'D{l}'] = D
        
    W_last = model[f'W{num_layers}']
    b_last = model[f'b{num_layers}']
    Z = np.dot(W_last, A) + b_last
    A = softmax(Z)
    cache[f'Z{num_layers}'] = Z
    cache[f'A{num_layers}'] = A
    
    return A, cache

# ==================== Loss Function ====================

def cross_entropy_loss(output, label, model=None, lambda_reg=0.0):
    batch_size = label.shape[1]
    epsilon = 1e-15
    output_clipped = np.clip(output, epsilon, 1 - epsilon)
    ce_loss = -np.sum(label * np.log(output_clipped)) / batch_size
    
    l2_loss = 0
    if model is not None and lambda_reg > 0:
        for l in range(1, model['num_layers'] + 1):
            l2_loss += np.sum(model[f'W{l}'] ** 2)
        l2_loss = (lambda_reg / 2) * l2_loss
        
    return ce_loss + l2_loss

# ==================== Backward Propagation ====================

def backward_propagation(model, cache, X, label, keep_prob=1.0):
    gradients = {}
    num_layers = model['num_layers']
    batch_size = X.shape[1]
    
    dZ = cache[f'A{num_layers}'] - label
    
    for l in range(num_layers, 0, -1):
        A_prev = cache[f'A{l-1}']
        gradients[f'dW{l}'] = np.dot(dZ, A_prev.T) / batch_size
        gradients[f'db{l}'] = np.sum(dZ, axis=1, keepdims=True) / batch_size
        
        if l > 1:
            dA_prev = np.dot(model[f'W{l}'].T, dZ)
            if cache.get(f'D{l-1}') is not None:
                dA_prev = dA_prev * cache[f'D{l-1}'] / keep_prob
            dZ = dA_prev * relu_derivative(cache[f'Z{l-1}'])
            
    return gradients

def calculate_gradient(model, cache, X, label, lambda_reg=0.0, keep_prob=1.0):
    gradients = backward_propagation(model, cache, X, label, keep_prob)
    if lambda_reg > 0:
        for l in range(1, model['num_layers'] + 1):
            gradients[f'dW{l}'] += lambda_reg * model[f'W{l}']
    return gradients

def update_parameters(model, gradients, learning_rate):
    for l in range(1, model['num_layers'] + 1):
        model[f'W{l}'] -= learning_rate * gradients[f'dW{l}']
        model[f'b{l}'] -= learning_rate * gradients[f'db{l}']
    return model
