# Backprop through time in numpy
# motivated from WILDML page(https://bit.ly/1R2jmNz)
def bptt(x, y):
    T = len(y) # length of the sequence
    # TODO a prediction function
    y_pred, h = prediction(x)

    grads = {}
    grads['U'] = np.zeros(U.shape)
    grads['V'] = np.zeros(V.shape)
    grads['W'] = np.zeros(W.shape)

    delta_pred = pred
    delta_pred[np.arange(T), y] -= 1
    for t in range(T, 0, -1):
        grads['V'] += np.outer(delta_pred, h[t].T) # add the contribution of each output
        delta_t = np.dot(V.T, delta_pred[t]) * (1 - (h[t] ** 2))
        # Full backprop till the beginning 
        for i in range(0, t+1):
           grads['W'] += np.outer(delta_t, h[i-1])
           grads['U'][:, x[i]] += delta_t

           delta_t = np.dot(W.T, delta_t) * (1-(h[i-1] ** 2))
    
    return grads
