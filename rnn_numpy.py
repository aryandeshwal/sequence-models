def prediction(x):
    T = len(x)
    # TODO reference HIDDEN_DIM somewhere
    h = np.zeros((T + 1, HIDDEN_DIM)) # hidden states
    h[-1] = np.zeros(HIDDEN_DIM)

    # TODO reference WORD_DIM somewhere
    preds = np.zeros((T, WORD_DIM))
    for t in range(T):
        h[t] = np.tanh(np.dot(U, x) + np.dot(W, h[t-1]))
        pred[t] = np.softmax(np.dot(V, h[t]))
    # return best output(max prob at each time step), hidden states, full output
    return np.argmax(preds, axis=1), h, preds

# Backprop through time in numpy
# motivated from WILDML page(https://bit.ly/1R2jmNz)
def bptt(x, y):
    T = len(y) # length of the sequence
    # TODO a prediction function
    y_pred, h, _ = prediction(x)

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
