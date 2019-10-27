import numpy as np 

def mse(y,tx,w):
    """
    Calculate the Mean Square Error of the given paramaters

    Parameters
    ----------
    y : np.array
        Array of labels (N,)
    tx : np.array
        Array of the features (N,D)
    w : np.array
        The weights of the model (D,)

    Returns:
        The result of the MSE calculation
    """
    e = y- tx@w
    return (e.T@e) / (2*len(y))

def least_squares(y, tx):
    """
    Least squares regression using normal equations

    Parameters
    ----------
    y : np.array
        Array of labels (N,)
    tx : np.array
        Array of the features  (N,D)

    Returns:
        (w, loss) the last weight vector of the calculation, and the corresponding loss value (cost function).

    """
#Calculate the weight through the normal equation solution
    gram_matrix=tx.T@tx
    w= np.linalg.solve(gram_matrix, tx.T@y)
    loss=mse(y,tx,w)
    return w,loss


def ridge_regression(y, tx, lambda_):
    """
    Calculate the Ridge Regression using normal equations

    Parameters
    ----------
    y : np.array
        Array of labels (N,)
    tx : np.array
        Array of the features (N,D)
    lambda_ : np.float64
        Regularization parameter

    Returns:
        (w, loss) the last weight vector of the calculation, and the corresponding loss value (cost function).
    """
    gram_matrix=np.dot(tx.T,tx)
    lambda_prime=lambda_*2*len(y)
    w=np.linalg.solve(gram_matrix + (lambda_prime*np.identity(tx.shape[1])),tx.T@y)
    loss=mse(y,tx,w)
    return w,loss

#     aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
#     a = tx.T.dot(tx) + aI
#     b = tx.T.dot(y)
#     w= np.linalg.solve(a, b)
#     loss=mse(y,tx,w)
#     return w,loss


    """
    Compute the gradient

    Parameters
    ----------
    y : np.array
        Array of labels (N,)
    tx : np.array
        Array of the features (N,D)
    w : np.array
        The weights of the model (D,)

    Returns:
        The gradient 
    """
def compute_gradient(y, tx, w):
    e = y- np.dot(tx,w)
    return -np.dot(tx.T,e)/len(y)

    """
    Linear regression using gradient descent

    Parameters
    ----------
    y : np.array
        Array of labels (N,)
    tx : np.array
        Array of the features  (N,D)
    initial_w : np.array
        Initial random weights of the model (D,)
    max_iters: int
        The maximum number of iterations
    gamma: float
        The step size

    Returns:
        (w, loss) the last weight vector of the calculation, and the corresponding loss value (cost function).

    """
def gradient_descent(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        gradient=compute_gradient(y, tx, w)
        w=w-gamma*gradient
        loss=mse(y, tx, w)
        # print("Gradient Descent({bi}/{ti}): loss={l}, w={w1}".format(bi=n_iter, ti=max_iters - 1, l=loss, w1=w))
    return w,loss


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
        
        Parameters
    ----------
    y : np.array
        Array of labels (N,)
    tx : np.array
        Array of the features  (N,D)
    batch_size : int
        Number of samples of the batch
    num_batches: int
        The number of batches (default is 1)
    shuffle: bool
        Randomize the dataset (default is True)
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


    """
    Linear regression using stochastic gradient descent

    Parameters
    ----------
    y : np.array
        Array of labels (N,)
    tx : np.array
        Array of the features  (N,D)
    initial_w : np.array
        Initial random weights of the model (D,)
    batch_size : int
        Number of samples of the batch
    max_iters: int
        The maximum number of iterations
    gamma: float
        The step size

    Returns:
        (w, loss) the last weight vector of the calculation, and the corresponding loss value (cost function).
    """
def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):

    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size=batch_size,num_batches=1):
            gradient=compute_gradient(minibatch_y, minibatch_tx, w)
            w=w-gamma*gradient
            loss=mse(minibatch_y, minibatch_tx, w)

    return w,loss


    """
    Compute the sigmoid

    Parameters
    ----------
    t : float
        
    Returns:
        The sigmoid of t
    """
def sigmoid(t):
    return 1/(1+np.exp(-t))


    """
    Logistic regression using gradient descent

    Parameters
    ----------
    y : np.array
        Array of labels (N,)
    tx : np.array
        Array of the features  (N,D)
    initial_w : np.array
        Initial random weights of the model (D,)
    max_iters: int
        The maximum number of iterations
    gamma: float
        The step size

    Returns:
        (w, loss) the last weight vector of the calculation, and the corresponding loss value (cost function).

    """
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    # init parameters
    threshold = 1e-8
    losses = []
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        pred = sigmoid(tx.dot(w))
        loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
        loss = np.squeeze(- loss)
        # loss= (np.log(1+np.exp(tx @ w)).sum()- y.T @ tx @ w).squeeze()
        # gradient= tx.T @ (sigmoid(tx @ w) - y)
        pred = sigmoid(tx.dot(w))
        gradient  = tx.T.dot(pred - y)
        w= w - gamma*gradient

        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
            
    return w.squeeze(),loss

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad


def reg_logistic_regression(y, tx, lambda_ ,initial_w, max_iters, gamma):
        # init parameters
    threshold = 1e-8
    losses = []

    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        # loss= (np.log(1+np.exp(tx @ w)).sum()- y.T @ tx @ w).squeeze() + lambda_/2 * np.linalg.norm(w)
        # gradient= tx.T @ (sigmoid(tx @ w) - y) + 2* lambda_*w
        loss = calculate_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
        gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w
        w= w - gamma*gradient

        # log info
        #if iter % 100 == 0:
            #print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
            
    return w.squeeze(),loss
    
