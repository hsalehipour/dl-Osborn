import numpy as np


def lrelu(x, alpha=0.001):
    """Leaky Relu activation function"""
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def add_noise(data, nrand = 10):
    nexample = np.int32(data.shape[0]/nz_profile)
    for ie in range(nexample):
        istart = ie * nz_profile
        iend = istart + nz_profile-1
        profiles = data.loc[istart:iend,:]
        for i in range(nrand):
            data.append(add_rand_logn_noise(profiles))
    return data


def add_gaussian_noise(data, coefficient=0.1, mu=0.0, sigma=0.5):
    """
   :param data: Data to add Gaussian noise to.
   :param coefficient: Noise factor.
   :param mu: Mean of the distribution.
   :param sigma: Standard deviation of the distribution.
   :return: Noisy copy of the input data.
   """
    return data + coefficient * np.random.normal(loc=mu, scale=sigma, size=data.shape)


def add_rand_logn_noise(data, amp=0.01):
    """
    add random lognormal noise
    """
    m = np.mean(data, axis=0)
    v = np.var(data , axis=0)
    mu = np.log((m**2)/np.sqrt(v+m**2))
    sigma = np.sqrt(np.log(v/(m**2)+1.))
    return data + amp * np.random.lognormal(mean=mu, sigma=sigma, size=data.shape)


def r2_score(label, prediction):
    """
    calculate the R2 score of a regression
    """
    total_error = np.sum((label-np.mean(label))**2)
    residual_error = np.sum((label-prediction)**2)
    r2 = 1.0 - residual_error/(total_error + np.finfo(np.float).eps)

    return r2

def rmse(label, prediction):
    """
    calculate the RMSE error of predictions
    """
    return np.sqrt(np.mean((label-prediction)**2))


def normalize_data(x, axis=0):
    """"
    Normalizes the data to range between 0-1
    """
    x = np.array(x, dtype=np.float32)
    xmin = np.expand_dims(x.min(axis=axis), axis=axis)
    xmax = np.expand_dims(x.max(axis=axis), axis=axis)

    return (x-xmin)/(xmax-xmin)

