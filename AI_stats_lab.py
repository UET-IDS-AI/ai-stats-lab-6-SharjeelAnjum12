import math
import numpy as np


def bernoulli_log_likelihood(data, theta):
    """
    Compute the Bernoulli log-likelihood for binary data.

    Parameters
    ----------
    data : array-like
        Sequence of 0/1 observations.
    theta : float
        Bernoulli parameter, must satisfy 0 < theta < 1.

    Returns
    -------
    float
        Log-likelihood:
            sum_i [x_i log(theta) + (1-x_i) log(1-theta)]

    Requirements
    ------------
    - Raise ValueError if data is empty
    - Raise ValueError if theta is not in (0,1)
    - Raise ValueError if data contains values other than 0 and 1
    """
    """
    Compute the Bernoulli log-likelihood for binary data.
    """
    # Validation
    if len(data) == 0:
        raise ValueError("Data must not be empty")

    if not (0 < theta < 1):
        raise ValueError("Theta must be between 0 and 1")

    for x in data:
        if x not in [0, 1]:
            raise ValueError("Data must contain only 0 and 1")

    # Log-likelihood calculation
    log_likelihood = 0.0

    for x in data:
        log_likelihood += x * math.log(theta) + (1 - x) * math.log(1 - theta)

    return log_likelihood


def bernoulli_mle_with_comparison(data, candidate_thetas=None):
    """
    Estimate the Bernoulli MLE and compare candidate theta values.

    Parameters
    ----------
    data : array-like
        Sequence of 0/1 observations.
    candidate_thetas : array-like or None
        Optional candidate theta values to compare using log-likelihood.
        If None, use [0.2, 0.5, 0.8].

    Returns
    -------
    dict
        A dictionary with:
        - 'mle': float
            The Bernoulli MLE
        - 'num_successes': int
        - 'num_failures': int
        - 'log_likelihoods': dict
            Mapping candidate theta -> log-likelihood
        - 'best_candidate': float
            Candidate theta with highest log-likelihood

    Requirements
    ------------
    - Validate data
    - Compute MLE analytically
    - Compute candidate log-likelihoods using bernoulli_log_likelihood
    - In case of ties in best candidate, return the first one encountered
    """
    """
    Estimate the Bernoulli MLE and compare candidate theta values.
    """
    # Validation
    if len(data) == 0:
        raise ValueError("Data must not be empty")

    for x in data:
        if x not in [0, 1]:
            raise ValueError("Data must contain only 0 and 1")

    if candidate_thetas is None:
        candidate_thetas = [0.2, 0.5, 0.8]

    data = np.array(data)

    # Successes and failures
    num_successes = np.sum(data)
    num_failures = len(data) - num_successes

    # MLE (mean of data)
    mle = num_successes / len(data)

    # Log-likelihoods for candidates
    log_likelihoods = {}
    for theta in candidate_thetas:
        ll = bernoulli_log_likelihood(data, theta)
        log_likelihoods[theta] = ll

    best_candidate = None
    best_value = -float('inf')

    for theta in candidate_thetas:
        if log_likelihoods[theta] > best_value:
            best_value = log_likelihoods[theta]
            best_candidate = theta

    # Result dictionary
    return {
        'mle': mle,
        'num_successes': int(num_successes),
        'num_failures': int(num_failures),
        'log_likelihoods': log_likelihoods,
        'best_candidate': best_candidate
    }


def poisson_log_likelihood(data, lam):
    """
    Compute the Poisson log-likelihood for count data.

    Parameters
    ----------
    data : array-like
        Sequence of nonnegative integer counts.
    lam : float
        Poisson rate, must satisfy lam > 0.

    Returns
    -------
    float
        Log-likelihood:
            sum_i [x_i log(lam) - lam - log(x_i!)]

    Requirements
    ------------
    - Raise ValueError if data is empty
    - Raise ValueError if lam <= 0
    - Raise ValueError if data contains negative or non-integer values

    Notes
    -----
    You may use math.lgamma(x + 1) for log(x!) since log(x!) = lgamma(x+1).
    """
    """
    Compute the Poisson log-likelihood for count data.
    """
    # Validation
    if len(data) == 0:
        raise ValueError("Data must not be empty")

    if lam <= 0:
        raise ValueError("Lambda must be greater than 0")

    for x in data:
        if (not isinstance(x, (int, np.integer))) or x < 0:
            raise ValueError("Data must contain nonnegative integers only")

    # Log-likelihood 
    log_likelihood = 0.0

    for x in data:
        log_likelihood += x * math.log(lam) - lam - math.lgamma(x + 1)

    return log_likelihood


def poisson_mle_analysis(data, candidate_lambdas=None):
    """
    Estimate the Poisson MLE and compare candidate lambda values.

    Parameters
    ----------
    data : array-like
        Sequence of nonnegative integer counts.
    candidate_lambdas : array-like or None
        Optional candidate lambdas to compare using log-likelihood.
        If None, use [1.0, 3.0, 5.0].

    Returns
    -------
    dict
        A dictionary with:
        - 'mle': float
            The Poisson MLE
        - 'sample_mean': float
        - 'total_count': int
        - 'n': int
        - 'log_likelihoods': dict
            Mapping candidate lambda -> log-likelihood
        - 'best_candidate': float
            Candidate lambda with highest log-likelihood

    Requirements
    ------------
    - Validate data
    - Compute MLE analytically
    - Compute candidate log-likelihoods using poisson_log_likelihood
    - In case of ties in best candidate, return the first one encountered
    """
    # Validation
    if len(data) == 0:
        raise ValueError("Data must not be empty")

    for x in data:
        if (not isinstance(x, (int, np.integer))) or x < 0:
            raise ValueError("Data must contain nonnegative integers only")

    if candidate_lambdas is None:
        candidate_lambdas = [1.0, 3.0, 5.0]

    data = np.array(data)

    n = len(data)
    total_count = int(np.sum(data))
    sample_mean = total_count / n

    # MLE
    mle = sample_mean

    # Log-likelihoods
    log_likelihoods = {}
    for lam in candidate_lambdas:
        ll = poisson_log_likelihood(data, lam)
        log_likelihoods[lam] = ll

    # Best candidate
    best_candidate = None
    best_value = -float('inf')

    for lam in candidate_lambdas:
        if log_likelihoods[lam] > best_value:
            best_value = log_likelihoods[lam]
            best_candidate = lam

    # Results
    return {
        'mle': mle,
        'sample_mean': sample_mean,
        'total_count': total_count,
        'n': n,
        'log_likelihoods': log_likelihoods,
        'best_candidate': best_candidate
    }
