# https://warrenmar.wordpress.com/tag/seasonal-hybrid-esd/
# Comments by Erik Rodner

def generalized_esd(x, alpha, r):
    """ Run the ESD algorithm with r being the number of outliers and alpha being a hyperparameter"""

    # Comment from the theoretical side:
    # The algorithm is basically the hotelings-T algorithm in the univariate case
    # with proper thresholds derived from the distribution

    # calculate lambda critical (the thresholds used)
    lambda_critical = []
    for i in range(1, r): # anomalies
        p = 1.0 - alpha/(2.0 * (n - i + 1))
        t = stats.t.ppf(p, n - i - 1)
        lambda_critical.append((n - i) * t / np.sqrt((n - i - 1 + t**2) * (n - i + 1)))
 
    def remove_largest(values):
        # get the maximum in the vector of mean/std-normalized values together with the index
        k, value = max(enumerate(abs((values - np.mean(values)) / np.std(values))), key=lambda x:x[1])
        # remove the maximum element
        new_values = [v for i, v in enumerate(values) if i != k]
        return new_values, value, k
 
    number_of_anomalies = 0
    count = 0
    new_x = x
    extreme_points = []
    for l_crit in lambda_critical:
        count = count + 1
        x, v, x_index = remove_largest(x)
        if v > l_crit:
            number_of_anomalies = count
            extreme_points.append(x_index)
            new_x = x
    return number_of_anomalies, extreme_points
