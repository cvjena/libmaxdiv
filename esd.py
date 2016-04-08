# https://warrenmar.wordpress.com/tag/seasonal-hybrid-esd/

def generalized_esd(x, alpha, r):
    # calculate lambda critical
    lambda_critical = []
    for i in range(1, r): # anomalies
        p = 1.0 - alpha/(2.0 * (n - i + 1))
        t = stats.t.ppf(p, n - i - 1)
        lambda_critical.append((n - i) * t / np.sqrt((n - i - 1 + t**2) * (n - i + 1)))
 
    def remove_largest(values):
        k, value = max(enumerate(abs((values - np.mean(values)) / np.std(values))), key=lambda x:x[1])
        new_values = [v for i, v in enumerate(values) if i != k]
        return new_values, value
 
    number_of_anomalies = 0
    count = 0
    new_x = x
    for l_crit in lambda_critical:
        count = count + 1
        x, v = remove_largest(x)
        if v > l_crit:
            number_of_anomalies = count
            new_x = x
    return number_of_anomalies, new_x
