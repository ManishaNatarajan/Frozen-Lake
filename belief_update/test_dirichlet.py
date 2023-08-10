import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, dirichlet

import matplotlib
matplotlib.use('TkAgg')

if __name__=='__main__':
    # Estimate a multinomial Distribution
    mu_star = [0.2, 0.65, 0.15]  # True parameters

    num_categories = len(mu_star)
    # Generate samples from the true distribution
    num_samples = 1000
    np.random.seed(0)

    interval = num_samples // 50

    # Generate samples at once using mu_star
    data = np.random.choice(np.arange(num_categories), num_samples, p=mu_star)
    _, counts = np.unique(data, return_counts=True)
    print(f"Total counts: {counts}")

    # Estimate hidden parameter (multinomial distribution) from prior
    prior_counts = [1, 1, 1]

    # Plot the initial distribution
    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 10000)
    ax.set_xlim(0.0, 1)
    # ax.set_ylim(0, 14)
    ax.plot(x, beta.pdf(x, counts[2], np.sum(counts)-counts[2]), label=0)
    ax.legend()
    plt.xlabel('Values of Random Variable x_1', fontsize='15')
    plt.ylabel('Probability', fontsize='15')
    plt.show()

    # # Calculate the different a and b parameter values at regular intervals of data observations
    # for i in range(0, num_samples, interval):
    #     start = i
    #     end = i + interval
    #     _, counts = np.unique(data[start:end], return_counts=True)
    #     total_samples = len(data[start:end])
    #     ax.plot(x, beta.pdf(x, counts[0], np.sum(counts)), label=end)
    #
    #     ax.legend()
    #     plt.title('Beta Distribution for mu={}'.format(mu), fontsize='15')
    #     plt.xlabel('Values of Random Variable X (0, 1)', fontsize='15')
    #     plt.ylabel('Probability', fontsize='15')

