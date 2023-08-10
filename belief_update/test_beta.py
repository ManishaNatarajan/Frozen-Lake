import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta


if __name__=='__main__':
    # Estimate a hidden Bernoulli parameter
    mu = 0.65

    # Generate samples ~ mu
    data = []
    num_samples = 100
    interval = num_samples // 5
    np.random.seed(0)
    for i in range(num_samples):
        if np.random.random() > mu:
            data.append(0)
        else:
            data.append(1)

    print("Total # 1s: {}".format(np.sum(data)))

    # Estimate the hidden parameter from beta prior
    a, b = 1, 1   # Prior Parameters (assume uniform distribution)

    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 1000)

    # Plot the initial distribution
    ax.set_xlim(0.0, 1)
    ax.set_ylim(0, 14)
    ax.plot(x, beta.pdf(x, a, b), label=0)
    ax.legend()
    plt.title('Beta Distribution for mu={}'.format(mu), fontsize='15')
    plt.xlabel('Values of Random Variable X (0, 1)', fontsize='15')
    plt.ylabel('Probability', fontsize='15')
    plt.savefig("test.png")

    # Calculate the different a and b parameter values at regular intervals of data observations
    for i in range(0, num_samples, interval):
        start = i
        end = i+interval
        num_ones = np.sum(data[start:end])
        total_samples = len(data[start:end])
        a = a + num_ones
        b = b + (total_samples - num_ones)
        ax.plot(x, beta.pdf(x, a, b), label=end)

        ax.legend()
        plt.title('Beta Distribution for mu={}'.format(mu), fontsize='15')
        plt.xlabel('Values of Random Variable X (0, 1)', fontsize='15')
        plt.ylabel('Probability', fontsize='15')
        plt.savefig("test_{}.png".format(i))

    # Shifting beta distribution
    mu_prime = 0.3

    # Generate samples ~ mu
    data = []
    num_samples = 100
    interval = num_samples // 5
    np.random.seed(0)
    for i in range(num_samples):
        if np.random.random() > mu_prime:
            data.append(0)
        else:
            data.append(1)

    print("Total # 1s: {}".format(np.sum(data)))

    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 1000)

    # Plot the initial distribution
    ax.set_xlim(0.0, 1)
    ax.set_ylim(0, 14)
    ax.plot(x, beta.pdf(x, a, b), label=0)

    # Calculate the different a and b parameter values at regular intervals of data observations
    for i in range(0, num_samples, interval):
        start = i
        end = i + interval
        num_ones = np.sum(data[start:end])
        total_samples = len(data[start:end])
        a = a + num_ones
        b = b + (total_samples - num_ones)
        ax.plot(x, beta.pdf(x, a, b), label=end)

    ax.legend()
    plt.title('Beta Distribution for mu\'={}'.format(mu_prime), fontsize='15')
    plt.xlabel('Values of Random Variable X (0, 1)', fontsize='15')
    plt.ylabel('Probability', fontsize='15')
    plt.show()