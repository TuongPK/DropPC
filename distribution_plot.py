import glob
import cPickle
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    path = './Result/*.pkl'

    files = glob.glob(path)

    # Labels
    name = []

    count = -1

    for result in files:
        print result

        count += 1
        with open(result, 'rb') as f:
            data = cPickle.load(f)

            [train_data, valid_data, test_data, gradient_data, duration, weights] = data[0]
            f.close()

            print(\
                'Last training error %f \n'
                'Last validation error %f\n'
                'Best testing error %f \n'
                'Duration %d'
            ) % (train_data[-1]*100, valid_data[-1]*100, test_data[-1]*100, duration/60)

            # Concatenate and reshape weights
            temp = weights
            weight = []
            for w in temp:
                weight.extend(w.ravel())

            weight = np.asarray(weight)

            density = stats.kde.gaussian_kde(weight)
            xs = np.linspace(min(weight), max(weight), 1000)
            plt.plot(xs, density(xs), label=name[count])

    plt.legend()
    plt.savefig('distribution.eps', format='eps', dpi=1000, bbox_inches='tight')

