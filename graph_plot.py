import glob
import cPickle
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from mnist_sgd import mnist_load_data

if __name__ == '__main__':
    path = './Result/*.pkl'


    files = glob.glob(path)
    """
    datasets = mnist_load_data('mnist.pkl.gz')
    train_set_x, train_set_y = datasets[0]

    data = train_set_x.get_value(borrow=True)
    img = data[0]
    print(data.shape)
    vmin = min(img)
    vmax = max(img)
    plt.matshow(img.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin, vmax=.5 * vmax)
    """
    for result in files:
        print result
        with open(result, 'rb') as f:
            data = cPickle.load(f)

            train_data = data[0][0]
            valid_data = data[0][1]
            test_data = data[0][2]
            gradient_data = data[0][3]
            duration = data[0][4]
            weights = data[0][5]
            f.close()

            print(\
                'Last training error %f \n'
                'Last validation error %f\n'
                'Best testing error %f \n'
                'Duration %d'
            ) % (train_data[-1]*100, valid_data[-1]*100, test_data[-1]*100, duration/60)

            fig, axes = plt.subplots(10, 10)

            weight = []

            for w_index in xrange(len(weights)-1):
                if w_index % 2 == 0:
                    weight.extend(weights[w_index].T)

            weight = np.asarray(weight)

            print(weight[0])
            break
            # use global min / max to ensure all weights are shown on the same scale
            vmin, vmax = weight.min(), weight.max()

            for coef, ax in zip(weight, axes.ravel()):
                ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
                           vmax=.5 * vmax)
                ax.set_xticks(())
                ax.set_yticks(())
            """
            temp = weights
            weight = []
            for w in temp:
                weight.extend(w.ravel())

            weight = np.asarray(weight)
            print(len(weight))

            density = stats.kde.gaussian_kde(weight)
            xs = np.linspace(min(weight), max(weight), 1000)
            plt.plot(xs, density(xs), label=result)
            """
            # fit = stats.norm.pdf(weight, np.mean(weight), np.std(weight)) # this is a fitting indeed
            # plt.plot(weight, fit, '.', label=result)
            # plt.hist(weight, normed=True, histtype='stepfilled')  # use this to draw histogram of your data
    plt.legend()
    plt.show()
    #plt.savefig('feature_pc.eps', format='eps', dpi=500, bbox_inches='tight')

