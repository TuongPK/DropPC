import glob
import cPickle
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    path = './Result/*.pkl'

    files = glob.glob(path)

    for result in files:
        print result

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

            fig, axes = plt.subplots(10, 10)

            weight = []

            for w_index in xrange(len(weights)-1):
                if w_index % 2 == 0:
                    weight.extend(weights[w_index].T)

            weight = np.asarray(weight)

            # use global min / max to ensure all weights are shown on the same scale
            vmin, vmax = weight.min(), weight.max()

            for coef, ax in zip(weight, axes.ravel()):
                ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
                           vmax=.5 * vmax)
                ax.set_xticks(())
                ax.set_yticks(())

    plt.legend()
    plt.savefig('feature_pc.eps', format='eps', dpi=500, bbox_inches='tight')

