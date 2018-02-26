import glob
import cPickle
import matplotlib.pyplot as plt

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

        gradient_data = [gradient_data[i] / (0.784) for i in xrange(len(gradient_data))]

        plt.plot(gradient_data, label=name[count])

    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Gradient Magnitude (x$10^{-5}$)')

    plt.savefig('gradient.eps', format='eps', dpi=1000, bbox_inches='tight')
