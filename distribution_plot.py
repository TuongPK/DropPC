import glob
import cPickle
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    path = './Result/*.pkl'

    files = glob.glob(path)

    name = ['PCDC_10',
            'SCDO',
            'SC']

    count = -1

    for result in files:
        print result
        count += 1
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

            temp = weights
            weight = []
            for w in temp:
                weight.extend(w.ravel())

            weight = np.asarray(weight)
            print(len(weight))
           density = stats.kde.gaussian_kde(weight)
            xs = np.linspace(min(weight), max(weight), 1000)
            plt.plot(xs, density(xs), label=name[count])

            #fit = stats.norm.pdf(weight, np.mean(weight), np.std(weight)) # this is a fitting indeed
            #plt.plot(weight, fit, '.', label=result)
            # plt.hist(weight, normed=True, histtype='stepfilled')  # use this to draw histogram of your data
    plt.legend()
    #plt.show()
    plt.savefig('distribution.eps', format='eps', dpi=1000, bbox_inches='tight')

