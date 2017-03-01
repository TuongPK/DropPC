import glob
import cPickle
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    path = './Result/*.pkl'

    files = glob.glob(path)

    name = ['PCDC_10',
            'PCDC_5',
            'SCDO',
            'PCDC_2',
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

        gradient_data = [gradient_data[i] / (0.784) for i in xrange(len(gradient_data))]

        plt.plot(gradient_data, label=name[count])
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Gradient Magnitude (x$10^{-5}$)')
    #plt.show()
    plt.savefig('gradient.eps', format='eps', dpi=1000, bbox_inches='tight')
