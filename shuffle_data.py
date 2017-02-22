import numpy

dataset = "breast_tissue.data"

input = numpy.genfromtxt(dataset, delimiter='\n', dtype=None)

output = input
nRow = len(input)

sIndex = range(nRow)
numpy.random.shuffle(sIndex)

for i in xrange(nRow):
    output[i] = input[sIndex[i]]

f = open("processed_breast_tissue.data", "w")

for i in sIndex:
    f.write(input[sIndex[i]] + "\n")

f.close()




