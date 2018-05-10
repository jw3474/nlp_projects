import numpy as np
import scipy.io as sio
from scipy import sparse


num_words = 3012
num_docs = 8447

X = np.zeros((num_words, num_docs), dtype=np.int32)

with open('nyt_data.txt', 'r') as f:
	line = f.readline()
	c = 1
	while line != '':
		word_counts = line.split(',')
		for ctr in word_counts:
			k = int(ctr.split(':')[0])
			v = int(ctr.split(':')[1])
			assert k <= num_words
			X[k-1, c-1] = v
		c += 1
		line = f.readline()

X_mat = {}
X_mat['X'] = sparse.csr_matrix(X)
sio.savemat('X', X_mat)