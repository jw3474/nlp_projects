import numpy as np
import scipy.io as sio


# read in dictionary
word_dict = {}
index = 1
with open('nyt_vocab.dat', 'r') as f:
	line = f.readline()
	while line != '':
		word_dict[index] = line.strip()
		index += 1
		line = f.readline()


assert len(word_dict.keys()) == 3012

# load in mat
sorted_w = sio.loadmat('sorted_w_matrix.mat')['sorted_w_matrix']
sorted_I = sio.loadmat('I_w.mat')['I_w']


# find top 10 for each column
cell_dict = {}

num_top = 10
num_cols = 25
for c in range(num_cols):
	cell_dict[c+1] = []
	for t in range(num_top):
		cell_dict[c+1].append((word_dict[int(sorted_I[t, c])], sorted_w[t, c]))


# generate a 5-by-5 table
rows = 5
cols = 5
s = '\\hline\n'
for r in range(rows):
	for t in range(num_top):
		line = ''
		for c in range(cols):
			i = r*cols + c + 1
			line += cell_dict[i][t][0] + ': ' + ('%.4f' % cell_dict[i][t][1])
			if c < cols-1:
				line += '&'
		s += line
		s += '\\\\'
		if t < num_top-1:
			s += '\n'
	s += '\\hline\n'

print s