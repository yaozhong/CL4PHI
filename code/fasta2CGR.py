# implemenation of Chaos Game Represenation of a genetic sequence.
# code reference: https://towardsdatascience.com/chaos-game-representation-of-a-genetic-sequence-4681f1a67e14

import collections, sys, io
from collections import OrderedDict
from matplotlib import pyplot as plt
from matplotlib import cm
import pylab, math
from pyfaidx import Fasta
from collections import defaultdict

from PIL import Image
import numpy as np
#from torchvision import transforms

def empty_dict():
	"""
	None type return vessel for defaultdict
	:return:
	"""
	return None

def count_kmers(sequence, k):
	d = collections.defaultdict(int)
	for i in range(len(sequence)-(k-1)):
		d[sequence[i:i+k]] += 1

	for key in list(d.keys()):
		if "N" in key:
			del d[key]
	return d

def probabilities(data, kmer_count, k):
	probabilities = collections.defaultdict(float)
	N = len(data)
	for key, value in kmer_count.items():
		probabilities[key] = float(value) / (N - k + 1)
	return probabilities

def chaos_game_representation(probabilities, k):
	array_size = int(math.sqrt(4**k))
	chaos = []
	for i in range(array_size):
		chaos.append([0]*array_size)

	maxx, maxy = array_size, array_size
	posx, posy = 1, 1

	for key, value in probabilities.items():
		for char in key:
			if char == "T" or char == "t":
				posx += maxx / 2
			elif char == "C" or char == "c":
				posy += maxy / 2
			elif char == "G" or char == "g":
				posx += maxx / 2
				posy += maxy / 2
			maxx = maxx / 2
			maxy /= 2

		chaos[int(posy)-1][int(posx)-1] = value
		maxx = array_size
		maxy = array_size
		posx = 1
		posy = 1

	return chaos


def seq_to_fcgr(sequence, k):
	"""
    Vectorized frequency chaos-game representation (FCGR) of a DNA sequence,
	returned as a (2^k x 2^k) numpy array.

	Ambiguity handling (v1.2.2): any k-mer containing a non-ACGT character is
	EXCLUDED from the count -- the standard k-mer-counter convention (KMC,
	Jellyfish), which is deterministic and unbiased. 
    """
	size = int(math.sqrt(4 ** k))
	n = len(sequence)
	fcgr = np.zeros((size, size), dtype=np.float64)
	if n < k:
		return fcgr
	arr = np.frombuffer(sequence.upper().encode('ascii', 'replace'), dtype=np.uint8)
	# base -> (x-bit, y-bit): A=(0,0) C=(0,1) G=(1,1) T=(1,0); any other char is invalid
	xb = np.zeros(256, dtype=np.int64)
	yb = np.zeros(256, dtype=np.int64)
	invalid = np.ones(256, dtype=bool)
	for ch, x, y in [('A', 0, 0), ('C', 0, 1), ('G', 1, 1), ('T', 1, 0)]:
		xb[ord(ch)] = x
		yb[ord(ch)] = y
		invalid[ord(ch)] = False
	X = xb[arr]
	Y = yb[arr]
	inv = invalid[arr].astype(np.int64)
	m = n - k + 1
	powers = (2 ** np.arange(k - 1, -1, -1)).astype(np.int64)
	xi = np.zeros(m, dtype=np.int64)
	yi = np.zeros(m, dtype=np.int64)
	nInv = np.zeros(m, dtype=np.int64)
	for j in range(k):   # k vectorized adds; no [n, k] window materialization
		xi += X[j:j + m] * powers[j]
		yi += Y[j:j + m] * powers[j]
		nInv += inv[j:j + m]
	valid = nInv == 0    # exclude any k-mer that contains a non-ACGT base
	flat = yi[valid] * size + xi[valid]
	counts = np.bincount(flat, minlength=size * size).reshape(size, size).astype(np.float64)
	return counts / (n - k + 1)   # divide by total window count (matches probabilities())


def cgr_positions(seq):

	# CGR_CENTER = (0.5, 0.5)
	CGR_X_MAX, CGR_Y_MAX = 1, 1
	CGR_X_MIN, CGR_Y_MIN = 0, 0

	CGR_A = (CGR_X_MIN, CGR_Y_MIN)
	CGR_T = (CGR_X_MAX, CGR_Y_MIN)
	CGR_G = (CGR_X_MAX, CGR_Y_MAX)
	CGR_C = (CGR_X_MIN, CGR_Y_MAX)
	CGR_CENTER = ((CGR_X_MAX - CGR_Y_MIN) / 2, (CGR_Y_MAX - CGR_Y_MIN) / 2)

	CGR_DICT = defaultdict(
	empty_dict,
	[
		('A', CGR_A),  # Adenine
		('T', CGR_T),  # Thymine
		('G', CGR_G),  # Guanine
		('C', CGR_C),  # Cytosine
		('U', CGR_T),  # Uracil demethylated form of thymine
		('a', CGR_A),  # Adenine
		('t', CGR_T),  # Thymine
		('g', CGR_G),  # Guanine
		('c', CGR_C),  # Cytosine
		('u', CGR_T)  # Uracil/Thymine
		]
	)

	cgr = []
	cgr_marker = CGR_CENTER[:
		]    # The center of square which serves as first marker
	for s in seq:
		cgr_corner = CGR_DICT[s]
		if cgr_corner:
			cgr_marker = (
				(cgr_corner[0] + cgr_marker[0]) / 2,
				(cgr_corner[1] + cgr_marker[1]) / 2
			)
			cgr.append([s, cgr_marker])
		else:
			#sys.stderr.write("Bad Nucleotide: " + s + " \n")
			pass

	return cgr

