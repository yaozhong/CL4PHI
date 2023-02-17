# data loading for the paired samples

import torch, io
from torch.utils.data import Dataset, DataLoader
from functools import partial

from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from fasta2CGR import *
import numpy as np
from PIL import Image

def get_data_host_sets(file_name_list):

	labels = []

	for fn in file_name_list:
		s_in = open(fn)
		
		for line in s_in:
			line_info = line.strip("\n")
			labels.append(line_info)

		s_in.close()

	return list(set(labels))


def get_label_map(species_file):
    s_in = open(species_file)
    spiece_dic, label2species = {}, []

    for idx, line in enumerate(s_in):
        line_info = line.strip("\n")
        specie_label = line_info.split("\t")[1]
        spiece_dic[specie_label] = idx
        label2species.append(specie_label)

    s_in.close()

    return spiece_dic, label2species

def load_host_label(file_name, l2s_dic):

	if file_name == '':
		return []

	s_in = open(file_name)
	labels = []

	for line in s_in:
		line_info = line.strip("\n")
		labels.append(l2s_dic[line_info])

	s_in.close()

	return labels

# get all host representation of the host_fa_file according to the keepList.
def get_host_fa(s2l_dic, host_fa_file, kmer, keep_list=[]):

	l2fa = {}
	# loading host fa information into 
	wgs = Fasta(host_fa_file)
	for bn in wgs.keys():
		# filtering labels not in the keep_list
		check = bn.replace("_", " ")
		if len(keep_list) > 0 and check not in keep_list:
			continue

		seq = wgs[bn][:].seq
		fc = count_kmers(seq, kmer)
		f_prob = probabilities(seq, fc, kmer)
		chaos_k = chaos_game_representation(f_prob, kmer)

		label = s2l_dic[bn.replace("_", " ")]
		l2fa[label] = chaos_k

	return l2fa


def my_collate_fn(batch, kmer, l2fa):

	images, hosts, labels = [],[],[]
	for name, seq, label in batch:
		phage_name = name
		seq = seq

		# FCGR, represntation for phage
		fc = count_kmers(seq, kmer)
		f_prob = probabilities(seq, fc, kmer)
		# consider the evluation of the k.
		chaos_k = chaos_game_representation(f_prob, kmer)
		img = chaos_k

		# consider the efficiency here.
		for l in l2fa.keys():
			if l == label:
				labels.append(1)
				#print("Hit label:", l)
			else:
				labels.append(0)

			images.append(img)
			hosts.append(l2fa[l])

	return np.array(images), np.array(hosts), np.array(labels)


# standard approach of loading data for valdiation and testing.
def my_collate_fn2(batch, kmer):

	images, labels, phage_name_list = [],[],[]
	for name, seq, label in batch:
		phage_name = name
		seq = seq
		labels.append(label)

		# FCGR
		fc = count_kmers(seq, kmer)
		f_prob = probabilities(seq, fc, kmer)
		# consider the evluation of the k.
		chaos_k = chaos_game_representation(f_prob, kmer)
		img = chaos_k
		images.append(img)
		phage_name_list.append(phage_name)

	return np.array(images), np.array(labels), phage_name_list


class fasta_dataset(Dataset):
	def __init__(self, file_name, label_file, host_file):

		wgs = Fasta(file_name)
		self.name = []
		self.seq = []

		self.s2l_dic, self.l2s = get_label_map(label_file)

		# sequence process an put it in queue
		for pn in wgs.keys():
			self.name.append(pn)
			self.seq.append(wgs[pn][:].seq)

		self.label = load_host_label(host_file, self.s2l_dic)	

	def __len__(self):
		return	len(self.name)

	def __getitem__(self, idx):
		if(len(self.label) == 0):
			return self.name[idx], self.seq[idx], []
		return self.name[idx], self.seq[idx], self.label[idx]

	def get_s2l_dic(self):
		return self.s2l_dic

	def get_l2s_dic(self):
		return self.l2s


