# prediction 
import argparse
import torch
from data_loading import *
from model import *

import numpy as np
from sklearn import metrics


def test(model, cached_ph, l2fa, cached_label, device, threshold=1, verbose=False):

	# first generate embeddings for the host.
	host_vec = np.array([l2fa[l] for l in l2fa.keys()])
	host_vec = torch.tensor(host_vec, dtype=torch.float32).to(device)
	host_vec = torch.unsqueeze(host_vec, dim=1)

	embed_bts = model(host_vec)
	label_list = list(l2fa.keys())

	pred_list, pred_dist, gold_list = [], [], []
	total_batch = len(cached_ph)

	with torch.no_grad():
		for i in range(total_batch):
			phs, labels = cached_ph[i], cached_label[i]

			phs = phs.to(device)
			labels = labels.to(device)
			embed_phs = model(phs)

			# local calcuation of the distance scores
			for e_ph in embed_phs:
				diff =  embed_bts - e_ph
				dist_sq = torch.sum(torch.pow(diff, 2), 1)
				dist = torch.sqrt(dist_sq)

				pred_dist.append(dist.to("cpu").detach().numpy())
				idx = torch.argmin(dist).to("cpu").detach().numpy()
				
				pred_list.extend([label_list[idx]])
				
			gold_list.extend(labels.to("cpu").detach().numpy())

	acc = metrics.accuracy_score(gold_list, pred_list)
	if verbose:
		print(f"@ Given set  Accuracy is {acc}")

	return acc, pred_dist, gold_list


# prediction without provide gold standard
def predict(model, cached_ph, l2fa, device):

	# first generate embeddings for the host.
	host_vec = np.array([l2fa[l] for l in l2fa.keys()])
	host_vec = torch.tensor(host_vec, dtype=torch.float32).to(device)
	host_vec = torch.unsqueeze(host_vec, dim=1)

	embed_bts = model(host_vec)
	label_list = list(l2fa.keys())

	pred_list, pred_dist = [], []
	total_batch = len(cached_ph)

	with torch.no_grad():
		for i in range(total_batch):
			phs = cached_ph[i]
			phs = phs.to(device)
			embed_phs = model(phs)

			# local calcuation of the distance scores
			for e_ph in embed_phs:
				diff =  embed_bts - e_ph
				dist_sq = torch.sum(torch.pow(diff, 2), 1)
				dist = torch.sqrt(dist_sq)

				pred_dist.append(dist.to("cpu").detach().numpy())

				#idx = torch.argmin(dist).to("cpu").detach().numpy()	
				#pred_list.extend([label_list[idx]])
				
	return pred_dist


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='<Prediciton of using contrastive learning for the phage-host identification>')

	parser.add_argument('--model',       default="CNN", type=str, required=True, help='contrastive learning encoding model')
	parser.add_argument('--model_dir', action="store",   type=str, required=True,  help="directory for saving the trained model.")
	parser.add_argument('--device',       default="cuda:0", type=str, required=False, help='GPU Device(s) used for training')

	parser.add_argument('--kmer',       default=6,       type=int, required=True, help='kmer length')
	parser.add_argument('--batch_size' ,default=64,      type=int,  required=False, help="batch_size of the training.")
	parser.add_argument('--workers',     default=8,       type=int, required=False, help='number of worker for data loading')

	# data related input
	parser.add_argument('--host_fa',   default="",  type=str, required=True, help='Host fasta files')
	parser.add_argument('--host_list', default="",  type=str, required=True, help='Host species list')
	parser.add_argument('--test_phage_fa', default="",   type=str, required=True, help='Test phage fasta file')
	parser.add_argument('--test_host_gold', default="",  type=str, required=False, help='Infecting host gold list')

	parser.add_argument('--use_train_bn', action='store_true', required=False, help='use the batch norm statistics in the train')
	
	args = parser.parse_args()

	kmer = args.kmer
	num_workers = args.workers
	batch_size = args.batch_size

	# preparing the test data for the evaluation.
	## 1. loading model
	#print("@ Loading model ... ", end="")
	## parparing host data information.
	if args.model == "CNN":
		model = cnn_module(7, 0)
	
	model.load_state_dict(torch.load(args.model_dir))
	model = model.to(args.device)

	if args.use_tran_bn:
		model.eval()

	#print("[ok]")

	## 2. loading data
	#print("@ Loading phage dataset ... ", end="")
	spiece_file = args.host_list
	host_fa_file = args.host_fa	
	phage_test_file = args.test_phage_fa
	host_test_file = args.test_host_gold

	fa_test_dataset = fasta_dataset(phage_test_file, spiece_file, host_test_file)
	test_generator = DataLoader(fa_test_dataset, batch_size, collate_fn=partial(my_collate_fn2, kmer=kmer), num_workers=num_workers) 

	cached_test_ph,  cached_test_label, test_phName = [], [], []

	for phs, labels, phName in test_generator:
		imgs_ph = torch.tensor(phs, dtype = torch.float32)
		cached_test_ph.append(torch.unsqueeze(imgs_ph, dim=1))
		cached_test_label.append(torch.tensor(labels))
		test_phName.extend(phName)

	#print("[ok]")

	# viualizaiton
	#print("@ Loading Host dataset ... ", end="")
	s2l_dic = fa_test_dataset.get_s2l_dic()
	l2fa = get_host_fa(s2l_dic, host_fa_file, kmer)
	l2sn = fa_test_dataset.get_l2s_dic()
	label_list = list(l2fa.keys())
	
	#print("[ok]")
	#print("@ Start prediction ...")
	if args.test_host_gold != "":
		acc_test, host_pred_list, gold_list  = test(model, cached_test_ph, l2fa, cached_test_label, args.device)
	else:
		host_pred_list = predict(model, cached_test_ph, l2fa, args.device)

	for i in range(len(host_pred_list)):

		print(test_phName[i], end="\t")
		idxs = np.argsort(host_pred_list[i])
		for idx in idxs:
			print(l2sn[label_list[idx]]+"_"+str(host_pred_list[i][idx]), end=" ")

		print("")

