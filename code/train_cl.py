# train the constrastive learning 
import argparse, time, random

from data_loading import *
from model import *
from eval import *

import torch
import multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F

from sklearn import metrics


import warnings
warnings.filterwarnings("ignore")


def set_seed(s):                                                                                              
	random.seed(s)
	np.random.seed(s)
	torch.manual_seed(s)

	torch.cuda.manual_seed_all(s)
	#add additional seed
	torch.backends.cudnn.deterministic=True
	torch.use_deterministic_algorithms = True


def train(dl_model, data_set, model_path, kmer, margin, batch_size, lr, epoch,\
	device="cuda:0", num_workers=64, verbose=True):

	cores = mp.cpu_count()
	if num_workers > 0 and num_workers < cores:
		cores = num_workers

	## data loading phase
	if verbose:
		print(" |- Start preparing dataset...")

	host_fa_file, spiece_file, phage_train_file, host_train_file, phage_valid_file, host_valid_file = data_set
	
	start_dataload = time.time()

	train_data_labels = get_data_host_sets([host_train_file, host_valid_file])
	print("	|-* Provided training sets totally has [", len(train_data_labels),"] hosts.")

	fa_train_dataset = fasta_dataset(phage_train_file, spiece_file, host_train_file)
	l2fa = get_host_fa(fa_train_dataset.get_s2l_dic(), host_fa_file, kmer)
	# add filters for host label here
	l2fa_filter = get_host_fa(fa_train_dataset.get_s2l_dic(), host_fa_file, kmer, train_data_labels)
	print("	|-[!] Checking host label information filtering for the training [non_filter:", len(l2fa.keys()), ", filtered:", len(l2fa_filter.keys()), "].")

	train_generator = DataLoader(fa_train_dataset, batch_size, collate_fn=partial(my_collate_fn, kmer=kmer, l2fa=l2fa_filter), num_workers=num_workers) 
    
	fa_valid_dataset = fasta_dataset(phage_valid_file, spiece_file, host_valid_file)
	valid_generator = DataLoader(fa_valid_dataset, batch_size, collate_fn=partial(my_collate_fn2, kmer=kmer), num_workers=num_workers) 

	# cached the data set
	cached_train_ph, cached_train_bt, cached_train_label = [], [], []
	cached_valid_ph, cached_valid_label, cached_valid_phageName = [], [], []

	## loading the image pairs for constrastive training
	for phs, bts, labels in train_generator:
		#X = torch.tensor(X, dtype = torch.float32).transpose(1,2)
		imgs_ph = torch.tensor(phs, dtype = torch.float32)
		imgs_bt = torch.tensor(bts, dtype = torch.float32)
		
		cached_train_ph.append(torch.unsqueeze(imgs_ph, dim=1))
		cached_train_bt.append(torch.unsqueeze(imgs_bt, dim=1))
		cached_train_label.append(torch.tensor(labels))

	for phs, labels, phName in valid_generator:
		imgs_ph = torch.tensor(phs, dtype = torch.float32)
		cached_valid_ph.append(torch.unsqueeze(imgs_ph, dim=1))
		cached_valid_label.append(torch.tensor(labels))
		cached_valid_phageName.append(phName)
	
	print(" |- loading [ok].")
	used_dataload = time.time() - start_dataload
	print("  |-@ used time:", round(used_dataload,2), "s")

	start_train = time.time()

	# model 2 (using CNN module)
	if dl_model == "CNN":
		model = cnn_module(7).to(device)
		optimizer = optim.Adam(model.parameters(), lr=lr)

	criterion = ContrastiveLoss(margin) 

	if verbose:
		print(f" |- Total number of {dl_model} has parameters %d:" %(sum([p.nelement() for p in model.parameters()])))
		print("  |- Training started ...")

	# start training
	epoch_acc_valid, epoch_acc_test, epoch_cm = [], [], []
	current_best_valid_acc = -100
	for ep in range(epoch):
		epoch_loss = 0
		for i in range(len(cached_train_ph)):
			phs, bts, labels = cached_train_ph[i], cached_train_bt[i], cached_train_label[i]

			phs = phs.to(device)
			bts = bts.to(device)
			labels = labels.to(device)

			embed_ph = model(phs)
			embed_bt = model(bts)

			loss = criterion(embed_ph, embed_bt, labels)
			epoch_loss += loss.item()

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		print("Epoch-%d, Loss=%f" %(ep,epoch_loss))
		acc_valid, _, _ = test(model, cached_valid_ph, l2fa_filter, cached_valid_label, device, 1, True)
		epoch_acc_valid.append(acc_valid)

		if acc_valid > current_best_valid_acc: # to be consistent with the following one. 
			current_best_valid_acc = acc_valid
			torch.save(model.state_dict(), model_path)
	
	idx = epoch_acc_valid.index(max(epoch_acc_valid))
	print(f"[Valid epoch idx/epoch]:{idx}/{epoch}, [valid acc]:{epoch_acc_valid[idx]}")
	used_train = time.time() - start_train
	print(" @ used training time:", round(used_train,2), "s. Total time:", round(used_train+used_dataload,2))



if __name__ == "__main__":

	set_seed(123)

	parser = argparse.ArgumentParser(description='<Contrastive learning for the phage-host identification>')

	parser.add_argument('--model',       default="CNN", type=str, required=True, help='contrastive learning encoding model')
	parser.add_argument('--model_dir', action="store",   type=str, required=True,  help="directory for saving the trained model.")
	parser.add_argument('--device',       default="cuda:0", type=str, required=False, help='GPU Device(s) used for training')
	
	parser.add_argument('--kmer',       default=5,       type=int, required=True, help='kmer length')
	parser.add_argument('--margin',     default=1,       type=int, required=True, help='Margins used in the contrastive training')
	parser.add_argument('--lr',     	default=1e-3,   type=float, required=False, help='Learning rate')
	parser.add_argument('--epoch',      default=20,       type=int, required=False, help='Training epcohs')
	parser.add_argument('--batch_size' ,default=64,      type=int,  required=False, help="batch_size of the training.")
	parser.add_argument('--workers',     default=64,       type=int, required=False, help='number of worker for data loading')

	# data related input
	parser.add_argument('--host_fa',   default="",  type=str, required=True, help='Host fasta files')
	parser.add_argument('--host_list', default="",  type=str, required=True, help='Host species list')

	parser.add_argument('--train_phage_fa', default="",   type=str, required=True, help='Host species list')
	parser.add_argument('--train_host_gold', default="",  type=str, required=True, help='Host species list')
	parser.add_argument('--valid_phage_fa', default="",   type=str, required=True, help='Host species list')
	parser.add_argument('--valid_host_gold', default="",  type=str, required=True, help='Host species list')


	
	args = parser.parse_args()

	data_set=[args.host_fa, args.host_list, args.train_phage_fa, args.train_host_gold, args.valid_phage_fa, args.valid_host_gold]

	# model train
	train(args.model, data_set, args.model_dir, args.kmer,args.margin, args.batch_size, \
		args.lr, args.epoch, args.device, args.workers)


