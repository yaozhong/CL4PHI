# Contrastive Learning for Phage-host interaction (CL4PHI)

![](figure/pipeline.png)


## Datasets

We store related data and model on Google cloud:
https://drive.google.com/drive/folders/1GUlI9h24pANmcq_7qULDQxz2cYtXntY2?usp=sharing

### Dataset

### Trained models



## Running thde code
### Training

lr=1e-3
epoch=10
batch_size=32
margin=1

model_save_path="results/"
device="cuda:0"
CODE="code/train_cl.py"

kmer=6
model="CNN"
model_info="CL_model_margin-{$margin}-epoch-${epoch}"

host_fa="data/CHERRY_benchmark_datasplit/cherry_host.fasta"
host_list="data/CHERRY_benchmark_datasplit/species.txt"
train_phage_fa="data/CHERRY_benchmark_datasplit/CHERRY_train.fasta"
train_host_gold="data/CHERRY_benchmark_datasplit/CHERRY_y_train.csv"
valid_phage_fa="data/CHERRY_benchmark_datasplit/CHERRY_val.fasta"
valid_host_gold="data/CHERRY_benchmark_datasplit/CHERRY_y_val.csv"

python $CODE --model $model --model_dir $model_save_path/${model_info}.pth --kmer $kmer --margin $margin \
	--host_fa $host_fa --host_list $host_list \
	--train_phage_fa $train_phage_fa  --train_host_gold $train_host_gold \
	--valid_phage_fa $valid_phage_fa  --valid_host_gold  $valid_host_gold \
	--device $device --lr $lr --epoch $epoch --batch_size $batch_size 



### Prediction

model_file="model/CL4PHI/DeepHostDATA_CL_CNN_kmer-6_lr-1e-3_batch-32_margin-1.pth"
OUTPUT="results/CL4PHI_pred_results.txt"
CODE="code/eval.py"

host_fa="data/CHERRY_benchmark_datasplit/cherry_host.fasta"
host_list="data/CHERRY_benchmark_datasplit/species.txt"
test_phage_fa="data/CHERRY_benchmark_datasplit/CHERRY_test.fasta"
test_host_gold="data/CHERRY_benchmark_datasplit/CHERRY_y_test.csv"

python $CODE --model "CNN" --model_dir $model_file \
 --host_fa $host_fa --host_list $host_list \
 --test_phage_fa $test_phage_fa  --test_host_gold  $test_host_gold \
 --kmer $kmer --device $device 



