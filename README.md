# Contrastive Learning for Phage-host interaction (CL4PHI)

![](figures/pipeline.png)

## Enviroments and Package dependency

- python 3
- Pytorch 1.11 
- pyfaidx


## Datasets and Pre-trained models

We store related data and model on Google cloud:
https://drive.google.com/drive/folders/1GUlI9h24pANmcq_7qULDQxz2cYtXntY2?usp=sharing

### Dataset

We provided processed datasets of the benchmark datasets used in [DeepHost](https://github.com/deepomicslab/DeepHost/tree/master/data) and [CHERRY](https://github.com/KennthShang/CHERRY/tree/main/Interactiondata).

Phage fasta files and gold standard of each data split can be found:
- **DeepHost benchmark dataset**
  - https://drive.google.com/drive/folders/1bC-YeEc_72ZvjRH1CxKZJ2CLbtg6jQ7W?usp=share_link

- **CHERRY benchmark dataset**
  - https://drive.google.com/drive/folders/1RBT-zIXZEc_6vdPVWG2lXJ1SRc7fastl?usp=share_link

- **Hosts fasta files** used in training and testing CHERRY models for the two benchmark datasets are stored in 
  - https://drive.google.com/drive/folders/1FeoiGM_yt2r4Kosn0I2EApEdGxwSkHU0?usp=share_link



### Trained models

We provided trained models of CL4PHI, DeepHost and CHERRY trained on each data split
under the fold [/trained_models](https://drive.google.com/drive/folders/1hnvj7gbJ1kpJ3uGegmqGB-mF7y_B71k3?usp=share_link)


## Running thde code

### Training
```
lr=1e-3
epoch=150
batch_size=32
margin=1

model_save_path="model_save_path/" 
device="cuda:0"  
CODE="code/train_cl.py"

kmer=6
model="CNN"

model_info="CL_model_margin-${margin}-epoch-${epoch}" 

# host data
host_fa="data/CHERRY_benchmark_datasplit/cherry_host.fasta"
host_list="data/CHERRY_benchmark_datasplit/species.txt"

# phage data
train_phage_fa="data/CHERRY_benchmark_datasplit/CHERRY_train.fasta"
train_host_gold="data/CHERRY_benchmark_datasplit/CHERRY_y_train.csv"
valid_phage_fa="data/CHERRY_benchmark_datasplit/CHERRY_val.fasta"
valid_host_gold="data/CHERRY_benchmark_datasplit/CHERRY_y_val.csv"

python $CODE --model $model --model_dir $model_save_path/${model_info}.pth --kmer $kmer --margin $margin \
	--host_fa $host_fa --host_list $host_list \
	--train_phage_fa $train_phage_fa  --train_host_gold $train_host_gold \
	--valid_phage_fa $valid_phage_fa  --valid_host_gold  $valid_host_gold \
	--device $device --lr $lr --epoch $epoch --batch_size $batch_size 
```


### Prediction

```
model_file="model/CL4PHI/DeepHostDATA_CL_CNN_kmer-6_lr-1e-3_batch-32_margin-1.pth"
OUTPUT="results/CL4PHI_pred_results.txt"

# host data
host_fa="data/CHERRY_benchmark_datasplit/cherry_host.fasta"
host_list="data/CHERRY_benchmark_datasplit/species.txt"

# test data
test_phage_fa="data/CHERRY_benchmark_datasplit/CHERRY_test.fasta"


python code/eval.py --model "CNN" --model_dir $model_file \
 --host_fa $host_fa --host_list $host_list \
 --test_phage_fa $test_phage_fa \
 --kmer $kmer --device $device > $OUTPUT
```

## Update
2024/09/11  Comment verbose printing in eval.py and add an option to use the learned BatchNorm statistics from training data (--use_train_bn). 

2024/08/06  The text output format bug has been fixed (line 161 in eval.py, with no impact on the evaluation of benchmark datasets). 
