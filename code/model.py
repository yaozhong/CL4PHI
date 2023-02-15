# preparing for the constrastive learning

import torch
from torch import nn

class ContrastiveLoss(torch.nn.Module):
	def __init__(self, margin=1.0):
		super(ContrastiveLoss, self).__init__()
		self.margin = margin

	def forward(self, x0, x1, y):
		diff = x0 - x1
		dist_sq = torch.sum(torch.pow(diff, 2), 1)
		dist = torch.sqrt(dist_sq)

		mdist = self.margin - dist
		dist = torch.clamp(mdist, min=0.0)
		loss = y * dist_sq + (1-y) * torch.pow(dist,2)
		loss = torch.sum(loss) / 2.0 / x0.size()[0]

		return loss


def distance(x1, x2, dist_type="euc"):

	if dist_type == "euc":
		dist = torch.cdist(x1,x2)**2

	if dist_type == "cos":
		cos = nn.CosineSimilarity(dim=1, eps=1e-6)
		dist = cos(x1, x2)

	return dist


# define the basic module
class cnn_module(nn.Module):
	def __init__(self, kernel_size=7, dr=0):
		super(cnn_module, self).__init__()
		self.conv1 = nn.Conv2d(1,64,kernel_size=kernel_size, stride=2)
		self.bn1 = nn.BatchNorm2d(64)
		self.conv2 = nn.Conv2d(64,128, kernel_size=kernel_size, stride=2)
		self.bn2 = nn.BatchNorm2d(128)
		self.relu = nn.ReLU()

		self.maxpool = nn.MaxPool2d(2)
		self.dropout = nn.Dropout(dr)
		
		self.fc1 = nn.Linear(4608, 512)
	

	def forward(self, x):
		x = self.bn1(self.relu(self.conv1(x)))
		x = self.bn2(self.relu(self.conv2(x)))
		x = self.maxpool(x)

		x = self.fc1(torch.flatten(x, 1))
	
		return x
