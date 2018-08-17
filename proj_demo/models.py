import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, dist, label):
        dist = torch.squeeze(dist)
        #print(dist)
        loss = torch.mean((1-label) * torch.pow(dist, 2) +
                (label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))
        return loss

class VAMetric(nn.Module):
	def __init__(self):
		super(VAMetric, self).__init__()
		self.conv1 = nn.Conv2d(1,300,(1,1152))
		self.lstm1 = nn.LSTMCell(300, 128)
		self.lstm2 = nn.LSTMCell(128, 128)
		self.fc1 = nn.Linear(128, 1)

	def init_params(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform(m.weight)
				nn.init.constant(m.bias, 0)

	def forward(self, vfeat, afeat):
		#vfeat 128*120*1024
		#afeat 128*120*128
		feat = torch.cat((vfeat,afeat), 2) #128*120*1152
		feat = torch.unsqueeze(feat, 1) #128*1*120*1152
		#print(feat.size())

		feat1 = F.relu(self.conv1(feat)) #128*300*120*1
		feat1 = torch.squeeze(feat1) #128*300*120

		h_t1 = Variable(torch.zeros(feat1.size(0), 128).float(), requires_grad=False)
		c_t1 = Variable(torch.zeros(feat1.size(0), 128).float(), requires_grad=False)
		h_t2 = Variable(torch.zeros(feat1.size(0), 128).float(), requires_grad=False)
		c_t2 = Variable(torch.zeros(feat1.size(0), 128).float(), requires_grad=False)

		h_t1 = h_t1.cuda()
		c_t1 = c_t1.cuda()
		h_t2 = h_t2.cuda()
		c_t2 = c_t2.cuda()

		for _, feat_t in enumerate(feat1.chunk(feat1.size(2), dim=2)):
			feat_t = torch.squeeze(feat_t)
			h_t1, c_t1 = self.lstm1(feat_t, (h_t1, c_t1))
			h_t2, c_t2 = self.lstm2(h_t1, (h_t2, c_t2))

		#print(h_t2.size())
		pair = F.relu(self.fc1(h_t2)) #128*1
		return pair
