#!/usr/bin/env python

from __future__ import print_function
from __future__ import division

import os
import numpy as np
from optparse import OptionParser

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import models
from dataset import VideoFeatDataset as dset
from tools.config_tools import Config
from tools import utils


parser = OptionParser()
parser.add_option('--config',
                  type=str,
                  help="evaluation configuration",
                  default="./configs/test_config.yaml")

(opts, args) = parser.parse_args()
assert isinstance(opts, object)
opt = Config(opts.config)
print(opt)

if opt.checkpoint_folder is None:
    opt.checkpoint_folder = 'checkpoints'

test_video_dataset = dset(opt.data_dir, opt.video_flist, which_feat='vfeat')
test_audio_dataset = dset(opt.data_dir, opt.audio_flist, which_feat='afeat')

print('number of test samples is: {0}'.format(len(test_video_dataset)))
print('finished loading data')


if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with \"cuda: True\"")
else:
    if int(opt.ngpu) == 1:
        print('so we use gpu 1 for testing')
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
        cudnn.benchmark = True
        print('setting gpu on gpuid {0}'.format(opt.gpu_id))

# test function for metric learning
def test(video_loader, audio_loader, model, opt):
    """
    train for one epoch on the training set
    """
    # training mode
    model.eval()  #Sets the module in evaluation mode.

    sim_mat = []
    right = 0
    for i, vfeat in enumerate(video_loader):
        for j, afeat in enumerate(audio_loader):
            # transpose feats
            #vfeat = vfeat.transpose(2,1)
            #afeat = afeat.transpose(2,1)

            # shuffling the index orders
            if i == j:
                sim_mat = []
                right = 0
                bz = vfeat.size()[0]
                print(bz)
                for k in np.arange(bz):
                    #print(k)
                    cur_vfeat = vfeat[k].clone()  #1*120*1024
                    cur_vfeats = cur_vfeat.repeat(bz, 1, 1)  #bz*120*1024
                    vfeat_var = Variable(cur_vfeats)
                    afeat_var = Variable(afeat)

                    if opt.cuda:
                        vfeat_var = vfeat_var.cuda()
                        afeat_var = afeat_var.cuda()

                    cur_sim = model(vfeat_var, afeat_var)  #bz*1
                    if k == 0:
                        simmat = cur_sim.clone()
                    else:
                        simmat = torch.cat((simmat, cur_sim), 1)  #bz*bz
                sorted, indices = torch.sort(simmat, 0)   #bz*bz
                np_indices = indices.cpu().data.numpy()
                topk = np_indices[:opt.topk,:]
                #print(topk)
                for k in np.arange(bz):
                    order = topk[:,k]
                    if k in order:
                        right = right + 1
                print('The similarity matrix: \n {}'.format(simmat))
                #print(simmat.size())
                #print(indices)
                print('Testing accuracy (top{}): {:.3f}'.format(opt.topk, right/bz))
                #txtName = "accuracy12.txt"
                #f = open(txtName, "a+")
                #new_context = '{:.3f}\n'.format(right/bz)
                #f.write(new_context)
                #f.close()

def main():
    global opt
    # test data loader
    test_video_loader = torch.utils.data.DataLoader(test_video_dataset, batch_size=opt.batchSize,
                                     shuffle=False, num_workers=int(opt.workers))
    test_audio_loader = torch.utils.data.DataLoader(test_audio_dataset, batch_size=opt.batchSize,
                                     shuffle=False, num_workers=int(opt.workers))

    # create model
    model = models.VAMetric()

    #for k1 in np.arange(1,30):
    if opt.init_model != '':
        model.load_state_dict(torch.load(opt.init_model))

    if opt.cuda:
        print('shift model to GPU .. ')
        model = model.cuda()

    test(test_video_loader, test_audio_loader, model, opt)
if __name__ == '__main__':
    main()
