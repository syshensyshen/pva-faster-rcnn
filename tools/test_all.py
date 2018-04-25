#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import _init_paths
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys
import numpy as np

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--netdir', dest='netdir',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--netname', dest='netname',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--netstart', dest='netstart',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--netend', dest='netend',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--accuOutput', dest='accuOutput',
                        help='model to test',
                        default=None, type=str)
    #parser.add_argument('--cfg', dest='cfg_file',
    #                    help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--vis', dest='vis', help='visualize detections',
                        action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    #if args.cfg_file is not None:
    #    cfg_from_file(args.cfg_file)
    #if args.set_cfgs is not None:
    #    cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id
    cfg.TEST.HAS_RPN = True
    cfg.TEST.SCALE_MULTIPLE_OF = 32
    cfg.TEST.MAX_SIZE = 2000
    scales = [512, 544, 576, 608, 640, 672, 704, 736]
                                #768, 800, 832, 864, 896, 928, 960, 992, 1024], dtype=np.int32)

    cfg.TEST.BBOX_VOTE = True
    cfg.TEST.BBOX_VOTE_N_WEIGHTED_SCORE = 5
    cfg.TEST.BBOX_VOTE_WEIGHT_EMPTY = 0.3
    cfg.TEST.NMS = 0.4
    cfg.TEST.RPN_PRE_NMS_TOP_N = 12000
    cfg.TEST.RPN_POST_NMS_TOP_N = 200
    
    THRESH=0.4
    
    TEST_GAP = 5000
    
    print('Using config:')
    pprint.pprint(cfg)
    
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    fobj=open(args.accuOutput,'w')
    for scale in scales:
        cfg.TEST.SCALES = (scale,)
        for i in range(int(args.netstart)/TEST_GAP,int(args.netend)/TEST_GAP+1):
            while not os.path.exists(args.netdir) and args.wait:
                print('Waiting for {} to exist...'.format(args.netdir))
                time.sleep(10)
            modelfile=args.netdir+'/'+args.netname+'_iter_'+str(TEST_GAP*i)+'.caffemodel'
            net = caffe.Net(args.prototxt, modelfile, caffe.TEST)
            
            net.name = os.path.splitext(os.path.basename(modelfile))[0]
            
    
            imdb = get_imdb(args.imdb_name)
            imdb.competition_mode(args.comp_mode)
            if not cfg.TEST.HAS_RPN:
                imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)
            ap=test_net(net, imdb, max_per_image=args.max_per_image, thresh=THRESH, vis=args.vis)
            fobj.writelines(args.netname+'_iter_'+str(TEST_GAP*i)+'.caffemodel: ')
            fobj.writelines(' ' + str(scale) + ' ')
            fobj.writelines(str(ap))
            fobj.writelines('\n')
    fobj.close()


