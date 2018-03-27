python ./tools/train_net.py \
	--gpu 0 \
	--solver models/pvanet/example_train/solver.prototxt \
	--weights models/pvanet/pva9.1/pva9.1_pretrained_no_fc6.caffemodel \
	--iters 1000000 \
	--cfg models/pvanet/cfgs/train.yml \
	--imdb voc_2012_trainval $@ > log 2>&1 &
