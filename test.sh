python ./tools/test_net.py \
	--net output/faster_rcnn_pvanet/voc_2012_trainval/mobilenet-v1_frcnn_iter_100000.caffemodel \
	--def models/pvanet/example_train/mobilenet-v1-test.prototxt \
	--cfg models/pvanet/cfgs/submit_1019.yml \
	--gpu 0 \
	--imdb voc_2007_test
