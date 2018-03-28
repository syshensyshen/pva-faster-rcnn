python ./tools/test_net.py \
	--net output/faster_rcnn_pvanet/voc_2007_trainval/FPN-lite_frcnn_iter_60000.caffemodel \
	--def ./models/pvanet/example_train/FPN-lite-test.prototxt \
	--cfg ./models/pvanet/cfgs/submit_1019.yml \
	--gpu 0\
	--imdb voc_2007_test
