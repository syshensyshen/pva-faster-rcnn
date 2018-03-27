# --------------------------------------------------------
# Fast R-CNN with OHEM
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Abhinav Shrivastava
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr
import cv2
from fast_rcnn.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
from fast_rcnn.nms_wrapper import nms

def get_allrois_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)

    # Get the input image blob, formatted for caffe
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    blobs = {'data': im_blob}

    if cfg.TRAIN.HAS_RPN:
        # Now, build the region of interest and label blobs
        rois_blob = np.zeros((0, 5), dtype=np.float32)
        labels_blob = np.zeros((0), dtype=np.float32)
        bbox_targets_blob = np.zeros((0, 4 * num_classes), dtype=np.float32)
        bbox_inside_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)

        for im_i in xrange(num_images):
            labels, overlaps, im_rois, bbox_targets, bbox_inside_weights \
                = _all_rois(roidb[im_i], num_classes)

            # Add to RoIs blob
            rois = _project_im_rois(im_rois, im_scales[im_i])
            batch_ind = im_i * np.ones((rois.shape[0], 1))
            rois_blob_this_image = np.hstack((batch_ind, rois))
            rois_blob = np.vstack((rois_blob, rois_blob_this_image))

            # Add to labels, bbox targets, and bbox loss blobs
            labels_blob = np.hstack((labels_blob, labels))
            bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
            bbox_inside_blob = np.vstack((bbox_inside_blob, bbox_inside_weights))

        blobs['rois'] = rois_blob
        blobs['labels'] = labels_blob

        if True:#cfg.TRAIN.BBOX_REG:
            blobs['bbox_targets'] = bbox_targets_blob
            blobs['bbox_inside_weights'] = bbox_inside_blob
            blobs['bbox_outside_weights'] = \
                np.array(bbox_inside_blob > 0).astype(np.float32)


    return blobs


def get_ohem_minibatch(loss, rois, labels, bbox_targets=None,
                       bbox_inside_weights=None, bbox_outside_weights=None):
    """Given rois and their loss, construct a minibatch using OHEM."""
    loss = np.array(loss)

    if cfg.TRAIN.OHEM_USE_NMS:
        # Do NMS using loss for de-dup and diversity
        keep_inds = []
        nms_thresh = cfg.TRAIN.OHEM_NMS_THRESH
        source_img_ids = [roi[0] for roi in rois]
        for img_id in np.unique(source_img_ids):
            for label in np.unique(labels):# erease the same imdex of boundboxes
                sel_indx = np.where(np.logical_and(labels == label, \
                                    source_img_ids == img_id))[0]
                if not len(sel_indx):
                    continue
                boxes = np.concatenate((rois[sel_indx, 1:],
                        loss[sel_indx][:,np.newaxis]), axis=1).astype(np.float32)
                keep_inds.extend(sel_indx[nms(boxes, nms_thresh)])

        hard_keep_inds = select_hard_examples(loss[keep_inds])
        hard_inds = np.array(keep_inds)[hard_keep_inds]
    else:
        hard_inds = select_hard_examples(loss)

    blobs = {'rois_hard': rois[hard_inds, :].copy(),
             'labels_hard': labels[hard_inds].copy()}
    if bbox_targets is not None:
        #assert cfg.TRAIN.BBOX_REG
        blobs['bbox_targets_hard'] = bbox_targets[hard_inds, :].copy()
        blobs['bbox_inside_weights_hard'] = bbox_inside_weights[hard_inds, :].copy()
        blobs['bbox_outside_weights_hard'] = bbox_outside_weights[hard_inds, :].copy()

    return blobs

def select_hard_examples(loss):
    """Select hard rois."""
    # Sort and select top hard examples.
    sorted_indices = np.argsort(loss)[::-1]
    hard_keep_inds = sorted_indices[0:np.minimum(len(loss), 64)]
    # (explore more ways of selecting examples in this function; e.g., sampling)

    return hard_keep_inds

def _all_rois(roidb, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # label = class RoI has max overlap with
    labels = roidb['max_classes']
    overlaps = roidb['max_overlaps']
    rois = roidb['boxes']

    # To use custom cfg.TRAIN.BG_THRESH_LO, comment the following assert.
    assert cfg.TRAIN.BG_THRESH_LO == 0.0, \
        "OHEM works best with BG_THRESH_LO = 0.0 (current value is {}).".format(cfg.TRAIN.BG_THRESH_LO)

    # Select foreground (background) RoIs.
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    bg_inds = np.where(overlaps < cfg.TRAIN.BG_THRESH_HI)[0]

    # All RoIs.
    keep_inds = np.append(fg_inds, bg_inds)

    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[len(fg_inds):] = 0
    overlaps = overlaps[keep_inds]
    rois = rois[keep_inds]

    bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(
            roidb['bbox_targets'][keep_inds, :], num_classes)

    return labels, overlaps, rois, bbox_targets, bbox_inside_weights


