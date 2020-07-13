"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datasets
import numpy as np
from model.utils.config import cfg
from datasets.factory2 import get_imdb
import PIL
import pdb

def prepare_roidb(imdb):
  """Enrich the imdb's roidb by adding some derived quantities that
  are useful for training. This function precomputes the maximum
  overlap, taken over ground-truth boxes, between each ROI and
  each ground-truth box. The class with maximum overlap is also
  recorded.
  """

  roidb = imdb.roidb
  if not (imdb.name.startswith('coco')):
    sizes = [PIL.Image.open(imdb.image_path_at(i)).size
         for i in range(imdb.num_images)]
         
  for i in range(len(imdb.image_index)):
    roidb[i]['img_id'] = imdb.image_id_at(i)
    roidb[i]['image'] = imdb.image_path_at(i)
    if not (imdb.name.startswith('coco')):
      roidb[i]['width'] = sizes[i][0]
      roidb[i]['height'] = sizes[i][1]
    # need gt_overlaps as a dense array for argmax
    gt_overlaps = roidb[i]['gt_overlaps'].toarray()
    # max overlap with gt over classes (columns)
    max_overlaps = gt_overlaps.max(axis=1)
    # gt class that had the max overlap
    max_classes = gt_overlaps.argmax(axis=1)
    roidb[i]['max_classes'] = max_classes
    roidb[i]['max_overlaps'] = max_overlaps
    # sanity checks
    # max overlap of 0 => class should be zero (background)
    zero_inds = np.where(max_overlaps == 0)[0]
    assert all(max_classes[zero_inds] == 0)
    # max overlap > 0 => class should not be zero (must be a fg class)
    nonzero_inds = np.where(max_overlaps > 0)[0]
    # assert all(max_classes[nonzero_inds] != 0)

def rank_roidb_ratio(roidb):
    # rank roidb based on the ratio between width and height.
    ratio_large = 2 # largest ratio to preserve.
    ratio_small = 0.5 # smallest ratio to preserve.    
    
    ratio_list = []
    for i in range(len(roidb)):
      width = roidb[i]['width']
      height = roidb[i]['height']
      ratio = width / float(height)

      if ratio > ratio_large:
        roidb[i]['need_crop'] = 1
        ratio = ratio_large
      elif ratio < ratio_small:
        roidb[i]['need_crop'] = 1
        ratio = ratio_small        
      else:
        roidb[i]['need_crop'] = 0

      ratio_list.append(ratio)

    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)
    return ratio_list[ratio_index], ratio_index

def filter_roidb(roidb):
    # filter the image without bounding box.
    print('before filtering, there are %d images...' % (len(roidb)))
    i = 0
    while i < len(roidb):
      if len(roidb[i]['boxes']) == 0:
        del roidb[i]
        i -= 1
      i += 1

    print('after filtering, there are %d images...' % (len(roidb)))
    return roidb

def test_rank_roidb_ratio(roidb, reserved):
    # rank roidb based on the ratio between width and height.
    ratio_large = 2 # largest ratio to preserve.
    ratio_small = 0.5 # smallest ratio to preserve.  
    

    # Image can show more than one time for test different category 
    ratio_list = []
    ratio_index = [] # image index reserved
    cat_list = [] # category list reserved
    for i in range(len(roidb)):
      width = roidb[i]['width']
      height = roidb[i]['height']
      ratio = width / float(height)

      if ratio > ratio_large:
        roidb[i]['need_crop'] = 1
        ratio = ratio_large
      elif ratio < ratio_small:
        roidb[i]['need_crop'] = 1
        ratio = ratio_small        
      else:
        roidb[i]['need_crop'] = 0


      for j in np.unique(roidb[i]['max_classes']):
        if j in reserved:
          ratio_list.append(ratio)
          ratio_index.append(i)
          cat_list.append(j)


    ratio_list = np.array(ratio_list)
    ratio_index = np.array(ratio_index)
    cat_list = np.array(cat_list)
    ratio_index = np.vstack((ratio_index,cat_list))

    return ratio_list, ratio_index

def combined_roidb(imdb_names, training=True, seen=1):
  """
  Combine multiple roidbs
  """

  def get_training_roidb(imdb, training):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED and training:
      print('Appending horizontally-flipped training examples...')
      # imdb.append_flipped_images()
      print('done')


    print('Preparing training data...')
    prepare_roidb(imdb)
    #ratio_index = rank_roidb_ratio(imdb)
    print('done')

    return imdb.roidb
  
  def get_roidb(imdb_name, training):
    imdb = get_imdb(imdb_name)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))

    imdb.filter(seen)

    roidb = get_training_roidb(imdb, training)



    return imdb, roidb, imdb.cat_data, imdb.inverse_list

  imdbs = []
  roidbs = []
  querys = []
  for s in imdb_names.split('+'):
    imdb, roidb, query, reserved = get_roidb(s, training)
    imdbs.append(imdb)
    roidbs.append(roidb)
    querys.append(query)
  imdb = imdbs[0]
  roidb = roidbs[0]
  query = querys[0]


  if len(roidbs) > 1 and training:
    for r in roidbs[1:]:
      roidb.extend(r)
    for r in range(len(querys[0])):
      query[r].extend(querys[1][r])
    tmp = get_imdb(imdb_names.split('+')[1])
    imdb = datasets.imdb.imdb(imdb_names, tmp.classes)

  if training:
    roidb = filter_roidb(roidb)
    ratio_list, ratio_index = rank_roidb_ratio(roidb)
  else:
    # Generate testing image, an image testing frequency(time) according to the reserved category
    ratio_list, ratio_index = test_rank_roidb_ratio(roidb, reserved)

  return imdb, roidb, ratio_list, ratio_index, query