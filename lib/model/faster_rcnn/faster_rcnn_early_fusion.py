import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg

from model.rpn.rpn import _RPN
import random

from model.roi_layers import ROIAlign, ROIPool

# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import RoIAlignAvg

from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
from model.utils.net_utils import *
from torch.autograd import Function


class attention(nn.Module):
    def __init__(self, inplanes):
        super(attention, self).__init__()
        self.in_channels = inplanes
        self.inter_channels = None

        if self.inter_channels is None:
            self.inter_channels = self.in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        conv_nd = nn.Conv2d
        self.maxpool_2d = nn.MaxPool2d(self.in_channels)

        bn = nn.BatchNorm2d

        self.theta_sketch = nn.Sequential(conv_nd(in_channels=self.in_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0, bias=True),
                         bn(self.in_channels),
                         nn.ReLU(),
                         conv_nd(in_channels=self.in_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0, bias=False))

        self.theta_image = nn.Sequential(conv_nd(in_channels=self.in_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0, bias=True),
                         bn(self.in_channels),
                         nn.ReLU(),
                         conv_nd(in_channels=self.in_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0, bias=False))

        self.op = nn.Sequential(conv_nd(in_channels=1, out_channels=1,
                         kernel_size=1, stride=1, padding=0, bias=True))

        nn.init.xavier_uniform_(self.theta_sketch[0].weight)
        nn.init.xavier_uniform_(self.theta_image[0].weight)
        nn.init.xavier_uniform_(self.op[0].weight)

    def forward(self, image_feats, sketch_feats):
        img_feats = self.theta_image(image_feats)
        sketch_feats_ = self.theta_sketch(sketch_feats)

        batch_size, n_channels, w, h = img_feats.shape
        image_feats = image_feats.view(batch_size, n_channels, -1)

        sketch_feats_ = sketch_feats_.view(batch_size, n_channels, -1)
        sketch_max_feats, _ = torch.max(sketch_feats_, dim=2)

        img_feats = img_feats.view(batch_size, n_channels, -1)


        attention_feats = torch.bmm(sketch_max_feats.unsqueeze(1), img_feats)

        sketch_max_feats = sketch_max_feats.unsqueeze(2).expand_as(img_feats)
        attention_feats = attention_feats.view(batch_size, 1, w, h)

        attention_feats = self.op(attention_feats)

        attention_feats = attention_feats.view(batch_size, 1, -1)
        attention_feats = attention_feats/256
        attention_map = attention_feats.clone().view(batch_size,1,w,h)


        attention_feats = image_feats*attention_feats.expand_as(image_feats)
        attention_feats = attention_feats.view(batch_size, n_channels, w, h)

        return attention_feats, sketch_feats,attention_map




class attention_early_fusion_multi_query(nn.Module):
    def __init__(self, inplanes):
        super(attention_early_fusion_multi_query, self).__init__()
        self.in_channels = inplanes
        self.inter_channels = None

        if self.inter_channels is None:
            self.inter_channels = self.in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        conv_nd = nn.Conv2d
        self.maxpool_2d = nn.MaxPool2d(self.in_channels)

        bn = nn.BatchNorm2d


        self.theta_sketch = nn.Sequential(conv_nd(in_channels=self.in_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0, bias=True),
                         bn(self.in_channels),
                         nn.ReLU(),
                         conv_nd(in_channels=self.in_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0, bias=False))

        self.theta_image = nn.Sequential(conv_nd(in_channels=self.in_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0, bias=True),
                         bn(self.in_channels),
                         nn.ReLU(),
                         conv_nd(in_channels=self.in_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0, bias=False))

        self.op = nn.Sequential(conv_nd(in_channels=1, out_channels=1,
                         kernel_size=1, stride=1, padding=0, bias=True))

        self.fusion_layer = nn.Sequential(conv_nd(in_channels=3, out_channels=1,
                         kernel_size=1, stride=1, padding=0, bias=True))

        self.sketch_fusion = nn.Sequential(conv_nd(in_channels=self.in_channels*3, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0, bias=False))

        nn.init.xavier_uniform_(self.theta_sketch[0].weight)
        nn.init.xavier_uniform_(self.theta_image[0].weight)
        nn.init.xavier_uniform_(self.op[0].weight)
        nn.init.xavier_uniform_(self.fusion_layer[0].weight)

    def forward(self, image_feats, sketch_feats_1, sketch_feats_2, sketch_feats_3):
        img_feats = self.theta_image(image_feats)
        sketch_feats__1 = self.theta_sketch(sketch_feats_1)
        sketch_feats__2 = self.theta_sketch(sketch_feats_2)
        sketch_feats__3 = self.theta_sketch(sketch_feats_3)

        sketches = self.sketch_fusion(torch.cat([sketch_feats_1, sketch_feats_2, sketch_feats_3], dim=1))


        batch_size, n_channels, w, h = img_feats.shape
        image_feats = image_feats.view(batch_size, n_channels, -1)

        sketch_feats__1 = sketch_feats__1.view(batch_size, n_channels, -1)
        sketch_max_feats_1, _ = torch.max(sketch_feats__1, dim=2)

        sketch_feats__2 = sketch_feats__2.view(batch_size, n_channels, -1)
        sketch_max_feats_2, _ = torch.max(sketch_feats__2, dim=2)

        sketch_feats__3 = sketch_feats__3.view(batch_size, n_channels, -1)
        sketch_max_feats_3, _ = torch.max(sketch_feats__3, dim=2)


        img_feats = img_feats.view(batch_size, n_channels, -1)


        attention_feats_1 = torch.bmm(sketch_max_feats_1.unsqueeze(1), img_feats)
        attention_feats_2 = torch.bmm(sketch_max_feats_2.unsqueeze(1), img_feats)
        attention_feats_3 = torch.bmm(sketch_max_feats_3.unsqueeze(1), img_feats)


        attention_feats_1 = attention_feats_1.view(batch_size, 1, w, h)
        attention_feats_2 = attention_feats_2.view(batch_size, 1, w, h)
        attention_feats_3 = attention_feats_3.view(batch_size, 1, w, h)

        attention_feats_1 = self.op(attention_feats_1)
        attention_feats_2 = self.op(attention_feats_2)
        attention_feats_3 = self.op(attention_feats_3)

        attention_map_1 = attention_feats_1.clone()
        attention_map_2 = attention_feats_2.clone()
        attention_map_3 = attention_feats_3.clone()


        attention_feats = torch.cat([attention_feats_1, attention_feats_2, attention_feats_3], dim=1)
        final_atten_map = attention_feats.clone()

        # attention_feats , _ = torch.max(attention_feats, dim=1)
        attention_feats = attention_feats.mean(1)

        attention_feats = attention_feats.view(batch_size, 1, -1)
        attention_feats = attention_feats/256
        attention_feats = image_feats*attention_feats.expand_as(image_feats)
        attention_feats = attention_feats.view(batch_size, n_channels, w, h)

        return attention_feats, sketches, [attention_map_1, attention_feats_2, attention_feats_3, final_atten_map]


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic, model_type="attention", fusion='query'):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        conv_nd = nn.Conv2d
        self.fusion = fusion

        if fusion == 'query':
            self.attention_net = attention(self.dout_base_model)
        elif fusion == 'attention':
            self.attention_net = attention_early_fusion_multi_query(self.dout_base_model)


        self.projection = conv_nd(in_channels=1024*2, out_channels=1024,
                          kernel_size=1, stride=1, padding=0, bias=False)

        nn.init.xavier_uniform_(self.projection.weight)



        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)
        self.triplet_loss = torch.nn.MarginRankingLoss(margin=cfg.TRAIN.MARGIN)

    def forward(self, im_data, query, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        detect_feat = self.RCNN_base(im_data)
        query_feat_1 = self.RCNN_base_sketch(query.permute(1,0,2,3,4)[0])
        query_feat_2 = self.RCNN_base_sketch(query.permute(1,0,2,3,4)[1])
        query_feat_3 = self.RCNN_base_sketch(query.permute(1,0,2,3,4)[2])

        comulative_query = torch.cat([query_feat_1.unsqueeze(4),query_feat_2.unsqueeze(4), query_feat_3.unsqueeze(4)], dim=4)

        comulative_query,_ = torch.max(comulative_query, dim=4)

        domain_loss = 0



        # rpn_feat, act_feat, act_aim, c_weight = self.match_net(detect_feat, query_feat)
        c_weight = None
        if self.fusion == 'query':
            act_feat, act_aim, attention_map = self.attention_net(detect_feat, query_feat_1, query_feat_2, query_feat_3)
        elif self.fusion == 'attention':
            act_feat, act_aim, attention_map = self.attention_net(detect_feat, comulative_query)
        act_aim = comulative_query
        # c_weight = None

        act_feat = torch.cat([act_feat, detect_feat], dim=1)
        act_feat = self.projection(act_feat)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(act_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            margin_loss = 0
            rpn_loss_bbox = 0
            score_label = None

        rois = Variable(rois)

        # do roi pooling based on predicted rois
        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(act_feat, rois.view(-1, 5))

        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(act_feat, rois.view(-1,5))


        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)
        # pooled_feat_atten = self._head_to_tail(pooled_feat_atten)
        query_feat  = self._head_to_tail(act_aim)
        batch_size = query_feat.shape[0]


        # domain_loss = 0

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)


        pooled_feat = pooled_feat.view(batch_size, rois.size(1), -1)
        query_feat = query_feat.unsqueeze(1).repeat(1,rois.size(1),1)


        pooled_feat = torch.cat((pooled_feat.expand_as(query_feat),query_feat), dim=2).view(-1, 4096)

        # compute object classification probability
        score = self.RCNN_cls_score(pooled_feat)

        score_prob = F.softmax(score, 1)[:, 1]

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0


        if self.training:
        # if True:
            # classification loss
            score_label = rois_label.view(batch_size, -1).float()
            gt_map = torch.abs(score_label.unsqueeze(1)-score_label.unsqueeze(-1))

            score_prob = score_prob.view(batch_size, -1)
            pr_map = torch.abs(score_prob.unsqueeze(1)-score_prob.unsqueeze(-1))
            target = -((gt_map-1)**2) + gt_map

            RCNN_loss_cls = F.cross_entropy(score, rois_label)

            margin_loss = 3 * self.triplet_loss(pr_map, gt_map, target)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = score_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
        # c_weight = None

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, margin_loss, RCNN_loss_bbox, rois_label, c_weight, domain_loss, attention_map

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score[0], 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score[1], 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)


    def create_architecture(self):
        self._init_modules()
        self._init_weights()
