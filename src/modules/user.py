#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import imp
import yaml
import json
import time
from PIL import Image
import __init__ as booger
import collections
import copy
import cv2
import os
import numpy as np

from modules.segmentator import *
from modules.trainer import *
from postproc.KNN import KNN

from aimet_torch.quantsim import QuantizationSimModel, load_encodings_to_sim
from aimet_torch.model_preparer import prepare_model
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.auto_quant import AutoQuant
from aimet_common.defs import QuantScheme 
import aimet_torch



class User():
  def __init__(self, ARCH, DATA, datadir, logdir, modeldir,cfg,quantized=False):
    # parameters
    self.ARCH = ARCH
    self.DATA = DATA
    self.datadir = datadir
    self.logdir = logdir
    self.modeldir = modeldir
    self.quantized=quantized
    self.cfg=cfg
    
    # get the data
    parserModule = imp.load_source("parserModule",
                                  'src/dataset/' +
                                   self.DATA["name"] + '/parser.py')
    self.parser = parserModule.Parser(root=self.datadir,
                                      train_sequences=self.DATA["split"]["train"],
                                      valid_sequences=self.DATA["split"]["valid"],
                                      test_sequences=self.DATA["split"]["test"],
                                      labels=self.DATA["labels"],
                                      color_map=self.DATA["color_map"],
                                      learning_map=self.DATA["learning_map"],
                                      learning_map_inv=self.DATA["learning_map_inv"],
                                      sensor=self.ARCH["dataset"]["sensor"],
                                      max_points=self.ARCH["dataset"]["max_points"],
                                      batch_size=1,
                                      workers=self.ARCH["train"]["workers"],
                                      gt=True,
                                      shuffle_train=False)
    
    self.post = None
    if self.ARCH["post"]["KNN"]["use"]:
      self.post = KNN(self.ARCH["post"]["KNN"]["params"],
                      self.parser.get_n_classes())
    
    # GPU?
    self.gpu = False
    
    # concatenate the encoder and the head
    with torch.no_grad():
      self.model = Segmentator(self.ARCH,
                               self.parser.get_n_classes(),
                               self.modeldir)
      self.model_single = self.model
      # self.device = torch.device("cpu")
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      print("Infering in device: ", self.device)
      if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        cudnn.benchmark = True
        cudnn.fastest = True
        self.gpu = True
        self.model.cuda()

      if self.quantized:
          if os.path.exists(self.cfg):
            with open(self.cfg) as f_in:
                self.cfg = json.load(f_in)
          self.input_shape=self.cfg['input_shape']
          self.dummy_input = torch.rand(self.input_shape, device=self.device)

          from aimet_torch.model_validator.model_validator import ModelValidator
          ModelValidator.validate_model(self.model, model_input=self.dummy_input)
          self.model = prepare_model(self.model)
          ModelValidator.validate_model(self.model, model_input=self.dummy_input)

          
          if "cle" in self.cfg["optimization_config"]["quantization_configuration"]["techniques"]:
            print("CLE...........")
            self.input_shape=(1,5,64,2048)
            equalize_model(self.model, self.input_shape)
              
          if "bn" in self.cfg["optimization_config"]["quantization_configuration"]["techniques"]:
            print("BN...........")
            fold_all_batch_norms(self.model, self.input_shape)
          
          dataloader=self.parser.get_train_set()
          if "adaround" in self.cfg["optimization_config"]["quantization_configuration"]["techniques"]:
                print("Adaround...........")
                
                params = AdaroundParameters(data_loader=dataloader, num_batches=1,default_num_iterations=1)
                
                self.model = Adaround.apply_adaround(self.model,
                                                    self.dummy_input,
                                                    params,
                                                    path=self.cfg['exports_path'],
                                                    filename_prefix='Adaround',
                                                    default_param_bw=8,
                                                    default_quant_scheme="tf_enhanced")
               
          kwargs = {
              "quant_scheme": QuantScheme.training_range_learning_with_tf_init,
              "default_param_bw": self.cfg["optimization_config"][
                  "quantization_configuration"
              ]["param_bw"],
              "default_output_bw": self.cfg["optimization_config"][
                  "quantization_configuration"
              ]["output_bw"], 
              "dummy_input": self.dummy_input,
          }
          print("QuantizationSIM")
          sim = QuantizationSimModel(self.model, **kwargs)
          if "adaround" in self.cfg["optimization_config"]["quantization_configuration"]["techniques"]:
              sim.set_and_freeze_param_encodings(encoding_path=self.cfg['exports_path']+'/Adaround.encodings')
              print("set_and_freeze_param_encodings finished!") 
          sim.compute_encodings(self.infer_subset, forward_pass_callback_args=self.parser.get_valid_set())
          self.model=sim.model
          self.sim=sim
          sim.export(path=self.cfg['exports_path'], filename_prefix=self.cfg['exports_name'], dummy_input=self.dummy_input.cpu(),onnx_export_args=(aimet_torch.onnx_utils.OnnxExportApiArgs (opset_version=11)))
        
      self.model.eval()
    

  

  def infer(self):
    # do train set
    if self.quantized:
      if self.cfg['qat']:
        print("QAT...........")
        trainer = Trainer(self.ARCH, self.DATA, self.datadir, "src/log", self.sim.model)
        self.sim.model = trainer.train()
        self.sim.export(path=self.cfg['exports_path'], filename_prefix=self.cfg['qat_name'], dummy_input=self.dummy_input.cpu(),onnx_export_args=(aimet_torch.onnx_utils.OnnxExportApiArgs (opset_version=11)))
        self.model= self.sim.model
        
    self.infer_subset(self.model,loader=self.parser.get_train_set(),
                      to_orig_fn=self.parser.to_original)

    # do valid set
    self.infer_subset(self.model,loader=self.parser.get_valid_set(),
                      to_orig_fn=self.parser.to_original)
    # do test set
    self.infer_subset(self.model,loader=self.parser.get_test_set(),
                      to_orig_fn=self.parser.to_original)

    print('Finished Infering')

    return

  def infer_subset(self,model, loader, to_orig_fn=None):
    to_orig_fn=self.parser.to_original
    # switch to evaluate mode 
    self.model.eval()
    

    # empty the cache to infer in high res
    if self.gpu:
      torch.cuda.empty_cache()

    with torch.no_grad():
      end = time.time()

      for i, (proj_in, proj_mask, _, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(loader):
        # first cut to rela size (batch size one allows it)
        # print(proj_in, proj_mask, path_seq, path_name, p_x, p_y, proj_range, unproj_range, npoints)
        p_x = p_x[0, :npoints]
        p_y = p_y[0, :npoints]
        proj_range = proj_range[0, :npoints]
        unproj_range = unproj_range[0, :npoints]
        path_seq = path_seq[0]
        path_name = path_name[0]

        if self.gpu:
          proj_in = proj_in.cuda()
          proj_mask = proj_mask.cuda()
          p_x = p_x.cuda()
          p_y = p_y.cuda()
          if self.post:
            proj_range = proj_range.cuda()
            unproj_range = unproj_range.cuda()

        # compute output
        proj_output = model(proj_in, proj_mask)
        proj_argmax = proj_output[0].argmax(dim=0)

        if self.post:
          # knn postproc
          unproj_argmax = self.post(proj_range,
                                    unproj_range,
                                    proj_argmax,
                                    p_x,
                                    p_y)
        else:
          # put in original pointcloud using indexes
          unproj_argmax = proj_argmax[p_y, p_x]

        # measure elapsed time
        if torch.cuda.is_available():
          torch.cuda.synchronize()

        # print("Infered seq", path_seq, "scan", path_name,
        #       "in", time.time() - end, "sec")
        end = time.time()

        # save scan
        # get the first scan in batch and project scan
        pred_np = unproj_argmax.cpu().numpy()
        pred_np = pred_np.reshape((-1)).astype(np.int32)

        # map to original label
        pred_np = to_orig_fn(pred_np)

        # save scan
        path = os.path.join(self.logdir, "sequences",
                            path_seq, "predictions", path_name)
        pred_np.tofile(path)

  