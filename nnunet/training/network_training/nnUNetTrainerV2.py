#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
# GK: change for target
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation, get_moreDA_augmentation_target
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *
# GK: change for target
import torch.optim as optim
import torch.nn.functional as F
from nnunet.domain_adaptation.discriminator import get_fc_discriminator
from nnunet.domain_adaptation.utils import prob_2_entropy, bce_loss, adjust_learning_rate_discriminator

class nnUNetTrainerV2(nnUNetTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, target=None):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16, target)
        # GK: Change the max_num_epochs here
        self.max_num_epochs = 1000
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None
        self.pin_memory = True
        # GK: change for target
        if target:
            self.learning_rate_D = 1e-4
            self.source_label = 0
            self.target_label = 1
            self.input_size_target = (512,512)
            self.LAMBDA_ADV_MAIN = 1.0
            self.LAMBDA_ADV_AUX = 0.1
            # 250000 is used in advent, also matches with nnUNet as 1000 epochs, and 250 batches per epoch with batch size of 1
            self.max_iters = 250000 
            self.power = 0.9

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                # GK: change for target: if self.target is set then the self.dl_target is also created
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                )
                # GK: change for target
                if self.target:
                    self.target_gen = get_moreDA_augmentation_target(
                        self.dl_target,
                        self.data_aug_params[
                            'patch_size_for_spatialtransform'],
                        self.data_aug_params,
                        deep_supervision_scales=self.deep_supervision_scales,
                        pin_memory=self.pin_memory,
                        use_nondetMultiThreadedAugmenter=False
                    )

                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
                # GK: change for target
                if self.target:
                    self.print_to_log_file("TARGET KEYS:\n %s" % (str(self.dataset_target.keys())),
                                        also_print_to_console=False)

            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()
            # GK: change for target: discriminator network
            if self.target:
                self.initial_discriminator_network_and_optimizer()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        if torch.cuda.is_available():
            # GK: here the network is send to GPU
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = None

    def initial_discriminator_network_and_optimizer(self):
        # discriminator network
        # feature-level
        print("GK: num_classes: (should be 3):", self.num_classes)
        self.d_aux = get_fc_discriminator(num_classes=self.num_classes) # defined as: self.num_classes = plans['num_classes'] + 1 
        
        # seg maps, i.e. output, level
        self.d_main = get_fc_discriminator(num_classes=self.num_classes)
        
        if torch.cuda.is_available():
            self.d_aux.cuda()
            self.d_main.cuda()

        self.optimizer_d_aux = optim.Adam(self.d_aux.parameters(), lr=self.learning_rate_D,
                                betas=(0.9, 0.99))
        self.optimizer_d_main = optim.Adam(self.d_main.parameters(), lr=self.learning_rate_D,
                                betas=(0.9, 0.99))

        self.interp_target_256 = nn.Upsample(size=(self.input_size_target[1], self.input_size_target[0]), mode='bilinear',
                            align_corners=True)


    def run_online_evaluation(self, output, target):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """
        target = target[0]
        output = output[0]
        return super().run_online_evaluation(output, target)

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)

        self.network.do_ds = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().predict_preprocessed_data_return_seg_and_softmax(data,
                                                                       do_mirroring=do_mirroring,
                                                                       mirror_axes=mirror_axes,
                                                                       use_sliding_window=use_sliding_window,
                                                                       step_size=step_size, use_gaussian=use_gaussian,
                                                                       pad_border_mode=pad_border_mode,
                                                                       pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                       verbose=verbose,
                                                                       mixed_precision=mixed_precision)
        self.network.do_ds = ds
        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        # GK: GET THE BATCH

        # GK: don't get confuse with target here. It means label. While I referred target as data coming from Henry Ford
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']
        data = maybe_to_torch(data) # GK: data is [12,1,512,512] and 12 batch size becuase of 12 processes perhaps
        target = maybe_to_torch(target)
        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        # GK: change for target: perhaps not required for validation process
        if self.target and do_backprop:
            #print("GK: current iteration", self.i_iter)
            #print("GK: source case:", data_dict['keys'])
            # get the target domain data
            target_data_dict = next(self.target_gen)
            target_data = target_data_dict['data']
            #print("GK: target case:", target_data_dict['keys'])
            # label is 1 for target cases and 0 for source
            assert target_data_dict['properties'][0]['label'] == 1 # I think 0 (in ['properties'][0]) is the stage
            target_data = maybe_to_torch(target_data)
            if torch.cuda.is_available():
                target_data = to_cuda(target_data)
            
            self.optimizer_d_aux.zero_grad()
            self.optimizer_d_main.zero_grad()

            # GK: adjust learning rate if required, doing only for discriminator, expecting nnUNet takes care of itself
            # like for scheduler
            adjust_learning_rate_discriminator(self.optimizer_d_aux, self.learning_rate_D, self.i_iter, self.max_iters, self.power)
            adjust_learning_rate_discriminator(self.optimizer_d_main, self.learning_rate_D, self.i_iter, self.max_iters, self.power)

            # only train segnet. Don't accumulate grads in disciminators
            for param in self.d_aux.parameters():
                param.requires_grad = False
            for param in self.d_main.parameters():
                param.requires_grad = False

        # GK: TRAINING

        if self.fp16:
            #print("GK: nnUNetTrainerv2.py: adversarial training not done in fp16 branch")
            with autocast():
                output = self.network(data) # GK: deep supervision: output is tuple of len 7 and objects as [12,3,512,512] -> 256,128,...,8
                del data
                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                # GK: Unscale has to be done here if we are refering network parameters before optimizer step()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                if not self.target:                    
                    self.amp_grad_scaler.step(self.optimizer)
                    self.amp_grad_scaler.update()

            # GK: change for target: adversarial training to fool the discriminator
            if self.target and do_backprop:
                with autocast():
                    target_output = self.network(target_data)
                    del target_data
                    # aux feature
                    pred_trg_aux = self.interp_target_256(target_output[1]) # interpolate h,w from 256(aux) to 512; now it's a different reference
                    d_out_aux = self.d_aux(prob_2_entropy(F.softmax(pred_trg_aux))) # [batch_size=1,num_classes=3,512,512] -> [1,1,16,16]
                    loss_adv_trg_aux = bce_loss(d_out_aux, self.source_label)
                    # main feature
                    d_out_main = self.d_main(prob_2_entropy(F.softmax(target_output[0])))
                    loss_adv_trg_main = bce_loss(d_out_main, self.source_label)
                    loss = (self.LAMBDA_ADV_MAIN * loss_adv_trg_main
                        + self.LAMBDA_ADV_AUX * loss_adv_trg_aux)
                    loss = loss
                
                # scale loss
                self.amp_grad_scaler.scale(loss).backward()

                # Train discriminator networks:
                # enable training mode on discriminator networks
                for param in self.d_aux.parameters():
                    param.requires_grad = True
                for param in self.d_main.parameters():
                    param.requires_grad = True

                # train with source
                with autocast():
                    # aux:
                    pred_src_aux = self.interp_target_256(output[1]) # detach here so not backpropagate gradients along this var
                    pred_src_aux = pred_src_aux.detach()
                    d_out_aux = self.d_aux(prob_2_entropy(F.softmax(pred_src_aux)))
                    loss_d_aux = bce_loss(d_out_aux, self.source_label)
                    loss_d_aux = loss_d_aux / 2
                    #loss_d_aux.backward()
                    # main:
                    pred_src_main = output[0]
                    pred_src_main = pred_src_main.detach()
                    d_out_main = self.d_main(prob_2_entropy(F.softmax(pred_src_main)))
                    loss_d_main = bce_loss(d_out_main, self.source_label)
                    loss_d_main = loss_d_main / 2
                    #loss_d_main.backward()
                
                # scale the loss
                self.amp_grad_scaler.scale(loss_d_aux).backward()
                self.amp_grad_scaler.scale(loss_d_main).backward()

                # train with target
                with autocast():
                    # aux:
                    pred_trg_aux = pred_trg_aux.detach() # already interpolated
                    d_out_aux = self.d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
                    loss_d_aux = bce_loss(d_out_aux, self.target_label)
                    loss_d_aux = loss_d_aux / 2
                    #loss_d_aux.backward()
                    # main:
                    pred_trg_main = target_output[0]
                    pred_trg_main = pred_trg_main.detach()
                    d_out_main = self.d_main(prob_2_entropy(F.softmax(pred_trg_main)))
                    loss_d_main = bce_loss(d_out_main, self.target_label)
                    loss_d_main = loss_d_main / 2
                    #loss_d_main.backward()
                
                # scale the loss
                self.amp_grad_scaler.scale(loss_d_aux).backward()
                self.amp_grad_scaler.scale(loss_d_main).backward()

                # GK: Though I am using a single scaler, so wondering if it's apt to have a one scale for 
                # all the three optimizers. Can be checked using self.amp_grad_scaler.get_scale()
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.step(self.optimizer_d_aux)
                self.amp_grad_scaler.step(self.optimizer_d_main)
                self.amp_grad_scaler.update()

        else: # GK: fp32 training
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12) # GK: I think it changes the parameters gradients var but not the parameters yet
                # GK: can't do here for adversarial training, do it later 
                if not self.target:
                    self.optimizer.step()
            
            # GK: change for target: adversarial training to fool the discriminator
            if self.target and do_backprop:
                target_output = self.network(target_data)
                del target_data
                # aux feature
                pred_trg_aux = self.interp_target_256(target_output[1]) # interpolate h,w from 256(aux) to 512; now it's a different reference
                d_out_aux = self.d_aux(prob_2_entropy(F.softmax(pred_trg_aux))) # [batch_size=1,num_classes=3,512,512] -> [1,1,16,16]
                loss_adv_trg_aux = bce_loss(d_out_aux, self.source_label)
                # main feature
                d_out_main = self.d_main(prob_2_entropy(F.softmax(target_output[0])))
                loss_adv_trg_main = bce_loss(d_out_main, self.source_label)
                loss = (self.LAMBDA_ADV_MAIN * loss_adv_trg_main
                    + self.LAMBDA_ADV_AUX * loss_adv_trg_aux)
                loss = loss
                loss.backward()

                # Train discriminator networks:
                # enable training mode on discriminator networks
                for param in self.d_aux.parameters():
                    param.requires_grad = True
                for param in self.d_main.parameters():
                    param.requires_grad = True
                # train with source
                # aux:
                pred_src_aux = self.interp_target_256(output[1]) # detach here so not backpropagate gradients along this var
                pred_src_aux = pred_src_aux.detach()
                d_out_aux = self.d_aux(prob_2_entropy(F.softmax(pred_src_aux)))
                loss_d_aux = bce_loss(d_out_aux, self.source_label)
                loss_d_aux = loss_d_aux / 2
                loss_d_aux.backward()
                # main:
                pred_src_main = output[0]
                pred_src_main = pred_src_main.detach()
                d_out_main = self.d_main(prob_2_entropy(F.softmax(pred_src_main)))
                loss_d_main = bce_loss(d_out_main, self.source_label)
                loss_d_main = loss_d_main / 2
                loss_d_main.backward()
                
                # train with target
                # aux:
                pred_trg_aux = pred_trg_aux.detach() # already interpolated
                d_out_aux = self.d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
                loss_d_aux = bce_loss(d_out_aux, self.target_label)
                loss_d_aux = loss_d_aux / 2
                loss_d_aux.backward()
                # main:
                pred_trg_main = target_output[0]
                pred_trg_main = pred_trg_main.detach()
                d_out_main = self.d_main(prob_2_entropy(F.softmax(pred_trg_main)))
                #print("GK: train discriminator networks for target", d_out_aux.shape, d_out_main.shape)
                loss_d_main = bce_loss(d_out_main, self.target_label)
                loss_d_main = loss_d_main / 2
                loss_d_main.backward()

                # for the seg next, this has to be done now
                self.optimizer.step()
                self.optimizer_d_aux.step()
                self.optimizer_d_main.step()

                if self.i_iter % self.num_batches_per_epoch == 0 and self.i_iter != 0:
                    # print at epoch end
                    self.print_to_log_file("Segmentation loss:", l)
                    self.print_to_log_file("Adversarial target loss: aux, main:", loss_adv_trg_aux, loss_adv_trg_main)
                    self.print_to_log_file("Discriminator loss: aux, main:", loss_d_aux, loss_d_main)

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        # GK: change for target: default case
        if not self.target:
            if self.fold == "all":
                # if fold==all then we use all images for training and validation
                tr_keys = val_keys = list(self.dataset.keys())
            else:
                splits_file = join(self.dataset_directory, "splits_final.pkl")

                # if the split file does not exist we need to create it
                if not isfile(splits_file):
                    self.print_to_log_file("Creating new 5-fold cross-validation split...")
                    splits = []
                    all_keys_sorted = np.sort(list(self.dataset.keys()))
                    kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                    for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                        train_keys = np.array(all_keys_sorted)[train_idx]
                        test_keys = np.array(all_keys_sorted)[test_idx]
                        splits.append(OrderedDict())
                        splits[-1]['train'] = train_keys
                        splits[-1]['val'] = test_keys
                    save_pickle(splits, splits_file)

                else:
                    self.print_to_log_file("Using splits from existing split file:", splits_file)
                    splits = load_pickle(splits_file)
                    self.print_to_log_file("The split file contains %d splits." % len(splits))

                self.print_to_log_file("Desired fold for training: %d" % self.fold)
                if self.fold < len(splits):
                    tr_keys = splits[self.fold]['train']
                    val_keys = splits[self.fold]['val']
                    self.print_to_log_file("This split has %d training and %d validation cases."
                                        % (len(tr_keys), len(val_keys)))
                else:
                    self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                        "contain only %d folds. I am now creating a "
                                        "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                    # if we request a fold that is not in the split file, create a random 80:20 split
                    rnd = np.random.RandomState(seed=12345 + self.fold)
                    keys = np.sort(list(self.dataset.keys()))
                    idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                    idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                    tr_keys = [keys[i] for i in idx_tr]
                    val_keys = [keys[i] for i in idx_val]
                    self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                        % (len(tr_keys), len(val_keys)))
                    print("GK: do_split(): training cases:", tr_keys)
                    print("GK: do_split(): validation cases:", val_keys)

            tr_keys.sort()
            val_keys.sort()
            self.dataset_tr = OrderedDict()
            for i in tr_keys:
                self.dataset_tr[i] = self.dataset[i]
            self.dataset_val = OrderedDict()
            for i in val_keys:
                self.dataset_val[i] = self.dataset[i]
    
        # GK: change for target: 
        if self.target:
            source_keys = []
            target_keys = []
            # self.dataset contains all the cases from source and target combined
            for case in self.dataset.keys():
                #print(f"GK: case:{case} and self.dataset[case]:{self.dataset[case]}")
                if self.dataset[case]["properties"]["label"] == 0:
                    source_keys.append(case)
                else: # label == 1
                    target_keys.append(case)
            
            if self.fold == 'all': # Presently, in this framework, only writing for fold 'all', no kfold
                source_tr_keys = source_val_keys = source_keys
            elif self.fold < 5:
                # could have referred and saved splits, just creating every time with a same random seed
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                source_keys = np.sort(source_keys)
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(source_keys)):
                    train_keys = np.array(source_keys)[train_idx]
                    test_keys = np.array(source_keys)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                
                source_tr_keys = splits[self.fold]['train']
                source_val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                    % (len(source_tr_keys), len(source_val_keys)))
                print("GK: do_split(): source training cases:", source_tr_keys)
                print("GK: do_split(): source validation cases:", source_val_keys)
            else:
                # create a random 80:20 split for any other fold value
                rnd = np.random.RandomState(seed=12345 + self.fold)
                source_keys = np.sort(source_keys)
                idx_tr = rnd.choice(len(source_keys), int(len(source_keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(source_keys)) if i not in idx_tr]
                source_tr_keys = [source_keys[i] for i in idx_tr]
                source_val_keys = [source_keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                    % (len(source_tr_keys), len(source_val_keys)))
                print("GK: do_split(): source training cases:", source_tr_keys)
                print("GK: do_split(): source validation cases:", source_val_keys)
            
            source_tr_keys.sort()
            source_val_keys.sort()
            target_keys.sort()

            self.dataset_tr = OrderedDict()
            for i in source_tr_keys:
                self.dataset_tr[i] = self.dataset[i]

            self.dataset_val = OrderedDict()
            for i in source_val_keys:
                self.dataset_val[i] = self.dataset[i]
            
            self.dataset_target = OrderedDict()
            for i in target_keys:
                self.dataset_target[i] = self.dataset[i]

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """

        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
            patch_size_for_spatialtransform = self.patch_size[1:]
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            patch_size_for_spatialtransform = self.patch_size

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = patch_size_for_spatialtransform

        self.data_aug_params["num_cached_per_thread"] = 2

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs

        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.apply(InitWeights_He(1e-2))
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                       "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                       "sometimes causes issues such as this one. Momentum has now been reduced to "
                                       "0.95 and network weights have been reinitialized")
        return continue_training

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        self.network.do_ds = True
        ret = super().run_training()
        self.network.do_ds = ds
        return ret
