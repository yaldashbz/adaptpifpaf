import random
import time
from os.path import isfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from src import Bar
from src import datasets, models
from src.losses.jointsmseloss import JointsMSELoss, CurriculumLoss, symmetric_mse_loss
from src.models.respose_refine import init_pretrained
from src.models.utils import detach_model
from src.train.utils import mixup_withindomain, mixup
from src.utils import AverageMeter, fliplr, flip_back, accuracy, adjust_learning_rate_main
from src.utils.utils_mt import get_current_target_weight, get_current_consistency_weight, update_ema_variables


class MeanTeacherTrainer:
    def __init__(self, args, **kwargs):
        super().__init__(**kwargs)
        self.args = args
        self.mt_factory = models.Factory()
        self.device = torch.device(self.args.device_name)
        self.best_acc = 0
        self.best_epoch_acc = 0
        self.global_step = 0
        self.idx = []

    @classmethod
    def _check_args(cls, args):
        assert args.pretrained_path != args.resume

    def _apply_dp(self, model):
        if self.args.dp:
            return torch.nn.DataParallel(model)
        return model

    def _prepare_models(self, device):
        n_joints = 18
        teacher = self.mt_factory.factory(n_joints)
        student = self.mt_factory.factory(n_joints)
        detach_model(teacher)

        teacher = self._apply_dp(teacher).to(device)
        student = self._apply_dp(student).to(device)
        return teacher, student

    def _apply_resume(self, student, teacher):
        args = self.args
        if args.resume:
            if isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                student.load_state_dict(checkpoint['state_dict'])
                teacher.load_state_dict(checkpoint['state_dict_ema'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

    def _apply_load_pretrained(self, student):
        args = self.args
        if args.pretrained_path:
            if isfile(args.pretrained_path):
                init_pretrained(student, args.pretrained_path)
            else:
                print("=> no pretrained model found at '{}'".format(args.pretrained_path))
        else:
            print("Training from sctrach")

    def _prepare_dataloaders(self):
        args = self.args
        train_dataset = datasets.factory(args.src_dataset, is_train=True, **vars(args))
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch, shuffle=True,
            num_workers=args.workers, drop_last=True
        )

        real_dataset_train = datasets.factory(args.trg_dataset, is_train=True, is_aug=False, **vars(args))
        real_loader_train = torch.utils.data.DataLoader(
            real_dataset_train,
            batch_size=args.train_batch, shuffle=False,
            num_workers=args.workers
        )
        real_dataset_valid = datasets.factory(args.trg_dataset, is_train=False, is_aug=False, **vars(args))
        real_loader_valid = torch.utils.data.DataLoader(
            real_dataset_valid,
            batch_size=args.test_batch, shuffle=False,
            num_workers=args.workers
        )
        return train_loader, real_loader_train, real_loader_valid

    def validate(self, val_loader, model, criterion, flip=True):
        device = self.device
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_re = AverageMeter()
        acces_re = AverageMeter()

        model.eval()
        end = time.time()
        bar = Bar('Eval ', max=len(val_loader))

        with torch.no_grad():
            for i, (input, target, meta) in enumerate(val_loader):
                data_time.update(time.time() - end)
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                target_weight = meta['target_weight'].to(device, non_blocking=True)

                model_out, model_out_refine = model(input, 1, return_domain=False)
                hm1, hm2 = model_out_refine
                score_map = model_out.cpu()
                score_map_refine = hm1.cpu()

                if flip:
                    flip_input = torch.from_numpy(fliplr(input.clone().cpu().numpy())).float().to(device)
                    flip_output, flip_out_refine = model(flip_input, 1, return_domain=False)
                    hm1_flip, hm2_flip = flip_out_refine
                    flip_output = flip_back(flip_output.cpu(), 'real_animal')
                    flip_out_refine = flip_back(hm1_flip.cpu(), 'real_animal')
                    score_map += flip_output
                    score_map_refine += flip_out_refine

                loss_re = criterion(hm1, target, target_weight, len(self.idx))
                acc_re, _ = accuracy(score_map_refine, target.cpu(), self.idx)

                losses_re.update(loss_re.item(), input.size(0))
                acces_re.update(acc_re[0], input.size(0))
                batch_time.update(time.time() - end)
                end = time.time()

                wandb.log(dict(
                    loss_re=losses_re.avg,
                    acc_re=acces_re.avg
                ))
                bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} ' \
                             '| Loss_re: {loss_re:.8f}  | Acc_re: {acc_re: .8f}'.format(
                    batch=i + 1,
                    size=len(val_loader),
                    data=data_time.val,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss_re=losses_re.avg,
                    acc_re=acces_re.avg
                )
                bar.next()

            bar.finish()
            return losses_re.avg, acces_re.avg

    def train(self):
        args, device = self.args, self.device
        njoints = datasets.get_joints_num(args.src_dataset)
        teacher, student = self._prepare_models(device)
        criterion = JointsMSELoss().to(device)
        criterion_oekm = CurriculumLoss().to(device)
        bce_loss = nn.BCEWithLogitsLoss().to(device)
        optimizer = torch.optim.Adam(
            student.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )

        # load pretrained model
        self._apply_resume(student, teacher)
        self._apply_load_pretrained(student)
        train_loader, real_loader_train, real_loader_valid = self._prepare_dataloaders()
        print('    Total params in one pose model : %.2fM'
              % (sum(p.numel() for p in student.parameters()) / 1000000.0))

        if self.args.evaluate:
            print('\nEvaluation only')

            loss, acc = self.validate(real_loader_valid, student, criterion, args.flip)
            print(f'Loss: {loss}, Acc: {acc}')
            return

        for epoch in range(args.start_epoch, args.epochs):
            lr = adjust_learning_rate_main(optimizer, epoch, args)
            # compute the number of joints selected from each image, gradually drop more and more joints with large loss
            topk = njoints - (max(epoch - args.start_mine_epoch, 0) // args.reduce_interval)
            topk = int(max(topk, args.min_kpts))

            # gradually decrease the weight for initial pseudo labels
            target_weight = get_current_target_weight(args, epoch)

            # gradually increase the weight for self-distillation loss, namely updated pseudo labels
            c2rconsistency_weight = get_current_consistency_weight(args, epoch)
            print('\nEpoch: %d | LR: %.6f | Trg_weight: %.6f | C2rcons_weight: %.6f' % (epoch + 1, lr,
                                                                                        target_weight,
                                                                                        c2rconsistency_weight))
            # generate pseudo labels using consistency check
            if epoch == args.start_epoch:
                # if args.generate_pseudol:
                #     student.eval()
                #     for animal in ['horse', 'tiger']:
                #         # switch animal to single category to generate pseudo labels separately
                #         args.animal = animal
                #         real_dataset_train = datasets.__dict__[args.dataset_real](is_train=True, is_aug=False,
                #                                                                   **vars(args))
                #         real_loader_train = torch.utils.data.DataLoader(
                #             real_dataset_train,
                #             batch_size=args.train_batch, shuffle=False,
                #             num_workers=args.workers
                #         )
                #         ssl_kpts = {}
                #         acces1 = AverageMeter()
                #         previous_img = None
                #         previous_kpts = None
                #         for _, (trg_img, trg_lbl, trg_meta) in enumerate(real_loader_train):
                #             trg_img = trg_img.to(device)
                #             trg_lbl = trg_lbl.to(device, non_blocking=True)
                #             for i in range(trg_img.size(0)):
                #                 score_map, generated_kpts = prediction_check(previous_img, previous_kpts, trg_img[i],
                #                                                              student,
                #                                                              real_dataset_train, device,
                #                                                              num_transform=5)
                #                 ssl_kpts[int(trg_meta['index'][i].cpu().numpy().astype(np.int32))] = generated_kpts
                #                 acc1, _ = accuracy(score_map, trg_lbl[i].cpu().unsqueeze(0), self.idx)
                #                 acces1.update(acc1[0], 1)
                #                 previous_img = trg_img[i]
                #                 previous_kpts = generated_kpts
                #         print('Acc on target {} training set (psedo-labels): {}'.format(args.animal, acces1.avg))
                #         np.save('./animal_data/psudo_labels/stage1/all/ssl_labels_train_{}.npy'.format(args.animal),
                #                 ssl_kpts)
                #     break
                # construct dataloader based on generated pseudolabels, switch animal to 'all' to train on all categories
                args.animal = 'all'
                real_dataset_train = datasets.factory(args.trg_dataset_crop, is_train=True, is_aug=True, **vars(args))
                real_loader_train = torch.utils.data.DataLoader(
                    real_dataset_train,
                    batch_size=args.train_batch, shuffle=True,
                    num_workers=args.workers,
                    drop_last=True
                )
                print("======> start training")

            loss_src_log = AverageMeter()
            loss_trg_log = AverageMeter()
            loss_cons_log = AverageMeter()
            joint_loader = zip(train_loader, real_loader_train)
            num_iter = len(real_loader_train) if len(train_loader) > len(real_loader_train) else len(train_loader)
            student.train()
            teacher.train()
            bar = Bar('Train ', max=num_iter)

            wandb.config = {
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "train_batch": args.train_batch,
                "test_batch": args.test_batch,
                "tent_is_used": args.tent
            }
            idx = self.idx

            for i, (
                    (src_img, src_lbl, src_meta),
                    (trg_img, trg_img_t, trg_img_s, trg_lbl, trg_lbl_t, trg_lbl_s, trg_meta)) \
                    in enumerate(joint_loader):
                # gradually increase the magnitude of grl as the discriminator gets better and better
                p = float(i + epoch * num_iter) / args.epochs / num_iter
                lambda_ = 2. / (1. + np.exp(-10 * p)) - 1
                wandb.log({'lambda': lambda_})

                optimizer.zero_grad()
                src_img, src_lbl, src_weight = src_img.to(device), src_lbl.to(device, non_blocking=True), src_meta[
                    'target_weight'].to(device, non_blocking=True)
                trg_img, trg_lbl = trg_img.to(device), trg_lbl.to(device, non_blocking=True)
                warpmat, trg_weight = trg_meta['warpmat'].to(device, non_blocking=True), trg_meta['target_weight'].to(
                    device, non_blocking=True)
                trg_img_t, trg_img_s, trg_lbl_t, trg_lbl_s = trg_img_t.to(device), trg_img_s.to(device), trg_lbl_t.to(
                    device, non_blocking=True), trg_lbl_s.to(device, non_blocking=True)
                # add mixup between domains
                # if args.mixup:
                #     src_img, src_lbl, src_weight, trg_img, trg_lbl, trg_weight = mixup(src_img, src_lbl, src_weight,
                #                                                                        trg_img, trg_lbl, trg_weight,
                #                                                                        args.beta, device)
                # add mixup between domains and within domains, the later is more about balancing horse and tiger data
                # if args.mixup_dual:
                #     if random.random() <= 0.5:
                #         src_img, src_lbl, src_weight, trg_img, trg_lbl, trg_weight = mixup(src_img, src_lbl, src_weight,
                #                                                                            trg_img, trg_lbl, trg_weight,
                #                                                                            args.beta, device)
                #     else:
                #         trg_img, trg_lbl, trg_weight = mixup_withindomain(trg_img, trg_lbl, trg_weight, args.beta,
                #                                                           args.beta, device)

                src_out, src_out_refine, src_domain_out = student(src_img, lambda_)

                src_kpt_score, src_kpt_score_ = src_out_refine
                loss_kpt_src = criterion(src_out, src_lbl, src_weight, len(idx)) + \
                               criterion(src_kpt_score, src_lbl, src_weight, len(idx))
                loss_src_log.update(loss_kpt_src.item(), src_img.size(0))
                wandb.log({'loss_keypoint_source': loss_kpt_src})

                trg_out, trg_out_refine, trg_domain_out = student(trg_img, lambda_)
                trg_kpt_score, trg_kpt_score_ = trg_out_refine

                # wandb.log({'loss_keypoint_target': trg_kpt_score})
                # loss based on initial pseudo labels
                loss_kpt_trg_coarse = target_weight * criterion_oekm(trg_out, trg_lbl, trg_weight, topk)
                loss_kpt_trg_refine = target_weight * criterion_oekm(trg_kpt_score, trg_lbl, trg_weight, topk)

                # wandb.log({'loss_kpt_trg_coarse': loss_kpt_trg_coarse})
                # wandb.log({'loss_kpt_trg_refine': loss_kpt_trg_refine})

                loss_kpt_trg = loss_kpt_trg_coarse + loss_kpt_trg_refine
                loss_trg_log.update(loss_kpt_trg.item(), trg_img.size(0))

                # loss for discriminator
                src_label = torch.ones_like(src_domain_out).cuda(device)
                trg_label = torch.zeros_like(trg_domain_out).cuda(device)
                loss_adv_src = args.adv_w * bce_loss(src_domain_out, src_label) / 2
                loss_adv_trg = args.adv_w * bce_loss(trg_domain_out, trg_label) / 2

                loss_stu = loss_kpt_src + loss_kpt_trg + loss_adv_src + loss_adv_trg
                loss_stu.backward()
                # LTH(model.backbone)
                # LTH(model.domain_classifier)
                wandb.log({'loss_student': loss_kpt_trg})

                # forward again for student teacher consistency
                model_out_stu, model_out_stu_refine = student(trg_img_s, lambda_, return_domain=False)
                # The teacher network has the same three forward steps although we only use the last one. The reason is to
                # keep the data statistics the same for student and teacher branch, mainly for batchnorm.
                with torch.no_grad():
                    _, _ = teacher(src_img, lambda_, return_domain=False)
                    _, _ = teacher(trg_img, lambda_, return_domain=False)
                    model_out_ema, model_out_ema_refine = teacher(trg_img_t, lambda_, return_domain=False)
                hm1, hm2 = model_out_stu_refine
                hm_ema, _ = model_out_ema_refine

                # self-distillation loss, namely inner loop update
                hm1_clone = hm1.clone().detach().requires_grad_(False)
                weight_const_var = torch.ones_like(trg_weight).to(device)
                c2rconst_loss = c2rconsistency_weight * criterion_oekm(model_out_stu, hm1_clone, weight_const_var,
                                                                       topk)

                # consistency between two heads of the student network, adapted from original mean teacher
                res_loss = args.logit_distance_cost * symmetric_mse_loss(hm1, hm2)

                wandb.log({'res_loss': res_loss})
                # consistency loss between student and teacher network
                flip_var = (trg_meta['flip'] == trg_meta['flip_ema'])
                flip_var = flip_var.to(device)
                # gradually increase the weight for consistency loss, namely outer loop update
                consistency_weight = get_current_consistency_weight(args, epoch)
                # adding transformation to the output of the teacher network
                grid = F.affine_grid(warpmat, hm_ema.size())
                hm_ema_trans = F.grid_sample(hm_ema, grid)
                hm_ema_trans_flip = hm_ema_trans.clone()
                hm_ema_trans_flip = flip_back(hm_ema_trans_flip.cpu(), 'real_animal')
                hm_ema_trans_flip = hm_ema_trans_flip.to(device)
                hm_ema_trans = torch.where(flip_var[:, None, None, None], hm_ema_trans, hm_ema_trans_flip)
                weight_const_var = torch.ones_like(trg_weight).to(device)
                consistency_loss = consistency_weight * criterion(hm2, hm_ema_trans, weight_const_var, len(idx))
                loss_cons_log.update(consistency_loss.item(), trg_img_t.size(0))

                loss_mt = consistency_loss + res_loss + c2rconst_loss
                wandb.log(dict(
                    consistency_loss=consistency_loss,
                    loss_mt=loss_mt
                ))
                loss_mt.backward()
                # before updating ema variables
                # LTH(model.head)
                optimizer.step()
                self.global_step += 1
                # update parameters for teacher network
                update_ema_variables(student, teacher, args.ema_decay, self.global_step)

                wandb.log(dict(
                    loss_cons=loss_cons_log.avg,
                    loss_src=loss_src_log.avg,
                    loss_trg=loss_trg_log.avg))
                bar.suffix = '({batch}/{size}) Total: {total:} | ETA: {eta:} | Loss_cons: {loss_cons:.8f}s' \
                             '| Loss_src: {loss_src:.8f} | Loss_trg: {loss_trg:.8f}'.format(
                    batch=i + 1,
                    size=num_iter,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss_cons=loss_cons_log.avg,
                    loss_src=loss_src_log.avg,
                    loss_trg=loss_trg_log.avg
                )
                bar.next()
            bar.finish()

            _, trg_val_acc_s = self.validate(real_loader_valid, student, criterion, args.flip)
            _, trg_val_acc_t = self.validate(real_loader_valid, teacher, criterion, args.flip)

            trg_val_acc = trg_val_acc_t
            if trg_val_acc > self.best_acc:
                self.best_epoch_acc = epoch + 1
            self.best_acc = max(trg_val_acc, self.best_acc)
            # save_checkpoint({
            #     'epoch': epoch + 1,
            #     'arch': args.arch,
            #     'state_dict': model.state_dict(),
            #     'state_dict_ema': model_ema.state_dict(),
            #     'best_acc': best_acc,
            #     'optimizer': optimizer.state_dict()
            # }, is_best, checkpoint=args.checkpoint, snapshot=args.snapshot)

        print(self.best_epoch_acc, self.best_acc)
