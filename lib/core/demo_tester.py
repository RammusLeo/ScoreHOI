import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import copy
import pickle
import os
from models.denoising_model import FC
from models.denoising_model_attcond import IPFC, IPFC_Obj
# from models.denoising_model_attcond_abla import IPFC_Obj_Noimg, IPFC_Obj_Noobj
from diffusion.scorehmr_diffusion import GaussianDiffusion
from diffusion.scorehmr_diffusion_betas import GaussianDiffusionBetas
from core.config import cfg, logger
from models.templates import smplh, obj_dict
from train_utils import AverageMeterDict, load_checkpoint, get_dataloader
from funcs_utils import rotmat_to_6d, rot6d_to_rotmat, rot6d_to_aa, axis_angle_to_6d
from vis_utils import vis_results_batch
from eval_utils import eval_chamfer_distance, eval_contact_estimation, eval_contact_score
from dataset.demo_dataset import DemoDataset
from torch.utils.data import DataLoader
import time


def prepare_pointnet_network(args, load_dir='', load_dir_backbone='', is_train=True): 
    from lib.models.feature_backbone import get_backbone  
    # from train_utils import count_parameters
    model = get_backbone()
    # logger.info(f'# of model parameters: {count_parameters(model)}')
    if load_dir:
        print('Loading checkpoint from', load_dir) 
        checkpoint = load_checkpoint(load_dir=load_dir)
        if args.not_continue:
            checkpoint['epoch'] = 0
        try:
            model.load_weights(checkpoint['model_state_dict'], strict=True)
        except:
            model.load_weights(checkpoint, strict=True)
    else:
        checkpoint = None
    if load_dir_backbone:
        print('Loading backbone checkpoint from', load_dir_backbone) 
        checkpoint = load_checkpoint(load_dir=load_dir_backbone)
        model.load_weights(checkpoint['model_state_dict'])
        print('Backbone loaded')
    return model

def prepare_diffusion_model(args, load_dir='', model_type='FC'):
    if model_type == 'FC':
        model = FC(cfg)
    elif model_type == 'IPFC':
        model = IPFC(cfg)
    elif model_type == 'IPFC_Obj':
        model = IPFC_Obj(cfg)
    # elif model_type == 'IPFC_Obj_Noimg':
    #     model = IPFC_Obj_Noimg(cfg)
    # elif model_type == 'IPFC_Obj_Noobj':
    #     model = IPFC_Obj_Noobj(cfg)
    else:
        raise ValueError('Unknown model type: {}'.format(model_type))
    if cfg.DIFFUSION.use_betas:
        diffusion_model = GaussianDiffusionBetas(cfg, model, device=torch.device('cuda'))
    else:
        diffusion_model = GaussianDiffusion(cfg, model, device=torch.device('cuda'))
    if load_dir != '':
        print('Loading diffusion checkpoint from', load_dir)         
        checkpoint = load_checkpoint(load_dir)
        if args.not_continue:
            checkpoint['epoch'] = 0
        diffusion_model.load_weights(checkpoint['model_state_dict'], strict=False)
    else:
        checkpoint = None
    return diffusion_model


class BaseTester:
    def __init__(self, args, load_dir='', backbone_dir='',model_type="pointnet"):
        if model_type == "pointnet":
            self.model = prepare_pointnet_network(args, load_dir, backbone_dir, False)
        else:
            raise Exception("Support pointnet backbone")
        self.model = self.model.cuda()
        dataset = DemoDataset()
        self.val_loaders = [DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)]
        self.eval_metrics = cfg.TEST.eval_metrics

        self.eval_history = {}
        self.eval_vals = {}
        self.outputs_db = {}
        
    def save_history(self, eval_history):
        if hasattr(self, 'eval_metrics'):
            for k in self.eval_metrics:
                if k in self.eval_vals:
                    eval_history[k].append(self.eval_vals[k])

class DemoTester(BaseTester):
    def __init__(self, args, load_dir='',backbone_dir='', diffusion_load_dir='', device='cuda', model_type="pointnet", diff_model_type="IPFC_Obj"):
        super(DemoTester, self).__init__(args, load_dir,backbone_dir, model_type=model_type)         
        self.eval_metrics = cfg.TEST.eval_metrics
        self.eval_history = {}
        for k in self.eval_metrics: self.eval_history[k] = []

        self.save_results = {}
        self.device = device
        self.diffusion_model = prepare_diffusion_model(args, diffusion_load_dir, model_type=diff_model_type)

        # torch.save(self.ema.ema_model.state_dict(), 'checkpoints/debug/loadmodel3.pth')
        self.smplh_layer = copy.deepcopy(smplh.layer['neutral']).to(self.device)   

        self.eval_vals_intermid = {}

    def run(self,current_model=None, current_diffusion_model=None, eval_mode="pointnet"):
        if current_model:
            self.model = current_model
        if current_diffusion_model:
            self.diffusion_model = current_diffusion_model
        self.model.eval()
        self.diffusion_model = self.diffusion_model.to(self.device)
        self.diffusion_model.eval()
        if eval_mode == "pointnet":
            for val_loader in self.val_loaders:
                self.run_orig(val_loader)           
        elif eval_mode == "intermid":
            for val_loader in self.val_loaders:
                self.run_one_dataset(val_loader)
        elif eval_mode == "after":
            for val_loader in self.val_loaders:
                self.run_after(val_loader)
        elif eval_mode == "generate":
            for val_loader in self.val_loaders:
                self.generate(val_loader)
        elif eval_mode == "demo":
            for val_loader in self.val_loaders:
                self.demo_test(val_loader)

    def run_orig(self, val_loader):
        running_evals = AverageMeterDict(self.eval_metrics)
        running_evals_intermid = AverageMeterDict(self.eval_metrics)
        val_loader = tqdm(val_loader)
        outputs_dict = {}
        for i, (inputs, targets, meta_info) in enumerate(val_loader):
            for k,v in inputs.items():
                if torch.is_tensor(v): inputs[k] = v.to(self.device)
            with torch.no_grad():
                coarse_outputs, contact_predict = self.model.init_coarse_inference(inputs)
                coarse_outputs_init = copy.deepcopy(coarse_outputs)
                coarse_outputs = self.diffusion_opt_iter(coarse_outputs, contact_predict,inputs=inputs, meta_info=meta_info)
                contact_predict = self.model.predict_contact(coarse_outputs)
                refined_outputs = self.model.refine_inference(contact_predict)

                # init_outputs = self.model(inputs)

            for k,v in coarse_outputs_init.items():
                if k=='smpl_verts': 
                    coarse_outputs_init[k] = smplh.upsample(coarse_outputs_init[k]).cpu().numpy()
                elif k=='obj_pose':
                    coarse_outputs_init[k] = rot6d_to_rotmat(coarse_outputs_init[k]).cpu().numpy()
                else:
                    if torch.is_tensor(v): coarse_outputs_init[k] = v.cpu().numpy()
            for k,v in contact_predict.items():
                if torch.is_tensor(v): contact_predict[k] = v.cpu().numpy()
            for k,v in refined_outputs.items():
                # if k=='smpl_verts': 
                #     coarse_outputs[k] = smplh.upsample(coarse_outputs[k]).cpu().numpy()
                # elif k=='obj_pose':
                #     coarse_outputs[k] = rot6d_to_rotmat(coarse_outputs[k]).cpu().numpy()
                # else:
                if torch.is_tensor(v): refined_outputs[k] = v.cpu().numpy()
                
            # Evaluation
            if cfg.TEST.do_eval:
                eval_dict = self.evaluation(refined_outputs, contact_predict, targets, meta_info, est_contact=True) 
                eval_dict_intermid = self.evaluation(coarse_outputs_init, contact_predict, targets, meta_info)
            if cfg.DIFFUSION.visualize:
                for k,v in meta_info.items():
                    if torch.is_tensor(v): meta_info[k] = v.cpu().numpy()
                for k,v in refined_outputs.items():
                    if torch.is_tensor(v): refined_outputs[k] = v.cpu().numpy()
                self.visualization(meta_info, refined_outputs, coarse_outputs_init)
            for k in self.eval_metrics:
                running_evals[k].update(sum(eval_dict[k]), len(eval_dict[k]))
                if len(eval_dict[k]) == 0: eval_dict[k] = 0.0
                else: eval_dict[k] = sum(eval_dict[k])/len(eval_dict[k])
                if k not in ['contact_est_p', 'contact_est_r']:
                    running_evals_intermid[k].update(sum(eval_dict_intermid[k]), len(eval_dict_intermid[k]))
                    if len(eval_dict_intermid[k]) == 0: eval_dict_intermid[k] = 0.0
                    else: eval_dict_intermid[k] = sum(eval_dict_intermid[k])/len(eval_dict_intermid[k])
            if i % cfg.TEST.print_freq == 0:
                message = f'({i}/{len(val_loader)})'
                for k in self.eval_metrics: message += f' {k}: {eval_dict[k]:.2f}'
                val_loader.set_description(message)

            # Save Results
            for idx, output in enumerate(coarse_outputs['smpl_verts']):
                outputs_dict[int(meta_info['ann_id'][idx])] = {
                    'smpl_verts': output,
                    'obj_pose': coarse_outputs['obj_pose'][idx],
                    'obj_trans': coarse_outputs['obj_trans'][idx],
                    'obj_name': meta_info['obj_name'][idx],
                    'cam_trans': coarse_outputs['cam_trans'][idx]
                }
        
        pickle.dump(outputs_dict, open((os.path.join(cfg.res_dir,'outputs.pkl')), 'wb'))

        for k in self.eval_metrics:
            self.eval_vals[k] = running_evals[k].avg
            if k not in ['contact_est_p', 'contact_est_r']:
                self.eval_vals_intermid[k] = running_evals_intermid[k].avg

        message = 'Finished Evaluation!\n'
        message += f'-------- Evaluation Results (Contact estimation) ------\n'
        message += f'Precision: {self.eval_vals["contact_est_p"]:.3f} / Recall: {self.eval_vals["contact_est_r"]:.3f}\n'
        message += f'---------- Evaluation Results (Reconstruction) --------\n'
        message += '>> Chamfer Distance\n'
        message += f'Human: {self.eval_vals["cd_human"]:.2f} / Object: {self.eval_vals["cd_object"]:.2f}\n'
        if "fscore_human" in self.eval_vals:
            message += '>> F-Scores\n'
            message += f'Human: {self.eval_vals["fscore_human"]:.2f} / Object: {self.eval_vals["fscore_object"]:.2f}\n'
        message += '>> Contact from reconstruction\n'
        message += f'Precision: {self.eval_vals["contact_rec_p"]:.3f} / Recall: {self.eval_vals["contact_rec_r"]:.3f}\n'
        fscore = 2*self.eval_vals["contact_rec_p"]*self.eval_vals["contact_rec_r"]/(self.eval_vals["contact_rec_p"]+self.eval_vals["contact_rec_r"])
        message += f'Contact F-score: {fscore:.3f}\n'

        message += f'INTERMID---------- Evaluation Results (Reconstruction) --------\n'
        message += '>> Chamfer Distance\n'
        message += f'Human: {self.eval_vals_intermid["cd_human"]:.2f} / Object: {self.eval_vals_intermid["cd_object"]:.2f}\n'
        # message += '>> F-Scores\n'
        # message += f'Human: {self.eval_vals["fscore_human"]:.2f} / Object: {self.eval_vals["fscore_object"]:.2f}\n'
        message += '>> Contact from reconstruction\n'
        message += f'Precision: {self.eval_vals_intermid["contact_rec_p"]:.3f} / Recall: {self.eval_vals_intermid["contact_rec_r"]:.3f}'

        logger.info(message)

    def demo_test(self, val_loader):
        val_loader = tqdm(val_loader)
        for i, (inputs, targets, meta_info) in enumerate(val_loader):
            for k,v in inputs.items():
                if torch.is_tensor(v): inputs[k] = v.to(self.device)
            with torch.no_grad():
                time1 = time.time()
                coarse_outputs, contact_predict = self.model.init_coarse_inference(inputs)
                coarse_outputs_init = copy.deepcopy(coarse_outputs)
                batch_val, x_t, y_input, time_pairs_inv, time_pairs, shape_val = self.prepare_x0(coarse_outputs, contact_predict,inputs=inputs, meta_info=meta_info)
                
                for i in range(cfg.DIFFUSION.GUIDANCE.optim_iters):
                    coarse_outputs = self.diffusion_opt(batch_val, x_t, y_input, time_pairs_inv, time_pairs, shape_val)
                    # update x_t
                    x_t = coarse_outputs['x_t']
                    coarse_outputs['img_feat'] = coarse_outputs_init['img_feat']
                    # update contact_labels
                    coarse_outputs['obj_verts_orig'] = coarse_outputs_init['obj_verts_orig']
                    contact_predict = self.model.predict_contact(coarse_outputs)
                # refined_outputs = self.model.refine_inference(contact_predict)
                refined_outputs = {
                    "smpl_verts": coarse_outputs["smpl_verts"],
                    # "refined_obj_verts": refined_obj_verts,
                    "obj_pose": rot6d_to_rotmat(coarse_outputs['obj_pose']),
                    "obj_trans": coarse_outputs['obj_trans'],
                    "cam_trans": coarse_outputs['cam_trans']
                }
                time2 = time.time()
                print("Time taken for one iteration: ", time2-time1)
                init_outputs = self.model(inputs)

            for k,v in refined_outputs.items():
                if torch.is_tensor(v): refined_outputs[k] = v.cpu().numpy()
            for k,v in contact_predict.items():
                if torch.is_tensor(v): contact_predict[k] = v.cpu().numpy()
            # for k,v in coarse_outputs.items():
            #     if k=='smpl_verts': 
            #         coarse_outputs[k] = smplh.upsample(coarse_outputs[k]).cpu().numpy()
            #     elif k=='obj_pose':
            #         coarse_outputs[k] = rot6d_to_rotmat(coarse_outputs[k]).cpu().numpy()
            #     else:
            #         if torch.is_tensor(v): coarse_outputs[k] = v.cpu().numpy()
            #     if torch.is_tensor(v): contact_predict[k] = v.cpu().numpy()
            

            for k,v in meta_info.items():
                if torch.is_tensor(v): meta_info[k] = v.cpu().numpy()
            for k,v in refined_outputs.items():
                if torch.is_tensor(v): refined_outputs[k] = v.cpu().numpy()
            self.visualization(meta_info, refined_outputs)


    def generate(self, val_loader):
        val_loader = tqdm(val_loader)
        for i, (inputs, targets, meta_info) in enumerate(val_loader):
            for k,v in inputs.items():
                if torch.is_tensor(v): inputs[k] = v.to(self.device)
            with torch.no_grad():
                coarse_outputs, contact_predict = self.model.init_coarse_inference(inputs)
                # coarse_outputs_init = copy.deepcopy(coarse_outputs)
                coarse_outputs = self.diffusion_opt_iter(coarse_outputs, contact_predict,inputs=inputs, meta_info=meta_info, use_guidance=False)

            for k,v in coarse_outputs.items():
                if k=='smpl_verts':
                    coarse_outputs[k] = smplh.upsample(coarse_outputs[k]).cpu().numpy()
                elif k=='obj_pose':
                    coarse_outputs[k] = rot6d_to_rotmat(coarse_outputs[k]).cpu().numpy()
                elif torch.is_tensor(v): coarse_outputs[k] = v.detach().cpu().numpy()
            # for k,v in contact_predict.items():
            #     if torch.is_tensor(v): contact_predict[k] = v.cpu().numpy()
            
            for k,v in meta_info.items():
                if torch.is_tensor(v): meta_info[k] = v.cpu().numpy()
            for k,v in coarse_outputs.items():
                if torch.is_tensor(v): coarse_outputs[k] = v.cpu().numpy()
            self.visualization(meta_info, coarse_outputs)


    def run_one_dataset(self, val_loader):
        running_evals = AverageMeterDict(self.eval_metrics)
        running_evals_intermid = AverageMeterDict(self.eval_metrics)
        val_loader = tqdm(val_loader)
        outputs_dict = {}
        for i, (inputs, targets, meta_info) in enumerate(val_loader):
            
            for k,v in inputs.items():
                if torch.is_tensor(v): inputs[k] = v.to(self.device)
            with torch.no_grad():
                time1 = time.time()
                coarse_outputs, contact_predict = self.model.init_coarse_inference(inputs)
                coarse_outputs_init = copy.deepcopy(coarse_outputs)
                batch_val, x_t, y_input, time_pairs_inv, time_pairs, shape_val = self.prepare_x0(coarse_outputs, contact_predict,inputs=inputs, meta_info=meta_info)
                
                for i in range(cfg.DIFFUSION.GUIDANCE.optim_iters):
                    coarse_outputs = self.diffusion_opt(batch_val, x_t, y_input, time_pairs_inv, time_pairs, shape_val)
                    # update x_t
                    x_t = coarse_outputs['x_t']
                    coarse_outputs['img_feat'] = coarse_outputs_init['img_feat']
                    # update contact_labels
                    coarse_outputs['obj_verts_orig'] = coarse_outputs_init['obj_verts_orig']
                    contact_predict = self.model.predict_contact(coarse_outputs)
                refined_outputs = self.model.refine_inference(contact_predict)

                time2 = time.time()
                print("Time taken for one iteration: ", time2-time1)
                print(inputs['img'].shape)
                init_outputs = self.model(inputs)

            for k,v in refined_outputs.items():
                if torch.is_tensor(v): refined_outputs[k] = v.cpu().numpy()
            for k,v in contact_predict.items():
                if torch.is_tensor(v): contact_predict[k] = v.cpu().numpy()
            for k,v in coarse_outputs.items():
                if k=='smpl_verts': 
                    coarse_outputs[k] = smplh.upsample(coarse_outputs[k]).cpu().numpy()
                elif k=='obj_pose':
                    coarse_outputs[k] = rot6d_to_rotmat(coarse_outputs[k]).cpu().numpy()
                else:
                    if torch.is_tensor(v): coarse_outputs[k] = v.cpu().numpy()
                
            # Evaluation
            if cfg.TEST.do_eval:
                eval_dict = self.evaluation(refined_outputs, contact_predict, targets, meta_info, est_contact=True) 
                eval_dict_intermid = self.evaluation(coarse_outputs, contact_predict, targets, meta_info)
            if cfg.DIFFUSION.visualize:
                for k,v in meta_info.items():
                    if torch.is_tensor(v): meta_info[k] = v.cpu().numpy()
                for k,v in refined_outputs.items():
                    if torch.is_tensor(v): refined_outputs[k] = v.cpu().numpy()
                self.visualization(meta_info, refined_outputs, init_outputs)
            for k in self.eval_metrics:
                running_evals[k].update(sum(eval_dict[k]), len(eval_dict[k]))
                if len(eval_dict[k]) == 0: eval_dict[k] = 0.0
                else: eval_dict[k] = sum(eval_dict[k])/len(eval_dict[k])
                if k not in ['contact_est_p', 'contact_est_r']:
                    running_evals_intermid[k].update(sum(eval_dict_intermid[k]), len(eval_dict_intermid[k]))
                    if len(eval_dict_intermid[k]) == 0: eval_dict_intermid[k] = 0.0
                    else: eval_dict_intermid[k] = sum(eval_dict_intermid[k])/len(eval_dict_intermid[k])
            if i % cfg.TEST.print_freq == 0:
                message = f'({i}/{len(val_loader)})'
                for k in self.eval_metrics: message += f' {k}: {eval_dict[k]:.2f}'
                val_loader.set_description(message)

            # Save Results
            for idx, output in enumerate(refined_outputs['smpl_verts']):
                outputs_dict[int(meta_info['ann_id'][idx])] = {
                    'smpl_verts': output,
                    'obj_pose': refined_outputs['obj_pose'][idx],
                    'obj_trans': refined_outputs['obj_trans'][idx],
                    'obj_name': meta_info['obj_name'][idx],
                    'cam_trans': refined_outputs['cam_trans'][idx]
                }
        
        pickle.dump(outputs_dict, open((os.path.join(cfg.res_dir,'outputs.pkl')), 'wb'))

        for k in self.eval_metrics:
            self.eval_vals[k] = running_evals[k].avg
            if k not in ['contact_est_p', 'contact_est_r']:
                self.eval_vals_intermid[k] = running_evals_intermid[k].avg

        message = 'Finished Evaluation!\n'
        message += f'-------- Evaluation Results (Contact estimation) ------\n'
        message += f'Precision: {self.eval_vals["contact_est_p"]:.3f} / Recall: {self.eval_vals["contact_est_r"]:.3f}\n'
        message += f'---------- Evaluation Results (Reconstruction) --------\n'
        message += '>> Chamfer Distance\n'
        message += f'Human: {self.eval_vals["cd_human"]:.2f} / Object: {self.eval_vals["cd_object"]:.2f}\n'
        if "fscore_human" in self.eval_vals:
            message += '>> F-Scores\n'
            message += f'Human: {self.eval_vals["fscore_human"]:.2f} / Object: {self.eval_vals["fscore_object"]:.2f}\n'
        message += '>> Contact from reconstruction\n'
        message += f'Precision: {self.eval_vals["contact_rec_p"]:.3f} / Recall: {self.eval_vals["contact_rec_r"]:.3f}\n'
        fscore = 2*self.eval_vals["contact_rec_p"]*self.eval_vals["contact_rec_r"]/(self.eval_vals["contact_rec_p"]+self.eval_vals["contact_rec_r"])
        message += f'Contact F-score: {fscore:.3f}\n'
        message += f'INTERMID---------- Evaluation Results (Reconstruction) --------\n'
        message += '>> Chamfer Distance\n'
        message += f'Human: {self.eval_vals_intermid["cd_human"]:.2f} / Object: {self.eval_vals_intermid["cd_object"]:.2f}\n'
        # message += '>> F-Scores\n'
        # message += f'Human: {self.eval_vals["fscore_human"]:.2f} / Object: {self.eval_vals["fscore_object"]:.2f}\n'
        message += '>> Contact from reconstruction\n'
        message += f'Precision: {self.eval_vals_intermid["contact_rec_p"]:.3f} / Recall: {self.eval_vals_intermid["contact_rec_r"]:.3f}'

        logger.info(message)


    def run_after(self, val_loader):
        running_evals = AverageMeterDict(self.eval_metrics)
        running_evals_intermid = AverageMeterDict(self.eval_metrics)
        val_loader = tqdm(val_loader)
        for i, (inputs, targets, meta_info) in enumerate(val_loader):
            
            for k,v in inputs.items():
                if torch.is_tensor(v): inputs[k] = v.to(self.device)
            with torch.no_grad():
                coarse_outputs, contact_predict = self.model.init_coarse_inference(inputs)
                # for i in range(cfg.DIFFUSION.GUIDANCE.optim_iters):
                #     coarse_outputs = self.diffusion_opt(coarse_outputs, contact_predict, meta_info)
                #     contact_predict = self.model.predict_contact(coarse_outputs)
                refined_outputs = self.model.refine_inference(contact_predict)
                smplh_pose, smplh_shape = self.humverts2smplh(refined_outputs['smpl_verts'])
                bs = smplh_pose.shape[0]
                refined_outputs['human_pose'] = axis_angle_to_6d(smplh_pose).reshape(bs, -1)
                refined_outputs['human_shape'] = smplh_shape
                refined_outputs['h_contact'] = contact_predict['h_contacts']
                refined_outputs['o_contact'] = contact_predict['o_contacts']
                refined_outputs['obj_pose'] = rotmat_to_6d(refined_outputs['obj_pose'])
                refined_outputs['obj_verts_orig'] = contact_predict['obj_verts_orig']
                refined_outputs['h2d_keypoints'] = contact_predict['h2d_keypoints']
                refined_outputs['img_feat'] = coarse_outputs['img_feat']
                
                refined_outputs = self.diffusion_opt(refined_outputs, contact_predict,inputs=inputs, meta_info=meta_info)
                init_outputs = self.model(inputs)

            for k,v in coarse_outputs.items():
                if k=='smpl_verts': 
                    coarse_outputs[k] = smplh.upsample(coarse_outputs[k]).cpu().numpy()
                elif k=='obj_pose':
                    coarse_outputs[k] = rot6d_to_rotmat(coarse_outputs[k]).cpu().numpy()
                else:
                    if torch.is_tensor(v): coarse_outputs[k] = v.cpu().numpy()
            for k,v in contact_predict.items():
                if torch.is_tensor(v): contact_predict[k] = v.cpu().numpy()
            for k,v in refined_outputs.items():
                if k=='obj_pose':
                    refined_outputs[k] = rot6d_to_rotmat(refined_outputs[k]).cpu().numpy()
                else:
                    if torch.is_tensor(v): refined_outputs[k] = v.cpu().numpy()
                
            # Evaluation
            if cfg.TEST.do_eval:
                eval_dict = self.evaluation(refined_outputs, contact_predict, targets, meta_info, est_contact=True) 
                eval_dict_intermid = self.evaluation(coarse_outputs, contact_predict, targets, meta_info)
            if cfg.DIFFUSION.visualize:
                for k,v in meta_info.items():
                    if torch.is_tensor(v): meta_info[k] = v.cpu().numpy()
                for k,v in refined_outputs.items():
                    if torch.is_tensor(v): refined_outputs[k] = v.cpu().numpy()
                self.visualization(meta_info, refined_outputs, init_outputs)
            for k in self.eval_metrics:
                running_evals[k].update(sum(eval_dict[k]), len(eval_dict[k]))
                if len(eval_dict[k]) == 0: eval_dict[k] = 0.0
                else: eval_dict[k] = sum(eval_dict[k])/len(eval_dict[k])
                if k not in ['contact_est_p', 'contact_est_r']:
                    running_evals_intermid[k].update(sum(eval_dict_intermid[k]), len(eval_dict_intermid[k]))
                    if len(eval_dict_intermid[k]) == 0: eval_dict_intermid[k] = 0.0
                    else: eval_dict_intermid[k] = sum(eval_dict_intermid[k])/len(eval_dict_intermid[k])
            if i % cfg.TEST.print_freq == 0:
                message = f'({i}/{len(val_loader)})'
                for k in self.eval_metrics: message += f' {k}: {eval_dict[k]:.2f}'
                val_loader.set_description(message)

        for k in self.eval_metrics:
            self.eval_vals[k] = running_evals[k].avg
            if k not in ['contact_est_p', 'contact_est_r']:
                self.eval_vals_intermid[k] = running_evals_intermid[k].avg

        message = 'Finished Evaluation!\n'
        message += f'-------- Evaluation Results (Contact estimation) ------\n'
        message += f'Precision: {self.eval_vals["contact_est_p"]:.3f} / Recall: {self.eval_vals["contact_est_r"]:.3f}\n'
        message += f'---------- Evaluation Results (Reconstruction) --------\n'
        message += '>> Chamfer Distance\n'
        message += f'Human: {self.eval_vals["cd_human"]:.2f} / Object: {self.eval_vals["cd_object"]:.2f}\n'
        # message += '>> F-Scores\n'
        # message += f'Human: {self.eval_vals["fscore_human"]:.2f} / Object: {self.eval_vals["fscore_object"]:.2f}\n'
        message += '>> Contact from reconstruction\n'
        message += f'Precision: {self.eval_vals["contact_rec_p"]:.3f} / Recall: {self.eval_vals["contact_rec_r"]:.3f}\n'

        message += f'INTERMID---------- Evaluation Results (Reconstruction) --------\n'
        message += '>> Chamfer Distance\n'
        message += f'Human: {self.eval_vals_intermid["cd_human"]:.2f} / Object: {self.eval_vals_intermid["cd_object"]:.2f}\n'
        # message += '>> F-Scores\n'
        # message += f'Human: {self.eval_vals["fscore_human"]:.2f} / Object: {self.eval_vals["fscore_object"]:.2f}\n'
        message += '>> Contact from reconstruction\n'
        message += f'Precision: {self.eval_vals_intermid["contact_rec_p"]:.3f} / Recall: {self.eval_vals_intermid["contact_rec_r"]:.3f}'

        logger.info(message)

    def diffusion_opt(self, batch_val, x_t, y_input, time_pairs_inv, time_pairs, shape_val):
        with torch.no_grad():
            sampling_output = self.diffusion_model.one_step_ddim_with_guidance(
                batch_val,
                x_t,
                y_input,
                time_pairs_inv,
                time_pairs,
                shape_val,
                )
            x_0 = sampling_output['x_0']
            camera_translation = sampling_output['camera_translation'] if sampling_output['camera_translation'] is not None else batch_val['inputs']['pred_cam_trans']
            x_t = sampling_output['x_t']

        human_pose_ = rot6d_to_aa(x_0[:, : 52 * 6].reshape(-1,6)).reshape(shape_val[0],-1)
        human_shape_ =  x_0[:, -10:] if cfg.DIFFUSION.use_betas else batch_val['inputs']['pred_betas']
        human_shape_ = self.diffusion_model.unnormalize_betas(human_shape_).to(torch.float32)
        smpl_output = self.smplh_layer(betas=human_shape_, global_orient=human_pose_[:, :3], body_pose=human_pose_[:, 3:66], left_hand_pose=human_pose_[:, 66:111], right_hand_pose=human_pose_[:, 111:])
        smpl_verts, smpl_joints = smpl_output.vertices - smpl_output.joints[:, [smplh.root_joint_idx]], smpl_output.joints - smpl_output.joints[:, [smplh.root_joint_idx]]
        # smpl_verts = smplh.downsample(smpl_verts)
        outputs={
            "smpl_verts": smpl_verts,
            'obj_pose': x_0[:, 52*6 : 52*6 + 6],
            'obj_trans': x_0[:, 52*6 + 6:52*6 + 9],
            'cam_trans': camera_translation,
            'smpl_joints': smpl_joints,
            'human_pose': x_0[:, : 52*6],
            'human_shape': human_shape_,
            'x_t': x_t,
        }
        return outputs
    
    def diffusion_opt_iter(self, coarse_outputs, contact_predict,inputs=None, meta_info=None, use_guidance=True):
        for k,v in coarse_outputs.items():
            if torch.is_tensor(v): coarse_outputs[k] = v.to(self.device)
        for k,v in contact_predict.items():
            if torch.is_tensor(v): contact_predict[k] = v.to(self.device)

        bs_times_samples_val = coarse_outputs["human_pose"].shape[0]
        batch_val = {
            "inputs": { "pred_betas": coarse_outputs["human_shape"],
                        "pred_pose": coarse_outputs["human_pose"],
                        "pred_obj_pose": coarse_outputs["obj_pose"],
                        "pred_obj_trans": coarse_outputs["obj_trans"],
                        "obj_verts_orig": coarse_outputs["obj_verts_orig"],
                        "pred_cam_trans": coarse_outputs["cam_trans"],
                        "pred_h_contacts": contact_predict["h_contacts"],
                        "pred_o_contacts": contact_predict["o_contacts"],
                        "h2d_keypoints": contact_predict['h2d_keypoints']},
            "meta_info": {"obj_mask": meta_info["obj_mask"] if meta_info is not None else None},
        }
        if coarse_outputs["img_feat"].dim() > 2:
            cond_img_feats_val = coarse_outputs["img_feat"].mean((2,3))
        else:
            cond_img_feats_val = coarse_outputs["img_feat"]
        if 'obj_feat' in inputs:
            cond_geo_feats_val = inputs['obj_feat']
        else:
            cond_geo_feats_val = coarse_outputs["obj_verts_orig"]
        with torch.no_grad():
            sampling_output = self.diffusion_model.sample(
                    batch_val,
                    cond_img_feats_val,
                    cond_geo_feats_val,
                    bs_times_samples_val,
                    use_guidance
                )
            x_0 = sampling_output['x_0']
            camera_translation = sampling_output['camera_translation'] if sampling_output['camera_translation'] is not None else batch_val['inputs']['pred_cam_trans']
        human_pose_ = rot6d_to_aa(x_0[:, : 52 * 6].reshape(-1,6)).reshape(bs_times_samples_val,-1)
        human_shape_ = batch_val['inputs']['pred_betas']
        smpl_output = self.smplh_layer(betas=human_shape_, global_orient=human_pose_[:, :3], body_pose=human_pose_[:, 3:66], left_hand_pose=human_pose_[:, 66:111], right_hand_pose=human_pose_[:, 111:])
        smpl_verts, smpl_joints = smpl_output.vertices - smpl_output.joints[:, [smplh.root_joint_idx]], smpl_output.joints - smpl_output.joints[:, [smplh.root_joint_idx]]
        smpl_verts = smplh.downsample(smpl_verts)
        outputs={
            "smpl_verts": smpl_verts,
            'obj_pose': x_0[:, 52*6 : 52*6 + 6],
            'obj_trans': x_0[:, 52*6 + 6:52*6 + 9],
            'cam_trans': camera_translation,
            'smpl_joints': smpl_joints,
            'human_pose': x_0[:, : 52*6],
            'human_shape': human_shape_,
            'img_feat': coarse_outputs["img_feat"],
            'obj_verts_orig': batch_val['inputs']['obj_verts_orig'],
        }
        return outputs

    def prepare_x0(self, coarse_outputs, contact_predict,inputs=None, meta_info=None):
        for k,v in coarse_outputs.items():
            if torch.is_tensor(v): coarse_outputs[k] = v.to(self.device)
        for k,v in contact_predict.items():
            if torch.is_tensor(v): contact_predict[k] = v.to(self.device)

        bs_times_samples_val = coarse_outputs["human_pose"].shape[0]
        batch_val = {
            "inputs": { "pred_betas": coarse_outputs["human_shape"],
                        "pred_pose": coarse_outputs["human_pose"],
                        "pred_obj_pose": coarse_outputs["obj_pose"],
                        "pred_obj_trans": coarse_outputs["obj_trans"],
                        "obj_verts_orig": coarse_outputs["obj_verts_orig"],
                        "pred_cam_trans": coarse_outputs["cam_trans"],
                        "pred_h_contacts": contact_predict["h_contacts"],
                        "pred_o_contacts": contact_predict["o_contacts"],
                        "pred_o_f_contacts": contact_predict["o_f_contacts"],
                        "h2d_keypoints": contact_predict['h2d_keypoints']},
            "meta_info": {"obj_mask": meta_info["obj_mask"] if meta_info is not None else None},
        }
        if coarse_outputs["img_feat"].dim() > 2:
            cond_img_feats_val = coarse_outputs["img_feat"].mean((2,3))
        else:
            cond_img_feats_val = coarse_outputs["img_feat"]
        if 'obj_feat' in inputs:
            cond_geo_feats_val = inputs['obj_feat']
        else:
            cond_geo_feats_val = coarse_outputs["obj_verts_orig"]
        with torch.no_grad():
            shape_val = (bs_times_samples_val, self.diffusion_model.model.diffusion_dim)
            x_t, y_input, time_pairs_inv, time_pairs, shape_val = self.diffusion_model.prepare_xt(
                    batch_val,
                    cond_img_feats_val,
                    cond_geo_feats_val,
                    shape_val,
                )
        return batch_val, x_t, y_input, time_pairs_inv, time_pairs, shape_val

    def visualization(self, inputs, outputs, init_outputs=None):
        rot_along_y = np.array([[0,0,1],
                               [0,1,0],
                               [-1,0,0]])
        batch_size = outputs["smpl_verts"].shape[0]
        # sample_idx = np.random.choice(batch_size, (16,))
        sample_idx = np.arange(0, batch_size, 1)
        for idx in sample_idx:
            vis_results_batch(inputs, outputs, idx, obj_dict, step=0, init_outputs=init_outputs, transform_rot = rot_along_y)
        return
    
    def evaluation(self, refined_outputs, contact_predict, targets, meta_info, est_contact=False):
        tar_smpl_mesh_cam = (targets['smpl_mesh_cam']).numpy()

        pred_smpl_mesh_cam, pred_obj_mesh_cam, tar_obj_mesh_cam, obj_faces = [], [], [], []
        for i in range(len(tar_smpl_mesh_cam)):
            smpl_mesh_cam = refined_outputs['smpl_verts'][i]
                
            smpl_joint_cam = smplh.joint_regressor @ smpl_mesh_cam
            
            smpl_mesh_cam = smpl_mesh_cam - smpl_joint_cam[smplh.root_joint_idx]
            pred_smpl_mesh_cam.append(smpl_mesh_cam)

            obj_name = meta_info['obj_name'][i]
            obj_mesh = obj_dict[obj_name].load_template()
            obj_faces.append(np.array(obj_mesh.faces))
            obj_mesh_cam = obj_dict.transform_object(obj_mesh.vertices, refined_outputs['obj_pose'][i], refined_outputs['obj_trans'][i])
            pred_obj_mesh_cam.append(obj_mesh_cam)
            obj_mesh_cam = obj_dict.transform_object(obj_mesh.vertices, targets['obj_pose'][i].numpy(), targets['obj_trans'][i].numpy())
            tar_obj_mesh_cam.append(obj_mesh_cam)
        pred_smpl_mesh_cam = np.stack(pred_smpl_mesh_cam)

        eval_dict = {}
        if 'cd_human' in self.eval_metrics:
            eval_dict['cd_human'], eval_dict['cd_object'], eval_dict['fscore_human'], eval_dict['fscore_object'] = eval_chamfer_distance(pred_smpl_mesh_cam, tar_smpl_mesh_cam, pred_obj_mesh_cam, tar_obj_mesh_cam, obj_faces)
            eval_dict['cd_human'], eval_dict['cd_object'] = eval_dict['cd_human']*100, eval_dict['cd_object']*100

        if est_contact and 'contact_est_p' in self.eval_metrics:
            eval_dict['contact_est_p'], eval_dict['contact_est_r'], eval_dict['contact_e2p'], eval_dict['contact_e2r'] = eval_contact_estimation(contact_predict['h_contacts'], contact_predict['o_contacts'], targets['h_contacts'], targets['o_contacts'])

        if 'fscore_human' in self.eval_metrics:
            eval_dict['fscore_human'], eval_dict['fscore_object'] = eval_dict['fscore_human']*100, eval_dict['fscore_object']*100

        if 'contact_rec_p' in self.eval_metrics:
            eval_dict['contact_rec_p'], eval_dict['contact_rec_r'] = eval_contact_score(pred_smpl_mesh_cam, pred_obj_mesh_cam,  targets['h_contacts'].numpy())
        return eval_dict