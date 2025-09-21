import os
import torch
import shutil
import datetime
import numpy as np
import os.path as osp
from easydict import EasyDict as edict

from core.logger import ColorLogger


def init_dirs(dir_list):
    for dir in dir_list:
        # if osp.exists(dir) and osp.isdir(dir):
        #     shutil.rmtree(dir)
        os.makedirs(dir, exist_ok=True)

cfg = edict()


""" Directory """
cfg.cur_dir = osp.dirname(osp.abspath(__file__))
cfg.root_dir = osp.join(cfg.cur_dir, '../../')
cfg.data_dir = osp.join(cfg.root_dir, 'data')
KST = datetime.timezone(datetime.timedelta(hours=8)) # CHANGE TIMEZONE FROM HERE


""" Dataset """
cfg.DATASET = edict()
cfg.DATASET.name = ''
cfg.DATASET.workers = 4
cfg.DATASET.random_seed = 321
cfg.DATASET.bbox_expand_ratio = 1.3
cfg.DATASET.obj_set = 'behave'
cfg.DATASET.merge = True


""" Model - HMR """
cfg.MODEL = edict()
cfg.MODEL.input_img_shape = (512, 512)
cfg.MODEL.input_condition_shape = (32, 32)
cfg.MODEL.input_body_shape = (256, 256)
cfg.MODEL.input_hand_shape = (256, 256)
cfg.MODEL.img_feat_shape = (8, 8, 8)
cfg.MODEL.weight_path = ''

''' Model - Diffusion '''
cfg.DIFFUSION = edict({
    "DIFFUSION_PROCESS": {},
    "DENOISING_MODEL": {},
    "TRAIN": {},
    "GUIDANCE": {}

})
cfg.DIFFUSION.use_betas = True
cfg.DIFFUSION.split_obj = True
cfg.DIFFUSION.visualize = False
cfg.DIFFUSION.beta_stats = 'data/base_data/human_models/betas_stats.pkl' 
cfg.DIFFUSION.result_dir = './diffusion_exp'
# cfg.DIFFUSION.checkpoint_dir = './experiment/checkpoints'
cfg.DIFFUSION.use_obj_feat = True


cfg.DIFFUSION.DIFFUSION_PROCESS.timesteps =1000
cfg.DIFFUSION.DIFFUSION_PROCESS.beta_schedule = 'cosine'

cfg.DIFFUSION.DENOISING_MODEL.loss_type = 'l2'
cfg.DIFFUSION.DENOISING_MODEL.objective = 'pred_noise'
cfg.DIFFUSION.DENOISING_MODEL.num_block_pose = 3
cfg.DIFFUSION.DENOISING_MODEL.num_block_shape = 2
cfg.DIFFUSION.DENOISING_MODEL.hidden_layer_dim = 1024


cfg.DIFFUSION.TRAIN.lr = 1e-4
cfg.DIFFUSION.TRAIN.batch_size = 256
cfg.DIFFUSION.TRAIN.batch_size_val = 8
cfg.DIFFUSION.TRAIN.num_workers = 8
cfg.DIFFUSION.TRAIN.total_steps = 20000
cfg.DIFFUSION.TRAIN.log_freq = 500
cfg.DIFFUSION.TRAIN.checkpoint_freq = 10000
cfg.DIFFUSION.TRAIN.num_train_samples = 1
cfg.DIFFUSION.TRAIN.num_val_samples = 1
cfg.DIFFUSION.TRAIN.sample_start = 999
cfg.DIFFUSION.TRAIN.ddim_step_size = 10

cfg.DIFFUSION.GUIDANCE.optim_iters = 20
cfg.DIFFUSION.GUIDANCE.sample_start = 50
cfg.DIFFUSION.GUIDANCE.ddim_step_size = 5
cfg.DIFFUSION.GUIDANCE.gradient_scale = 1.0
cfg.DIFFUSION.GUIDANCE.use_guidance = True
cfg.DIFFUSION.GUIDANCE.earlystop = False
cfg.DIFFUSION.GUIDANCE.w_inter = 0.5  #0.1
cfg.DIFFUSION.GUIDANCE.w_kp2d = 0
cfg.DIFFUSION.GUIDANCE.w_colli = 0.1  #0.5
cfg.DIFFUSION.GUIDANCE.w_omask = 0
cfg.DIFFUSION.GUIDANCE.w_inter_o_f = 0.5  #0.5
# W_MULTIVIEW = 0.005
# W_SMOOTH = 30
cfg.DIFFUSION.GUIDANCE.use_hips = True

""" Train Detail """
cfg.TRAIN = edict()
cfg.TRAIN.batch_size = 64
cfg.TRAIN.shuffle = True
cfg.TRAIN.begin_epoch = 1
cfg.TRAIN.end_epoch = 50
cfg.TRAIN.warmup_epoch = 3
cfg.TRAIN.scheduler = 'step'
cfg.TRAIN.lr = 1.0e-4
cfg.TRAIN.min_lr = 1e-6
cfg.TRAIN.lr_step = [30]
cfg.TRAIN.lr_factor = 0.1
cfg.TRAIN.optimizer = 'adam'
cfg.TRAIN.momentum = 0
cfg.TRAIN.weight_decay = 0
cfg.TRAIN.beta1 = 0.5
cfg.TRAIN.beta2 = 0.999
cfg.TRAIN.print_freq = 10

cfg.TRAIN.loss_names = ['contact', 'vert', 'edge', 'param', 'coord', 'hand_bbox']
cfg.TRAIN.contact_loss_weight = 1.0
cfg.TRAIN.smpl_vert_loss_weight = 1.0
cfg.TRAIN.obj_vert_loss_weight = 1.0
cfg.TRAIN.smpl_edge_loss_weight = 1.0
cfg.TRAIN.smpl_pose_loss_weight = 1.0
cfg.TRAIN.smpl_shape_loss_weight = 1.0
cfg.TRAIN.smpl_trans_loss_weight = 1.0
cfg.TRAIN.obj_pose_loss_weight = 1.0
cfg.TRAIN.obj_trans_loss_weight = 1.0
cfg.TRAIN.smpl_3dkp_loss_weight = 1.0
cfg.TRAIN.smpl_2dkp_loss_weight = 1.0
cfg.TRAIN.pos_2dkp_loss_weight = 1.0
cfg.TRAIN.hand_bbox_loss_weight = 1.0


""" Augmentation """
cfg.AUG = edict()
cfg.AUG.scale_factor = 0.2
cfg.AUG.rot_factor = 30
cfg.AUG.shift_factor = 0
cfg.AUG.color_factor = 0.2
cfg.AUG.blur_factor = 0
cfg.AUG.flip = False


""" Test Detail """
cfg.TEST = edict()
cfg.TEST.batch_size = 24
cfg.TEST.shuffle = False
cfg.TEST.do_eval = True
cfg.TEST.eval_metrics_diffusion = ['cd_human', 'cd_object', 'contact_rec_p', 'contact_rec_r']
cfg.TEST.eval_metrics = ['cd_human', 'cd_object', 'contact_rec_p', 'contact_rec_r']
# cfg.TEST.eval_metrics = ['cd_human', 'cd_object', 'contact_rec_p', 'contact_rec_r']
cfg.TEST.print_freq = 5
cfg.TEST.contact_thres = 0.05


""" CAMERA """
cfg.CAMERA = edict()
cfg.CAMERA.focal = (2500, 2500)
cfg.CAMERA.princpt = (cfg.MODEL.input_img_shape[1]/2, cfg.MODEL.input_img_shape[0]/2)
cfg.CAMERA.depth_factor = 4.4
cfg.CAMERA.obj_depth_factor = 2.2*2

np.random.seed(cfg.DATASET.random_seed)
torch.manual_seed(cfg.DATASET.random_seed)
torch.backends.cudnn.benchmark = True
logger = None

    
def update_config(dataset_name='', exp_dir='', ckpt_path='', visualize=False):
    if dataset_name != '':
        dataset_name_dict = {'behave': 'BEHAVE', 'intercap': 'InterCap'}
        cfg.DATASET.name = dataset_name_dict[dataset_name.lower()]
        cfg.DATASET.obj_set = dataset_name

    if exp_dir == '':
        save_folder = 'exp_' + str(datetime.datetime.now(tz=KST))[5:-13]
        save_folder = save_folder.replace(" ", "_")
        save_folder_path = 'experiment/{}'.format(save_folder)
    else:
        save_folder_path = 'experiment/{}'.format(exp_dir)

    if ckpt_path != '':
        cfg.MODEL.weight_path = ckpt_path

    if visualize:
        cfg.DIFFUSION.visualize = True

    cfg.output_dir = osp.join(cfg.root_dir, save_folder_path)
    cfg.graph_dir = osp.join(cfg.output_dir, 'graph')
    cfg.vis_dir = osp.join(cfg.output_dir, 'vis')
    cfg.res_dir = osp.join(cfg.output_dir, 'results')
    cfg.log_dir = osp.join(cfg.output_dir, 'log')
    cfg.checkpoint_dir = osp.join(cfg.output_dir, 'checkpoints')

    print("Experiment Data on {}".format(cfg.output_dir))

    init_dirs([cfg.output_dir, cfg.log_dir, cfg.res_dir, cfg.vis_dir, cfg.checkpoint_dir])
    os.system(f'cp -r lib {cfg.output_dir}/codes')
    
    global logger; logger = ColorLogger(cfg.log_dir)