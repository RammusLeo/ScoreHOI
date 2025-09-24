import torch
import numpy as np
import os.path as osp
import torch.optim as optim
from collections import Counter
from collections import OrderedDict
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from core.config import cfg, logger
# from dataset.merge_dataset import MergeDataset
# from dataset.merge_dataset_pointnet import MergeDatasetObj
# from dataset.merge_dataset_pointnet_trans import MergeDatasetObjTrans
# from dataset.merge_dataset_pointnet_imhd import MergeDatasetObjIMHD
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

def get_dataloader(dataset_name='', is_train=False, is_dist=False, merge=False, train_diffusion=False, use_trans=False):
    if dataset_name == '': return None, None
    dataset_split = 'TRAIN' if is_train else 'TEST'   
    batch_per_dataset = cfg[dataset_split].batch_size
    dataset_list, dataloader_list = [], []
    if logger is not None:
        logger.info(f"==> Preparing {dataset_split} Dataloader...")
    transform = transforms.ToTensor()
    if merge:
        transform = transforms.ToTensor()
        dataset_split = 'TRAIN' if is_train else 'TEST'   
        batch_per_dataset = cfg[dataset_split].batch_size
        if train_diffusion:
            dataset = MergeDatasetObjIMHD(transform, dataset_split.lower())
        else:
            if use_trans:
                dataset = MergeDatasetObjTrans(transform, dataset_split.lower())
            else:
                dataset = MergeDatasetObj(transform, dataset_split.lower())
        if logger is not None:
            logger.info(f"# of {dataset_split} MergeDataset data: {len(dataset)}")
    else:
        exec(f'from {dataset_name}.dataset import {dataset_name}')
        
        dataset = eval(f'{dataset_name}')(transform, dataset_split.lower())
        if logger is not None:
            logger.info(f"# of {dataset_split} {dataset_name} data: {len(dataset)}")
    dataset_list.append(dataset)


    if is_dist:
        sampler = DistributedSampler(dataset)
        if not is_train:
            dataloader = DataLoader(dataset=dataset, batch_size=batch_per_dataset, shuffle=False, 
                                    num_workers=cfg.DATASET.workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn, sampler=sampler)
            dataloader_list.append(dataloader)
        else:
            dataloader_list = DataLoader(dataset=dataset, batch_size=batch_per_dataset, shuffle=False,
                                        num_workers=cfg.DATASET.workers, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn, sampler=sampler)
        return dataloader_list, sampler

    if not is_train:
        dataloader = DataLoader(dataset=dataset, batch_size=batch_per_dataset, shuffle=cfg[dataset_split].shuffle, 
                                num_workers=cfg.DATASET.workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)
        dataloader_list.append(dataloader)
    else:
        dataloader_list = DataLoader(dataset=dataset, batch_size=batch_per_dataset, shuffle=cfg[dataset_split].shuffle,
                                    num_workers=cfg.DATASET.workers, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)

    return dataloader_list, dataset_list 


def train_setup(model, checkpoint):    
    loss_history, eval_history = None, None
    optimizer = get_optimizer(model=model, lr=cfg.TRAIN.lr, name=cfg.TRAIN.optimizer)
    lr_scheduler = get_scheduler(optimizer=optimizer)
    
    if checkpoint is not None:
        # if 'optim_state_dict' in checkpoint:
        #     optimizer.load_state_dict(checkpoint['optim_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        curr_lr = 0.0

        for param_group in optimizer.param_groups:
            curr_lr = param_group['lr']

        if 'scheduler_state_dict' in checkpoint:
            lr_state = checkpoint['scheduler_state_dict']
            lr_state['milestones'], lr_state['gamma'] = Counter(cfg.TRAIN.lr_step), cfg.TRAIN.lr_factor
            lr_scheduler.load_state_dict(lr_state)

        if 'train_log' in checkpoint:
            loss_history = checkpoint['train_log']
        if 'test_log' in checkpoint:
            eval_history = checkpoint['test_log']
        if 'epoch' in checkpoint:
            cfg.TRAIN.begin_epoch = checkpoint['epoch'] + 1
        # logger.info("===> resume from epoch {:d}, current lr: {:.0e}, milestones: {}, lr factor: {:.0e}"
        #             .format(cfg.TRAIN.begin_epoch, curr_lr, lr_state['milestones'], lr_state['gamma']))

    return optimizer, lr_scheduler, loss_history, eval_history
    

class AverageMeterDict(object):
    def __init__(self, names):
        for name in names:
            value = AverageMeter()
            setattr(self, name, value)

    def __getitem__(self,key):
        return getattr(self, key)

    def update(self, name, val, n=1):
        getattr(self, name).update(val, n)


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


def check_data_parallel(train_weight):
    new_state_dict = OrderedDict()
    for k, v in train_weight.items():
        name = k[7:]  if k.startswith('module') else k  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def get_optimizer(model, lr=1.0e-4, name='adam'):
    total_params = []
    
    if hasattr(model, 'trainable_modules'):
        for module in model.trainable_modules:
            total_params += list(module.parameters())
    else:
        total_params += list(model.parameters())
    
    if hasattr(model, 'trainable_params'):
        total_params = model.trainable_params

    optimizer = None
    if name == 'sgd':
        optimizer = optim.SGD(
            total_params,
            lr=lr,
            momentum=cfg.TRAIN.momentum,
            weight_decay=cfg.TRAIN.weight_decay
        )
    elif name == 'rmsprop':
        optimizer = optim.RMSprop(
            total_params,
            lr=lr
        )
    elif name == 'adam':
        optimizer = optim.Adam(
            total_params,
            lr=lr,
            betas=(cfg.TRAIN.beta1, cfg.TRAIN.beta2)
        )
    elif name == 'adamw':
        optimizer = optim.AdamW(
            total_params,
            lr=lr,
            weight_decay=cfg.TRAIN.weight_decay
        )
    return optimizer


def get_scheduler(optimizer):
    scheduler = None
    if cfg.TRAIN.scheduler == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.lr_step, gamma=cfg.TRAIN.lr_factor)
    elif cfg.TRAIN.scheduler == 'platue':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.TRAIN.lr_factor, patience=10, min_lr=1e-5)
    elif cfg.TRAIN.scheduler == 'cosine':
        from warmup_scheduler import GradualWarmupScheduler
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.TRAIN.end_epoch-cfg.TRAIN.warmup_epoch, eta_min=cfg.TRAIN.min_lr)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=cfg.TRAIN.warmup_epoch, after_scheduler=scheduler_cosine)
        optimizer.zero_grad(); optimizer.step()
        scheduler.step()

    return scheduler


def save_checkpoint(states, epoch, file_path=None, is_best=None):
    if file_path is None:
        file_name = f'epoch_{epoch}.pth.tar'
        output_dir = cfg.checkpoint_dir
        if states['epoch'] == cfg.TRAIN.end_epoch:
            file_name = 'final.pth.tar'
        file_path = osp.join(output_dir, file_name)
            
    torch.save(states, file_path)

    if is_best:
        torch.save(states, osp.join(output_dir, 'best.pth.tar'))


def load_checkpoint(load_dir, epoch=0, pick_best=False):
    try:
        checkpoint = torch.load(load_dir, map_location='cpu')
        return checkpoint
    except Exception as e:
        raise ValueError("No checkpoint exists!\n", e)


def worker_init_fn(worder_id):
        np.random.seed(np.random.get_state()[1][0] + worder_id)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def cycle(dl):
    """
    Generate data from dataloader.
    """
    while True:
        for data in dl:
            yield data


def filter_based_on_pose(batch):
    """
    Keep only samples with pseudo-GT pose annotations.
    """
    has_smpl_params = batch["has_smpl_params"]["body_pose"] > 0
    batch_size = has_smpl_params.sum().item()
    for key in batch:
        if key == "imgname":
            batch["imgname"] = list(
                np.array(batch["imgname"])[has_smpl_params.cpu().numpy()]
            )
        elif key in ["smpl_params", "has_smpl_params", "smpl_params_is_axis_angle"]:
            for nested_key in batch[key]:
                batch[key][nested_key] = batch[key][nested_key][has_smpl_params]
        else:
            batch[key] = batch[key][has_smpl_params]
    assert (
        batch["img"].size(0) == batch["smpl_params"]["body_pose"].size(0) == batch_size
        and torch.all(batch["has_smpl_params"]["body_pose"]).item()
        and torch.all(batch["has_smpl_params"]["global_orient"]).item()
    ), "Error in discarding images with no SMPL pseudo-GT"
    return batch

def recursive_to(x, target):
    """
    Recursively transfer a batch of data to the target device
    Args:
        x (Any): Batch of data.
        target (torch.device): Target device.
    Returns:
        Batch of data where all tensors are transfered to the target device.
    """
    if isinstance(x, dict):
        return {k: recursive_to(v, target) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.to(target)
    elif isinstance(x, list):
        return [recursive_to(i, target) for i in x]
    else:
        return x