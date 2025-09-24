import os
import argparse
import __init_path
import torch

from core.config import update_config, cfg
parser = argparse.ArgumentParser(description='Test ScoreHOI')
parser.add_argument('--gpu', type=str, default='0', help='assign multi-gpus by comma concat')
parser.add_argument('--dataset', type=str, default='behave', choices=['behave', 'intercap'], help='dataset')
parser.add_argument('--checkpoint', type=str, default='', help='model path for evaluation')
parser.add_argument('--exp', type=str, default='', help='assign experiments directory')
parser.add_argument("--ckpt_path", type=str, required=True)
parser.add_argument("--sample_start", type=int, default=-1)
parser.add_argument("--optim_iters" , type=int, default=-1)
parser.add_argument("--ddim_step_size" , type=int, default=-1)
parser.add_argument("--w_inter", type=float, default=-1)
parser.add_argument("--w_colli", type=float, default=-1)
parser.add_argument("--w_inter_o_f", type=float, default=-1)
parser.add_argument("--not_continue", action="store_true", help="not continue training")
parser.add_argument("--visualize", action="store_true", help="visualize the results")
parser.add_argument("--eval_mode", type=str, default="demo")

# Organize arguments
args = parser.parse_args()
update_config(dataset_name=args.dataset, exp_dir=args.exp, ckpt_path=args.checkpoint, visualize=args.visualize)

def update_cfg(args):
    if args.w_inter >= 0:
        cfg.DIFFUSION.GUIDANCE.w_inter = args.w_inter
    if args.w_colli >= 0:
        cfg.DIFFUSION.GUIDANCE.w_colli = args.w_colli
    if args.w_inter_o_f >= 0:
        cfg.DIFFUSION.GUIDANCE.w_inter_o_f = args.w_inter_o_f
    if args.optim_iters > 0:
        cfg.DIFFUSION.GUIDANCE.optim_iters = args.optim_iters
    if args.ddim_step_size > 0:
        cfg.DIFFUSION.GUIDANCE.ddim_step_size = args.ddim_step_size
    if args.sample_start > 0:
        cfg.DIFFUSION.GUIDANCE.sample_start = args.sample_start

update_cfg(args)

from core.config import logger
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
logger.info(f"Work on GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
logger.info(f"Args: {args}")
logger.info(f"Cfg: {cfg}")

device = torch.device(f"cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Prepare tester
# from core.base import TesterOptimizeIP
from core.demo_tester import DemoTester
# from core.diffusion_trainer import TesterOptimizeIP
tester = DemoTester(args, load_dir=cfg.MODEL.weight_path, diffusion_load_dir=args.ckpt_path, device=device, model_type="pointnet", diff_model_type="IPFC_Obj")


# Test ScoreHOI
print("===> Start Evaluation...")
# tester.run(0)
tester.run(eval_mode=args.eval_mode)