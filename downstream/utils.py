import argparse
import datetime
import multiprocessing
import random
import threading
import os

import numpy as np
import torch
from dateutil import tz


def str2bool(value):
    if value.lower() in ['yes', 'true', 't', '1']:
        return True
    elif value.lower() in ['no', 'false', 'f', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def setup_arguments():
    # -------------------------------
    # load hyper-param
    # -------------------------------
    parse = argparse.ArgumentParser()

    # ==========================task and dataset configurations ===============================#
    parse.add_argument('--task', type=str, default='classification',
                       choices=['classification', 'segmentation'],
                       help="the downstream tasks.")
    parse.add_argument('--phase', type=str, default='inference',
                       choices=['finetune', 'inference'],
                       help="experiment phase")
    parse.add_argument('--data_name', type=str,
                       choices=['rsna', 'nih', 'siim', 'shenzhen', 'tbx11k'],
                       default='siim', help='the name of the dataset')
    parse.add_argument('--view_position_path', type=str, help='the local path of view positions dict',
                       default='/MIMIC-CXR/view-positions-dict-mimic.json'
                       )
    parse.add_argument('--batch_size', default=8, type=int, help="the batch size for the training phase")
    parse.add_argument('--test_batch_size', default=8, type=int, help="the batch size for the inference phase")
    parse.add_argument('--num_workers', type=int, default=0, help="Cpu num for dataloaders")

    # ========================== model configuration ===============================#
    parse.add_argument('--encoder_max_length', type=int, default=300)
    parse.add_argument('--global_token', type=str, default='avg', choices=['avg', 'pool'])
    parse.add_argument('--loss_type', type=str, default='focal', choices=['focal', 'sigmoid', 'cross-entropy'])
    parse.add_argument('--knowledge_feat', type=str2bool, help="whether use indication", default='no')
    # freeze models setup
    parse.add_argument('--frozen_backbone', type=str2bool, help="whether frozen backbone", default='no')

    # perceiver
    parse.add_argument('--text_encoder_num_blocks', type=int, default=6)
    parse.add_argument('--perceiver_num_blocks', type=int, default=3)
    parse.add_argument('--perceiver_num_latents', type=int, default=128, help='the number of latent array')
    parse.add_argument('--perceiver_num_heads', type=int, default=8)

    # =========================== Learning configurations ===========================#
    parse.add_argument('--learning_rate', type=float, default=5.0e-5, help='initial learning rate')  # 5.0e-5
    parse.add_argument('--patience', type=int, default=5, help='the patience of lr scheduler')
    parse.add_argument('--seed', type=int, default=9233, help='random seed')

    # ====================== Pytorch Lightning configurations ===========================
    parse.add_argument('--devices', type=int, default=1, help='how many gpus to use')
    parse.add_argument('--num_nodes', type=int, default=1, help='Number of GPU nodes for distributed training.')
    parse.add_argument('--accelerator', type=str, default="gpu", help='accelerator types')
    parse.add_argument('--strategy', type=str, default="ddp", help='default ddp for multi-gpus')
    parse.add_argument('--precision', type=str, default='bf16-mixed',
                       help='16 or 32 bf16-mixed, using for original pytorch amp auto cast')
    parse.add_argument('--limit_val_batches', type=float, default=1.0,
                       help='How much of validation dataset to check (float = fraction, int = num_batches).')
    parse.add_argument('--limit_test_batches', type=float, default=1.0,
                       help='How much of test dataset to check (float = fraction, int = num_batches).')
    parse.add_argument('--limit_train_batches', type=float, default=1.0,
                       help='How much of training dataset to check (float = fraction, int = num_batches)')
    parse.add_argument('--max_epochs', type=int, default=100, help='Stop training once this number of epochs is reached')
    parse.add_argument('--log_every_n_steps', type=int, default=500, help='How often to log within steps.')
    parse.add_argument('--every_n_train_steps', type=int, default=0,
                       help='How many training steps to save a checkpoint')
    parse.add_argument('--val_check_interval', type=float, default=1.0, help='How often to check the validation set')
    parse.add_argument('--accumulate_grad_batches', type=int, default=2,
                       help='Accumulates gradients over k batches before stepping the optimizer')
    parse.add_argument("--num_sanity_val_steps", type=int, default=2,
                       help='Sanity check runs n validation batches before starting the training routine')

    # ======================== Checkpoints configurations ===========================
    parse.add_argument('--save_best_model', type=str2bool, default='no', help='whether save model')
    parse.add_argument('--save_last_model', type=str2bool, default='no', help='whether save model')
    parse.add_argument('--online_ckpt', type=str2bool, default='no',
                       help='whether using online checkpoint for the image encoder and text decoder')
    parse.add_argument('--ckpt_zoo_dir', type=str,
                       help='the directory of local checkpoints, e.g., chexbert and radgraph',
                       default='/checkpoints/'
                       )
    parse.add_argument('--resume', type=str, help='the path of the ckpt for resuming training.',
                       )
    parse.add_argument('--load', type=str, help='the path of the ckpt for finetune.',
                       )
    parse.add_argument('--test_ckpt_path', type=str, help='the path of the ckpt for inference',
                       )
    parse.add_argument('--version', type=str, default='CoGaze-without-context', help='the name of experiment')
    parse.add_argument('--exp_dir_trial', type=str, default='results', help='the directory for recording experimental results')

    # model ckpt
    parse.add_argument('--rad_dino_path', type=str, default='microsoft/rad-dino', help='image encoder')
    parse.add_argument('--cxr_bert_path', type=str, default='microsoft/BiomedVLP-CXR-BERT-specialized', help='text encoder')
    parse.add_argument('--distilgpt2_path', type=str, default='distilbert/distilgpt2', help='text encoder')
    parse.add_argument('--vila_m3_path', type=str, default='MONAI/Llama3-VILA-M3-3B', help='large-language-model')
    parse.add_argument('--libra_path', type=str, default='X-iZhang/libra-v1.0-3b', help='large-language-model')
    parse.add_argument('--meditron_path', type=str, default='epfl-llm/meditron-7b', help='large-language-model')
    parse.add_argument('--chexagent_path', type=str, default='StanfordAIMI/CheXagent-2-3b', help='large-language-model')
    # report generation metrics ckpt
    parse.add_argument('--chexbert_path', type=str, default='chexbert.pth', help='checkpoint for chexbert')
    parse.add_argument('--bert_path', type=str, default='bert-base-uncased', help='checkpoint')
    parse.add_argument('--radgraph_path', type=str, default='radgraph', help='checkpoint')

    # ======================== ablation study configurations ===========================
    # parse.add_argument('--use_eye_gaze', type=str2bool, help="whether use eye-gaze data for align", default='yes')
    parse.add_argument('--ratio_eye_gaze', type=float, help="the ratio of eye-gaze data for align",
                       default=1.0)
    parse.add_argument('--data_ratio', type=float, help="the ratio of training set for downstream task",
                       default=1.0)

    # =============finish=====================#

    args = parse.parse_args()
    args = vars(args)
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H")
    # args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    args['exp_dir_trial'] = f'{args["exp_dir_trial"]}/{args["data_name"]}/{args["task"]}/{args["phase"]}_{args["version"]}_{extension}'
    os.makedirs(args['exp_dir_trial'], exist_ok=True)

    # config logger
    logger = SetLogger(f'{args["exp_dir_trial"]}/log_{extension}.log', 'a')

    # determine absolute path for all checkpoints
    ckpt_name_list = ['chexbert_path', 'radgraph_path', "bert_path", 'vila_m3_path', 'chexagent_path', 'libra_path']

    if not args['online_ckpt']:
        ckpt_name_list.extend(['rad_dino_path', 'meditron_path', 'cxr_bert_path', 'distilgpt2_path'])
    for ckpt_name in ckpt_name_list:
        args[ckpt_name] = os.path.join(args['ckpt_zoo_dir'], args[ckpt_name])

    # determine the monitor_mode

    args['monitor_mode'] = 'max'
    args['monitor_metric'] = 'val_monitor'

    args['prefetch_factor'] = None
    if args['num_workers'] != 0:
        args['prefetch_factor'] = 2

    checkpoint_dir = os.path.join(args['exp_dir_trial'], 'checkpoint')
    os.makedirs(checkpoint_dir, exist_ok=True)

    args['ckpt_dir'] = checkpoint_dir
    args['time'] = extension
    # # save parameters
    # config_dir = f"{args['exp_dir_trial']}/configs"
    # os.makedirs(config_dir, exist_ok=True)
    # file_name = f"{config_dir}/config_{extension}.yaml"
    # print(f'parameters is saved in {file_name}')
    # with open(file_name, 'w') as file:
    #     yaml.dump(args, file, default_flow_style=False)
    return args, logger


def setup_seed(seed):
    # seed init
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch seed init
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SetLogger:
    def __init__(self, filepath, mode='a', lock=None):
        """
        Implements write routine
        :param filepath: the file where to write
        :param mode: can be 'w' or 'a'
        :param lock: pass a shared lock for multi-process write access
        """
        self.filepath = filepath
        if mode not in ['w', 'a']:
            raise ValueError("Mode must be 'w' or 'a'")
        self.mode = mode
        self.lock = lock or (multiprocessing.Lock() if 'multiprocessing' in globals() else threading.Lock())

        try:
            self.file = open(self.filepath, self.mode)
        except Exception as e:
            print(f"Failed to open log file: {e}")
            raise

    def info(self, message):
        """
        Log an info message to the file.
        :param message: The message to log
        """
        with self.lock:
            try:
                self.file.write(message + '\n')
                self.file.flush()
            except Exception as e:
                print(f"Failed to write to log file: {e}")

    def __del__(self):
        """Ensure that the file is closed when the logger is destroyed."""
        try:
            if not self.file.closed:
                self.file.close()
        except Exception as e:
            print(f"Failed to close log file: {e}")
