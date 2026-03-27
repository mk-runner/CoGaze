import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from downstream.utils import setup_arguments, setup_seed
from downstream.model_classification import Classifier, Segmenter

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
torch.set_float32_matmul_precision('medium')


def train():

    args, logger = setup_arguments()
    setup_seed(args['seed'])
    seed_everything(args['seed'])
    # modify torch type

    # save the config
    params = ''
    for key, value in args.items():
        params += f'{key}:\t{value}\n'
    logger.info(params)
    print(params)

    # Trainer
    trainer = pl.Trainer(
        devices=args['devices'],
        num_nodes=args['num_nodes'],
        accelerator=args['accelerator'],
        val_check_interval=args['val_check_interval'],
        limit_val_batches=args['limit_val_batches'],
        max_epochs=args['max_epochs'],
        num_sanity_val_steps=args['num_sanity_val_steps'],
        accumulate_grad_batches=2,
        log_every_n_steps=500,
        callbacks=None,
        logger=None,
        deterministic=True,
        benchmark=False,
        enable_checkpointing=False
    )
    if 'nih' not in args['data_name']:
        num_classes = 2
    else:
        num_classes = 14

    if args['task'] == 'classification':
        model = Classifier(args, logger, num_classes=num_classes)
    elif args['task'] == 'segmentation':
        model = Segmenter(args, logger)
    else:      # segmentation task
        raise NotImplementedError

    try:
        if args['phase'] != 'inference':
            if args['resume'] is None and args['load'] is None:
                pass
            else:
                if args['resume'] is not None:
                    checkpoint = torch.load(args['resume'])
                    optimizer_state = checkpoint['optimizer_state']
                else:  # args['load']
                    checkpoint = torch.load(args['load'])
                    optimizer_state = None
                cur_model_state = model.state_dict()
                pre_model_state = checkpoint['state_dict']
                valid_state = {k: v for k, v in pre_model_state.items() if
                               k in cur_model_state and v.shape == cur_model_state[k].shape}
                invalid_state = {k for k in pre_model_state.keys() if k not in valid_state}
                print(f"missing {invalid_state}")
                cur_model_state.update(valid_state)
                # load adapter state dict
                model.load_state_dict(cur_model_state)
                model.optimizer_state_dict = optimizer_state

            trainer.fit(model=model)

        else:   # test
            cur_model_state = model.state_dict()
            checkpoint = torch.load(args['test_ckpt_path'])
            pre_model_state = checkpoint['state_dict']
            valid_state = {k: v for k, v in pre_model_state.items() if
                           k in cur_model_state and v.shape == cur_model_state[k].shape}
            invalid_state = {k for k in pre_model_state.keys() if k not in valid_state}
            print(f"missing {invalid_state}")
            cur_model_state.update(valid_state)
            # load adapter state dict
            model.load_state_dict(cur_model_state)
            trainer.test(model=model)
    except KeyboardInterrupt:
        print('Interrupted! Cleaning up...')
    finally:
        torch.cuda.empty_cache()


if __name__ == '__main__':
    train()
