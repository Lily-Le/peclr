import os
from pprint import pformat

from easydict import EasyDict as edict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CometLogger
from src.constants import (
    COMET_KWARGS,
    HYBRID2_CONFIG,
    SIMCLR_CONFIG ,
    BASE_DIR,
    TRAINING_CONFIG_PATH,
)
from src.data_loader.data_set_cbg import Data_Set_cbg
from src.data_loader.utils import get_data_cbg, get_train_val_split
from src.experiments.utils import (
    get_callbacks,
    get_general_args,
    get_model,
    prepare_name,
    save_experiment_key,
    update_model_params,
    update_train_params,
)
from src.utils import get_console_logger, read_json


def main():
    # get configs
    experiment_type = "hybrid2"
    # experiment_type = "simclr"
    console_logger = get_console_logger(__name__)
    args = get_general_args("Hybrid model 2 training script.")
    # args = get_general_args("Simclr model training script.")
    train_param = edict(read_json(TRAINING_CONFIG_PATH))
    train_param = update_train_params(args, train_param)
    # model_param_path = SIMCLR_CONFIG
    model_param_path = HYBRID2_CONFIG  # SIMCLR_CONFIG
    model_param = edict(read_json(model_param_path))
    console_logger.info(f"Train parameters {pformat(train_param)}")
    seed_everything(train_param.seed)
    '''
    # data preperation
    # DataLoader: return  sample = {
        #     "image": fg_img,
        #     "mask": fg_mask, 
        #     "K": camera_param,
        #     "joints3D": joints3D,
        #     "joints_valid": joints_valid,
        # }
    # Data_set: (in hybrid2 experiment I commented the to tensor & norm here)
    #  return {
    #         **{"transformed_image1": img1, "transformed_image2": img2},
    #         **{"mask": sample["mask"]},
    #         **{f"{k}_1": v for k, v in param1.items() if v is not None},
    #         **{f"{k}_2": v for k, v in param2.items() if v is not None},
    #     }
    # 
    # get_data: dataset concatnate
    '''
    data = get_data_cbg(
        Data_Set_cbg, train_param, sources=args.sources, experiment_type=experiment_type
    )

    # Control the backround of positive and negative samples
    # Dataloader: just do basic transformations like crop, translation, etc
    train_data_loader, val_data_loader = get_train_val_split(
        data, batch_size=train_param.batch_size, num_workers=train_param.num_workers
    )


    
    # For the returned samples in one batch, change the backgrounds accordingly
    # Logger
    experiment_name = prepare_name(
        f"{experiment_type}_", train_param, hybrid_naming=False
    )
    comet_logger = CometLogger(**COMET_KWARGS, experiment_name=experiment_name)

    # model
    model_param = update_model_params(model_param, args, len(data), train_param)
    model_param.augmentation = [
        key for key, value in train_param.augmentation_flags.items() if value
    ]
    console_logger.info(f"Model parameters {pformat(model_param)}")
    model = get_model(
        experiment_type="hybrid2",#"simclr"
        heatmap_flag=args.heatmap,
        denoiser_flag=args.denoiser,
    )(config=model_param)

    # callbacks
    callbacks = get_callbacks(
        logging_interval=args.log_interval,
        experiment_type="hybrid2",#"simclr"
        save_top_k=args.save_top_k,
        period=args.save_period,
    )
    # trainer
    trainer = Trainer(
        accumulate_grad_batches=train_param.accumulate_grad_batches,
        # resume_from_checkpoint='/home/d3-ai/cll/peclr/data/models/Hybrid2-Frei-cgbgr/9a5d29db6f584042a53343cbe9faa5c0/checkpoints/epoch=69.ckpt',
        # resume_from_checkpoint='/home/d3-ai/cll/peclr/data/models/Hybrid2-Frei-cgbgr/9a5d29db6f584042a53343cbe9faa5c0/checkpoints/epoch=69.ckpt',
        # resume_from_checkpoint = '/home/zlc/cll/code/peclr_cbg/data/models_res18/hybrid2-frei-cgbg/6a61d1fc5a254615a6d2f9071e1e7a44/checkpoints/epoch=14.ckpt',
        gpus="0",
        logger=comet_logger,
        max_epochs=train_param.epochs,
        precision=train_param.precision,
        amp_backend="native",
        **callbacks,
    )
    trainer.logger.experiment.set_code(
        overwrite=True,
        filename=os.path.join(
            BASE_DIR, "src", "experiments", "peclr_training.py"
        ),
    )
    if args.meta_file is not None:
        save_experiment_key(
            experiment_name, trainer.logger.experiment.get_key(), args.meta_file
        )
    trainer.logger.experiment.log_parameters(train_param)
    trainer.logger.experiment.log_parameters(model_param)
    trainer.logger.experiment.add_tags(["pretraining", "HYBRID2"] + args.tag)
    # trainer.logger.experiment.add_tags(["pretraining", "SIMCLR"] + args.tag)
    
    # training
    trainer.fit(model, train_data_loader, val_data_loader)


if __name__ == "__main__":
    main()
