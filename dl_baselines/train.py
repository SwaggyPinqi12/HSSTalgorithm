import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import shutil
import segmentation_models_pytorch as smp
from py.evaluate import evaluate
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss
import wandb

# set the dataset name
dataset_name = 'FabricStain'

project_name = 'UnetPP'  # set the project name
dir_train = Path(f'/mnt/e/Run/FabricStain/train/')
dir_val = Path(f'/mnt/e/Run/FabricStain/val/')
dir_checkpoint = Path(f'./checkpoints/{dataset_name}/')

USE_WANDB = False  # global switch, turn off all wandb functions when False


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data)

def load_model(model,optim,model_dir,resume=True,best=False):

    latest_model_path = os.path.join(model_dir, f"{project_name}/latest.pkl")
    resume_path=os.path.join(model_dir, "*")
    if not resume:
        os.system('rm -rf {}'.format(resume_path))

    if not os.path.exists(latest_model_path):
        return 1,-1
    if not best:
        pretrained_model = torch.load(latest_model_path)
    else:
        pretrained_model = torch.load(os.path.join(model_dir, f"{project_name}/best.pkl"))

    print('Load model epoch: {}'.format(pretrained_model['epoch']))
    best_validate_value=pretrained_model['validate_value']
    model.load_state_dict(pretrained_model['model_state_dict'])
    optim.load_state_dict(pretrained_model['optim'])
    return pretrained_model['epoch']+1,best_validate_value

def save_model(model, epoch, optim, model_dir, validate_value, best_validate_value):
    latest_path = os.path.join(model_dir, f"{project_name}/latest.pkl")
    best_path = os.path.join(model_dir, f"{project_name}/best.pkl")
    os.makedirs(os.path.dirname(latest_path), exist_ok=True)
    torch.save(
        {
            'optim': optim.state_dict(),
            "model_state_dict": model.state_dict(),
            'epoch': epoch,
            'validate_value': best_validate_value
        },
        latest_path,
    )
    if validate_value >= best_validate_value:
        best_validate_value = validate_value
        shutil.copy(latest_path, best_path)
    return best_validate_value


def train_model(
        model,
        device,
        epochs: int = 200,
        batch_size: int = 2,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 1,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    train_img_dir = dir_train / 'images'
    train_mask_dir = dir_train / 'masks'
    val_img_dir = dir_val / 'images'
    val_mask_dir = dir_val / 'masks'

    train_set = BasicDataset(train_img_dir, train_mask_dir, img_scale)
    val_set = BasicDataset(val_img_dir, val_mask_dir, img_scale)
    n_train = len(train_set)
    n_val = len(val_set)

    # 2. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    if USE_WANDB:
        experiment = wandb.init(project='ID', name=project_name, entity="ruscheng-zhejiang-university", resume='allow', anonymous='must')
        experiment.config.update(
            dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                 val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
        )
    else:
        experiment = None

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')
    model_dir=dir_checkpoint
    # 3. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0
    start_epochs,best_val=load_model(model,optimizer,model_dir)
    # 4. Begin training
    for epoch in range(start_epochs, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                # assert images.shape[1] == model.in_channels, \
                #     f'Network has been defined with {model.in_channels} input channels, ' \
                #     f'but loaded images have {images.shape[1]} channels. Please check that ' \
                #     'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    true_masks = F.one_hot(true_masks.squeeze_(1), 2).permute(0, 3, 1, 2).float()
                    loss = criterion(masks_pred, true_masks) \
                        + dice_loss(F.softmax(masks_pred, dim=1).float(),
                        true_masks,
                        multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                if USE_WANDB and experiment is not None:
                    experiment.log({
                        'train loss': loss.item(),
                        'step': global_step,
                        'epoch': epoch
                    })
                pbar.set_postfix(**{'loss (batch)': loss.item()})
        histograms = {}
        if USE_WANDB and experiment is not None:
            for tag, value in model.named_parameters():
                tag = tag.replace('/', '.')
                if not (torch.isinf(value) | torch.isnan(value)).any():
                    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                # if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                #     histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
        val_score = evaluate(model, val_loader, device, amp)
        scheduler.step(val_score)

        if USE_WANDB and experiment is not None:
            try:
                experiment.log({
                    'learning rate': optimizer.param_groups[0]['lr'],
                    'validation Dice': val_score,
                    'images': wandb.Image(images[0].cpu()),
                    'masks': {
                        'true': wandb.Image(true_masks[0].float().cpu()),
                        'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                    },
                    'step': global_step,
                    'epoch': epoch,
                    **histograms
                })
            except:
                pass


        logging.info('Validation Dice score: {}'.format(val_score))
        if save_checkpoint:
            best_val=save_model(model,epoch,optimizer,model_dir,val_score,best_val)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    model =smp.UnetPlusPlus( # other choices: UPerNet, DeepLabV3, SegFormer
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=None,
        in_channels=3,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
        classes=2,  # model output channels (number of classes in your dataset)
    )
    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    print("Model Initializing")
    model = model.to(memory_format=torch.channels_last)

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
