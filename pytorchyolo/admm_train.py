#! /usr/bin/env python3

from __future__ import division

import os
import argparse
import tqdm
import shutil

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from pytorchyolo.models import load_model
from pytorchyolo.utils.logger import Logger
from pytorchyolo.utils.utils import to_cpu, load_classes, print_environment_info, provide_determinism, worker_seed_set
from pytorchyolo.utils.datasets import ListDataset
from pytorchyolo.utils.augmentations import AUGMENTATION_TRANSFORMS
#from pytorchyolo.utils.transforms import DEFAULT_TRANSFORMS
from pytorchyolo.utils.parse_config import parse_data_config
from pytorchyolo.utils.loss import compute_loss
from pytorchyolo.utils.pattern_utils import *
from pytorchyolo.utils.admm import *
from pytorchyolo.test import _evaluate, _create_validation_data_loader

from terminaltables import AsciiTable

from torchsummary import summary


def _create_data_loader(img_path, batch_size, img_size, n_cpu, multiscale_training=False):
    """Creates a DataLoader for training.

    :param img_path: Path to file containing all paths to training images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :param multiscale_training: Scale images to different sizes randomly
    :type multiscale_training: bool
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = ListDataset(
        img_path,
        img_size=img_size,
        multiscale=multiscale_training,
        transform=AUGMENTATION_TRANSFORMS)
    print('n_cpu:')
    print(n_cpu)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_seed_set)
    return dataloader


def run():
    print_environment_info()
    parser = argparse.ArgumentParser(description="Trains the YOLO model.")
    parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("-d", "--data", type=str, default="config/coco.data", help="Path to data config file (.data)")
    parser.add_argument("-e", "--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("-v", "--verbose", action='store_true', help="Makes the training more verbose")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--pretrained_weights", type=str, help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="Interval of epochs between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="Interval of epochs between evaluations on validation set")
    parser.add_argument("--multiscale_training", action="store_true", help="Allow multi-scale training")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="Evaluation: IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.1, help="Evaluation: Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="Evaluation: IOU threshold for non-maximum suppression")
    parser.add_argument("--logdir", type=str, default="/mnt/Data-Weight/1xN_new/yolov3/log/admm", help="Directory for training log files (e.g. for TensorBoard)")
    parser.add_argument("--seed", type=int, default=-1, help="Makes results reproducable. Set -1 to disable.")
    parser.add_argument("--block_pattern_prune", action="store_true", help="block pattern prune")
    parser.add_argument("--N", type=int, default=4, help="size of N")
    parser.add_argument("--pr_rate", type=float, default=0.5, help="pruning rate")
    parser.add_argument("--only_1_N_prune", action="store_true", help="only 1xN prune")
    parser.add_argument("--kernel_pattern_num", type=int, default=4, help="pattern number of every layer")
    parser.add_argument("--gpu", type=int, default=0, help="which gpu")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint (.pth) to resume training.")
    parser.add_argument("--rho", type=float, default=0.01, help="rho for admm")
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    if args.seed != -1:
        provide_determinism(args.seed)

    logger = Logger(f"{args.logdir}/{args.kernel_pattern_num}_patterns")  # Tensorboard logger

    # Create output directories if missing
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(args.data)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # -----------------------------
    # Set the specified GPU device
    # -----------------------------
    # Use the specified GPU if available, otherwise fallback to CPU.
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ############
    # Create model
    # ############

    model = load_model(args.model, args.gpu, args.pretrained_weights)

    # Print model
    if args.verbose:
        summary(model, input_size=(3, model.hyperparams['height'], model.hyperparams['height']))

    mini_batch_size = model.hyperparams['batch'] // model.hyperparams['subdivisions']

    # #################
    # Create Dataloader
    # #################

    # Load training dataloader
    dataloader = _create_data_loader(
        train_path,
        mini_batch_size,
        model.hyperparams['height'],
        args.n_cpu,
        args.multiscale_training)

    # Load validation dataloader
    validation_dataloader = _create_validation_data_loader(
        valid_path,
        mini_batch_size,
        model.hyperparams['height'],
        args.n_cpu)

    # ################
    # Create optimizer
    # ################

    params = [p for p in model.parameters() if p.requires_grad]

    if (model.hyperparams['optimizer'] in [None, "adam"]):
        optimizer = optim.Adam(
            params,
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
        )
    elif (model.hyperparams['optimizer'] == "sgd"):
        optimizer = optim.SGD(
            params,
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
            momentum=model.hyperparams['momentum'])
    else:
        print("Unknown optimizer. Please choose between (adam, sgd).")

    print("\n---- Evaluating Model Before Training and Pruning ----")
    # Evaluate the model on the validation set
    metrics_output = _evaluate(
        args.gpu,
        model,
        validation_dataloader,
        class_names,
        img_size=model.hyperparams['height'],
        iou_thres=0.5,
        conf_thres=0.01,
        nms_thres=0.4,
        verbose=args.verbose
    )

    # if args.only_1_N_prune or args.block_pattern_prune:
    add_mask(model)

    N_cfg = [args.N] * 75
    pr_cfg = [args.pr_rate] * 75

    # no pruning for first layer
    pr_cfg[0] = 0
    
    if args.resume_from is None:
        # only 1xN prune for 1x1 conv layer before admm
        N_prune_admm(model, pr_cfg, N_cfg)
        layer_top_k_pattern_list = layer_pattern(model, args)
        admm_block_pattern_prune(model, N_cfg, layer_top_k_pattern_list)
        Z, U = initialize_Z_and_U(model)
        Y, V = initialize_Y_and_V(model)

    # -----------------------------
    # Optionally resume from checkpoint
    # -----------------------------
    start_epoch = 1
    best_map = 0.0
    current_map = 0.0
    is_best = False

    train_iou_loss_list = []
    train_obj_loss_list = []
    train_class_loss_list = []
    train_loss_list = []

    val_precision_list = []
    val_recall_list = []
    val_mAP_list = []
    val_f1_list = []



    if args.resume_from is not None:
        print(f"Resuming training from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_map = checkpoint['best_map']
        model.seen = checkpoint['model_seen']
        batches_done = checkpoint['batches_done']
        val_precision_list = checkpoint['val_precision_list']
        val_recall_list = checkpoint['val_recall_list']
        val_mAP_list = checkpoint['val_mAP_list']
        val_f1_list = checkpoint['val_f1_list']
        train_iou_loss_list = checkpoint['train_iou_loss_list']
        train_obj_loss_list = checkpoint['train_obj_loss_list']
        train_class_loss_list = checkpoint['train_class_loss_list']
        train_loss_list = checkpoint['train_loss_list']
        W = checkpoint["W"]
        # Z = checkpoint["Z"]
        # Y = checkpoint["Y"]
        # U = checkpoint["U"]
        # V = checkpoint["V"]

        Z = tuple(tensor.cpu() for tensor in checkpoint["Z"])
        Y = tuple(tensor.cpu() for tensor in checkpoint["Y"])
        U = tuple(tensor.cpu() for tensor in checkpoint["U"])
        V = tuple(tensor.cpu() for tensor in checkpoint["V"])
        layer_top_k_pattern_list = checkpoint["layer_top_k_pattern_list"]

        print(f"Resumed from epoch {checkpoint['epoch']}.")

        print('val_mAP_list:')
        print(val_mAP_list)

        print("\n---- Evaluating Model After Loading Checkpoint ----")
        # Evaluate the model on the validation set
        metrics_output = _evaluate(
            args.gpu,
            model,
            validation_dataloader,
            class_names,
            img_size=model.hyperparams['height'],
            iou_thres=0.5,
            conf_thres=0.01,
            nms_thres=0.4,
            verbose=args.verbose
        )
    
    # if args.only_1_N_prune:
    #     N_prune(model, pr_rate, N)

    # if metrics_output is not None:
    #     precision, recall, AP, f1, ap_class = metrics_output
    #     evaluation_metrics = [
    #         ("validation/precision", precision.mean()),
    #         ("validation/recall", recall.mean()),
    #         ("validation/mAP", AP.mean()),
    #         ("validation/f1", f1.mean())]
    #     # logger.list_of_scalars_summary(evaluation_metrics, epoch)
    #     logger.list_of_scalars_summary(evaluation_metrics, -1)




    # skip epoch zero, because then the calculations for when to evaluate/checkpoint makes more intuitive sense
    # e.g. when you stop after 30 epochs and evaluate every 10 epochs then the evaluations happen after: 10,20,30
    # instead of: 0, 10, 20
    print('args.rho')
    print(args.rho)
    for epoch in range(start_epoch, args.epochs+1):

        print("\n---- Training Model ----")

        model.train()  # Set model to training mode

        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")):
            batches_done = len(dataloader) * epoch + batch_i

            # if batch_i == 5:
            #     break

            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)

            outputs = model(imgs)

            loss, loss_components = compute_loss(outputs, targets, model)

            loss = admm_loss(args, device, model, Z, U, Y, V, loss)

            loss.backward()

            ###############
            # Run optimizer
            ###############

            if batches_done % model.hyperparams['subdivisions'] == 0:
                # Adapt learning rate
                # Get learning rate defined in cfg
                lr = model.hyperparams['learning_rate']
                if batches_done < model.hyperparams['burn_in']:
                    # Burn in
                    lr *= (batches_done / model.hyperparams['burn_in'])
                else:
                    # Set and parse the learning rate to the steps defined in the cfg
                    for threshold, value in model.hyperparams['lr_steps']:
                        if batches_done > threshold:
                            lr *= value
                # Log the learning rate
                logger.scalar_summary("train/learning_rate", lr, batches_done)
                # Set learning rate
                for g in optimizer.param_groups:
                    g['lr'] = lr

                # Run optimizer
                optimizer.step()
                # Reset gradients
                optimizer.zero_grad()

            # ############
            # Log progress
            # ############
            if args.verbose:
                print(AsciiTable(
                    [
                        ["Type", "Value"],
                        ["IoU loss", float(loss_components[0])],
                        ["Object loss", float(loss_components[1])],
                        ["Class loss", float(loss_components[2])],
                        ["Loss", float(loss_components[3])],
                        ["Batch loss", to_cpu(loss).item()],
                    ]).table)

            train_iou_loss = float(loss_components[0])
            train_obj_loss = float(loss_components[1])
            train_class_loss = float(loss_components[2])
            train_loss = to_cpu(loss).item()

            # Tensorboard logging
            tensorboard_log = [
                ("train/iou_loss", train_iou_loss),
                ("train/obj_loss", train_obj_loss),
                ("train/class_loss", train_class_loss),
                ("train/loss", train_loss)]
            logger.list_of_scalars_summary(tensorboard_log, batches_done)
            train_iou_loss_list.append(train_iou_loss)
            train_obj_loss_list.append(train_obj_loss)
            train_class_loss_list.append(train_class_loss)
            train_loss_list.append(train_loss)

            model.seen += imgs.size(0)

        

        # ########
        # Evaluate
        # ########

        if epoch % args.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            metrics_output = _evaluate(
                args.gpu,
                model,
                validation_dataloader,
                class_names,
                img_size=model.hyperparams['height'],
                iou_thres=args.iou_thres,
                conf_thres=args.conf_thres,
                nms_thres=args.nms_thres,
                verbose=args.verbose
            )

            if metrics_output is not None:
                precision, recall, AP, f1, ap_class = metrics_output
                precision_mean = precision.mean()
                recall_mean = recall.mean()
                AP_mean = AP.mean()
                f1_mean = f1.mean()

                evaluation_metrics = [
                    ("validation/precision", precision_mean),
                    ("validation/recall", recall_mean),
                    ("validation/mAP", AP_mean),
                    ("validation/f1", f1_mean)]
                logger.list_of_scalars_summary(evaluation_metrics, epoch)
                val_precision_list.append(precision_mean)
                val_recall_list.append(recall_mean)
                val_mAP_list.append(AP_mean)
                val_f1_list.append(f1_mean)


                current_map = AP_mean
                
                is_best = current_map > best_map
                print('is_best:')
                print(is_best)

                if current_map > best_map:
                    best_map = current_map
                
        W = update_W(model)
        Z = update_Z(W, U, N_cfg, layer_top_k_pattern_list, model)
        Y = update_Y(W, V, pr_cfg, N_cfg, model, Y, args)
        U = update_U(U, W, Z)
        V = update_V(V, W, Y)        
        # #############
        # Save progress
        # #############

        # Save model to checkpoint file
        if epoch % args.checkpoint_interval == 0:
            
            checkpoint_path = f"/mnt/Data-Weight/1xN_new/yolov3/checkpoint/admm/{args.kernel_pattern_num}_patterns/admm_yolov3_last.pth"
            
            print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
            # Save model and optimizer state, plus current epoch
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_map': best_map,
                'model_seen': model.seen,
                'batches_done': batches_done,
                'val_precision_list': val_precision_list,
                'val_recall_list': val_recall_list,
                'val_mAP_list': val_mAP_list,
                'val_f1_list': val_f1_list,
                'train_iou_loss_list': train_iou_loss_list,
                'train_obj_loss_list': train_obj_loss_list,
                'train_class_loss_list': train_class_loss_list,
                'train_loss_list': train_loss_list,
                'W': W,
                'Z': Z,
                'Y': Y,
                'U': U,
                'V': V,
                'layer_top_k_pattern_list': layer_top_k_pattern_list
            }, checkpoint_path)

            if is_best:
                shutil.copyfile(checkpoint_path, f"/mnt/Data-Weight/1xN_new/yolov3/checkpoint/admm/{args.kernel_pattern_num}_patterns/admm_yolov3_best.pth")

    
    best_model_path = f"/mnt/Data-Weight/1xN_new/yolov3/checkpoint/admm/{args.kernel_pattern_num}_patterns/admm_yolov3_best.pth"
    print(f"---- Loading best checkpoint after admm: '{best_model_path}' ----")
    ckpt_prune_admm = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt_prune_admm['state_dict'])
    block_pattern_prune(model, args, layer_top_k_pattern_list, N_cfg)
    retrain_1_N_prune(model, args, layer_top_k_pattern_list, pr_cfg, N_cfg)
    best_model_path_pruned = f"/mnt/Data-Weight/1xN_new/yolov3/checkpoint/admm/{args.kernel_pattern_num}_patterns/admm_yolov3_best_pruned.pth"
    torch.save({
        'state_dict': model.state_dict()
    }, best_model_path_pruned)
    print('best_map:')
    print(best_map)


if __name__ == "__main__":
    run()
