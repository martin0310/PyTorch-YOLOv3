#! /usr/bin/env python3

from __future__ import division
from collections import defaultdict

import argparse
import tqdm
import numpy as np

import torch
import torch.nn as nn

from pytorchyolo.models import load_model
from pytorchyolo.utils.pattern_utils import *
from pytorchyolo.utils.utils import print_environment_info

def calculate_conv_parameters(model, load_pruned_model):
    total_conv_params = 0
    total_depthwise_conv_params = 0
    non_zero_depthwise_weights = 0
    is_depth_wise = False

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            if hasattr(module, 'weight'):
            
                # check if this module is depth wise convolution:
                if module.groups == module.in_channels == module.out_channels:
                    is_depth_wise = True
                else:
                    is_depth_wise = False

                if load_pruned_model:
                    # if isinstance(module, BlockL1Conv):
                    if hasattr(module, 'mask'):
                        non_zero_weights = torch.nonzero(module.mask, as_tuple=False)
                        total_conv_params += non_zero_weights.size(0)
                    else:
                        # Use the tensor's nonzero elements to count non-zero parameters
                        non_zero_weights = torch.nonzero(module.weight, as_tuple=False)
                        total_conv_params += non_zero_weights.size(0)
                    
                    # sum only depth-wise convolution weights
                    if is_depth_wise:
                        
                        #check if it is depthwise and kernel is 3x3
                        if module.weight.shape[-2:] == (3, 3):
                            total_depthwise_conv_params += module.weight.numel()
                            if hasattr(module, 'mask'):
                                non_zero_weights_in_depth_wise = torch.nonzero(module.mask, as_tuple=False)
                                non_zero_depthwise_weights += non_zero_weights_in_depth_wise.size(0)
                            else:
                                non_zero_weights_in_depth_wise = torch.nonzero(module.weight, as_tuple=False)
                                non_zero_depthwise_weights += non_zero_weights_in_depth_wise.size(0)
                else:
                    
                    total_conv_params += module.weight.numel()
    return total_conv_params, total_depthwise_conv_params, non_zero_depthwise_weights


def calculate_flops(model, input_size, device, load_pruned_model):
    total_flops = 0
    total_depthwise_flops = 0 
    total_non_zero_depthwise_flops = 0

    input_tensor = torch.rand(1, *input_size).to(device)  # Move input tensor to the device
    
    def hook_fn_forward(module, input, output):
        nonlocal total_flops
        nonlocal total_depthwise_flops
        nonlocal total_non_zero_depthwise_flops

        if isinstance(module, nn.Conv2d):
            # check if this module is depth wise convolution:
            if module.groups == module.in_channels == module.out_channels:
                is_depth_wise = True
            else:
                is_depth_wise = False

            output_height, output_width = output.shape[2:]
            
            in_channels_per_group = module.in_channels // module.groups
            for out_channel in range(module.out_channels):
                for in_channel in range(in_channels_per_group):
                    for kernel_y in range(module.kernel_size[0]):
                        for kernel_x in range(module.kernel_size[1]):
                            # Check for non-zero weights
                            if load_pruned_model:
                                if hasattr(module, 'mask'):
                                    mask = module.mask[out_channel, in_channel, kernel_y, kernel_x]
                                    if mask.item() != 0:
                                        # total_flops += 2 * output_height * output_width
                                        total_flops +=  output_height * output_width
                                else:
                                    weight = module.weight[out_channel, in_channel, kernel_y, kernel_x]
                                    if weight.item() != 0:
                                        # total_flops += 2 * output_height * output_width
                                        total_flops += output_height * output_width
                                
                                #sum only depth-wise convolution FLOPs
                                if is_depth_wise:
                                    if module.weight.shape[-2:] == (3, 3):
                                        total_depthwise_flops += output_height * output_width
                                        if hasattr(module, 'mask'):
                                            depthwise_mask = module.mask[out_channel, in_channel, kernel_y, kernel_x]
                                            if depthwise_mask.item() != 0:
                                                # total_flops += 2 * output_height * output_width
                                                total_non_zero_depthwise_flops +=  output_height * output_width
                                        else:
                                            depthwise_weight = module.weight[out_channel, in_channel, kernel_y, kernel_x]
                                            if depthwise_weight.item() != 0:
                                                # total_flops += 2 * output_height * output_width
                                                total_non_zero_depthwise_flops += output_height * output_width

                            else:
                                weight = module.weight[out_channel, in_channel, kernel_y, kernel_x]
                                # total_flops += 2 * output_height * output_width
                                total_flops +=  output_height * output_width
            print('total_flops:')
            print(total_flops)
    # Register hook for all modules
    hooks = []
    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(hook_fn_forward))

    # Perform a forward pass to trigger the hooks
    model.eval()
    with torch.no_grad():
        model(input_tensor)

    return total_flops, total_depthwise_flops, total_non_zero_depthwise_flops

def to_human_readable(total):
    # Define the suffixes and their corresponding values
    suffixes = [("G", 1e9), ("M", 1e6), ("K", 1e3), ("F", 1)]
    for suffix, threshold in suffixes:
        if total >= threshold:
            return f"{total / threshold:.2f} {suffix}"
    return f"{total}"  # Fallback case

def check_only_normal_convs(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if module.groups != 1:
                raise RuntimeError(f"Non-normal Conv2d found in layer: {name} (groups={module.groups})")
    print("All Conv2d layers are normal (groups=1)")

def run():
    print_environment_info()
    parser = argparse.ArgumentParser(description="Evaluate validation data.")
    parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("-pretrained_w", "--pretrained_weights", type=str, default="weights/yolov3.weights", help="Path to weights or checkpoint file (.weights or .pth)")
    parser.add_argument("-w", "--weights", type=str, default="", help="Path to weights or checkpoint file (.weights or .pth)")
    parser.add_argument("--kernel_pattern_num", type=int, default=4, help="pattern number of every layer")
    parser.add_argument("--N", type=int, default=4, help="size of N")
    parser.add_argument("--gpu", type=int, default=0, help="which gpu")
    parser.add_argument("--load_pruned_model", help="check whether to load pruned model or not to calculate flops and parameters", action="store_true")
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    model = load_model(args.model, args.gpu, args.pretrained_weights)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    add_mask(model)
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    model.to('cpu')
    check_only_normal_convs(model)
    # print('model:')
    # print(model)
    
    # Calculate total convolutional parameters
    total_conv_params, total_depthwise_conv_params, non_zero_depthwise_weights = calculate_conv_parameters(model, args.load_pruned_model)
    total_conv_params_human_readable = to_human_readable(total_conv_params)
    total_depthwise_conv_params_human_readable = to_human_readable(total_depthwise_conv_params)
    non_zero_depthwise_weights_human_readable = to_human_readable(non_zero_depthwise_weights)

    input_size = (3, 416, 416) # Adjust based on your model's input
    device = 'cpu'
    total_flops, total_depthwise_flops, total_non_zero_depthwise_flops = calculate_flops(model, input_size, device, args.load_pruned_model)
    # print(f"Total FLOPs considering only non-zero weights in kernels: {total_flops}")
    total_human_readable_flops = to_human_readable(total_flops)
    total_depthwise_flops_human_readable = to_human_readable(total_depthwise_flops)
    total_non_zero_depthwise_flops_human_readable = to_human_readable(total_non_zero_depthwise_flops)

    print(f"Total Convolutional Parameters: {total_conv_params_human_readable}")
    print(f"Total Depth-wise Parameters: {total_depthwise_conv_params_human_readable}")
    print(f"Total Non-zero Depth-wise Parameters: {non_zero_depthwise_weights_human_readable}")
    print(f"Total FLOPs: {total_human_readable_flops}")
    print(f"Total Depth-wise FLOPs: {total_depthwise_flops_human_readable}")
    print(f"Total Non-zero Depth-wise FLOPs: {total_non_zero_depthwise_flops_human_readable}")

    

    


if __name__ == "__main__":
    run()