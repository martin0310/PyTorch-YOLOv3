from collections import defaultdict

import torch
import torch.nn as nn
import types


def conv2d_forward_with_mask(self, x):
    return nn.functional.conv2d(
        x, self.weight * self.mask, bias=self.bias,
        stride=self.stride, padding=self.padding, dilation=self.dilation,
        groups=self.groups
    )

def add_mask(model):
    print('==> add_mask function..')

    conv_num = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if hasattr(module, "weight"):
                mask = torch.ones_like(module.weight)
                module.register_buffer("mask", mask)
                module.forward = types.MethodType(conv2d_forward_with_mask, module)
                conv_num += 1

    print('conv_num:')
    print(conv_num)





def N_prune(model, pr_cfg, N_cfg):
    print('==> 1xN prune..')

    with torch.no_grad():

        conv_num = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if hasattr(module, "weight") and hasattr(module, "mask"):
                    if conv_num == 0:
                        conv_num += 1
                        continue

                    if module.weight.shape[-2:] == (1, 1):
                        if module.weight.shape[0] % N_cfg[conv_num] != 0:
                            conv_num += 1 
                            continue

                    w = module.weight.detach().clone().cpu()
                    c_out, c_in, k_1, k_2 = w.shape
                    w = w.permute(1, 0, 2, 3)
                    w = w.contiguous().view(-1,N_cfg[conv_num]*k_1*k_2) 
                    prune = int(w.size(0)*pr_cfg[conv_num])
                    w = torch.sum(torch.abs(w), 1)
                    _, indice = torch.topk(w, prune, largest=False)
                    m = torch.ones(w.size(0))
                    m[indice] = 0
                    m = torch.unsqueeze(m, 1)
                    m = m.repeat(1, N_cfg[conv_num]*k_1*k_2)
                    m = m.view(c_in, c_out, k_1, k_2)
                    m = m.permute(1, 0, 2, 3)

                    module.mask.copy_(m)

                    conv_num += 1

def layer_pattern(model, args):
    print('==> find layer pattern..')
    layer_top_k_pattern_list = []

    _3_3_layer_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if hasattr(module, "weight"):

                if module.weight.shape[-2:] == (3, 3):
                    _3_3_layer_count += 1
                    # print('_3_3_layer_count:')
                    # print(_3_3_layer_count)
                    if hasattr(module, "mask"):
                        mask_patterns = defaultdict(int)
                        for i in range(module.mask.size(0)):
                            for j in range(module.mask.size(1)):
                                if torch.count_nonzero(module.mask[i][j]) == 9:
                                    kernel = module.weight[i][j].detach().clone()
                                    flat_kernel = kernel.view(-1)

                                    abs_flat_kernel = torch.abs(flat_kernel)
                                    _, indices = torch.topk(abs_flat_kernel, 4)
                                    
                                    mask = torch.zeros_like(flat_kernel)

                                    mask[indices] = 1

                                    pattern_key = tuple(mask.tolist())

                                    mask_patterns[pattern_key] += 1

                        top_k_patterns = sorted(mask_patterns, key=mask_patterns.get, reverse=True)[:args.kernel_pattern_num]
                        top_k_masks = [torch.tensor(pattern).view(1, 3, 3) for pattern in top_k_patterns]

                        layer_top_k_pattern_list.append(top_k_masks)

    return layer_top_k_pattern_list


def block_pattern_prune(model, args, layer_top_k_pattern_list, N_cfg):
    print('==> enter block_pattern_prune..')

    _3_3_layer_count = 0
    conv_layer_index = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if hasattr(module, "weight") and hasattr(module, "mask"):
                
                #check if it is 3x3 kernel
                if module.weight.shape[-2:] == (3, 3):
                    top_k_masks_in_layer = layer_top_k_pattern_list[_3_3_layer_count]

                    # if (conv_layer_index == 0) and (_3_3_layer_count == 0):
                    #     _3_3_layer_count += 1
                    #     conv_layer_index += 1
                    #     continue

                    _3_3_layer_count += 1 
                    
                    for i in range(module.mask.size(1)):
                        for j in range(0, module.mask.size(0), N_cfg[conv_layer_index]):

                            #check 1xN block is all 0 or all 1
                            if (module.mask[j][i] == 1).sum().item() == 0:
                                for index in range(1, N_cfg[conv_layer_index]):
                                    if (module.mask[j + index][i] == 1).sum().item() != 0:
                                        print('conv_layer_index:')
                                        print(conv_layer_index)
                                        print('error')
                                        raise Exception(f"weight not in 1x{N_cfg[conv_layer_index]} form")
                    
                            elif (module.mask[j][i] == 1).sum().item() == 9:
                                for index in range(1, N_cfg[conv_layer_index]):
                                    if (module.mask[j + index][i] == 1).sum().item() != 9:
                                        print('error')
                                        print('conv_layer_index:')
                                        print(conv_layer_index)
                                        raise Exception(f"weight not in 1x{N_cfg[conv_layer_index]} form")
                        
                            if (module.mask[j][i] == 1).sum().item() == 9:
                                block_kernel_sum = 0
                                
                                for block_kernel_index in range(N_cfg[conv_layer_index]):
                                    block_kernel_sum += torch.abs(module.weight[j + block_kernel_index][i].detach().clone())

                                best_mask = None
                                best_sum = torch.tensor(float('-inf')).to(block_kernel_sum.device)

                                for mask in top_k_masks_in_layer:
                                    mask = mask.to(block_kernel_sum.device) 
                                    masked_kernel = block_kernel_sum * mask
                                    current_sum = masked_kernel.abs().sum()

                                    if current_sum > best_sum:
                                        best_sum = current_sum
                                        best_mask = mask

                                for block_kernel_index in range(N_cfg[conv_layer_index]):
                                    module.mask[j + block_kernel_index][i] = best_mask
                conv_layer_index += 1


def count_divisible(model, N_cfg):


    _3_3_count = 0
    _1_1_count = 0
    conv_layer_index = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if hasattr(module, "weight") and hasattr(module, "mask"):
                if module.weight.shape[-2:] == (3, 3):
                    if module.weight.shape[0] % N_cfg[conv_layer_index] != 0:
                        print('module:')
                        print(module)
                        print('module.weight.shape:')
                        print(module.weight.shape)
                        print('module.weight.shape[0]:')
                        print(module.weight.shape[0])
                        print('N_cfg[conv_layer_index]:')
                        print(N_cfg[conv_layer_index])
                        print('==================')
                        _3_3_count += 1
                elif module.weight.shape[-2:] == (1, 1):
                    if module.weight.shape[0] % N_cfg[conv_layer_index] != 0:
                        print('module:')
                        print(module)
                        print('module.weight.shape:')
                        print(module.weight.shape)
                        print('module.weight.shape[0]:')
                        print(module.weight.shape[0])
                        print('N_cfg[conv_layer_index]:')
                        print(N_cfg[conv_layer_index])
                        print('==================')
                        _1_1_count += 1

            conv_layer_index += 1

    print(f"3x3 Not divisible by N_cfg: {_3_3_count} layers")
    print(f"1x1 Not divisible by N_cfg: {_1_1_count} layers")