import torch
import torch.nn as nn

def admm_block_pattern_prune(model, N_cfg, layer_top_k_pattern_list):
    print('==> Enter ADMM  block_pattern_prune..')

    with torch.no_grad():
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
                            
                                if (module.mask[j][i] == 1).sum().item() == 9:
                                    block_kernel_sum = 0
                                    
                                    for block_kernel_index in range(N_cfg[conv_layer_index]):
                                        block_kernel_sum += torch.abs(module.weight[j + block_kernel_index][i].detach().clone())

                                    best_mask = None
                                    best_sum = torch.tensor(float('-inf')).to(block_kernel_sum.device)

                                    for mask in top_k_masks_in_layer:
                                        mask = mask.to(block_kernel_sum.device) 
                                        masked_kernel = block_kernel_sum * mask
                                        # current_sum = masked_kernel.abs().sum()
                                        current_sum = (masked_kernel ** 2).sum()

                                        if current_sum > best_sum:
                                            best_sum = current_sum
                                            best_mask = mask

                                    for block_kernel_index in range(N_cfg[conv_layer_index]):
                                        # module.mask[j + block_kernel_index][i] = best_mask
                                        module.weight[j + block_kernel_index][i].mul_(torch.squeeze(best_mask))
                    conv_layer_index += 1


def admm_loss(args, device, model, Z, U, Y, V, loss):
    idx = 0
    # _3_3_layer_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if hasattr(module, "weight") and hasattr(module, "mask"):
                if module.weight.shape[-2:] == (3, 3):

                    # if _3_3_layer_count == 0:
                    #     _3_3_layer_count += 1
                    #     continue

                    u = U[idx].to(device)
                    z = Z[idx].to(device)

                    if idx == 0:
                        loss += args.rho / 2 * (module.weight - z + u).norm() ** 2
                    else:
                        v = V[idx].to(device)
                        y = Y[idx].to(device)
                        loss += (args.rho / 2 * (module.weight - z + u).norm() ** 2 + args.rho / 2 * (module.weight - y + v).norm() ** 2)
                    
                    
                    idx += 1
                    # _3_3_layer_count += 1
    # print('\n==> admm_loss\n')
    return loss





def initialize_Z_and_U(model):
    print('\n==> initialize_Z_and_U..\n')

    Z = ()
    U = ()

    # idx = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if hasattr(module, "weight") and hasattr(module, "mask"):
                if module.weight.shape[-2:] == (3, 3):
                    
                    # if idx == 0:
                    #     idx += 1
                    #     continue

                    Z += (module.weight.detach().cpu().clone(),)
                    U += (torch.zeros_like(module.weight.detach().clone()).cpu(),)

                    # idx += 1

    return Z, U



def initialize_Y_and_V(model):
    print('\n==> initialize_Y_and_V..\n')

    Y = ()
    V = ()

    # idx = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if hasattr(module, "weight") and hasattr(module, "mask"):
                if module.weight.shape[-2:] == (3, 3):
                    # if idx == 0:
                    #     idx += 1
                    #     continue
                    Y += (module.weight.detach().cpu().clone(),)
                    V += (torch.zeros_like(module.weight.detach().clone()).cpu(),)

                    # idx += 1
    
    return Y, V


def update_W(model):
    print('\n==> Update_W..\n')

    # _3_3_layer_count = 0
    W = ()
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if hasattr(module, "weight") and hasattr(module, "mask"):
                if module.weight.shape[-2:] == (3, 3):

                    # if _3_3_layer_count == 0:
                    #     _3_3_layer_count += 1
                    #     continue
                    
                    W += (module.weight.detach().cpu().clone(),)

                    # _3_3_layer_count += 1
    return W


def update_Z(W, U, N_cfg, layer_top_k_pattern_list, model):
    print('\n==> update_Z..\n')

    new_Z = ()

    _3_3_layer_count = 0
    conv_layer_index = 0

    # idx = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if hasattr(module, "weight") and hasattr(module, "mask"):
                
                if module.weight.shape[-2:] == (3, 3):
                    # z = W[idx] + U[idx]
                    z = W[_3_3_layer_count] + U[_3_3_layer_count]
                    top_k_masks_in_layer = layer_top_k_pattern_list[_3_3_layer_count]

                    # if _3_3_layer_count == 0:
                    #     _3_3_layer_count += 1
                    #     conv_layer_index += 1
                    #     continue

                    for i in range(module.mask.size(1)):
                        for j in range(0, module.mask.size(0), N_cfg[conv_layer_index]):
                        
                            if (module.mask[j][i] == 1).sum().item() == 9:
                                block_kernel_sum = 0
                                
                                for block_kernel_index in range(N_cfg[conv_layer_index]):
                                    block_kernel_sum += torch.abs(z[j + block_kernel_index][i].detach().clone())

                                best_mask = None
                                best_sum = torch.tensor(float('-inf')).to(block_kernel_sum.device)

                                for mask in top_k_masks_in_layer:
                                    mask = mask.to(block_kernel_sum.device) 
                                    masked_kernel = block_kernel_sum * mask
                                    # current_sum = masked_kernel.abs().sum()
                                    current_sum = (masked_kernel ** 2).sum()

                                    if current_sum > best_sum:
                                        best_sum = current_sum
                                        best_mask = mask

                                for block_kernel_index in range(N_cfg[conv_layer_index]):
                                    # module.mask[j + block_kernel_index][i] = best_mask
                                    with torch.no_grad():
                                        z[j + block_kernel_index][i].mul_(torch.squeeze(best_mask))
                    new_Z += (z,)
                    _3_3_layer_count += 1
                    # idx += 1

                conv_layer_index += 1

    return new_Z


def update_U(U, W, Z):
    print('\n==> update_U..\n')

    new_U = ()
    for u, w, z in zip(U, W, Z):
        new_u = u + w - z
        new_U += (new_u,)
    return new_U


def update_Y(W, V, pr_cfg, N_cfg, model, Y, args):
    print('\n==> update_Y..\n')
    new_Y = ()

    _3_3_layer_count = 0
    conv_layer_index = 0

    # idx = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if hasattr(module, "weight") and hasattr(module, "mask"):
                
                if module.weight.shape[-2:] == (3, 3):
                    
                    # y = W[idx] + V[idx]
                    if _3_3_layer_count == 0:
                        _3_3_layer_count += 1
                        conv_layer_index += 1 
                        new_Y += (Y[0],)
                        continue

                    y = W[_3_3_layer_count] + V[_3_3_layer_count]

                    # if _3_3_layer_count == 0:
                    #     _3_3_layer_count += 1
                    #     conv_layer_index += 1
                    #     continue


                    temp_y = y.detach().clone()
                    c_out, c_in, k_1, k_2 = temp_y.shape
                    temp_y = temp_y.permute(1, 0, 2, 3)
                    temp_y = temp_y.contiguous().view(-1, N_cfg[conv_layer_index]*k_1*k_2)
                    prune = int(temp_y.size(0) * pr_cfg[conv_layer_index])
                    temp_y = torch.sum(temp_y ** 2, 1)
                    _, indice = torch.topk(temp_y, prune, largest=False)
                    m = torch.ones(temp_y.size(0))
                    m[indice] = 0
                    m = torch.unsqueeze(m, 1)
                    m = m.repeat(1, N_cfg[conv_layer_index]*k_1*k_2)
                    m = m.view(c_in, c_out, k_1, k_2)
                    m = m.permute(1, 0, 2, 3)

                    with torch.no_grad():
                        y.mul_(m)


                    new_Y += (y,)
                    _3_3_layer_count += 1
                    # idx += 1

                conv_layer_index += 1
    
    return new_Y


def update_V(V, W, Y):
    print('\n==> update_V..\n')
    
    new_V = ()
    for v, w, y in zip(V, W, Y):
        new_v = v + w - y
        new_V += (new_v,)
        
    return new_V


def retrain_1_N_prune(model, args, layer_top_k_pattern_list, pr_cfg, N_cfg):
    print('\n==> retrain_1_N_prune..\n')

    with torch.no_grad():
        # _3_3_layer_count = 0
        conv_layer_index = 0

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if hasattr(module, "weight") and hasattr(module, "mask"):
                    
                    if conv_layer_index == 0:
                        conv_layer_index += 1
                        continue
                    #check if it is 3x3 kernel
                    # if module.weight.shape[-2:] == (3, 3):

                    if module.weight.shape[-2:] == (3, 3):
                        
                        sparse_weight_admm = (module.weight.detach().clone()) * (module.mask.detach().clone())
                        c_out, c_in, k_1, k_2 = sparse_weight_admm.shape
                        w = sparse_weight_admm.detach().clone()
                        w = w.permute(1, 0, 2, 3)
                        w = w.contiguous().view(-1,N_cfg[conv_layer_index]*k_1*k_2) 
                        prune = int(w.size(0)*pr_cfg[conv_layer_index])
                        w = torch.sum(torch.abs(w), 1)
                        _, indice = torch.topk(w, prune, largest=False)
                        m = torch.ones(w.size(0))
                        m[indice] = 0
                        m = torch.unsqueeze(m, 1)
                        m = m.repeat(1, N_cfg[conv_layer_index]*k_1*k_2)
                        m = m.view(c_in, c_out, k_1, k_2)
                        m = m.permute(1, 0, 2, 3)
                        
                        # with torch.no_grad():
                        m = m.to(module.mask.device)
                        module.mask.mul_(m)

                    # _3_3_layer_count += 1
                    conv_layer_index += 1


# 1xN prune for admm(only prune 1x1 conv layer)
def N_prune_admm(model, pr_cfg, N_cfg):
    print('==> 1xN prune..')

    with torch.no_grad():

        conv_num = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if hasattr(module, "weight"):
                    if hasattr(module, "mask"):
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

        print('conv_num in N_prune_admm function:')
        print(conv_num)
