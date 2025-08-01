import torch
import torch.nn as nn
from typing import Optional
import time
import re
import matplotlib.pyplot as plt
#################################  2-bit Datatypes  #################################
INT2 = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

#################################  3-bit Datatypes  #################################
INT3 = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
FP3 = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
FP3_ER_POS = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
FP3_ER_NEG = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
FP3_EA_POS = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0]
FP3_EA_NEG = [-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]

#################################  4-bit Datatypes  #################################
INT4 = [-7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
FLINT4 = [-16.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0]
FP4_E2M1 = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
FP4_ER_POS = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]
FP4_ER_NEG = [-12.0, -10.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
FP4_EA_POS = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
FP4_EA_NEG = [-16.0, -12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]

#################################  5-bit Datatypes  #################################
INT5 = [-15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
FLINT5 = [-64.0, -32.0, -24.0, -16.0, -14.0, -12.0, -10.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 24.0, 32.0, 64.0]
FP5_E2M2 = [-28.0, -24.0, -20.0, -16.0, -14.0, -12.0, -10.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 28.0]
FP5_E3M1 = [-192.0, -128.0, -96.0, -64.0, -48.0, -32.0, -24.0, -16.0, -12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 192.0]

#################################  6-bit Datatypes  #################################
INT6 = [
    -31.0, -30.0, -29.0, -28.0, -27.0, -26.0, -25.0, -24.0, -23.0, -22.0, -21.0, -20.0, -19.0, -18.0, -17.0, -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0
]
FP6_E2M3 = [
    -60.0, -56.0, -52.0, -48.0, -44.0, -40.0, -36.0, -32.0, -30.0, -28.0, -26.0, -24.0, -22.0, -20.0, -18.0, -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0
]
FP6_E3M2 = [
    -448.0, -384.0, -320.0, -256.0, -224.0, -192.0, -160.0, -128.0, -112.0, -96.0, -80.0, -64.0, -56.0, -48.0, -40.0, -32.0, -28.0, -24.0, -20.0, -16.0, -14.0, -12.0, -10.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 28.0, 32.0, 40.0, 48.0, 56.0, 64.0, 80.0, 96.0, 112.0, 128.0, 160.0, 192.0, 224.0, 256.0, 320.0, 384.0, 448.0
]

DATATYPE_MAPPING_3_BIT = {
    'int3': INT3, 'fp3': FP3, 
    'fp3_er_pos': FP3_ER_POS, 'fp3_er_neg': FP3_ER_NEG, 
    'fp3_ea_pos': FP3_EA_POS, 'fp3_ea_neg': FP3_EA_NEG, 
}
DATATYPE_MAPPING_3_BIT_MX = {
    'mx_int3': INT3, 'mx_fp3': FP3
}

DATATYPE_MAPPING_4_BIT = {
    'int4': INT4, 'fp4': FP4_E2M1, 'flint4': FLINT4,
    'fp4_er_pos': FP4_ER_POS, 'fp4_er_neg': FP4_ER_NEG, 
    'fp4_ea_pos': FP4_EA_POS, 'fp4_ea_neg': FP4_EA_NEG, 
}
DATATYPE_MAPPING_4_BIT_MX = {
    'mx_int4': INT4, 'mx_fp4': FP4_E2M1 
}

DATATYPE_MAPPING_5_BIT = {
    'int5': INT5, 'fp5': FP5_E2M2, 'flint5': FLINT5,
    'fp5_e2m2': FP5_E2M2, 'fp5_e3m1': FP5_E3M1
}

DATATYPE_MAPPING_6_BIT = {
    'int6': INT6, 'fp6': FP6_E2M3, 
    'fp6_e2m3': FP6_E2M3, 'fp6_e3m2': FP6_E3M2
}


def extract_info(input_str):
    match = re.search(r'layers\.(\d+)', input_str)
    if match:
        return int(match.group(1))
    
    if "lm_head" in input_str:
        return -1
    
    return None

@torch.no_grad()
def quant_int(w_fp16, wq_bits:int=4, group_size: Optional[int]=None):
    """
        Symmetric INT quantization.
    """    
    if (group_size is None) or (group_size <= 0):
        w_fp16_new = w_fp16.to(torch.float16)
    else:
        K, C = w_fp16.size() # output channel, input channel
        NUM_GROUP = C // group_size
        w_fp16_new = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size).to(torch.float16)
    
    rmax = torch.amax(w_fp16_new.abs(), dim=-1, keepdim=True)
    qmax = 2 ** (wq_bits - 1) - 1
    qmin = -qmax
    scale_fp = rmax / qmax
    scale_fp = scale_fp.clamp(min=1e-5, max=1e4)
    q_tensor = torch.clamp(torch.round(w_fp16_new / scale_fp), min=qmin, max=qmax)

    w_fp16_new = q_tensor * scale_fp
    if (group_size is None) or (group_size <= 0):
        return w_fp16_new
    else:
        return w_fp16_new.reshape(K, C)


@torch.no_grad()
def quant_int_asym(w_fp16, wq_bits:int=4, group_size: Optional[int]=None):
    """
        Asymmetric INT quantization.
    """    
    if (group_size is None) or (group_size <= 0):
        w_fp16_new = w_fp16.to(torch.float16)
    else:
        K, C = w_fp16.size() # output channel, input channel
        NUM_GROUP = C // group_size
        w_fp16_new = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size).to(torch.float16)
        # print("K: ", K, "C: ", C)
    
    rmin = torch.amin(w_fp16_new, dim=-1, keepdim=True)
    rmax = torch.amax(w_fp16_new, dim=-1, keepdim=True)
    qmin = 0
    qmax = 2**wq_bits - 1
    scale_fp = (rmax - rmin) / (qmax - qmin)
    scale_fp = scale_fp.clamp(min=1e-5, max=1e4)
    zeropoint = torch.round(-rmin / scale_fp).clamp(min=qmin, max=qmax)

    q_tensor = torch.clamp(torch.round(w_fp16_new / scale_fp) + zeropoint, min=qmin, max=qmax)

    w_fp16_new = (q_tensor - zeropoint) * scale_fp
    if (group_size is None) or (group_size <= 0):
        return w_fp16_new
    else:
        return w_fp16_new.reshape(K, C)

def tile_norm_vectorized(tensor, in_dim_tilesize, out_dim_tilesize, wq_bits, threshold, diff_num, high_bit_num, low_bit_num):
    m, n = tensor.shape
    reshaped = tensor.view(m//out_dim_tilesize, out_dim_tilesize, n//in_dim_tilesize, in_dim_tilesize)
    reshaped = reshaped.permute(0, 2, 1, 3)  # (m/group, n/group, group, group)
    tile_norm = torch.norm(reshaped, p=2, dim=(2, 3))
    wq_bit_mat = torch.full_like(tile_norm, fill_value = wq_bits)
    
    max_mask = (tile_norm >= threshold[0])
    min_mask = (tile_norm <= threshold[1])
    
    high_bit_num_temp = max_mask.float().sum().item()
    low_bit_num_temp = min_mask.float().sum().item()

    if high_bit_num < diff_num:
        if high_bit_num + high_bit_num_temp >= diff_num:
            rm_high = high_bit_num + high_bit_num_temp - diff_num
            rm_high = int(rm_high)
            true_indices = torch.where(max_mask)
            if len(true_indices[0]) >= rm_high:
                rows = true_indices[0][:rm_high]
                cols = true_indices[1][:rm_high]
                max_mask[rows, cols] = False
        high_bit_num_temp = max_mask.float().sum().item()
        wq_bit_mat[max_mask] = wq_bit_mat[max_mask] + 1
        high_bit_num = high_bit_num + high_bit_num_temp
    
    if low_bit_num < diff_num:
        if low_bit_num + low_bit_num_temp >= diff_num:
            rm_high = low_bit_num + low_bit_num_temp - diff_num
            rm_high = int(rm_high)
            true_indices = torch.where(min_mask)
            if len(true_indices[0]) >= rm_high:
                rows = true_indices[0][:rm_high]
                cols = true_indices[1][:rm_high]
                min_mask[rows, cols] = False
        low_bit_num_temp = min_mask.float().sum().item()
        wq_bit_mat[min_mask] = wq_bit_mat[min_mask] - 1
        low_bit_num = low_bit_num + low_bit_num_temp
        
    return  wq_bit_mat, high_bit_num, low_bit_num

@torch.no_grad()
def quant_int_asym_tile(w_fp16, gradient = None, wq_bits:int=4, group_size: Optional[int]=None, threshold = None, diff_num = None, high_bit_num = None, low_bit_num = None, in_dim_tilesize = None, out_dim_tilesize = None):
    """
        Gradient-Group-Asymmetric INT quantization.
    """
    if (group_size is None) or (group_size <= 0):
        w_fp16_new = w_fp16.to(torch.float16)
    else:
        K, C = w_fp16.size() # output channel, input channel
        C_GROUP = C // in_dim_tilesize
        K_GROUP = K // out_dim_tilesize
        w_fp16_new = w_fp16.view(K_GROUP, out_dim_tilesize, C_GROUP, in_dim_tilesize).to(torch.float16)
        w_fp16_new = w_fp16_new.permute(0, 2, 1, 3)  # (m/group, n/group, group, group)
        wq_bit_mat, high_bit_num, low_bit_num = tile_norm_vectorized(gradient, in_dim_tilesize, out_dim_tilesize, wq_bits, threshold, diff_num, high_bit_num, low_bit_num)
    
    rmin = torch.amin(w_fp16_new, dim=-1, keepdim=True)
    rmax = torch.amax(w_fp16_new, dim=-1, keepdim=True)
    qmin = torch.zeros_like(rmax)
    qmax = 2**wq_bit_mat - 1
    qmax = qmax.unsqueeze(-1).repeat(1, 1, qmin.shape[2])
    qmax = qmax.unsqueeze(-1)
    
    scale_fp = (rmax - rmin) / (qmax - qmin)
    scale_fp = scale_fp.clamp(min=1e-5, max=1e4)
    zeropoint = torch.round(-rmin / scale_fp).clamp(min=qmin, max=qmax)

    q_tensor = torch.clamp(torch.round(w_fp16_new / scale_fp) + zeropoint, min=qmin, max=qmax)

    w_fp16_new = (q_tensor - zeropoint) * scale_fp
    if (group_size is None) or (group_size <= 0):
        return w_fp16_new
    else:
        w_fp16_new = w_fp16_new.permute(0, 2, 1, 3)
        return w_fp16_new.reshape(K, C), high_bit_num, low_bit_num

@torch.no_grad()
def quant_mx(w_fp16, wq_bits:int=4, datatype: str="", group_size: int=32):
    """
        MX quantization.
        Reference: https://github.com/microsoft/microxcaling/blob/7bc41952de394f5cc5e782baf132e7c7542eb4e4/mx/mx_ops.py
    """ 
    if wq_bits == 3:
        DATATYPE_MAPPING = DATATYPE_MAPPING_3_BIT_MX
    elif wq_bits == 4:
        DATATYPE_MAPPING = DATATYPE_MAPPING_4_BIT_MX
    else:
        raise ValueError(f"Currently only support 3-bit, 4-bit quantization, not {wq_bits}-bit")

    assert datatype in DATATYPE_MAPPING, f"unexpected data type {datatype}."

    allow_value = DATATYPE_MAPPING[datatype]
    mid_value = [(allow_value[i] + allow_value[i + 1]) / 2 for i in range(len(allow_value) - 1)]
    K, C = w_fp16.size() # output channel, input channel
    NUM_GROUP = C // group_size
    w_fp16_new = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size).to(torch.float32)
    
    shared_exp, _ = torch.max(w_fp16_new.abs(), dim=-1, keepdim=True)
    shared_exp = torch.floor(torch.log2(shared_exp))
    w_fp16_new = w_fp16_new / (2**shared_exp)
    qmax = max([abs(x) for x in allow_value])
    scale = 1 / (qmax / 2)
    x = w_fp16_new / scale

    q_tensor = torch.zeros_like(x)
    for i in range(len(allow_value)):
        data = allow_value[i]
        if i == 0:
            q_tensor += torch.where(x <= mid_value[i], data, 0)
        elif i == len(allow_value) - 1:
            q_tensor += torch.where(x > mid_value[i - 1], data, 0)
        else:
            q_tensor += torch.where((mid_value[i - 1] < x) & (x <= mid_value[i]), data, 0)

    w_fp16_new = q_tensor * scale * (2**shared_exp)
    return w_fp16_new.reshape(K, C).to(torch.float16)


@torch.no_grad()
def quant_datatype(w_fp16, wq_bits:int=4, datatype: str="", group_size: Optional[int]=None):
    if wq_bits == 3:
        DATATYPE_MAPPING = DATATYPE_MAPPING_3_BIT
    elif wq_bits == 4:
        DATATYPE_MAPPING = DATATYPE_MAPPING_4_BIT
    elif wq_bits == 5:
        DATATYPE_MAPPING = DATATYPE_MAPPING_5_BIT
    elif wq_bits == 6:
        DATATYPE_MAPPING = DATATYPE_MAPPING_6_BIT
    else:
        raise ValueError(f"Currently only support 3-, 4-, 5-, and 6-bit quantization, not {wq_bits}-bit")

    assert datatype in DATATYPE_MAPPING, f"unexpected data type {datatype}."

    allow_value = DATATYPE_MAPPING[datatype]
    mid_value = [(allow_value[i] + allow_value[i + 1]) / 2 for i in range(len(allow_value) - 1)]

    if (group_size is None) or (group_size <= 0):
        w_fp16_new = w_fp16.to(torch.float16)
    else:
        K, C = w_fp16.size() # output channel, input channel
        NUM_GROUP = C // group_size
        w_fp16_new = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size).to(torch.float16)

    rmax = torch.amax(w_fp16_new.abs(), dim=-1, keepdim=True)
    qmax = max([abs(x) for x in allow_value])
    scale_fp = rmax / qmax
    scale_fp = scale_fp.clamp(min=1e-5, max=1e4)
    x = w_fp16_new / scale_fp

    q_tensor = torch.zeros_like(x)
    for i in range(len(allow_value)):
        data = allow_value[i]
        if i == 0:
            q_tensor += torch.where(x <= mid_value[i], data, 0)
        elif i == len(allow_value) - 1:
            q_tensor += torch.where(x > mid_value[i - 1], data, 0)
        else:
            q_tensor += torch.where((mid_value[i - 1] < x) & (x <= mid_value[i]), data, 0)

    w_fp16_new = q_tensor * scale_fp 

    if (group_size is None) or (group_size <= 0):
        return w_fp16_new
    else:
        return w_fp16_new.reshape(K, C)

@torch.no_grad()
def search_datatype_tile(w_fp16, gradient, wq_bits:int=4, datatype: str='mixed_bitmod_tile', group_size: Optional[int]=None, threshold = None, diff_num = None, high_bit_num = None, low_bit_num = None, in_dim_tilesize = None, out_dim_tilesize = None):
    if (group_size is None) or (group_size <= 0):
        w_fp16_new = w_fp16.to(torch.float16)
        return w_fp16_new, high_bit_num, low_bit_num
    else:
        K, C = w_fp16.size() # output channel, input channel
        C_GROUP = C // in_dim_tilesize
        K_GROUP = K // out_dim_tilesize
        w_fp16_new = w_fp16.view(K_GROUP, out_dim_tilesize, C_GROUP, in_dim_tilesize).to(torch.float16)
        w_fp16_new = w_fp16_new.permute(0, 2, 1, 3)  # (m/group, n/group, group, group)
        wq_bit_mat, high_bit_num, low_bit_num = tile_norm_vectorized(gradient, in_dim_tilesize, out_dim_tilesize, wq_bits, threshold, diff_num, high_bit_num, low_bit_num)
    
        for i in range(K_GROUP):
            for j in range(C_GROUP):
                sub_w_fp16 = w_fp16_new[i, j, :, :] 
                sub_wq_bits = int(wq_bit_mat[i, j].item())
                datatype_list = [] 
                if sub_wq_bits == 3:
                    if datatype == 'mixed_bitmod_tile':
                        datatype_list = ['fp3_er_pos', 'fp3_er_neg', 'fp3_ea_pos', 'fp3_ea_neg']
                    elif datatype == 'mixed_er_group':
                        datatype_list = ['fp3_er_pos', 'fp3_er_neg']
                    elif datatype == 'mixed_ea_group':
                        datatype_list = ['fp3_ea_pos', 'fp3_ea_neg']
                    elif datatype == 'mixed_ant_group':
                        datatype_list = ['int3', 'fp3']
                elif sub_wq_bits == 4:
                    if datatype == 'mixed_bitmod_tile':
                        datatype_list = ['fp4_er_pos', 'fp4_er_neg', 'fp4_ea_pos', 'fp4_ea_neg']
                    elif datatype == 'mixed_er_group':
                        datatype_list = ['fp4_er_pos', 'fp4_er_neg']
                    elif datatype == 'mixed_ea_group':
                        datatype_list = ['fp4_ea_pos', 'fp4_ea_neg']
                    elif datatype == 'mixed_ant_group':
                        datatype_list = ['int4', 'flint4']
                else:
                    datatype_list = ["int_asys"]

                if datatype_list[0] == "int_asys":
                    q_tensor = quant_int_asym(sub_w_fp16, wq_bits=sub_wq_bits, group_size=group_size)
                else:
                    sub_K, sub_C = sub_w_fp16.size() # output channel, input channel
                    if (group_size is None) or (group_size <= 0):
                        group_size = sub_C
                    sub_NUM_GROUP = sub_C // group_size
                    sub_w_fp16 = sub_w_fp16.unsqueeze(-1).reshape(sub_K, sub_NUM_GROUP, group_size)
                    q_tensor = torch.zeros_like(sub_w_fp16)
                    
                    error = torch.full([sub_K, sub_NUM_GROUP], 1e4, dtype=sub_w_fp16.dtype, device=sub_w_fp16.device)
                    for datatype_temp in datatype_list:
                        w_fp16_tmp = quant_datatype(sub_w_fp16, wq_bits=sub_wq_bits, datatype=datatype_temp, group_size=None)
                        quant_error = (w_fp16_tmp - sub_w_fp16).pow(2).mean(-1)
                        update_mask = torch.lt(quant_error, error)
                        error[update_mask] = quant_error[update_mask]
                        q_tensor[update_mask] = w_fp16_tmp[update_mask]
                        del w_fp16_tmp, quant_error, update_mask
                    q_tensor = q_tensor.reshape(sub_K, sub_C)
                w_fp16_new[i, j, :, :] = q_tensor
                
        w_fp16_new = w_fp16_new.permute(0, 2, 1, 3)
        return w_fp16_new.reshape(K, C), high_bit_num, low_bit_num
                

@torch.no_grad()
def search_datatype(w_fp16, wq_bits:int=4, datatype: str='mixed_bitmod', group_size: Optional[int]=None):
    if wq_bits == 3:
        if datatype == 'mixed_bitmod':
            datatype_list = ['fp3_er_pos', 'fp3_er_neg', 'fp3_ea_pos', 'fp3_ea_neg']
        elif datatype == 'mixed_er':
            datatype_list = ['fp3_er_pos', 'fp3_er_neg']
        elif datatype == 'mixed_ea':
            datatype_list = ['fp3_ea_pos', 'fp3_ea_neg']
        elif datatype == 'mixed_ant':
            datatype_list = ['int3', 'fp3']
    elif wq_bits == 4:
        if datatype == 'mixed_bitmod':
            datatype_list = ['fp4_er_pos', 'fp4_er_neg', 'fp4_ea_pos', 'fp4_ea_neg']
        elif datatype == 'mixed_er':
            datatype_list = ['fp4_er_pos', 'fp4_er_neg']
        elif datatype == 'mixed_ea':
            datatype_list = ['fp4_ea_pos', 'fp4_ea_neg']
        elif datatype == 'mixed_ant':
            datatype_list = ['int4', 'flint4']
    else:
        raise ValueError(f"Currently only support 3-bit and 4-bit mixed quantization, not {wq_bits}-bit")

    K, C = w_fp16.size() # output channel, input channel
    if (group_size is None) or (group_size <= 0):
        group_size = C
    NUM_GROUP = C // group_size
    w_fp16 = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size)
    q_tensor = torch.zeros_like(w_fp16)
    
    error = torch.full([K, NUM_GROUP], 1e3, dtype=w_fp16.dtype, device=w_fp16.device)
    for datatype in datatype_list:
        w_fp16_tmp = quant_datatype(w_fp16, wq_bits=wq_bits, datatype=datatype, group_size=None)
        quant_error = (w_fp16_tmp - w_fp16).pow(2).mean(-1)
        update_mask = torch.lt(quant_error, error)
        error[update_mask] = quant_error[update_mask]
        q_tensor[update_mask] = w_fp16_tmp[update_mask]

        del w_fp16_tmp, quant_error, update_mask
    
    return q_tensor.reshape(K, C)
    
def get_threshold(model, model_name, loaded_grad, in_dim_tilesize, out_dim_tilesize, th_percent):
    norm_ls = []
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            gradient = loaded_grad.get(n+'.weight')
            if (gradient is None):
                continue
            dim_out, dim_in = gradient.shape
            reshaped = gradient.view(dim_out//out_dim_tilesize, out_dim_tilesize, dim_in//in_dim_tilesize, in_dim_tilesize)
            reshaped = reshaped.permute(0, 2, 1, 3)  # (m/group, n/group, group, group)
            tile_norm = torch.norm(reshaped, p=2, dim=(2, 3))
            norm_list_temp = tile_norm.flatten().tolist()
            norm_ls = norm_ls + norm_list_temp
            
    sorted_list = sorted(norm_ls, reverse=True)
    x = range(len(sorted_list))
    y = sorted_list
    threshold = [0, 0]
    
    if th_percent == None:
        x_max = max(x)
        y_max = max(y)
        value = 0
        max_threshold_id = 0
        num_inf = 0
        for i in range(len(sorted_list)):
            if torch.isinf(torch.tensor(y[i])): 
                num_inf += 1
                if torch.isfinite(torch.tensor(y[i+1])):
                    y_max = y[i+1]
                continue
            temp_value = (x_max-num_inf-x[i])*(y_max-y[i])
            if value <= temp_value:
                value = temp_value
                threshold[0] = y[i]
                max_threshold_id = i
    else:
        max_threshold_id = int(th_percent * len(sorted_list))
    
    threshold[0] = sorted_list[max_threshold_id]
    threshold[1] = sorted_list[-max_threshold_id]
    
    return threshold, max_threshold_id, len(sorted_list)

def quant_model(model, model_name: Optional[str]=None, wq_bits: Optional[int]=None, wq_datatype: Optional[str]=None, wq_groupsize: Optional[int]=None, in_dim_tilesize = None, out_dim_tilesize = None, loaded_grad = None, th_percent = None):
    if "tile" in wq_datatype:
        threshold, max_threshold_id, tile_num = get_threshold(model, model_name, loaded_grad, in_dim_tilesize, out_dim_tilesize, th_percent)
        high_bit_num = 0
        low_bit_num = 0
        if (wq_datatype.startswith("int")) and ("asym" in wq_datatype):
            print(f"Applying group-tile-asymmetric INT quantization with bits: {wq_bits}, group size: {wq_groupsize}")
            for n, m in model.named_modules():
                if isinstance(m, torch.nn.Linear):
                    gradient = loaded_grad.get(n+'.weight')
                    if (gradient is None):
                        continue
                    else:
                        m.weight.data, high_bit_num, low_bit_num = quant_int_asym_tile(m.weight.data, gradient, wq_bits=wq_bits, group_size=wq_groupsize, threshold = threshold, diff_num = max_threshold_id, high_bit_num = high_bit_num, low_bit_num = low_bit_num, in_dim_tilesize = in_dim_tilesize, out_dim_tilesize = out_dim_tilesize)
            print("high_bit_total_num: ", high_bit_num)
            print("low_bit_total_num: ", low_bit_num)
            print("diff_num: ", max_threshold_id)
            return max_threshold_id, tile_num 
        
        elif ("mixed" in wq_datatype):
            print(f"Applying group-tile-mixed datatype quantization with bits: {wq_bits}, datatype: {wq_datatype}, group size: {wq_groupsize}")
            for n, m in model.named_modules():
                if isinstance(m, torch.nn.Linear):
                    gradient = loaded_grad.get(n+'.weight')
                    if (gradient is None):
                        continue
                    else:
                        m.weight.data, high_bit_num, low_bit_num = search_datatype_tile(m.weight.data, gradient, wq_bits=wq_bits, datatype=wq_datatype, group_size=wq_groupsize, threshold = threshold, diff_num = max_threshold_id, high_bit_num = high_bit_num, low_bit_num = low_bit_num, in_dim_tilesize = in_dim_tilesize, out_dim_tilesize = out_dim_tilesize)
            
            print("high_bit_total_num: ", high_bit_num)
            print("low_bit_total_num: ", low_bit_num)
            print("diff_num: ", max_threshold_id)
            return max_threshold_id, tile_num 
        else:
            raise ValueError(f"Unsupported datatype {wq_datatype}")
    else:
        if (wq_datatype is None) or (wq_datatype in ["fp16", "fp32"]):
            print("Not applying quantization")
            time.sleep(2)
            
        elif (wq_datatype.startswith("int")) and ("asym" in wq_datatype):
            print(f"Applying asymmetric INT quantization with bits: {wq_bits}, group size: {wq_groupsize}")
            time.sleep(2)
            for n, m in model.named_modules():
                if isinstance(m, torch.nn.Linear):
                    gradient = loaded_grad.get(n+'.weight')
                    if (gradient is None):
                        continue
                    else:
                        m.weight.data = quant_int_asym(m.weight.data, wq_bits=wq_bits, group_size=wq_groupsize)
      
        elif (wq_datatype.startswith("int")) and ("asym" not in wq_datatype):
            print(f"Applying symmetric INT quantization with bits: {wq_bits}, group size: {wq_groupsize}")
            time.sleep(2)
            for n, m in model.named_modules():
                if isinstance(m, torch.nn.Linear):
                    print(f'Quantizing layer: {n}')
                    m.weight.data = quant_int(m.weight.data, wq_bits=wq_bits, group_size=wq_groupsize)
                        
        elif ("mx" in wq_datatype):
            '''
                We use hard-coded group size 32 based on the Open Compute Standard
                https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
            '''
            print(f"Applying MX quantization with bits: {wq_bits}, datatype: {wq_datatype}, group size: 32")
            time.sleep(2)
            for n, m in model.named_modules():
                if isinstance(m, torch.nn.Linear):
                    print(f'Quantizing layer: {n}')
                    m.weight.data = quant_mx(m.weight.data, wq_bits=wq_bits, datatype=wq_datatype, group_size=32)
                        
        elif ("mixed" in wq_datatype):
            print(f"Applying mixed datatype quantization with bits: {wq_bits}, datatype: {wq_datatype}, group size: {wq_groupsize}")
            time.sleep(2)
            for n, m in model.named_modules():
                if isinstance(m, torch.nn.Linear):
                    print(f'Quantizing layer: {n}')
                    m.weight.data = search_datatype(m.weight.data, wq_bits=wq_bits, datatype=wq_datatype, group_size=wq_groupsize)
        elif ("fp" in wq_datatype):
            print(f"Applying floating-point datatype quantization with bits: {wq_bits}, group size: {wq_groupsize}")
            time.sleep(2)
            for n, m in model.named_modules():
                if isinstance(m, torch.nn.Linear):
                    print(f'Quantizing layer: {n}')
                    m.weight.data = quant_datatype(m.weight.data, wq_bits=wq_bits, datatype=wq_datatype, group_size=wq_groupsize)
        else:
            raise ValueError(f"Unsupported datatype {wq_datatype}")
        return None, None 