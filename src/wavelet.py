import os
import pandas as pd
import numpy as np
import pywt
import torch
from typing import List, Optional, Union, Dict, Tuple
from pytorch_wavelets import DWT1DForward
import ptwt.stationary_transform


def wavelet_coeff_raw(signal: Union[pd.Series, np.ndarray, torch.Tensor],
                     wavelet: str = 'sym4',
                     level: int = 6,
                     device: Optional[Union[str, torch.device]] = None) -> Dict[str, torch.Tensor]:
    """
    小波分解：获取原始小波系数频域表示
    
    Parameters
    ----------
    signal  : 支持 pandas.Series、numpy.ndarray 或 torch.Tensor，长度轴为最后一维
    wavelet : 小波基，默认 'sym4'
    level   : 分解层数，默认 6
    device  : torch.device 或 str，可选。如果未指定，自动使用可用 GPU，否则回退 CPU

    Returns
    -------
    coeff_dict : Dict[str, torch.Tensor]，直接返回 GPU/CPU 张量形式的小波系数
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    signal_tensor = _as_wavelet_tensor(signal, device)

    dwt = DWT1DForward(J=level, wave=wavelet, mode='symmetric').to(device)
    with torch.no_grad():
        approx, details = dwt(signal_tensor)

    coeff_tensors = [approx.squeeze(0).squeeze(0).contiguous()]
    coeff_tensors.extend(detail.squeeze(0).squeeze(0).contiguous() for detail in reversed(details))
    coeff_names = [f'cA{level}'] + [f'cD{i}' for i in range(level, 0, -1)]
    coeff_dict = {name: coef for name, coef in zip(coeff_names, coeff_tensors)}
    return coeff_dict


def wavelet_coeff_swt(signal: Union[pd.Series, np.ndarray, torch.Tensor],
                     wavelet: str = 'sym4',
                     level: int = 6,
                     device: Optional[Union[str, torch.device]] = None) -> Dict[str, torch.Tensor]:
    """
    平稳小波变换（SWT）：获取原始小波系数（不进行下采样，所有层保持相同长度）
    
    Parameters
    ----------
    signal  : 支持 pandas.Series、numpy.ndarray 或 torch.Tensor，长度轴为最后一维
    wavelet : 小波基，默认 'sym4'
    level   : 分解层数，默认 6
    device  : torch.device 或 str，可选。如果未指定，自动使用可用 GPU，否则回退 CPU

    Returns
    -------
    coeff_dict : Dict[str, torch.Tensor]，直接返回 GPU/CPU 张量形式的小波系数
                每层系数长度与输入信号相同（平稳小波变换的特点）
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    signal_tensor = _as_wavelet_tensor(signal, device)
    
    # 确保输入是 2D: [batch_size, signal_length]
    # ptwt.stationary_transform.swt 接受 2D 输入
    original_shape = signal_tensor.shape
    if signal_tensor.dim() == 3:
        batch_size, channels, seq_len = signal_tensor.shape
        signal_tensor = signal_tensor.reshape(batch_size * channels, seq_len)
    else:
        batch_size, seq_len = signal_tensor.shape
        channels = 1

    with torch.no_grad():
        # ptwt.stationary_transform.swt 返回列表: [cAn, cDn, cDn-1, ..., cD1]
        # coeffs[0] = cA{level} (近似系数)
        # coeffs[1:] = [cD{level}, cD{level-1}, ..., cD1] (细节系数)
        coeffs = ptwt.stationary_transform.swt(signal_tensor, wavelet, level=level)
    
    # 组织输出格式，与 wavelet_coeff_raw 保持一致
    # coeffs[0] = cA{level}
    # coeffs[1:] = [cD{level}, cD{level-1}, ..., cD1]
    coeff_tensors = [coeff.squeeze(0).contiguous() for coeff in coeffs]
    
    coeff_names = [f'cA{level}'] + [f'cD{i}' for i in range(level, 0, -1)]
    coeff_dict = {name: coef for name, coef in zip(coeff_names, coeff_tensors)}
    
    return coeff_dict


def wavelet_coeff_packet(signal: Union[pd.Series, np.ndarray, torch.Tensor],
                        wavelet: str = 'sym4',
                        level: int = 6,
                        device: Optional[Union[str, torch.device]] = None) -> Dict[str, torch.Tensor]:
    """
    小波包分解：获取第 level 层的所有节点系数（按频率排序）
    
    Parameters
    ----------
    signal  : 支持 pandas.Series、numpy.ndarray 或 torch.Tensor，长度轴为最后一维
    wavelet : 小波基，默认 'sym4'
    level   : 分解层数，默认 6
    device  : torch.device 或 str，可选。如果未指定，自动使用可用 GPU，否则回退 CPU

    Returns
    -------
    coeff_dict : Dict[str, torch.Tensor]，直接返回 GPU/CPU 张量形式的小波包系数
                Key 为 'node_0', 'node_1', ... 其中索引对应频率从低到高
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    signal_tensor = _as_wavelet_tensor(signal, torch.device('cpu')) # 先在 CPU 上处理
    
    signal_np = signal_tensor.numpy()
    batch_size, channels, seq_len = signal_np.shape
    
    # 扁平化处理: 将 batch 和 channels 合并，统一循环
    signal_flat = signal_np.reshape(-1, seq_len)
    
    results = {} # key -> list of arrays
    
    for i in range(signal_flat.shape[0]):
        sig = signal_flat[i]
        wp = pywt.WaveletPacket(data=sig, wavelet=wavelet, mode='symmetric', maxlevel=level)
        
        # 获取第 level 层的所有节点，按频率排序
        nodes = wp.get_level(level, order='freq')
        
        for j, node in enumerate(nodes):
            key = f'node_{j}'
            if key not in results:
                results[key] = []
            results[key].append(node.data)
            
    # 将结果重新组装为 Tensor
    final_dict = {}
    for key, data_list in results.items():
        # data_list 是 [total_samples, coeff_len]
        data_arr = np.array(data_list)
        # 还原 batch, channels
        data_arr = data_arr.reshape(batch_size, channels, -1)
        
        # 转为 Tensor 并移至目标 device
        tensor = torch.from_numpy(data_arr).to(dtype=torch.float32, device=device)
        
        if tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.dim() > 0 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
            
        final_dict[key] = tensor.contiguous()

    return final_dict


def _as_wavelet_tensor(signal: Union[pd.Series, np.ndarray, torch.Tensor],
                       device: torch.device) -> torch.Tensor:
    """将不同类型的信号输入规范化为 [batch, channels, seq_len] 张量。"""
    if isinstance(signal, pd.Series):
        signal = signal.to_numpy()

    if isinstance(signal, np.ndarray):
        tensor = torch.as_tensor(signal, dtype=torch.float32, device=device)
    elif torch.is_tensor(signal):
        tensor = signal.to(device=device, dtype=torch.float32)
    else:
        raise TypeError("signal 必须是 pandas.Series、numpy.ndarray 或 torch.Tensor")

    if tensor.dim() == 1:
        tensor = tensor.view(1, 1, -1)
    elif tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 3:
        pass
    else:
        raise ValueError("signal 张量维度需为 1、2 或 3")

    return tensor.contiguous()


def wavelet_coeff_time(coeffs: Union[List[np.ndarray], pd.DataFrame],
                      wavelet: str = 'sym4',
                      signal_length: Optional[int] = None,
                      index: Optional[pd.Index] = None) -> pd.DataFrame:
    """
    按层级对已分解的小波系数进行时域重建
    
    Parameters
    ----------
    coeffs : Union[List[np.ndarray], pd.DataFrame]
        小波分解得到的系数，可以是：
        1. 系数列表（第一个为最高层近似系数，其后依次为各层细节系数）
        2. wavelet_coeff_raw 输出的 DataFrame
    wavelet : str, optional
        小波基名称，默认 'sym4'
    signal_length : int, optional
        重建后信号的长度，若为 None，则自动取完整重建信号长度
    index : pandas.Index, optional
        重建后 DataFrame 的行索引，长度需与 signal_length 相同

    Returns
    -------
    pd.DataFrame
        每列为一层系数在时域的重建结果，列名依次为 cA{级数}, cD{级数}, …, cD1
    """
    # 如果输入是 DataFrame，转换为系数列表
    if isinstance(coeffs, pd.DataFrame):
        coeffs = [coeffs[col].iloc[0] for col in coeffs.columns]
    
    # 计算层数（近似层 + 细节层总数 = len(coeffs)）
    n_levels = len(coeffs) - 1
    # 构造列名：['cA{n_levels}', 'cD{n_levels}', ..., 'cD1']
    coeff_names = [f'cA{n_levels}'] + [f'cD{i}' for i in range(n_levels, 0, -1)]

    # 如果没有指定输出长度，就先做一次完整重建以获取长度
    if signal_length is None:
        full_rec = pywt.waverec(coeffs, wavelet)
        signal_length = len(full_rec)

    # 按层重建
    coeff_ts = {}
    for idx, (name, coef) in enumerate(zip(coeff_names, coeffs)):
        # 构造一个只保留当前层系数的列表，其余层填零
        zeros_list = [np.zeros_like(c) for c in coeffs]
        zeros_list[idx] = coef
        rec = pywt.waverec(zeros_list, wavelet)
        # 截取到指定长度
        coeff_ts[name] = rec[:signal_length]

    # 生成 DataFrame
    df_coeff_time = pd.DataFrame(coeff_ts)
    if index is not None:
        df_coeff_time.index = index

    return df_coeff_time
