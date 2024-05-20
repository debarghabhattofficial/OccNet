# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Xiaoyu Tian
# ---------------------------------------------
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results

import mmcv
import numpy as np
import pycocotools.mask as mask_util

# NOTE: Following code was added by DEB to 
# set up a logger for profiling the FPS of the
# model.
# ================================================
import time

import logging
from .logger import Logger
# ================================================

def custom_encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code. Semantic Masks only
    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).
    Returns:
        list | tuple: RLE encoded mask.
    """
    cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = []
    for i in range(len(cls_segms)):
        encoded_mask_results.append(
            mask_util.encode(
                np.array(
                    cls_segms[i][:, :, np.newaxis], order='F',
                        dtype='uint8'))[0])  # encoded with RLE
    return [encoded_mask_results]

def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    model.eval()
    bbox_results = []
    mask_results = []
    occ_results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    assert world_size == 1

    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    have_mask = False
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            occ_results.append(result)
            
            #if isinstance(result[0], tuple):
            #    assert False, 'this code is for instance segmentation, which our code will not utilize.'
            #    result = [(bbox_results, encode_mask_results(mask_results))
            #              for bbox_results, mask_results in result]
        if rank == 0:
            prog_bar.update()

    return occ_results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        '''
        bacause we change the sample of the evaluation stage to make sure that each gpu will handle continuous sample,
        '''
        #for res in zip(*part_list):
        for res in part_list:  
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results

def single_gpu_test(model,
                    data_loader,
                    show=False,
                    debug=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    premature_stop=False,
                    premature_stop_num=100):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to save viualization results.
            Default: True.
        out_dir (str): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    # Set up logger.
    logger = Logger.get_logger(
        name=__name__, 
        level=logging.DEBUG if debug else logging.INFO
    )

    model.eval()
    results = []
    dataset = data_loader.dataset
    prig_bar = None

    if premature_stop:
        prog_bar = mmcv.ProgressBar(premature_stop_num)
    else:
        prog_bar = mmcv.ProgressBar(len(dataset))

    total_process_time = 0.0
    
    for i, data in enumerate(data_loader):
        if debug:
            # Record the time when we start running inference.
            start_time = time.time()
            # Run inference on batch of sample.
            # NOTE: When debugging to profile FPS, ensure that
            # each batch has only one data sample.
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
            # Record the time when we stop running inference.
            end_time = time.time()
            # Update the total process time.
            total_process_time += end_time - start_time
        else:
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)

        results.extend([result])

        prog_bar.update()

        # Stop prematurely when debugging (if required).
        if premature_stop:
            if i + 1 == premature_stop_num:
                break
        # =============================================================
        
    print("")
    avg_fps = 0.0
    if debug:
        # Calculate average FPS.
        if premature_stop:
            avg_fps = premature_stop_num / total_process_time
        else:
            avg_fps = len(data_loader) / total_process_time
        logger.debug(f"Average FPS: {avg_fps:.2f}")

    return results, avg_fps


def collect_results_gpu(result_part, size):
    collect_results_cpu(result_part, size)