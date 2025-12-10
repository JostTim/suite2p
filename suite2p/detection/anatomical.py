"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import numpy as np
from pathlib import Path
from typing import Any, Dict
from scipy.ndimage import find_objects, gaussian_filter
from cellpose.models import CellposeModel
from cellpose import transforms, dynamics
from cellpose.utils import fill_holes_and_remove_small_masks
from cellpose.transforms import normalize99
import time
import cv2
import os

from typing import cast

from . import utils
from .stats import roi_stats
from .chan2detect import detect as chan2_detect

from logging import getLogger

logger = getLogger("suite2p.anatomical")


def mask_centers(masks):
    centers = np.zeros((masks.max(), 2), np.int32)
    diams = np.zeros(masks.max(), np.float32)
    slices = find_objects(masks)
    for i, si in enumerate(slices):
        if si is not None:
            sr, sc = si
            ymed, xmed, diam = utils.mask_stats(masks[sr, sc] == (i + 1))
            centers[i] = np.array([ymed, xmed])
            diams[i] = diam
    return centers, diams


def patch_detect(patches, diam):
    """anatomical detection of masks from top active frames for putative cell"""
    logger.info("refining masks using cellpose")
    npatches = len(patches)
    ly = patches[0].shape[0]
    model = CellposeModel()
    imgs = np.zeros((npatches, ly, ly, 2), np.float32)
    for i, m in enumerate(patches):
        imgs[i, :, :, 0] = transforms.normalize99(m)
    rsz = 30.0 / diam
    imgs = transforms.resize_image(imgs, rsz=rsz).transpose(0, 3, 1, 2)
    imgs, ysub, xsub = transforms.pad_image_ND(imgs)  # type: ignore

    pmasks = np.zeros((npatches, ly, ly), np.uint16)
    batch_size = 8 * 224 // ly
    tic = time.time()
    for j in np.arange(0, npatches, batch_size):
        y = model.cp.network(imgs[j : j + batch_size])[0]
        y = y[:, :, ysub[0] : ysub[-1] + 1, xsub[0] : xsub[-1] + 1]
        y = y.asnumpy()
        for i, yi in enumerate(y):
            cellprob = yi[-1]
            dP = yi[:2]
            niter = 1 / rsz * 200
            p = dynamics.follow_flows(-1 * dP * (cellprob > 0) / 5.0, niter=niter)
            maski = dynamics.get_masks(p, iscell=(cellprob > 0), flows=dP, threshold=1.0)
            maski = fill_holes_and_remove_small_masks(maski)
            maski = transforms.resize_image(maski, ly, ly, interpolation=cv2.INTER_NEAREST)
            pmasks[j + i] = maski
        if j % 5 == 0:
            logger.info("%d / %d masks created in %0.2fs" % (j + batch_size, npatches, time.time() - tic))
    return pmasks


def refine_masks(stats, patches, seeds, diam, Lyc, Lxc):
    nmasks = len(patches)
    patch_masks = patch_detect(patches, diam)
    ly = patches[0].shape[0] // 2
    igood = np.zeros(nmasks, "bool")
    for i, (patch_mask, stat, (yi, xi)) in enumerate(zip(patch_masks, stats, seeds)):
        mask = np.zeros((Lyc, Lxc), np.float32)
        ypix0, xpix0 = stat["ypix"], stat["xpix"]
        mask[ypix0, xpix0] = stat["lam"]
        func_mask = utils.square_mask(mask, ly, yi, xi)
        ious = utils.mask_ious(patch_mask.astype(np.uint16), (func_mask > 0).astype(np.uint16))[0]
        if len(ious) > 0 and ious.max() > 0.45:
            mask_id = np.argmax(ious) + 1
            patch_mask = patch_mask[
                max(0, ly - yi) : min(2 * ly, Lyc + ly - yi), max(0, ly - xi) : min(2 * ly, Lxc + ly - xi)
            ]
            func_mask = func_mask[
                max(0, ly - yi) : min(2 * ly, Lyc + ly - yi), max(0, ly - xi) : min(2 * ly, Lxc + ly - xi)
            ]
            ypix0, xpix0 = np.nonzero(patch_mask == mask_id)
            lam0 = func_mask[ypix0, xpix0]
            lam0[lam0 <= 0] = lam0.min()
            ypix0 = ypix0 + max(0, yi - ly)
            xpix0 = xpix0 + max(0, xi - ly)
            igood[i] = True
            stat["ypix"] = ypix0
            stat["xpix"] = xpix0
            stat["lam"] = lam0
            stat["anatomical"] = True
        else:
            stat["anatomical"] = False
    return stats


def roi_detect(mproj, diameter=None, cellprob_threshold=0.0, flow_threshold=1.5, pretrained_model=None):
    if pretrained_model is None:
        pretrained_model = "cpsam"
    elif not os.path.exists(pretrained_model):
        pretrained_model = "cpsam"

    model = CellposeModel(gpu=True, pretrained_model=pretrained_model)
    logger.info(
        f"Running evaluation with {diameter=}, {flow_threshold=}, {cellprob_threshold=} on model {pretrained_model}"
    )
    masks = model.eval(mproj, diameter=diameter, cellprob_threshold=cellprob_threshold, flow_threshold=flow_threshold)[
        0
    ]
    shape = masks.shape
    _, masks = np.unique(np.int32(masks), return_inverse=True)
    masks = masks.reshape(shape)
    centers, mask_diams = mask_centers(masks)
    median_diam = np.median(mask_diams)
    logger.info(">>>> %d masks detected, median diameter = %0.2f " % (masks.max(), median_diam))
    return masks, centers, median_diam, mask_diams.astype(np.int32)


def cellpose_to_stats(ops: dict, /, save=True, remove_old_results=True, compute_additionnal_stats=True, compute_chan_2stats = True):

    save_path = Path(ops["save_path"])

    cellpose_image_key = ops.get("cellpose_image_key")
    if cellpose_image_key is None :
        raise ValueError("cellpose_image_key not found in ops. Be sure that you ran prepare_cellpose_from_ops and cellpose.gui.run first.")

    cellpose_seg = open_cellpose_seg_file(save_path / f"{cellpose_image_key}_seg.npy")

    # import cv2
    # image_used = cv2.imread(cellpose_seg["filename"], -1)  # cv2.LOAD_IMAGE_ANYDEPTH)
    # if image_used.ndim > 2:
    #     image_used = image_used[..., [2, 1, 0]]

    # image_used = cast(np.ndarray, image_used)

    image_used = ops[cellpose_image_key]

    # weights calculation only works for situation of anatomical_only = 2 wich is when meanImg was used.
    weights_image = 0.1 + np.clip(
        (image_used - np.percentile(image_used, 1)) / (np.percentile(image_used, 99) - np.percentile(image_used, 1)),
        0,
        1,
    )

    stats = masks_to_stats(cellpose_seg["masks"], weights_image)
    vars_to_copy = ["diameter", "cellprob_threshold", "flow_threshold"]
    for var in vars_to_copy:
        ops[var] = cellpose_seg[var]

    if ops.get("ops_path"):
        np.save(ops["ops_path"], ops)

    if remove_old_results:
        remove_previous_extraction_results(ops, remove_redcell = compute_chan_2stats)

    if compute_additionnal_stats:
        stats = roi_stats(
            stats,
            Ly=ops["Ly"],
            Lx=ops["Lx"],
            aspect=ops.get("aspect", None),
            diameter=ops.get("diameter", None),
            max_overlap=ops.get("max_overlap", None),
            do_crop=ops.get("soma_crop", 1),
        )
    if compute_chan_2stats and "meanImg_chan2" in ops.keys():
        if "chan2_thres" not in ops:
            ops["chan2_thres"] = 0.65
        ops, redcell = chan2_detect(ops, stats)
        # logger.info(f"saving redcell {type(redcell)} {redcell.shape} {redcell.dtype} {redcell}")
        np.save(save_path / "redcell.npy", redcell)

    if save:
        np.save(save_path / "stat.npy", stats)

    if ops.get("ops_path"):
        np.save(ops["ops_path"], ops)

    return stats, redcell


def remove_previous_extraction_results(ops: dict, remove_redcell = True):
    save_path = Path(ops["save_path"])
    files = list(save_path.glob("F*.npy"))
    files.append(save_path / "iscell.npy")
    if remove_redcell:
        files.append(save_path / "redcell.npy")
    files.append(save_path / "spks.npy")

    for file in files:
        file.unlink(missing_ok=True)


def open_cellpose_seg_file(cellpose_seg_file_path: str | Path):
    # cellpose gui saves a file name from the input you gave such that : {input}.png becomes {input}_seg.npy
    return np.load(cellpose_seg_file_path, allow_pickle=True).item()


def prepare_cellpose_from_ops(ops: dict, cellpose_image_key = "meanImg"):
    # saves a meanimage in the suite2p save folder
    import pImage
    from PIL.Image import fromarray

    cellpose_image_filename = f"{cellpose_image_key}.png"
    image_path = Path(ops["save_path"]) / cellpose_image_filename

    image = fromarray(pImage.transformations.rescale_to_8bit(ops[cellpose_image_key]), mode="L")
    image.save(image_path)

    # update the ops metadata file from newly added options
    ops["cellpose_image_key"] = cellpose_image_key
    ops["cellpose_image_filename"] = cellpose_image_filename

    if ops.get("ops_path"):
        # if we know where to save the metadata, save newly added options from the updated ops dict
        np.save(ops["ops_path"], ops)

    return str(image_path)


def masks_to_stats(masks, weights):
    stats = []
    slices = find_objects(masks)
    for i, si in enumerate(slices):
        sr, sc = si
        ypix0, xpix0 = np.nonzero(masks[sr, sc] == (i + 1))
        ypix0 = ypix0.astype(int) + sr.start
        xpix0 = xpix0.astype(int) + sc.start
        ymed = np.median(ypix0)
        xmed = np.median(xpix0)
        imin = np.argmin((xpix0 - xmed) ** 2 + (ypix0 - ymed) ** 2)
        xmed = xpix0[imin]
        ymed = ypix0[imin]
        stats.append({"ypix": ypix0, "xpix": xpix0, "lam": weights[ypix0, xpix0], "med": [ymed, xmed], "footprint": 1})
    stats = np.array(stats)
    return stats


def select_rois(ops: Dict[str, Any], mov: np.ndarray, diameter=None):
    """find ROIs in static frames

    Parameters:

        ops: dictionary
            requires keys "high_pass", "anatomical_only", optional "yrange", "xrange"

        mov: ndarray t x Lyc x Lxc, binned movie

    Returns:
        stats: list of dicts

    """
    Lyc, Lxc = mov.shape[1:]
    mean_img = mov.mean(axis=0)
    mov = utils.temporal_high_pass_filter(mov=mov, width=int(ops["high_pass"]))
    max_proj = mov.max(axis=0)
    # max_proj = np.percentile(mov, 90, axis=0) #.mean(axis=0)
    if ops["anatomical_only"] == 1:
        img = np.log(np.maximum(1e-3, max_proj / np.maximum(1e-3, mean_img)))
        weights = max_proj
    elif ops["anatomical_only"] == 2:
        img = mean_img
        weights = 0.1 + np.clip(
            (mean_img - np.percentile(mean_img, 1)) / (np.percentile(mean_img, 99) - np.percentile(mean_img, 1)), 0, 1
        )
    elif ops["anatomical_only"] == 3:
        if "meanImgE" in ops:
            img = ops["meanImgE"][ops["yrange"][0] : ops["yrange"][1], ops["xrange"][0] : ops["xrange"][1]]
        else:
            img = mean_img
            logger.warning("no enhanced mean image, using mean image instead")
        weights = 0.1 + np.clip(
            (mean_img - np.percentile(mean_img, 1)) / (np.percentile(mean_img, 99) - np.percentile(mean_img, 1)), 0, 1
        )
    else:
        img = max_proj.copy()
        weights = max_proj

    t0 = time.time()
    if diameter is not None:
        if isinstance(diameter, (list, np.ndarray)) and len(ops["diameter"]) > 1:
            rescale = diameter[1] / diameter[0]
            img = cv2.resize(img, (Lxc, int(Lyc * rescale)))
        else:
            rescale = 1.0
            diameter = [diameter, diameter]
        if diameter[1] > 0:
            logger.info("!NOTE! diameter set to %0.2f for cell detection with cellpose" % diameter[1])
        else:
            logger.info("!NOTE! diameter set to 0 or None, diameter will be estimated by cellpose")
    else:
        logger.info("!NOTE! diameter set to 0 or None, diameter will be estimated by cellpose")

    if ops.get("spatial_hp_cp", 0):
        img = np.clip(normalize99(img), 0, 1)
        img -= gaussian_filter(img, diameter[1] * ops["spatial_hp_cp"])

    masks, centers, median_diam, mask_diams = roi_detect(
        img,
        diameter=diameter[1],
        flow_threshold=ops["flow_threshold"],
        cellprob_threshold=ops["cellprob_threshold"],
        pretrained_model=ops["pretrained_model"],
    )
    if rescale != 1.0:
        masks = cv2.resize(masks, (Lxc, Lyc), interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, (Lxc, Lyc))
    stats = masks_to_stats(masks, weights)
    logger.info("Detected %d ROIs, %0.2f sec" % (len(stats), time.time() - t0))

    new_ops = {
        "diameter": median_diam,
        "max_proj": max_proj,
        "Vmax": 0,
        "ihop": 0,
        "Vsplit": 0,
        "Vcorr": img,
        "Vmap": 0,
        "spatscale_pix": 0,
    }
    ops.update(new_ops)

    return stats


# def run_assist():
#     nmasks, diam = 0, None
#     if anatomical:
#         try:
#             print(">>>> CELLPOSE estimating spatial scale and masks as seeds for functional algorithm")
#             from . import anatomical
#             mproj = np.log(np.maximum(1e-3, max_proj / np.maximum(1e-3, mean_img)))
#             masks, centers, diam, mask_diams = anatomical.roi_detect(mproj)
#             nmasks = masks.max()
#         except:
#             print("ERROR importing or running cellpose, continuing without anatomical estimates")
#         if tj < nmasks:
#             yi, xi = centers[tj]
#             ls = mask_diams[tj]
#             imap = np.ravel_multi_index((yi, xi), (Lyc, Lxc))
# if nmasks > 0:
#         stats = anatomical.refine_masks(stats, patches, seeds, diam, Lyc, Lxc)
#         for stat in stats:
#             if stat["anatomical"]:
#                 stat["lam"] *= sdmov[stat["ypix"], stat["xpix"]]
