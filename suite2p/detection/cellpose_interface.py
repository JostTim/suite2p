import numpy as np
from pathlib import Path
from warnings import warn
from .anatomical import masks_to_stats
from .stats import roi_stats
from .chan2detect import detect as chan2_detect


def cellpose_to_stats(settings: dict, /, save=True, remove_old_results=True, compute_additionnal_stats=True):

    warn("cellpose_to_stats is no longer tested in current suite2p version, expect issues.")

    save_path = Path(settings["save_path"]) # settings does not contain this anymore, 
    # because it gets "reduced" to settings["detection"] in detection.detection_wrapper in pipeline_s2p.py

    cellpose_image_key = settings.get("cellpose_image_key", "meanImg")
    # cellpose_seg = open_cellpose_seg_file(save_path / "meanImg_seg.npy")

    seg_file_path = save_path / f"{cellpose_image_key}_seg.npy"
    if not seg_file_path.exists():
        raise FileNotFoundError(f"Cellpose segmentation file not found at: {seg_file_path}\n"
                                "Please ensure you have saved from the cellpose GUI.")
    
    cellpose_seg = open_cellpose_seg_file(seg_file_path)


    # import cv2
    # image_used = cv2.imread(cellpose_seg["filename"], -1)  # cv2.LOAD_IMAGE_ANYDEPTH)
    # if image_used.ndim > 2:
    #     image_used = image_used[..., [2, 1, 0]]

    # image_used = cast(np.ndarray, image_used)

    if cellpose_image_key not in settings:
         raise KeyError(f"The key '{cellpose_image_key}' was not found in the suite2p ops dictionary.")
    
    image_used = settings[cellpose_image_key]
    # image_used = ops["meanImg"]

    # weights calculation only works for situation of anatomical_only = 2 wich is when meanImg was used.
    weights_image = 0.1 + np.clip(
        (image_used - np.percentile(image_used, 1)) / (np.percentile(image_used, 99) - np.percentile(image_used, 1)),
        0,
        1,
    )

    stats = masks_to_stats(cellpose_seg["masks"], weights_image)
    vars_to_copy = ["diameter", "cellprob_threshold", "flow_threshold"]
    for var in vars_to_copy:
        settings[var] = cellpose_seg[var]

    if settings.get("ops_path"):
        np.save(settings["ops_path"], settings)

    if remove_old_results:
        remove_previous_extraction_results(settings)

    if compute_additionnal_stats:
        stats = roi_stats(
            stats,
            Ly=settings["Ly"],
            Lx=settings["Lx"],
            diameter=settings.get("diameter", None),
            max_overlap=settings.get("max_overlap", None),
            do_soma_crop=settings.get("soma_crop", 1),
        )
        if "meanImg_chan2" in settings.keys():
            if "chan2_thres" not in settings:
                settings["chan2_thres"] = 0.65
            settings, redcell = chan2_detect(
                settings["meanImg"],
                settings["meanImg_chan2"], 
                stats,
                settings["diameter"], 
                settings=settings,
            )
            # logger.info(f"saving redcell {type(redcell)} {redcell.shape} {redcell.dtype} {redcell}")
            np.save(save_path / "redcell.npy", redcell)

    if save:
        np.save(save_path / "stat.npy", stats)

    if settings.get("ops_path"):
        np.save(settings["ops_path"], settings)

    return stats, redcell

def remove_previous_extraction_results(settings: dict, remove_redcell = True):
    save_path = Path(settings["save_path"])
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


def prepare_cellpose_from_ops(settings: dict, cellpose_image_key: str = "meanImg"):
    # saves a meanimage in the suite2p save folder
    import pImage
    from PIL.Image import fromarray

    warn("prepare_cellpose_from_ops is no longer tested in current suite2p version, expect issues.")

    save_path = Path(settings["save_path"])

    if cellpose_image_key not in settings:
        raise KeyError(f"The key '{cellpose_image_key}' was not found in the suite2p ops dictionary. "
                       f"Available keys include: {[k for k in settings if 'Img' in k or 'proj' in k]}")
    
    image_data = settings[cellpose_image_key]

    image = fromarray(pImage.transformations.rescale_to_8bit(image_data), mode="L")

    output_filename = f"{cellpose_image_key}.png"
    output_path = save_path / output_filename
    image.save(output_path)

    return str(output_path)