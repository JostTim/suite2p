import numpy as np


def update_ops(ops: dict, **kwargs):
    ops.update(kwargs)
    save_ops(ops)


def save_ops(ops: dict):
    if ops.get("ops_path"):
        np.save(ops["ops_path"], ops)
