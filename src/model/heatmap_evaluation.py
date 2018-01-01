import logging
import tensorflow as tf
import numpy as np

from model import base
from skimage.measure import block_reduce
from utils import logging as lg


lg.set_logging()



def aopc(model_obj: base.BaseNetwork, x, max_k=15, patch_size=(4,4), order="morf"):

    _, relevance_heatmaps = model_obj.rel_lrp_deep_taylor(x)

    rel_patches = block_reduce(relevance_heatmaps, block_size=(1, patch_size[0], patch_size[1]), func=np.sum)

    rel_patches_flatted = rel_patches.reshape(x.shape[0], -1)

    if order == "morf":
        logging.info("Using MoRF strategy")
        patch_indices = np.argsort(-rel_patches_flatted, axis=1)[:, :max_k]
    else:
        logging.info("Using random order strategy")
        patch_indices = np.zeros((x.shape[0], max_k))
        for i in range(x.shape[0]):
            patch_indices[i, :] = np.random.choice(rel_patches_flatted.shape[1], max_k, replace=False)

    patch_indices_i = np.floor( patch_indices / rel_patches.shape[1] )
    patch_indices_j = patch_indices % rel_patches.shape[2]

    ii_start, jj_start = (patch_indices_i * patch_size[0]).astype(int), (patch_indices_j * patch_size[1]).astype(int)
    ii_end, jj_end = ii_start + patch_size[0], jj_start + patch_size[1]

    with model_obj.get_session() as sess:
        rr_inputs = np.zeros((x.shape[0], model_obj.architecture.recur))
        y_pred = sess.run(model_obj.dag.y_pred, feed_dict={model_obj.dag.x: x, model_obj.dag.rx: rr_inputs,
                                                           model_obj.dag.keep_prob:1 })

        mark = y_pred == np.max(y_pred, axis=1).reshape(-1, 1)

        relevance_at_0 = np.sum(y_pred*mark, axis=1)

        aopc_at_k = [0]

        x_permuted = np.copy(x)

        np.random.seed(0)

        for i in range(max_k):
            for j in range(x_permuted.shape[0]):
                ix, iy = ii_start[j,i], ii_end[j,i]
                jx, jy = jj_start[j,i], jj_end[j,i]
                values = np.random.normal(0, 0.1, patch_size)
                x_permuted[j, ix:iy, jx:jy] = values

            relevance_at_k = sess.run(model_obj.dag.y_pred, feed_dict={
                model_obj.dag.x: x_permuted, model_obj.dag.rx: rr_inputs, model_obj.dag.keep_prob: 1
            })*mark


            relevance_at_k = np.sum(relevance_at_k, axis=1)

            m = np.sum((relevance_at_0 - relevance_at_k) / relevance_at_0) / x.shape[0]
            aopc_at_k.append(m)

    # print(aopc_at_k)
    # print('auto diff')
    # print(np.diff(np.array(aopc_at_k)))
    return np.cumsum(np.array(aopc_at_k) / (np.arange(0, max_k+1)+1))

