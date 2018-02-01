import logging
import numpy as np

from model import base
from skimage.measure import block_reduce
from utils import logging as lg

from skimage.filters.rank import entropy
from skimage.morphology import square, disk
from skimage import img_as_ubyte

FLIP_FUNCTION = {
    'zero': lambda x : np.zeros(x),
    'minus_one': lambda x: -np.ones(x)
}


lg.set_logging()


def aopc(model_obj: base.BaseNetwork, x, y, max_k=49, patch_size=(4,4), order="morf", method="deep_taylor",
         verbose=False, flip_function='minus_one'):

    if method == 'random':
        method = 'sensitivity'


    print('using %s flip' % flip_function)

    rel_prop = getattr(model_obj, 'rel_%s' % method)
    _, relevance_heatmaps = rel_prop(x, y)

    rel_patches = block_reduce(relevance_heatmaps, block_size=(1, patch_size[0], patch_size[1]), func=np.sum)

    rel_patches_flatted = rel_patches.reshape(x.shape[0], -1)
    if verbose:
        print('negative relevance : %f' % np.sum(rel_patches_flatted<0))

    logging.info('AOPC Using %s flipping' % flip_function)

    if order == "morf":
        logging.info("Using MoRF strategy")
        patch_indices = np.argsort(-rel_patches_flatted, axis=1)[:, :max_k]
    else:
        logging.info("Using random order strategy")
        patch_indices = np.zeros((x.shape[0], max_k))
        seed = 0
        np.random.seed(seed)
        logging.info('set seed to %d' % seed)
        for i in range(x.shape[0]):
            patch_indices[i, :] = np.random.choice(rel_patches_flatted.shape[1], max_k, replace=False)

    patch_indices_i = np.floor( patch_indices / rel_patches.shape[1] )
    patch_indices_j = patch_indices % rel_patches.shape[2]

    ii_start, jj_start = (patch_indices_i * patch_size[0]).astype(int), (patch_indices_j * patch_size[1]).astype(int)
    ii_end, jj_end = ii_start + patch_size[0], jj_start + patch_size[1]


    relevances = []

    with model_obj.get_session() as sess:
        rr_inputs = np.zeros((x.shape[0], model_obj.architecture.recur))
        relevance_at_0 = sess.run(model_obj.dag.y_pred_y_target,
                                  feed_dict={model_obj.dag.x: x, model_obj.dag.y_target: y,
                                             model_obj.dag.rx: rr_inputs, model_obj.dag.keep_prob: 1
                                             })

        relevance_at_0 = np.mean(np.sum(relevance_at_0, axis=1))

        relevances.append(relevance_at_0)

        x_permuted = np.copy(x)

        for i in range(max_k):
            for j in range(x_permuted.shape[0]):
                ix, iy = ii_start[j,i], ii_end[j,i]
                jx, jy = jj_start[j,i], jj_end[j,i]
                values = FLIP_FUNCTION[flip_function](patch_size)
                x_permuted[j, ix:iy, jx:jy] = values

            # for k in range(x.shape[0]):
            #     plt.subplot(2, 10, k+1)
            #     plt.imshow(rel_patches[k,:, :], cmap='Reds')
            #
            #     plt.subplot(2, 10, k+1 + 10)
            #     plt.imshow(x_permuted[k,:, :], cmap='Reds')
            #
            # plt.savefig('relevance-%s-permuted-%d.png' % (method, i+1))

            # todo: should we apply relu here?
            relevance_at_k = sess.run(model_obj.dag.y_pred_y_target, feed_dict={
                model_obj.dag.x: x_permuted, model_obj.dag.y_target:y,
                model_obj.dag.rx: rr_inputs, model_obj.dag.keep_prob: 1
            })

            relevance_at_k = np.mean(np.sum(relevance_at_k, axis=1))

            relevances.append(relevance_at_k)


    return relevances


def image_entropy(model_obj: base.BaseNetwork, x, patch_size=4):
    _, relevance_heatmaps = model_obj.rel_lrp_deep_taylor(x)

    entropies = []

    for i in range(relevance_heatmaps.shape[0]):
        img = relevance_heatmaps[i, :, :]
        img = img_as_ubyte(img)
        entropies.append(entropy(img, square(patch_size)))

    return np.mean(entropies)


def count_flip(model_obj: base.BaseNetwork, x, y_true, max_k=16, patch_size=(7,7), order="morf", method="deep_taylor",
               verbose=False, flip_function='minus_one'):
    if method == 'random':
        method = 'sensitivity'

    rel_prop = getattr(model_obj, 'rel_%s' % method)
    print('total x : %d' % x.shape[0])
    y_pred, relevance_heatmaps = rel_prop(x)
    print(y_pred)


    # correct_prediction_indices = np.argmax(y_true, axis=1) == y_pred
    # x = x[correct_prediction_indices, :, :]
    # relevance_heatmaps = relevance_heatmaps[correct_prediction_indices, :, :]
    # print('correctly predicted x : %d' % x.shape[0])

    rel_patches = block_reduce(relevance_heatmaps, block_size=(1, patch_size[0], patch_size[1]), func=np.sum)

    rel_patches_flatted = rel_patches.reshape(x.shape[0], -1)
    if verbose:
        print('negative relevance : %f' % np.sum(rel_patches_flatted<0))

    if order == "morf":
        logging.info("Using MoRF strategy")
        patch_indices = np.argsort(-rel_patches_flatted, axis=1)[:, :max_k]
    else:
        logging.info("Using random order strategy")
        patch_indices = np.zeros((x.shape[0], max_k))
        for i in range(x.shape[0]):
            np.random.seed(0)
            choice = np.random.choice(rel_patches_flatted.shape[1], max_k, replace=False)
            print(choice)
            patch_indices[i, :] = choice

    print('Rel patches shape')
    print(rel_patches.shape)
    patch_indices_i = np.floor( patch_indices / rel_patches.shape[2] )
    patch_indices_j = patch_indices % rel_patches.shape[2]

    ii_start, jj_start = (patch_indices_i * patch_size[0]).astype(int), (patch_indices_j * patch_size[1]).astype(int)
    # ii_start = (ii_start / patch_size[0]).astype(int)
    ii_end, jj_end = ii_start + patch_size[0], jj_start + patch_size[1]


    pred_indices = np.zeros((x.shape[0], 1+max_k))

    with model_obj.get_session() as sess:
        rr_inputs = np.zeros((x.shape[0], model_obj.architecture.recur))
        y_pred = sess.run(model_obj.dag.y_pred, feed_dict={model_obj.dag.x: x, model_obj.dag.rx: rr_inputs,
                                                           model_obj.dag.keep_prob:1 })

        pred_at_0 = np.argmax(y_pred, axis=1)
        pred_indices[:, 0] = pred_at_0

        x_permuted = np.copy(x)

        for i in range(max_k):
            for j in range(x_permuted.shape[0]):
                ix, iy = ii_start[j,i], ii_end[j,i]
                jx, jy = jj_start[j,i], jj_end[j,i]
                values = FLIP_FUNCTION[flip_function](patch_size)
                x_permuted[j, ix:iy, jx:jy] = values

            y_pred = sess.run(model_obj.dag.y_pred, feed_dict={
                model_obj.dag.x: x_permuted, model_obj.dag.rx: rr_inputs, model_obj.dag.keep_prob: 1
            })

            # for k in range(10):
            #     plt.subplot(2, 10, k+1)
            #     plt.imshow(rel_patches[k,:, :], cmap='Reds')
            #
            #     plt.subplot(2, 10, k+1 + 10)
            #     plt.imshow(x_permuted[k,:, :], cmap='Reds')
            #
            # plt.savefig('./tmp/no-flip-%s-permuted-%d.png' % (method, i+1))

            pred_at_k = np.argmax(y_pred, axis=1)

            pred_indices[:, i+1] = pred_at_k

    no_flips = - np.ones((x.shape[0], 1))
    for i in range(x.shape[0]):
        change_idxs = np.argwhere(pred_indices[i, :] != pred_indices[i, 0])

        if len(change_idxs) > 0:
            no_flips[i, 0] = change_idxs[0]


    return np.mean(no_flips)
