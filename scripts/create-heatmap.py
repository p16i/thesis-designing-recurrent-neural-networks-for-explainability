import tensorflow as tf
import numpy as np
import fire
import logging
from utils import logging as lg

from model import provider
from utils import data_provider

lg.set_logging()

def run(model_path):
    logging.info('Building relevance for %s' % model_path)

    model_object = provider.load(model_path)

    dataset = data_provider.get_data(model_object.experiment_artifact.dataset)

    pred, heatmaps = model_object.lrp(dataset.test2d.x)
    logging.info('heatmap shape :')
    logging.info(heatmaps.shape)

    heatmaps = np.expand_dims(heatmaps, axis=3)

    tf.reset_default_graph()

    with tf.Session() as sess:
        # Input placehoolders
        variables = dict()
        for k, l in dataset.labels.items():
            name = l.lower().replace(' ', '_')
            with tf.name_scope('class-%s' % name):
                indices = np.argwhere(np.argmax(dataset.test2d.y, axis=1) == k)
                logging.info("We have %d samples in class %s" % (len(indices), l))
                logging.info(indices.shape)
                x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='class_%s' % name)
                image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
                tf.summary.image('heatmap', image_shaped_input, max_outputs=len(indices))
                variables[k] = (x, indices)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('%s/boards/results' % model_path)

        feed_dict = dict()
        for k, v in variables.items():
            x, indices = variables[k]
            h = np.squeeze(heatmaps[indices, :], axis=1)
            feed_dict[x] = h

        summary = sess.run(merged, feed_dict=feed_dict)
        train_writer.add_summary(summary)
        train_writer.close()





    pass
    # readmodel


    # produce heatmap
    # visualize weigth?




if __name__ == '__main__':
   fire.Fire(run)