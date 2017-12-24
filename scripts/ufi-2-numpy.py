import fire
import glob
import numpy as np
from PIL import Image

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
INPUT_DIM = IMAGE_HEIGHT*IMAGE_WIDTH
NO_CLASSES = 605


def convert(input_dir, output_dir, verbose=False):
    print('Loading UFI data from %s' % input_dir)

    name = input_dir.split('/')[-1]
    print('> %s' % name)

    files = glob.glob('%s/*/*.pgm' % input_dir)

    print('we have %d files' % len(files))
    if verbose:
        print('First 5 files')
        print('\n'.join(files[:5]))

    x = np.zeros((len(files), INPUT_DIM))
    y = np.zeros((len(files), NO_CLASSES))

    for i in range(len(files)):
        f = files[i]
        person_idx = int(f.split('/')[-2][1:]) - 1
        if verbose:
            print('Reading %s' % f)
            print('This person is %d' % person_idx)
        im = Image.open(f)
        x[i, :] = np.asarray(im, dtype=np.uint8).reshape(-1, INPUT_DIM)
        y[i, person_idx] = 1

        if i % 100 == 0:
            print('Done with %d files' % i)

    np.save('%s/%s-x.npy' % (output_dir, name), x)
    np.save('%s/%s-y.npy' % (output_dir, name), y)



if __name__ == '__main__':
    fire.Fire(convert)
