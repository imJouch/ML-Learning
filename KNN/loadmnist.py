import struct
import array
import numpy
def load(path_img, path_lbl):
    labels = []
    with open(path_lbl, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049,'
                             'got {}'.format(magic))

        label_data = array.array("B", file.read())
        for i in range(size):
            labels.append(label_data[i])
    with open(path_img, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051,'
                             'got {}'.format(magic))
        image_data = array.array("B", file.read())
        images = numpy.zeros((size, rows * cols))

        for i in range(size):
            if ((i % 2000 == 0) or (i + 1 == size)):
                print("%d numbers imported" % (i))
            images[i, :] = image_data[i * rows * cols: (i + 1) * rows * cols]
    return images, labels, size