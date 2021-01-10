
import numpy as np


def export_learn_data(path, bin_labels, bin_data):
    with open(f'{path}/data.bin', 'ab') as data_file:
        with open(f'{path}/temp.bin', 'wb') as tmp:
            print(np.array(bin_data).shape)
            print(np.array(bin_data).reshape((-1,)))
            np.array(bin_data).reshape((-1,)).tofile(tmp)
        with open(f'{path}/temp.bin', 'rb') as tmp:
            data_file.write(tmp.read())
    if bin_labels is not None:
        with open(f'{path}/labels.bin', 'ab') as labels_file:
            with open(f'{path}/temp.bin', 'wb') as tmp:
                np.array(bin_labels).tofile(tmp)
            with open(f'{path}/temp.bin', 'rb') as tmp:
                labels_file.write(tmp.read())

