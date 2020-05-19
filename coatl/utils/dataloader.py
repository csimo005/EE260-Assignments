import numpy as np
import coatl

class dataloader():
    def __init__(self, dataset, batch_size, shuffle=True):
        self._dataset = dataset
        self._len = (len(self._dataset)+batch_size-1)//batch_size
        self._batch_size = batch_size

        self._indexes = np.arange(len(dataset), dtype=np.long)
        np.random.shuffle(self._indexes)
        self._n = 0

        sample = dataset[0]
        self._num_ret = len(sample)
        self._sizes = ()
        for i in range(self._num_ret):
            self._sizes += (sample[i].shape,)

    def __iter__(self):
        np.random.shuffle(self._indexes) #reshuffle indexes
        self._n = 0 #start at beginning
        return self

    def __next__(self):
        if self._n == len(self._dataset):
            raise StopIteration

        ret = ()
        for i in range(self._num_ret):
            ret += (coatl.tensor(shape=(self._batch_size,)+self._sizes[i]),)

        for i in range(self._batch_size):
            sample = self._dataset[self._indexes[self._n]]

            for j in range(self._num_ret):
                if isinstance(sample[j], coatl.tensor):
                    ret[j]._data[i] = sample[j]._data
                elif isinstance(sample[j], np.ndarray):
                    ret[j]._data[i] = sample[j]
                else:
                    raise TypeError('Dataset returned unhandled type \'%d\' to dataloader' % str(type(ret[j])))
            self._n += 1
            if self._n == len(self._dataset):
                break

        return ret