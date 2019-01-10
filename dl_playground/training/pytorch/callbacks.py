"""Classes for utilities called at certain points during model training

Reference Implementations:
- https://github.com/keras-team/keras/blob/master/keras/callbacks.py
"""

# TODO: Remove the "raise NotImplementedError" and make a note that these
# aren't there because each child class is *not required* to fill in all of the
# methods that are here
class Callback(object):
    """Abstract base class specifying the interface used to build new callbacks

    Reference Implementation:
    - https://github.com/keras-team/keras/blob/master/keras/callbacks.py
    """

    def __init__(self):
        """Init"""
        raise NotImplementedError

    def on_epoch_begin(self, idx_epoch):
        """Perform the specified computation at the beginning of an epoch

        :param idx_epoch: index of the upcoming epoch
        :type idx_epoch: integer
        """
        raise NotImplementedError

    def on_epoch_end(self, idx_epoch, logs=None):
        """Perform the specified computation at the end of an epoch

        :param idx_epoch: index of the most recently completed epoch
        :type idx_epoch: integer
        :param logs: holds the metric results for the most recently completed
         epoch
        :type logs: dict:
        """
        raise NotImplementedError

    def on_test_batch_end(self, idx_batch, logs=None):
        """Perform the specified computation at the end of a test batch

        :param idx_batch: index of the batch within the current epoch
        :type idx_batch: integer
        :param logs: holds the metric results for this batch
        :type logs: dict
        """
        raise NotImplementedError

    def on_train_batch_end(self, idx_batch, logs=None):
        """Perform the specified computation at the end of a training batch

        :param idx_batch: index of the batch within the current epoch
        :type idx_batch: integer
        :param logs: holds the metric results for this batch
        :type logs: dict
        """
        raise NotImplementedError
    
    def on_epoch_end(self, epoch, logs=None):
        """Perform the specified computation at the end of an epoch

        :param epoch: index of the most recently completed epoch
        :type epoch: integer
        :param logs: holds the metric results for the most recently completed
         epoch
        :type logs: dict:
        """
        raise NotImplementedError


class BaseLogger(Callback):
    """Callback that accumulates running averages of metrics during epochs

    Reference Implementation:
    - https://github.com/keras-team/keras/blob/master/keras/callbacks.py
    """

    def __init__(self):
        """Init"""

        pass

    def on_epoch_begin(self, idx_epoch):
        """Perform the specified computation at the beginning of an epoch

        :param idx_epoch: index of the upcoming epoch
        :type idx_epoch: integer
        """

        self.n_seen = 0
        self.totals = {}
    
    # TODO: Check and see if the `size` is "averaged" here in a standard Keras
    # callback
    def on_train_batch_end(self, idx_batch, logs=None)
        """Perform the specified computation at the end of a training batch

        Note: This method assumes that each metric value in `logs` is averaged
        across the batch.

        :param idx_batch: index of the batch within the current epoch
        :type idx_batch: integer
        :param logs: holds the metric results for this batch
        :type logs: dict
        """

        logs = logs or {}
        batch_size = logs.get('size', 0)
        self.n_seen += batch_size

        for key, batch_average_value in logs.items():
            if key in self.totals:
                self.totals[key] += batch_average_value * batch_size
            else:
                self.totals[key] = batch_average_value * batch_size

    def on_epoch_end
