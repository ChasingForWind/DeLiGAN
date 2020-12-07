import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def load_mnist(path="data/MNIST_data"):
    """
    导入mnist数据集
    :param path: 数据路径
    :return: 数据字典：{train_x,train_y,test_x,test_y}
    """

    mnist = input_data.read_data_sets(path, one_hot=True)

    data_dict = {
        "train_x": mnist.train.images,
        "train_y": mnist.train.labels,
        "test_x": mnist.test.images,
        "test_y": mnist.test.labels
    }
    return data_dict

