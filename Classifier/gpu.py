import os
print(os.listdir('.'))


import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))


# print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
# print("Built with CUDA:", tf.test.is_built_with_cuda())
# print("GPU device name:", tf.test.gpu_device_name())
