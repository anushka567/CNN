import numpy as np
import matplotlib.pyplot as plt
import scipy, h5py
import helper as help




def load_dataset():
    with h5py.File('E:\pycharmprojects\CNN\week2-programming-project\datasets/train_catvnoncat.h5',
                   "r") as train_dataset:
        train_set_x_orig = np.array(train_dataset["train_set_x"][:])
        train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    with h5py.File('E:\pycharmprojects\CNN\week2-programming-project\datasets/test_catvnoncat.h5', "r") as test_dataset:
        test_set_x_orig = np.array(test_dataset["test_set_x"][:])
        test_set_y_orig = np.array(test_dataset["test_set_y"][:])
        classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes



train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

print(classes[0].decode("utf-8"))
print(classes[1].decode("utf-8"))
index = 25

plt.subplot(141)
plt.imshow(train_set_x_orig[index])

print("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode(
    "utf-8") + "' picture.")

print(train_set_x_orig.shape)

# splitting the multicolor image at index into r g and b streams

test_r = train_set_x_orig[index].copy()
test_g = train_set_x_orig[index].copy()
test_b = train_set_x_orig[index].copy()

# idea behind this slicing : completely removing the r and b component from an image will give me the green component only
test_r[:, :, 1] = 0
test_r[:, :, 2] = 0
test_g[:, :, 0] = 0
test_g[:, :, 2] = 0
test_b[:, :, 0] = 0
test_b[:, :, 1] = 0

# playing around with image coloring
# test_random=train_set_x_orig[index].copy()
# test_random[:,:,0]=(test_random[:,:,0]+255*np.random.rand())%255
# test_random[:,:,1]=(test_random[:,:,1]+255*np.random.rand())%255
# test_random[:,:,2]=(test_random[:,:,2]+255*np.random.rand())%255

plt.subplot(142)
plt.imshow(test_r)
plt.subplot(143)
plt.imshow(test_g)
plt.subplot(144)
plt.imshow(test_b)
plt.show()

m_train = train_set_x_orig.shape[0]
m_test = test_set_y.shape[1]
img_height = train_set_x_orig.shape[1]
img_width = train_set_x_orig.shape[2]
print(f"{m_train}  {m_test} {img_height} {img_width}")

# flattening the vector
# x=(test_r+test_g+test_b).reshape(3*img_height*img_width,1)
# print(x)
flatten_train = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
flatten_test = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print(test_set_y.shape)
print(train_set_y.shape)
print(flatten_train.shape)
print(flatten_test.shape)

# standardizing the data as a part of preprocessing (improves the quality of data)
train_set_x = flatten_train / 255
test_set_x = flatten_test / 255

# preprocessing complete

# implementing the neural network for image classification

help.model(train_set_x_orig,train_set_y,train_set_x_orig,test_set_y)
