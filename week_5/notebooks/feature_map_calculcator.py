import numpy as np

input_given = np.array(
    [
        [20, 35, 35, 35, 35, 20],
        [29, 46, 44, 42, 42, 27],
        [16, 25, 21, 19, 19, 12],
        [66, 120, 116, 154, 114, 62],
        [74, 216, 174, 252, 172, 112],
        [70, 210, 170, 250, 170, 110],
    ]
)

filter_given = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

stride = 1


def calculate_feature_map(input_mat, kernel, stride):
    # Get dimensions
    in_h, in_w = input_mat.shape
    k_h, k_w = kernel.shape

    # Calculate output dimensions
    out_h = (in_h - k_h) // stride + 1
    out_w = (in_w - k_w) // stride + 1

    # Initialize output feature map
    feature_map = np.zeros((out_h, out_w))

    # Perform convolution
    for i in range(0, out_h):
        for j in range(0, out_w):
            # Extract the current region of interest
            region = input_mat[
                i * stride : i * stride + k_h, j * stride : j * stride + k_w
            ]
            # Element-wise multiplication and summation
            feature_map[i, j] = np.sum(region * kernel)

    return feature_map


result = calculate_feature_map(input_given, filter_given, stride)

print("Resulting Feature Map:")
print(result)
