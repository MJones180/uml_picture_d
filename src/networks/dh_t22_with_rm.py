# ==============================================================================
# This code runs the response matrix internally. Input data must be normalized
# between [0, 1] and output data must be normalized between [-1, 1].
# ==============================================================================
# `dh_t22_with_rm` network { 2x59x59 -> 1512 }.
# Trainable parameters: 140,245,992

import torch
import torch.nn as nn
import numpy as np

# ==============================================================================
# The indexes in the darkhole which are not being used. This is stored instead
# of the active indexes because this array has 825 values instead of 2656.
# ==============================================================================
# From `data/raw/darkhole_mask/0_data.h5`:
#   from h5py import File
#   import numpy as np
#   dh_mask = File('0_data.h5')['dark_zone_mask'][:]
#   dh_mask = dh_mask[21:80]
#   dh_mask = dh_mask[:, 21:80]
#   active_idxs = np.where(dh_mask.flatten())[0]
#   inactive_idxs = np.setdiff1d(np.arange(3481), active_idxs)
# ==============================================================================
INACTIVE_IDXS = np.array([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
    55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
    74, 75, 76, 77, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
    111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
    126, 127, 128, 129, 130, 131, 132, 133, 134, 160, 161, 162, 163, 164, 165,
    166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180,
    181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 221, 222, 223, 224,
    225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
    240, 241, 242, 243, 244, 245, 246, 247, 248, 282, 283, 284, 285, 286, 287,
    288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302,
    303, 304, 305, 306, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352,
    353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 403, 404, 405, 406,
    407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421,
    463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477,
    478, 479, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535,
    536, 537, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595,
    643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 702, 703, 704,
    705, 706, 707, 708, 709, 710, 711, 712, 762, 763, 764, 765, 766, 767, 768,
    769, 770, 822, 823, 824, 825, 826, 827, 828, 829, 881, 882, 883, 884, 885,
    886, 887, 941, 942, 943, 944, 945, 946, 1000, 1001, 1002, 1003, 1004, 1060,
    1061, 1062, 1063, 1119, 1120, 1121, 1179, 1180, 1238, 1239, 1297, 1383,
    1384, 1385, 1386, 1387, 1388, 1389, 1441, 1442, 1443, 1444, 1445, 1446,
    1447, 1448, 1449, 1499, 1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507,
    1508, 1509, 1557, 1558, 1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566,
    1567, 1568, 1569, 1616, 1617, 1618, 1619, 1620, 1621, 1622, 1623, 1624,
    1625, 1626, 1627, 1628, 1675, 1676, 1677, 1678, 1679, 1680, 1681, 1682,
    1683, 1684, 1685, 1686, 1687, 1734, 1735, 1736, 1737, 1738, 1739, 1740,
    1741, 1742, 1743, 1744, 1745, 1746, 1793, 1794, 1795, 1796, 1797, 1798,
    1799, 1800, 1801, 1802, 1803, 1804, 1805, 1852, 1853, 1854, 1855, 1856,
    1857, 1858, 1859, 1860, 1861, 1862, 1863, 1864, 1911, 1912, 1913, 1914,
    1915, 1916, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1971, 1972, 1973,
    1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 2031, 2032, 2033, 2034,
    2035, 2036, 2037, 2038, 2039, 2091, 2092, 2093, 2094, 2095, 2096, 2097,
    2183, 2241, 2242, 2300, 2301, 2359, 2360, 2361, 2417, 2418, 2419, 2420,
    2476, 2477, 2478, 2479, 2480, 2534, 2535, 2536, 2537, 2538, 2539, 2593,
    2594, 2595, 2596, 2597, 2598, 2599, 2651, 2652, 2653, 2654, 2655, 2656,
    2657, 2658, 2710, 2711, 2712, 2713, 2714, 2715, 2716, 2717, 2718, 2768,
    2769, 2770, 2771, 2772, 2773, 2774, 2775, 2776, 2777, 2778, 2826, 2827,
    2828, 2829, 2830, 2831, 2832, 2833, 2834, 2835, 2836, 2837, 2885, 2886,
    2887, 2888, 2889, 2890, 2891, 2892, 2893, 2894, 2895, 2896, 2897, 2943,
    2944, 2945, 2946, 2947, 2948, 2949, 2950, 2951, 2952, 2953, 2954, 2955,
    2956, 2957, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010,
    3011, 3012, 3013, 3014, 3015, 3016, 3017, 3059, 3060, 3061, 3062, 3063,
    3064, 3065, 3066, 3067, 3068, 3069, 3070, 3071, 3072, 3073, 3074, 3075,
    3076, 3077, 3117, 3118, 3119, 3120, 3121, 3122, 3123, 3124, 3125, 3126,
    3127, 3128, 3129, 3130, 3131, 3132, 3133, 3134, 3135, 3136, 3137, 3138,
    3174, 3175, 3176, 3177, 3178, 3179, 3180, 3181, 3182, 3183, 3184, 3185,
    3186, 3187, 3188, 3189, 3190, 3191, 3192, 3193, 3194, 3195, 3196, 3197,
    3198, 3232, 3233, 3234, 3235, 3236, 3237, 3238, 3239, 3240, 3241, 3242,
    3243, 3244, 3245, 3246, 3247, 3248, 3249, 3250, 3251, 3252, 3253, 3254,
    3255, 3256, 3257, 3258, 3259, 3289, 3290, 3291, 3292, 3293, 3294, 3295,
    3296, 3297, 3298, 3299, 3300, 3301, 3302, 3303, 3304, 3305, 3306, 3307,
    3308, 3309, 3310, 3311, 3312, 3313, 3314, 3315, 3316, 3317, 3318, 3319,
    3320, 3346, 3347, 3348, 3349, 3350, 3351, 3352, 3353, 3354, 3355, 3356,
    3357, 3358, 3359, 3360, 3361, 3362, 3363, 3364, 3365, 3366, 3367, 3368,
    3369, 3370, 3371, 3372, 3373, 3374, 3375, 3376, 3377, 3378, 3379, 3380,
    3381, 3403, 3404, 3405, 3406, 3407, 3408, 3409, 3410, 3411, 3412, 3413,
    3414, 3415, 3416, 3417, 3418, 3419, 3420, 3421, 3422, 3423, 3424, 3425,
    3426, 3427, 3428, 3429, 3430, 3431, 3432, 3433, 3434, 3435, 3436, 3437,
    3438, 3439, 3440, 3441, 3442, 3443, 3459, 3460, 3461, 3462, 3463, 3464,
    3465, 3466, 3467, 3468, 3469, 3470, 3471, 3472, 3473, 3474, 3475, 3476,
    3477, 3478, 3479, 3480
])


class LearnableAdd(nn.Module):

    def __init__(self, shape):
        super().__init__()
        # Initialize the learnable parameter with random values
        self.bias = nn.Parameter(torch.randn(shape))

    def forward(self, x):
        # Pointwise addition of the learned parameter to input
        return x + self.bias


def _make_conv_block(in_features, out_features, kernel_size):
    return nn.Sequential(
        nn.Conv2d(
            in_features,
            out_features,
            kernel_size,
            padding='same',
            bias=False,
        ),
        nn.BatchNorm2d(out_features),
        nn.LeakyReLU(),
    )


# Performs in-block downsizing
def _make_conv_block_and_downsize(in_features, out_features, kernel_size):
    return nn.Sequential(
        nn.Conv2d(
            in_features,
            out_features,
            kernel_size,
            padding=1,
            bias=False,
            stride=2,
        ),
        nn.BatchNorm2d(out_features),
        nn.LeakyReLU(),
    )


def _make_dense_block(in_features, out_features):
    return nn.Sequential(nn.Linear(in_features, out_features), nn.LeakyReLU())


class Network(nn.Module):

    def example_input():
        return torch.rand((1, 2, 59, 59))

    def __init__(self):
        super().__init__()

        # ======================================================================
        # Indexes needed to go from a 2D grid to a 1D grid
        # ======================================================================

        # Number of points along the 59x59 grid
        grid_pixels = 3481
        # Indexes of active darkhole pixels
        active_idxs = np.setdiff1d(np.arange(grid_pixels), INACTIVE_IDXS)
        # The values are copied for both the real and imaginary parts
        self.all_active_idxs = np.concatenate(
            (active_idxs, active_idxs + grid_pixels))

        # ======================================================================
        # Values to run the response matrix
        # ======================================================================

        self.input_min_x = nn.Parameter(torch.zeros(1))
        self.input_max_min_diff = nn.Parameter(torch.zeros(1))
        self.output_min_x = nn.Parameter(torch.zeros(1512))
        self.output_max_min_diff = nn.Parameter(torch.zeros(1512))
        self.resp_mat_layer = nn.Linear(5312, 1512, bias=False)

        # ======================================================================
        # The CNN layers
        # ======================================================================

        self.conv_block1 = _make_conv_block(2, 128, 3)
        self.conv_block2 = _make_conv_block(128, 128, 3)
        self.conv_block3 = _make_conv_block(128, 128, 3)
        # 59x59 -> 30x30
        self.conv_block4 = _make_conv_block_and_downsize(128, 256, 3)
        self.conv_block5 = _make_conv_block(256, 256, 3)
        self.conv_block6 = _make_conv_block(256, 256, 3)
        # 30x30 -> 15x15
        self.conv_block7 = _make_conv_block_and_downsize(256, 512, 3)
        self.conv_block8 = _make_conv_block(512, 512, 3)
        self.conv_block9 = _make_conv_block(512, 512, 3)
        # 15x15 -> 8x8
        self.conv_block10 = _make_conv_block_and_downsize(512, 1024, 3)
        self.conv_block11 = _make_conv_block(1024, 1024, 3)
        self.conv_block12 = _make_conv_block(1024, 1024, 3)
        # 8x8 -> 4x4
        self.conv_block13 = _make_conv_block_and_downsize(1024, 2048, 3)
        self.conv_block14 = _make_conv_block(2048, 2048, 3)
        self.conv_block15 = _make_conv_block(2048, 2048, 3)
        # 4x4 -> 1x1
        self.avgpool1 = nn.AvgPool2d(4)
        self.dense_block1 = _make_dense_block(2048, 4096)
        self.out_layer = nn.Linear(4096, 1512)

    def forward(self, x):

        # ======================================================================
        # Obtain the response matrix output
        # ======================================================================

        # Flatten the input
        x_flat = torch.flatten(x, 1, -1)
        # Grab out the active darkhole pixels
        x_flat_active = x_flat[:, self.all_active_idxs]
        # Denormalize the input data
        x_flat_active_denorm = (x_flat_active * self.input_max_min_diff +
                                self.input_min_x)
        # Run the response matrix
        rm_output = self.resp_mat_layer(x_flat_active_denorm)
        # Normalize the output data
        rm_output = 2 * (
            (rm_output - self.output_min_x) / self.output_max_min_diff) - 1

        # ======================================================================
        # Obtain the CNN output
        # ======================================================================

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        # Downsize
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)
        # Downsize
        x = self.conv_block7(x)
        x = self.conv_block8(x)
        x = self.conv_block9(x)
        # Downsize
        x = self.conv_block10(x)
        x = self.conv_block11(x)
        x = self.conv_block12(x)
        # Downsize
        x = self.conv_block13(x)
        x = self.conv_block14(x)
        x = self.conv_block15(x)
        x = self.avgpool1(x)
        x = torch.squeeze(x, 2)
        x = torch.squeeze(x, 2)
        x = self.dense_block1(x)
        cnn_output = self.out_layer(x)

        # ======================================================================
        # Combine the two outputs
        # ======================================================================

        return cnn_output + rm_output
