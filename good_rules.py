from dataclasses import dataclass
import numpy as np
import cv2

def function():
    ...
function = type(function)

def basic_convolution(field: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return cv2.filter2D(field, -1, kernel, borderType=cv2.BORDER_ISOLATED)

def fast_inv_gaussian_activation(field: np.ndarray) -> np.ndarray:
    return (-1/np.exp(0.42*field**2)+1)

def inv_gaussian_activation(field: np.ndarray) -> np.ndarray:
    return (-1/np.power(2, (0.6*np.power(field, 2)))+1)

def basic_intervention(field: np.ndarray, x: int, y:int) -> None:
    field[x, y] = 1

def checkerboard_intervetion(field: np.ndarray, x: int, y:int) -> None:
    field[x-5:x+5, y-5:y+5] = (
        np.array([[1,0]*5, [0,1]*5] * 5)
        * cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
    )


@dataclass
class CallableRuleset:

    kernel: np.ndarray
    convolution: function
    activation: function
    intervention: function
    steps: int = 1
    initialization_percentage: float = 0.5

    def __call__(self, field: np.ndarray) -> np.ndarray:
        for _ in range(self.steps):
            field = self.activation(self.convolution(field, self.kernel))
        return field



classic = CallableRuleset(
    kernel = np.array([
        [1,1,1],
        [1,9,1],
        [1,1,1],
    ], dtype='uint8'),
    convolution = basic_convolution,
    activation = lambda field: ((field == 3) | (field == 11) | (field == 12)),
    intervention = basic_intervention
)

slime_pulling_worms = CallableRuleset(
    kernel = np.array([
        [  0.74, -0.946,   0.74],
        [-0.946, -0.434, -0.946],
        [  0.74, -0.946,   0.74],
    ], dtype='float32'),
    convolution = basic_convolution,
    activation = inv_gaussian_activation,
    steps=4,
    intervention = checkerboard_intervetion
)

blood_pumping_worms = CallableRuleset(
    kernel = np.array([
        [ 0.742, -0.966, 0.742],
        [-0.966, -0.45, -0.966],
        [ 0.742, -0.966, 0.742],
    ], dtype='float32'),
    convolution = basic_convolution,
    activation = inv_gaussian_activation,
    intervention = checkerboard_intervetion,
    steps=4,
    initialization_percentage=0.2
)

pipes = CallableRuleset(
    kernel = np.array([
        [-.2,  -0.1,     0,    .1,   -.2],
        [ .1,     0,   .55,    .0,   -.1],
        [  0,   .55,   1.3,   .55,     0],
        [-.1,     0,   .55,    .0,    .1],
        [-.2,    .1,     0,   -.1,   -.2],
    ], dtype='float32'),
    convolution = basic_convolution,
    activation = inv_gaussian_activation,
    intervention = checkerboard_intervetion,
    initialization_percentage = 0.1
)


moving_rocks = CallableRuleset(
    kernel = np.array([
        [-.3,   -.3,    .1,    .3,   -.3],
        [ .3,    .0,   .55,    .0,   -.3],
        [ .1,   .55,   1.2,   .55,    .1],
        [-.3,    .0,   .55,    .0,    .3],
        [-.6,    .3,    .1,   -.3,   -.3],
    ], dtype='float32'),
    convolution = basic_convolution,
    activation = inv_gaussian_activation,
    intervention = checkerboard_intervetion,
    initialization_percentage = 0.2
)
