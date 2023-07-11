# ggol
Generalized Game Of Life.
The logic is implemented as a two stage operation:
1. convolution
2. activation

the main point is to have a well defined search space for a given activation function that allows to search for some cool kernels, while keeping the whole operation intuitive and simple. This way, im planning to fit a model that gauges the coolness of a given resulting field, where coolness is a proxy for some tangible metrics like contrast, mean value and frame-to-frame change. Of course, the whole idea is to allow the model to find its own metrics that satisfy the hard-coded rules and use it as a loss to optimise the kernel. Im not sure whether i want to keep the optuna as a kernel optimizer, but since im aiming for an emergent property i would prefer to start with something with more freedom (im not sure my loss will always be differentiable)
![image](https://github.com/iliya-malecki/ggol/assets/53195438/7ed57947-7221-47d6-9163-3867ce9e62c9)
