# Mechanical-MNIST-Fashion
similar to Mechanical MNIST -- but with fashion MNIST as input!

# Full Dataset
The full Mechanical MNIST - fashion dataset can be downloaded from <link forthcoming>

# This repository contains the following:

## 1) the code to generate the dataset

This code is idential to the Mechanical MNIST simulation code (https://github.com/elejeune11/Mechanical-MNIST/tree/master/generate_dataset) with two exceptions:
* the input pattern bitmap is from the Fashion MNIST dataset
* the maximum Young's modulus is 25 units in this code (it is 100 in the original Mechanical MNIST dataset) 
The file ``fashion_MNIST_UE.py'' corresponds to the Uniaxial Extension (UE) load case while the file ``fashion_MNIST_EE.py'' corresponds to the Equibiaxial Extension (EE) load case. 

Sample input bitmaps are included in the folder. All simulations are conducted with the FEniCS computing platform (https://fenicsproject.org).

## 
Note -- the original Fashion MNIST dataset is available here: https://github.com/zalandoresearch/fashion-mnist

The MIT License (MIT) Copyright © 2017 Zalando SE, https://tech.zalando.com

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


## 2) code to import the data into python 

In https://github.com/elejeune11/Mechanical-MNIST-fashion/tree/main/plot_data in ``plot_data.py'' we show how to import the files into python (100 samples of test and training data for UE are given in ``example_outputs''). Then we show how to plot the data. 

# Additional information

See:
https://github.com/elejeune11/Mechanical-MNIST
and
https://github.com/elejeune11/Mechanical-MNIST-Transfer-Learning
for more info on metamodeling 
