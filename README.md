# cnn_cpu
Reference-Implementation of a Convolutional Neural Network using C++ without parallelization.

Used libraries: Armadillo, Google Test


### Usage:

**1.** Clone repository  `git clone https://github.com/felixp98/cnn_cpu.git`<br>
**2.** Install Armadillo version 8.4 or higher (see below).<br>
**3.** Create a build directory, e.g. `mkdir cmake_build`<br>
**4.** Update the correct path for the MNIST data in `cnn_reference_test.cpp`<br>
**5.** cd into the build directory and generate build files: `cmake ../`<br>
**6.** Build the project using `make`<br>
**7.** Train and test the model using `./cnn_reference_test`
<br>
<br>

#### Armadillo Installation:
1. Install Armadillo prerequisities: `sudo apt install cmake libopenblas-dev liblapack-dev`
2. Download newest Armadillo version: http://arma.sourceforge.net/download.html
3. `cd arma*` into extracted directory
4. Generate build files: `cmake .`
5. `make`
6. `sudo make install`
