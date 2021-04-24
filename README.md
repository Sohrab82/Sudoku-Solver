## Project: Sudoko Solver

[//]: # (Image References)

[image1]: ./images/soduku.jpg "input"
[image2]: ./output_images/dist_train.jpg "Training data distrib"
[image3]: ./output_images/dist_valid.jpg "Validation data distrib"
[image4]: ./output_images/dist_test.jpg "Testing data distrib"
[image41]: ./output_images/sample_30.jpg "speed 80"
[image5]: ./output_images/lenet.jpg "lenet train"
[image6]: ./output_images/incept.jpg "incept train"
[image7]: ./output_images/new_images.jpg "new images"


### Overview

- Reads a picture of a sudoku puzzle
- Performs some image processing with OpenCV to find the numbers on the puzzle
- Uses the pre-trained CNN model to recognise the numbers
- Uses a developed algorithm (implemented in `utils/sudoku_solver.py`) to solve the puzzle

### Files:

`train_mnist.py`: How to load mnist dataset and train it with Tensorflow 2.x with the models defined in `utils/models.py` and save the models in `./models/*.h5`

`load_mnist.py`: How to load the pre-trained models from `./models/*.h5` and evaluate them with Tensorflow

`run_me.py`: Main file to read the image and solve the puzzle

