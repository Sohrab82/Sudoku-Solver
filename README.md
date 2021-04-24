## Project: Sudoko Solver

[//]: # (Image References)

[image1]: ./images/soduku.jpg "input"
[image2]: ./images/incept.jpg "Incption model training"


### Overview
![alt text][image1]

- Reads a picture of a sudoku puzzle
- Performs some image processing with OpenCV to find the numbers on the puzzle
- Uses the pre-trained CNN model to recognise the numbers
- Uses a developed algorithm (implemented in `utils/sudoku_solver.py`) to solve the puzzle

### Files:
![alt text][image2]

`train_mnist.py`: How to load mnist dataset and train it with Tensorflow 2.x with the models defined in `utils/models.py` and save the models in `./models/*.h5`

`load_mnist.py`: How to load the pre-trained models from `./models/*.h5` and evaluate them with Tensorflow

`run_me.py`: Main file to read the image and solve the puzzle

