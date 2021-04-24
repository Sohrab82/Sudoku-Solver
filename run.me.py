import cv2
import numpy as np
from utils.models import load_model, model_incept_fun
import tensorflow as tf
from utils.sudoku_solver import solve
from utils.misc import reorder, is_around_point

PLOTTING = False

# load image of the sudoku
image = cv2.imread('./images/sudoku.jpg')

# resize the image
image = cv2.resize(image, (600, 800))
img_width = image.shape[1]
img_height = image.shape[0]

# binary threshold the gray image
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_gray = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY_INV)[1]

# erode and dilate to remove noise and make lines thicker
kernel = np.ones((3, 3), np.uint8)
image_gray = cv2.erode(image_gray, kernel, iterations=1)
image_gray = cv2.dilate(image_gray, kernel, iterations=1)

# find contours and sort them by area
cnts = cv2.findContours(image_gray, cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)[0]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

# find the largest rectangle
largest_rect = []
for c in cnts:
    peri = cv2.arcLength(c, True)
    # approximate a polygon with the contour points
    approx = cv2.approxPolyDP(c, 0.01 * peri, True)
    if len(approx) == 4:
        largest_rect = approx
        cv2.drawContours(image, [approx], 0, (0, 255, 0), 3)
        break

if len(largest_rect) == 0:
    print('Main rectangle not found')
if PLOTTING:
    cv2.imshow("soduki_main", image)
    cv2.waitKey(0)

# perspective transform to get the main rectangle warped
# reorder the points for warp transform
largest_rect = reorder(largest_rect)
M = cv2.getPerspectiveTransform(np.float32(largest_rect),
                                np.float32([
                                    [0, 0],
                                    [img_width, 0],
                                    [0, img_height],
                                    [img_width, img_height],
                                ]))
image_warped = cv2.warpPerspective(
    image_gray, M, (img_width, img_height))

cell_h = int(img_height / 9.)
cell_w = int(img_width / 9.)
margin_h = int(cell_h / 10.)
margin_w = int(cell_w / 10.)
image_warped = image_warped[margin_h:img_height -
                            margin_h, margin_w:img_width - margin_w].copy()

if PLOTTING:
    cv2.imshow("warped", image_warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# load trained mnist model
model = load_model(model_incept_fun, image_shape=(28, 28, 1),
                   n_classes=10, learning_rate=0.001, h5_file='./models/incept.h5')

sudoku_grid = np.ones((9, 9), np.int) * (-1)
for row in range(9):
    for col in range(9):
        # part of the image containing cell[row, col]
        image_cell = image_warped[row * cell_h + margin_h:(row + 1) * cell_h - margin_h,
                                  col * cell_w + margin_w:(col + 1) * cell_w - margin_w]
        image_cell = cv2.erode(image_cell, kernel, iterations=1)

        # find contours inside each cell
        cnts = cv2.findContours(image_cell, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE)[0]

        # find a contour that has a point in radius of min_d to the center of the cell
        min_d = cell_w / 4.
        digit_cnt = []
        for c in cnts:
            if is_around_point(c, (cell_w / 2., cell_h / 2.), min_d):
                digit_cnt = c
                break
        if len(digit_cnt) == 0:
            continue

        # find bounding box the digit and discard outside of it
        x0, y0, w0, h0 = cv2.boundingRect(digit_cnt)

        # some cleaning outside of th detected digit
        mask = np.zeros_like(image_cell)
        mask[y0:y0 + h0, x0:x0 + w0] = 1
        image_cell = cv2.bitwise_and(image_cell, image_cell, mask=mask)

        # image containing only the number
        digit_image = image_cell[y0:y0 + h0, x0:x0 + w0].copy()
        # cv2.rectangle(image_cell, (x0, y0), (x0 + w0, y0 + h0), 255, 1)
        # cv2.imshow(f'{str(row)},{str(col)}',
        #            image_cell)

        # resize (keeping aspect ratio) the number to a (20, 20) square with the number centered
        if w0 > h0:
            digit_image = cv2.resize(digit_image, (20, int(20. / w0 * h0)))
        else:
            digit_image = cv2.resize(digit_image, (int(20. / h0 * w0), 20))

        # image to be used for prediction with NN model
        # The original black and white (bilevel) images from MNIST were size normalized to fit in a 20x20 pixel box while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

        mnist_image = np.zeros((28, 28), np.float)
        # put digit image in the center of mnist_image
        sr = int((28 - digit_image.shape[0]) / 2)
        sc = int((28 - digit_image.shape[1]) / 2)
        mnist_image[sr:sr + digit_image.shape[0],
                    sc:sc + digit_image.shape[1]] = digit_image
        if PLOTTING:
            cv2.imshow(f'{str(row)},{str(col)}',
                       mnist_image)
        mnist_image = mnist_image / 255.
        mnist_image = mnist_image.reshape(1, 28, 28, 1)
        mnist_image = tf.cast(mnist_image, dtype=tf.float32)
        prediction = np.argmax(model.predict([mnist_image]), axis=1)
        print('row={}, col={}, number={}'.format(row + 1, col + 1, prediction))
        sudoku_grid[row, col] = prediction
        if PLOTTING:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

solve(sudoku_grid)
