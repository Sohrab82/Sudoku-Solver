import numpy as np


def reject_row_col(grid, row, col):
    # change the numbers in row and col to zero,
    # i.e they cannot be the corresponding number
    idx = np.where(grid == 10)
    R, C = idx
    for r, c in zip(R, C):
        for r0 in range(9):
            if grid[r0, col] == -1:
                grid[r0, col] = 0
        for c0 in range(9):
            if grid[row, c0] == -1:
                grid[row, c0] = 0
    return grid


def add_new_number(grids, N, row, col):
    # puts 0 in every grid in the place of the new number
    # and puts 10 for the corresponding grid
    grids[0][row, col] = N
    grids[N][row, col] = 10
    for n in range(1, 10):
        if n != N:
            grids[n][row, col] = 0
        else:
            r0_cell = int(row / 3) * 3
            c0_cell = int(col / 3) * 3
            for r in range(r0_cell, r0_cell + 3):
                for c in range(c0_cell, c0_cell + 3):
                    if grids[n][r, c] == -1:
                        grids[n][r, c] = 0
    reject_row_col(grids[N], row, col)


def process_cell(grids, N, cell_no):
    grid = grids[N]
    div = int((cell_no - 1) / 3)
    rem = (cell_no - 1) % 3
    idx = np.where(grid[div * 3:(div + 1) * 3, rem * 3:(rem + 1) * 3] == -1)
    r, c = idx
    if len(r) == 1:
        add_new_number(grids, N, r + div * 3, c + rem * 3)
        return True
    return False


def process_all_cells(grids):
    new_number_added = True
    while new_number_added:
        new_number_added = False
        for n in range(1, 10):
            for cell_no in range(1, 10):
                new_n_added = True
                while new_n_added:
                    new_n_added = process_cell(grids, n, cell_no)
                    new_number_added = new_number_added or new_n_added


def process_all_lines(grids):
    new_number_added = False
    for line_no in range(1, 10):
        for num in range(1, 10):
            # skip if the number exists in the row
            if np.any(grids[num][line_no - 1, :] == 10):
                continue
            i = (grids[num][line_no - 1, :] == -1).sum()
            if i == 1:
                # only one -1, put the number there
                i = np.where(grids[num][line_no - 1, :] == -1)
                add_new_number(grids, num, line_no - 1, i[0])
                new_number_added = True
        for num in range(1, 10):
            # skip if the number exists in the col
            if np.any(grids[num][:, line_no - 1] == 10):
                continue
            i = (grids[num][:, line_no - 1] == -1).sum()
            if i == 1:
                # only one -1, put the number there
                i = np.where(grids[num][:, line_no - 1] == -1)
                add_new_number(grids, num, i[0], line_no - 1)
                new_number_added = True
    return new_number_added


def process_guess(grids, N, cell_no):
    grid_updated = False
    div = int((cell_no - 1) / 3)
    rem = (cell_no - 1) % 3
    idx = np.where(grids[N][div * 3:(div + 1) * 3,
                            rem * 3:(rem + 1) * 3] == -1)
    r_minus_one, c_minus_one = idx
    if len(r_minus_one) < 1:
        return
    if len(r_minus_one) == 1:
        add_new_number(grids, N, 3 * div +
                       r_minus_one[0], 3 * rem + c_minus_one[0])
        grid_updated = True
        return grid_updated
    if len(np.unique(r_minus_one)) == 1 or len(np.unicode(c_minus_one)) == 1:
        if len(np.unique(r_minus_one)) == 1:
            for c in range(9):
                if c in range(rem * 3, (rem + 1) * 3):
                    # inside the current cell
                    continue
                if grids[N][div * 3 + np.unique(r_minus_one)[0], c] == -1:
                    grids[N][div * 3 + np.unique(r_minus_one)[0], c] = 0
                    grid_updated = True
        else:
            for r in range(9):
                if r in range(div * 3, (div + 1) * 3):
                    # inside the current cell
                    continue
                if grids[N][r, rem * 3 + np.unique(c_minus_one)[0]] == -1:
                    grids[N][r, rem * 3 + np.unique(c_minus_one)[0]] = 0
                    grid_updated = True
    return grid_updated


def process_all_guesses(grids):
    grid_updated = False
    for num in range(1, 10):
        for cell_no in range(1, 10):
            grid_updated = grid_updated or process_guess(grids, num, cell_no)
    return grid_updated


def find_cells_with_only_one_option(grids):
    new_number_added = False
    for r in range(9):
        for c in range(9):
            if grids[0][r, c] != -1:
                continue
            possibilty_num = -1
            for n in range(1, 10):
                if grids[n][r, c] == -1:
                    if possibilty_num != -1:
                        # cell(r, c) has more than one possibilty
                        possibilty_num = 0
                        break
                    possibilty_num = n
            if possibilty_num > 0:
                add_new_number(grids, possibilty_num, r, c)
                new_number_added = True
    return new_number_added


# sudoku_grid = np.array([[-1, -1, -1, -1, -1, -1, 9, 1, -1],
#                         [-1, -1, -1, -1, 5, -1, -1, -1, -1],
#                         [5, -1, -1, -1, -1, 1, 2, 3, -1],
#                         [-1, 3, -1, -1, -1, -1, -1, -1, -1],
#                         [-1, -1, -1, -1, 8, -1, 7, -1, -1],
#                         [-1, 9, 4, -1, 1, 5, 8, -1, -1],
#                         [-1, -1, -1, 3, 6, -1, -1, -1, 9],
#                         [-1, -1, 7, -1, 4, -1, -1, -1, -1],
#                         [4, -1, -1, -1, -1, -1, 3, 2, -1]])


def solve(sudoku_grid):
    print('Unsolved grid:')
    print(sudoku_grid)
    grids = []
    for _ in range(10):
        grids.append(np.ones((9, 9), dtype=np.int) * (-1))

    grids[0] = sudoku_grid

    for n in range(1, 10):
        idx = np.where(grids[0] == n)
        R, C = idx
        grid = grids[n]
        for r, c in zip(R, C):
            grid[r, c] = 10
            add_new_number(grids, n, r, c)

    while process_all_cells(grids) or process_all_lines(grids) or process_all_guesses(grids) or find_cells_with_only_one_option(grids):
        pass
    print('Solved grid:')
    print(grids[0])
