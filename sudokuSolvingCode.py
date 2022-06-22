# sudoku solving algorithm


def find_next_empty(puzzle):
    '''
    Returns the next row, column thats not filled yet.
    If all the rows and columns are filled return None, None
    input:
        puzzle- an array of arrays of int ([[int],[int]]) (the sudoku board)
    output:
        (row,colum) (a tuple of next empty box)
    '''
    for r in range(9):
        for c in range(9):
            if puzzle[r][c] == 0:
                return r, c

    return None, None


def is_valid(puzzle, guess, row, col):
    '''
    Checks if the guess at the row and column of the puzzle is valid
    input:
        puzzle- an array of arrays of int ([[int],[int]]) (the sudoku board)
        guess- int
        row- int
        col - int
    output:
        Boolean (True/False)
    '''
    # if the number is repeated in row or column, then the guess is not valid
    row_vals = puzzle[row]
    if guess in row_vals:
        return False

    col_vals = [puzzle[i][col] for i in range(9)]
    if guess in col_vals:
        return False

    # Check if the number is repeated in the square
    row_start = (row // 3) * 3
    col_start = (col // 3) * 3

    for r in range(row_start, row_start + 3):
        for c in range(col_start, col_start + 3):
            if puzzle[r][c] == guess:
                return False

    # the number is valid so return true
    return True


def solve_sudoku(puzzle):
    '''
    Solves the sudoku using backtracking
    input:
        puzzle- an array of arrays of int ([[int],[int]])
        (each inner array is a row in the sudoku)
    output:
        Boolean (True if the solution exists and False if it doesn't exist)
        If the solution exists mutates the puzzle to be the solution and
        return that puzzle. else leave the puzzle as it is.
    '''

    # find an empty box to make the guess
    row, col = find_next_empty(puzzle)

    # check if an empty box exists(it it doesn't then we have found a solution)
    if row is None:
        return True

    # if there is empty space make a guess and check if its valid
    # if its valid place the guess at the specified box and
    # recursively call solve_sudoku
    for guess in range(1, 10):
        if is_valid(puzzle, guess, row, col):
            puzzle[row][col] = guess
            if solve_sudoku(puzzle):
                return True

        # if its not valid backtrack and try a new number
        puzzle[row][col] = 0

    # if none of the numbers tries works then its unsolvable
    return False
