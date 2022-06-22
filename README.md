# sudokuSolver
A python sudoku solver. It takes an image containing a sudoku. It finds the sudoku using image processing via openCV library in Python. Then it uses a convolutional neural network (CNN) model to detect and identify the numbers from the images (using digit recognition). After that it solves the sudoku using backtracking technique. And it displays the solved sudoku as an image and it also displays the solution by overlaying it with the original image.  
# Run the program
To run the program create/go (by using cd) to a directory of your choice in the command prompt and run git clone sdjalskd  
Then create a virtual environment using python -m venv env
Then activate the virtual environment by writing pathToVirtualEnvironment\env\Scripts\activate.bat
Then install the necessary dependencies using pip install -r /path/to/requirements.txt command
Then to run the program run this command- python SudokuSolverMain.py
# To create the CNN model.
The CNN model is already created and uploaded in this repository.
But if you want to create the model then run the command- python modelTraining.py and this will create the CNN model.
