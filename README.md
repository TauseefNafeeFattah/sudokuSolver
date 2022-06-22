# SudokuSolver
A python sudoku solver. It takes an image containing a sudoku. It finds the sudoku using image processing via openCV library in Python. Then it uses a convolutional neural network (CNN) model to detect and identify the numbers from the images (using digit recognition). After that it solves the sudoku using backtracking technique. And it displays the solved sudoku as an image and it also displays the solution by overlaying it with the original image.  
## Run the program
To run the program create/go (by using cd) to a directory of your choice in the command prompt and run **git clone https://github.com/TauseefNafeeFattah/sudokuSolver.git**  
Then create a virtual environment using **python -m venv env** (Also you can take a look at this [link](https://docs.python.org/3/library/venv.html) to see the oficial documentation on how to create a virtual environment)  
Then activate the virtual environment by writing **pathToVirtualEnvironment\env\Scripts\activate.bat**  
Then install the necessary dependencies using **pip install -r /path/to/requirements.txt** command  
Then to run the main program run this command- **python SudokuSolverMain.py**  
### To create the CNN model.
The CNN model is already created and uploaded in this repository.  
But if you want to create the model then run the command- **python modelTraining.py** and this will successfully create the CNN model.
#### Final result
final result of running the main program is shown below(this is the image 2 in the test_images folder)  
Input-![2](https://user-images.githubusercontent.com/57330415/174934814-a3ad4c13-a174-44ba-9404-bea7c0b10045.jpg)  
Output- ![solved2](https://user-images.githubusercontent.com/57330415/174934887-9299b44d-81f1-49f8-ad2d-89fdf45263ed.png)
