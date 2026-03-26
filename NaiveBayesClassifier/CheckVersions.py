# EECS 658 Assignment 1
# Name: Abinav Krishnan
# Date: 08 / 30 / 2024
# Description: Prints the version of python and libraries

# CheckVersions
def checkVersions():
    import sys
    print('Python: {}'.format(sys.version))
    # scipy
    import scipy
    print('scipy: {}'.format(scipy.__version__))
    # numpy
    import numpy
    print('numpy: {}'.format(numpy.__version__))
    # pandas
    import pandas
    print('pandas: {}'.format(pandas.__version__))
    # scikit-learn
    import sklearn
    print('sklearn: {}'.format(sklearn.__version__))
    print()
    
checkVersions() # Calls checkVersions
print("Hello World!") # Prints "Hello World!"
