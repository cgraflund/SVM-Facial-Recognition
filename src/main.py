import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from os.path import dirname, abspath, join

def read_pgm(pgmf):
    """Return a raster of integers from a PGM as a list of lists."""
    assert pgmf.readline() == b'P5\n'
    (width, height) = [int(i) for i in pgmf.readline().split()]
    depth = int(pgmf.readline())
    assert depth <= 255

    raster = []
    for y in range(height):
        row = []
        for y in range(width):
            row.append(ord(pgmf.read(1)))
        raster.append(row)
    return raster

def read():
    print("Reading input faces...")
    dir = join(dirname(dirname(abspath(__file__))),'data', 's')
    X = np.matrix(np.zeros((92*112,400)))
    y = np.zeros(400)
    for subject in range(40):
        for image in range(10):
            f = open(join(dir+str(subject+1), str(image+1) + ".pgm"), 'rb')
            img = read_pgm(f)
            X[:, 10 * subject + image] = np.mat(np.asarray(img).flatten()).T
            y[10 * subject + image] = subject+1
    return X.T, y

# Read in the inputs and confirm shape
X, y = read()
print(f"X data shape:    {X.shape}")
print(f"y data shape:    {y.shape}")

# Train test split
print("\nTrain test split...")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
print(f"All Data:        {len(X)} points")
print(f"Training data:   {len(X_train)} points")
print(f"Testing data:    {len(X_test)} points")

# SVM stuff
print("\nRunning SVM...")
svc = SVC()
svc.fit(X_train, y_train)

# Evaluate
accuracy = svc.score(X_test, y_test)
print(f"Testing accuracy: {accuracy}")