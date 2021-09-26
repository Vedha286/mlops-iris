from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# define a Gaussain NB classifier
GaussianNBClf = GaussianNB()
GaussianNBClf_acc = 0.0


# define the class encodings and reverse encodings
classes = {0: "Iris Setosa", 1: "Iris Versicolour", 2: "Iris Virginica"}
r_classes = {y: x for x, y in classes.items()}

# function to train and load the model during startup
def load_model():
    # load the dataset from the official sklearn datasets
    X, y = datasets.load_iris(return_X_y=True)

    # do the test-train split and train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    GaussianNBClf.fit(X_train, y_train)

    # calculate the print the accuracy score
    GaussianNBClf_acc = accuracy_score(y_test, GaussianNBClf.predict(X_test))
    
    print(f"GaussianNB Classifier trained with accuracy: " + str(GaussianNBClf_acc))
    

# function to predict the flower using the model
def predict(query_data):
    print(query_data)
    x = list(query_data.dict().values())
    print(x)
    prediction = GaussianNBClf.predict([x])[0]
    print(f"Model prediction: {classes[prediction]}")
    return classes[prediction]

# function to retrain the model as part of the feedback loop
def retrain(data):
    # pull out the relevant X and y from the FeedbackIn object
    X = [list(d.dict().values())[:-1] for d in data]
    y = [r_classes[d.flower_class] for d in data]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    GaussianNBClf.fit(X_train, y_train)

    # calculate the print the accuracy score
    GaussianNBClf_acc = accuracy_score(y_test, GaussianNBClf.predict(X_test))

    print(f"GaussianNB Classifier trained with accuracy: " + str(GaussianNBClf_acc))
   
