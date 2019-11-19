from sklearn import datasets 
from sklearn.tree import DecisionTreeClassifier, export_text
#from sklearn.tree import export_graphviz
import numpy as np
from sklearn.model_selection import train_test_split

decisionTreeClassifier = DecisionTreeClassifier(random_state=4)
export_text = export_text

if __name__ == "__main__":
    dataset = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.01)
    print(len(X_train), len(X_test))
    decisionTreeClassifier.fit(X_train, y_train)
    
    '''export_graphviz(
        treeClf,
        out_file=dot_data,
        feature_names=dataTraining.feature_names,
        class_names=dataTraining.target_names,
        rounded=True,
        filled=True
    )'''

    r = export_text(decisionTreeClassifier, feature_names=dataset['feature_names'])

    print(r)
    r = r.split('\n')
    for i in range(len(r)):
        r[i] = r[i]+'<br>'
    r = ''.join(r)
    r = r.split('   ')
    for i in range(len(r)):
        r[i] = r[i] + '&nbsp;&nbsp;&nbsp;&nbsp;'
    r = ''.join(r)

    print(dataset['feature_names'])
    print(dataset.data[0])
    print(y_test)
    output=decisionTreeClassifier.predict(X_test)
    print(output)
    
    print(decisionTreeClassifier.score(X_test, y_test))
