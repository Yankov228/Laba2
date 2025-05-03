import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import preprocessing 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Вхідний файл, який містить дані 
input_file = 'income_data.txt'

# Читання даних 
X = [] 
y = [] 
count_class1 = 0 
count_class2 = 0 
max_datapoints = 25000

with open(input_file, 'r') as f: 
    for line in f.readlines(): 
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break 
 
        if '?' in line: 
            continue
        data = line[:-1].split(', ') 

        if data[-1] == '<=50K' and count_class1 < max_datapoints: 
            X.append(data) 
            count_class1 += 1 
        
        if data[-1] == '>50K' and count_class2 < max_datapoints: 
            X.append(data) 
            count_class2 += 1

# Перетворення на масив numpy 
X = np.array(X)

# Перетворення рядкових даних на числові 
label_encoder = [] 
X_encoded = np.empty(X.shape) 
for i,item in enumerate(X[0]): 
    if item.isdigit(): 
        X_encoded[:, i] = X[:, i] 
    else: 
        label_encoder.append(preprocessing.LabelEncoder()) 
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i]) 
 
X = X_encoded[:, :-1].astype(int) 
Y = X_encoded[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

# Створення SVМ-класифікатора
classifiers = []
classifiers.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr'))) 
classifiers.append(('LDA', LinearDiscriminantAnalysis())) 
classifiers.append(('KNN', KNeighborsClassifier())) 
classifiers.append(('CART', DecisionTreeClassifier())) 
classifiers.append(('NB', GaussianNB()))
classifiers.append(('SVM', SVC(gamma='auto')))



# Передбачення результату для тестової точки даних 
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners', 'Not-in-family', 'White', 'Male', '0', '0', '40', 'United-States']

# Кодування тестової точки даних 
input_data_encoded = [-1] * len(input_data) 
count = 0
for i, item in enumerate(input_data): 
    if item.isdigit(): 
        input_data_encoded[i] = int(input_data[i])
    else: 
        input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]])[0]) 
        count += 1 

input_data_encoded = np.array(input_data_encoded)


# Навчання та оцінка моделей
for name, model in classifiers:
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    predicted_class = model.predict([input_data_encoded])

    print(f"\nClassifier: {name}")
    print("Predicted income class:", label_encoder[-1].inverse_transform(predicted_class)[0])
    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
    print("Classification Report:\n", classification_report(y_test, y_test_pred))




