# Завантаження бібліотек
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
import numpy as np

# Завантаження датасету
dataset = load_iris()
X = dataset.data
y = dataset.target

# Розділення на навчальну і валідаційну вибірки
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# Модель
model = SVC(gamma='auto')

# Крос-валідація
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
print("SVM: %f (%f)" % (cv_results.mean(), cv_results.std()))

# Навчання і прогноз
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Оцінка
print("Accuracy: ", accuracy_score(Y_validation, predictions))
print("Confusion Matrix:\n", confusion_matrix(Y_validation, predictions))
print("Classification Report:\n", classification_report(Y_validation, predictions))

# Нове передбачення
X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = model.predict(X_new)  
print("Прогноз: {}".format(prediction))  
print("Спрогнозована мітка: {}".format(dataset['target_names'][prediction]))  
