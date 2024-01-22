from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,jaccard_score,f1_score,confusion_matrix,ConfusionMatrixDisplay,precision_score,recall_score
import matplotlib.pyplot as plt
import numpy as np

def train_knn_model(X_train, X_test, y_train, y_test, n_neighbors=4):
    """
    Verilen girdi verileriyle KNN modeli oluşturur ve eğitir.
    
    Parameters:
    - X_train, X_test, y_train, y_test: Train ve test veri setleri.
    - n_neighbors (int): KNN modelinde kullanılacak komşu sayısı.

    Returns:
    - None
    """

    # KNN modelini oluştur ve eğit
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X_train, y_train)

    # Test verisi üzerinde tahmin yap
    y_pred = knn_model.predict(X_test)

    # Doğruluk (accuracy) hesapla
    accuracy = accuracy_score(y_test, y_pred)
    print(f"KNN Accuracy: {accuracy:.3f}")


def train_logistic_regression_model(X_train, X_test, y_train, y_test, C=1.0,solver='lbfgs',multi_class='multinomial'):
    """
    Verilen girdi verileriyle logistic regression modeli oluşturur ve eğitir.
    
    Parameters:
    - X_train, X_test, y_train, y_test: Train ve test veri setleri.
    - C (float): Inverse of regularization strength (düzenleme kuvvetinin tersi).

    Returns:
    - None
    """
    # Logistic regression modelini oluştur ve eğit
    logreg_model = LogisticRegression(C=C,solver=solver,multi_class=multi_class)
    logreg_model.fit(X_train, y_train)

    # Test verisi üzerinde tahmin yap
    y_pred = logreg_model.predict(X_test)

    # Doğruluk (accuracy) hesapla
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Accuracy: {accuracy:.3f}")

    # F1-score hesapla
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"F1-Score: {f1:.3f}")

    jac_score = jaccard_score(y_test, y_pred, average='micro')
    print(f"Jaccard Score: {jac_score:.3f}")

    # Confusion matrix hesapla
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1,2])
    # disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix, display_labels=LR.classes_)
    disp.plot(cmap='YlGn')
    

def train_decision_tree_model(X_train, X_test, y_train, y_test, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    """
    Verilen girdi verileriyle Decision Tree modeli oluşturur ve eğitir.
    
    Parameters:
    - X_train, X_test, y_train, y_test: Train ve test veri setleri.
    - max_depth (int): Ağacın maksimum derinliği.
    - min_samples_split (int veya float): Bir iç düğümün ikiye bölünmesi için gerekli örneklerin minimum sayısı.
    - min_samples_leaf (int veya float): Bir yaprak düğümünde bulunması gereken minimum örnek sayısı.

    Returns:
    - None
    """
    # Decision Tree modelini oluştur ve eğit
    dt_model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    dt_model.fit(X_train, y_train)

    # Test verisi üzerinde tahmin yap
    y_pred = dt_model.predict(X_test)

    # Doğruluk (accuracy) hesapla
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Decision Tree Accuracy: {accuracy:.3f}")

    # F1-score hesapla
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"F1-Score: {f1:.3f}")

    # Jaccard Score hesapla
    jaccard = jaccard_score(y_test, y_pred, average="weighted")
    print(f"Jaccard Score: {jaccard:.3f}")

    # Confusion matrix hesapla
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    disp.plot(cmap='YlGn')

def train_random_forest_model(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    """
    Verilen girdi verileriyle Random Forest modeli oluşturur ve eğitir.
    
    Parameters:
    - X_train, X_test, y_train, y_test: Train ve test veri setleri.
    - n_estimators (int): Oluşturulacak ağaç sayısı.
    - max_depth (int): Her bir ağacın maksimum derinliği.
    - min_samples_split (int veya float): Bir iç düğümün ikiye bölünmesi için gerekli örneklerin minimum sayısı.
    - min_samples_leaf (int veya float): Bir yaprak düğümünde bulunması gereken minimum örnek sayısı.

    Returns:
    - None
    """
    # Random Forest modelini oluştur ve eğit
    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=42)
    rf_model.fit(X_train, y_train)

    # Test verisi üzerinde tahmin yap
    y_pred = rf_model.predict(X_test)

    # Doğruluk (accuracy) hesapla
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {accuracy:.3f}")

    # F1-score hesapla
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"F1-Score: {f1:.3f}")

    # Jaccard Score hesapla
    jaccard = jaccard_score(y_test, y_pred, average="weighted")
    print(f"Jaccard Score: {jaccard:.3f}")

    # Confusion matrix hesapla
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    disp.plot(cmap='YlGn')

def train_naive_bayes(X_train, X_test, y_train, y_test):
    """
    Verilen girdi verileriyle Naive Bayes modeli oluşturur ve eğitir.
    
    Parameters:
    - X_train, X_test, y_train, y_test: Train ve test veri setleri.

    Returns:
    - None
    """
    # Naive Bayes modelini oluştur ve eğit
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    # Test verisi üzerinde tahmin yap
    y_pred = nb_model.predict(X_test)

    # Doğruluk (accuracy) hesapla
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Naive Bayes Accuracy: {accuracy:.3f}")

    # Precision, Recall, F1-Score ve Jaccard Score hesapla
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    jaccard = jaccard_score(y_test, y_pred, average='weighted')

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1:.3f}")
    print(f"Jaccard Score: {jaccard:.3f}")

    # Confusion matrix hesapla ve görselleştir
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    disp.plot(cmap='YlGn')