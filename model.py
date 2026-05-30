from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd 


iris = load_iris()
x = iris.data
y = iris.target

df = pd.DataFrame(x, columns=iris.feature_names)
df['target_label'] = y 
df['species_names'] = df['target_label'].map({0:'setosa',
                                            1:'versicolor',
                                            2:'virginica'})
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print("----Data Fram Preview----")
print(df)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                        test_size=0.2,
                                        random_state=42,
                                        shuffle=True)
         
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train_scaled, y_train)
predictions = model.predict(x_test_scaled)

conf_matrix = confusion_matrix(y_test, predictions)

print("----Confusion Matrix----\n")
print(f"\n{conf_matrix}")
print("\n----Classification Report----")
print(classification_report(y_test, predictions, target_names=iris.target_names))