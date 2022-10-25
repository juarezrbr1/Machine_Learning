from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

arquivo = read_csv('wine_dataset.csv')

arquivo['style'] = arquivo['style'].replace('red', 0)
arquivo['style'] = arquivo['style'].replace('white', 1)

y = arquivo['style']
x = arquivo.drop('style', axis=1)

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

modelo = ExtraTreesClassifier(n_estimators=100)
modelo.fit(x_treino, y_treino)

resultado = modelo.score(x_teste, y_teste)
print('Acurácia: ', resultado)

print(y_teste[30:33])

previsoes = modelo.predict(x_teste[30:33])
print(previsoes)