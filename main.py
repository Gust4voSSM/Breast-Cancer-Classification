from sklearn import (
    model_selection,
    tree,
    naive_bayes,
    linear_model,
    neighbors,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from abc import ABC, abstractmethod
import os
import json
from visualizador import VisualizadorInterativo

def analisar_dados(data):
    print('\n' "Numero de exemplos: ", data.shape[0], '\n',
          '\n' "Por classe:" '\n',
          data[1].value_counts(),
          '\n' "//// HEAD ////" '\n',
          data.head())

class Modelo(ABC):
    @abstractmethod
    def train(self, *, X_train, y_train, X_test, y_test, metrica='accuracy'):
        pass

class KNNModelo(Modelo):
    def train(self, *, X_train, y_train, X_test, y_test, metrica='accuracy'):
        #print(f"\nBuscando melhor valor de k para KNN com validação cruzada (métrica: {metrica})...")
        param_grid = {'n_neighbors': list(range(1, 6))}
        grid = model_selection.GridSearchCV(
            neighbors.KNeighborsClassifier(),
            param_grid,
            cv=5,
            scoring=metrica
        )
        grid.fit(X_train, y_train)
        return grid

class ArvoreDecisaoModelo(Modelo):
    def train(self, *, X_train, y_train, X_test, y_test, metrica='accuracy'):
        #print(f"\nBuscando melhores hiperparâmetros para árvore de decisão com validação cruzada (métrica: {metrica})...")
        param_grid = {
            'max_depth': [None, 3, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid = model_selection.GridSearchCV(
            tree.DecisionTreeClassifier(random_state=42),
            param_grid,
            cv=5,
            scoring=metrica
        )
        grid.fit(X_train, y_train)
        return grid

class NaiveBayesModelo(Modelo):
    def __init__(self):
        super().__init__("Naive Bayes")
        self.classificador = GaussianNB()

    def train(self, *, X_train, y_train, X_test, y_test, metrica='accuracy'):
        from sklearn.naive_bayes import GaussianNB
        grid = model_selection.GridSearchCV(
            GaussianNB(),
            param_grid={},  # não tem hiperparâmetros
            cv=5,
            scoring=metrica
        )
        grid.fit(X_train, y_train)
        return grid

class RegressaoLogisticaModelo(Modelo):
    def __init__(self):
        super().__init__("Regressão Logística")
        self.classificador = LogisticRegression(
            solver='liblinear',  # bom para conjuntos pequenos
            penalty='l2',
            max_iter=1000,
            random_state=42
        )

    def train(self, *, X_train, y_train, X_test, y_test, metrica='accuracy'):
        from sklearn.linear_model import LogisticRegression
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear'],
            'penalty': ['l2'],
            'max_iter': [1000]
        }
        grid = model_selection.GridSearchCV(
            LogisticRegression(random_state=42),
            param_grid,
            cv=5,
            scoring=metrica
        )
        grid.fit(X_train, y_train)
        return grid

def print_resumo_resultados(resultados):
    print("\nResumo dos resultados:")
    for nome_modelo, res_mod in resultados.items():
        print(f"\nModelo: {nome_modelo}")
        for proporcao, res_prop in res_mod.items():
            print(f"  Proporção treino/teste: {proporcao}")
            for _, res in res_prop.items():
                print(f"Treino={res['score_treino']:.4f}, Teste={res['score_teste']:.4f}")

def treinar_modelos(resultados_path="resultados_grid.json"):
    print("Importando dataset...")
    data = pd.read_csv("data/wdbc.data", header=None)
    analisar_dados(data)
    
    # Colunas de atributos
    X = data.iloc[:, 2:].values

    # Coluna da classe
    y = data.iloc[:, 1].values
    y = (y == 'M').astype(int) # Convertendo numérica (binário)
    metricas = ['accuracy', 'f1', 'recall', 'precision']
    modelos = [
        ("KNN", KNNModelo()),
        ("Árvore de Decisão", ArvoreDecisaoModelo()),
        ("Naive Bayes", NaiveBayesModelo()),
        ("Regressão Logística", RegressaoLogisticaModelo())
    ]
    
    proporcoes = [(treino/100, 1-treino/100) for treino in range(5, 100, 5)]
    resultados = {}
    for prop_treino, prop_teste in proporcoes:
        print(f"\n==== Proporção treino/teste: {int(prop_treino*100)}%/{int(prop_teste*100)}% ====")
        try:
            X_train, X_test, y_train, y_test = model_selection.train_test_split(
                X, y, test_size=prop_teste, random_state=42, stratify=y
            )
        except ValueError as e:
            print(f"[EXCEÇÃO] Falha ao dividir dados para proporção {int(prop_treino*100)}%/{int(prop_teste*100)}%: {e}")
            continue
        print(f"Tamanho treino: {X_train.shape[0]}")
        print(f"Tamanho teste: {X_test.shape[0]}")
        params = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }
        for nome_modelo, modelo in modelos:
            #print(f"\n==== {nome_modelo} ====")
            if nome_modelo not in resultados:
                resultados[nome_modelo] = {}
            if f"{int(prop_treino*100)}%/{int(prop_teste*100)}%" not in resultados[nome_modelo]:
                resultados[nome_modelo][f"{int(prop_treino*100)}%/{int(prop_teste*100)}%"] = {}
            for metrica in metricas:
                params['metrica'] = metrica
                #print(f"\n--- Avaliando métrica: {metrica} ---")
                grid = modelo.train(**params)
                best_estimator = grid.best_estimator_
                score_treino = best_estimator.score(X_train, y_train)
                score_teste = best_estimator.score(X_test, y_test)
                if nome_modelo == "KNN":
                    melhor = grid.best_params_['n_neighbors']
                else:
                    melhor = grid.best_params_
                resultados[nome_modelo][f"{int(prop_treino*100)}%/{int(prop_teste*100)}%"].setdefault(metrica, {})
                resultados[nome_modelo][f"{int(prop_treino*100)}%/{int(prop_teste*100)}%"] [metrica] = {
                    'melhor_param': melhor,
                    'score_treino': score_treino,
                    'score_teste': score_teste
                }
    with open(resultados_path, "w") as f:
        json.dump(resultados, f, indent=2)

def main():
    resultados_path = "resultados_grid.json"
    
    if os.path.exists(resultados_path):
        print("[INFO] Resultados já existem. Pulando treinamento e carregando do JSON...")
    else:
        treinar_modelos()

    with open(resultados_path, "r") as f:
            resultados = json.load(f)
    
    # Inicializar visualizador interativo
    print("\n[INFO] Abrindo visualizador interativo...")
    print("Instruções:")
    print("- Use o dropdown de radio buttons para selecionar a métrica")
    print("- Use os checkboxes 'Modelos' para mostrar/ocultar modelos")
    print("- Use os checkboxes 'Conjuntos' para mostrar Treino e/ou Teste")
    print("- Curvas sólidas = Conjunto de Treino, Curvas tracejadas = Conjunto de Teste")
    print("- O eixo Y está otimizado para melhor visualização (foco em 0.8-1.0)")
    
    visualizador = VisualizadorInterativo(resultados)

if __name__ == "__main__":
    main()