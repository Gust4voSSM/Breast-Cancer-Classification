from sklearn import (
    model_selection,
    tree,
    naive_bayes,
    linear_model,
    neighbors
)
import pandas as pd
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import os
import json

def analisar_dados(data):
    print('\n' "Numero de exemplos: ", data.shape[0], '\n',
          '\n' "Por classe:" '\n',
          data[1].value_counts(),
          '\n' "//// HEAD ////" '\n',
          data.head())

def treinar_arvore_decisao(X_train=None, y_train=None, X_test=None, y_test=None, metrica='accuracy'):
    print(f"\nBuscando melhores hiperparâmetros para árvore de decisão com validação cruzada (métrica: {metrica})...")
    param_grid = {
        'max_depth': [None],
        'min_samples_split': [2],
        'min_samples_leaf': [1]
    }
    grid = model_selection.GridSearchCV(
        tree.DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring=metrica
    )
    grid.fit(X_train, y_train)
    print(f"Melhores parâmetros encontrados: {grid.best_params_}")
    print(f"Média na validação cruzada ({metrica}): {grid.best_score_:.4f}")
    clf_best = grid.best_estimator_
    test_acc = clf_best.score(X_test, y_test)
    print(f"{metrica.capitalize()} no conjunto de teste: {test_acc:.4f}")
    return clf_best, grid
    
def treinar_knn_com_cv(X_train=None, y_train=None, X_test=None, y_test=None, metrica='accuracy'):
    print(f"\nBuscando melhor valor de k para KNN com validação cruzada (métrica: {metrica})...")
    param_grid = {'n_neighbors': list(range(1, 6))}
    grid = model_selection.GridSearchCV(
        neighbors.KNeighborsClassifier(),
        param_grid,
        cv=5,
        scoring=metrica
    )
    grid.fit(X_train, y_train)
    print(f"Melhor k encontrado: {grid.best_params_['n_neighbors']}")
    print(f"Média na validação cruzada ({metrica}): {grid.best_score_:.4f}")
    test_acc = grid.best_estimator_.score(X_test, y_test)
    print(f"{metrica.capitalize()} no conjunto de teste: {test_acc:.4f}")
    return grid

class Modelo(ABC):
    @abstractmethod
    def train(self, *, X_train, y_train, X_test, y_test, metrica='accuracy'):
        pass

class KNNModelo(Modelo):
    def train(self, *, X_train, y_train, X_test, y_test, metrica='accuracy'):
        print(f"\nBuscando melhor valor de k para KNN com validação cruzada (métrica: {metrica})...")
        param_grid = {'n_neighbors': list(range(1, 6))}
        grid = model_selection.GridSearchCV(
            neighbors.KNeighborsClassifier(),
            param_grid,
            cv=5,
            scoring=metrica
        )
        grid.fit(X_train, y_train)
        print(f"Melhor k encontrado: {grid.best_params_['n_neighbors']}")
        print(f"Média na validação cruzada ({metrica}): {grid.best_score_:.4f}")
        test_acc = grid.best_estimator_.score(X_test, y_test)
        print(f"{metrica.capitalize()} no conjunto de teste: {test_acc:.4f}")
        return grid

class ArvoreDecisaoModelo(Modelo):
    def train(self, *, X_train, y_train, X_test, y_test, metrica='accuracy'):
        print(f"\nBuscando melhores hiperparâmetros para árvore de decisão com validação cruzada (métrica: {metrica})...")
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
        print(f"Melhores parâmetros encontrados: {grid.best_params_}")
        print(f"Média na validação cruzada ({metrica}): {grid.best_score_:.4f}")
        test_acc = grid.best_estimator_.score(X_test, y_test)
        print(f"{metrica.capitalize()} no conjunto de teste: {test_acc:.4f}")
        return grid

def main():
    print("Importando dataset...")
    data = pd.read_csv("data/wdbc.data", header=None)
    analisar_dados(data)
    
    # Colunas de atributos
    X = data.iloc[:, 2:].values

    # Coluna da classe
    y = data.iloc[:, 1].values
    y = (y == 'M').astype(int) # Convertendo numérica (binário)

    # Divisão estratificada; 70% treino, 30% teste
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"Tamanho treino: {X_train.shape[0]}")
    print(f"Tamanho teste: {X_test.shape[0]}")

    metricas = ['accuracy', 'f1', 'recall', 'precision']
    modelos = [
        ("KNN", KNNModelo()),
        ("Árvore de Decisão", ArvoreDecisaoModelo())
    ]

    proporcoes = [(treino/100, 1-treino/100) for treino in range(5, 100, 5)]
    resultados_path = "resultados_grid.json"
    if os.path.exists(resultados_path):
        print("[INFO] Resultados já existem. Pulando treinamento e carregando do JSON...")
        with open(resultados_path, "r") as f:
            resultados = json.load(f)
    else:
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
            for nome_modelo, modelo in modelos:
                print(f"\n==== {nome_modelo} ====")
                if nome_modelo not in resultados:
                    resultados[nome_modelo] = {}
                if f"{int(prop_treino*100)}%/{int(prop_teste*100)}%" not in resultados[nome_modelo]:
                    resultados[nome_modelo][f"{int(prop_treino*100)}%/{int(prop_teste*100)}%"] = {}
                for metrica in metricas:
                    print(f"\n--- Avaliando métrica: {metrica} ---")
                    params = {
                        'X_train': X_train,
                        'y_train': y_train,
                        'X_test': X_test,
                        'y_test': y_test,
                        'metrica': metrica
                    }
                    grid = modelo.train(**params)
                    if nome_modelo == "KNN":
                        melhor = grid.best_params_['n_neighbors']
                    else:
                        melhor = grid.best_params_
                    resultados[nome_modelo][f"{int(prop_treino*100)}%/{int(prop_teste*100)}%"].setdefault(metrica, {})
                    resultados[nome_modelo][f"{int(prop_treino*100)}%/{int(prop_teste*100)}%"] [metrica] = {
                        'melhor_param': melhor,
                        'score_cv': grid.best_score_,
                        'score_teste': grid.best_estimator_.score(X_test, y_test)
                    }
        with open(resultados_path, "w") as f:
            json.dump(resultados, f, indent=2)

    print("\nResumo dos resultados:")
    for nome_modelo, res_mod in resultados.items():
        print(f"\nModelo: {nome_modelo}")
        for proporcao, res_prop in res_mod.items():
            print(f"  Proporção treino/teste: {proporcao}")
            for metrica, res in res_prop.items():
                print(f"    {metrica.capitalize()}: melhor_param={res['melhor_param']}, "
                      f"CV={res['score_cv']:.4f}, Teste={res['score_teste']:.4f}")

if __name__ == "__main__":
    main()