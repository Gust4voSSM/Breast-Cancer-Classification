from sklearn import (
    model_selection,
    tree,
    naive_bayes,
    linear_model,
    neighbors
)
import pandas as pd

def analisar_dados(data):
    print('\n' "Numero de exemplos: ", data.shape[0], '\n',
          '\n' "Por classe:" '\n',
          data[1].value_counts(),
          '\n' "//// HEAD ////" '\n',
          data.head())
    
def treinar_knn_com_cv(*, X_train, y_train, X_test, y_test, metrica='accuracy'):
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
    knn_best = grid.best_estimator_
    test_acc = knn_best.score(X_test, y_test)
    print(f"{metrica.capitalize()} no conjunto de teste: {test_acc:.4f}")
    return knn_best, grid


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
    resultados = {}

    for metrica in metricas:
        print(f"\n--- Avaliando métrica: {metrica} ---")
        params = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'metrica': metrica
        }
        knn_best, grid = treinar_knn_com_cv(**params)
        resultados[metrica] = {
            'melhor_k': grid.best_params_['n_neighbors'],
            'score_cv': grid.best_score_,
            'score_teste': knn_best.score(X_test, y_test)
        }

    print("\nResumo dos resultados:")
    for metrica, res in resultados.items():
        print(f"{metrica.capitalize()}: melhor_k={res['melhor_k']}, "
              f"CV={res['score_cv']:.4f}, Teste={res['score_teste']:.4f}")

if __name__ == "__main__":
    main()