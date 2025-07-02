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
import numpy as np
import os
import json

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

class VisualizadorInterativo:
    def __init__(self, resultados):
        self.resultados = resultados
        self.fig, self.ax = plt.subplots(figsize=(14, 9))
        plt.subplots_adjust(bottom=0.3, right=0.85)

        # Geração automática de cores para qualquer número de modelos
        modelo_labels = list(self.resultados.keys())
        cmap = plt.get_cmap('tab10')
        self.cores = {modelo: cmap(i % 10) for i, modelo in enumerate(modelo_labels)}

        # Extrair proporções de treino
        primeiro_modelo = modelo_labels[0]
        self.proporcoes = list(self.resultados[primeiro_modelo].keys())
        self.x_values = [int(prop.split('%')[0]) for prop in self.proporcoes]

        # Lista de métricas disponíveis (detecta automaticamente do primeiro modelo)
        primeira_prop = self.proporcoes[0]
        self.metricas = list(self.resultados[primeiro_modelo][primeira_prop].keys())
        self.metrica_atual = self.metricas[0] if self.metricas else 'accuracy'

        # Criar dropdown real para métricas
        ax_dropdown = plt.axes([0.15, 0.12, 0.15, 0.08])
        self.dropdown_metricas = mwidgets.RadioButtons(ax_dropdown, self.metricas)
        self.dropdown_metricas.on_clicked(self.mudar_metrica)

        # Criar checkboxes para modelos (dinâmico)
        ax_check_modelos = plt.axes([0.4, 0.12, 0.25, 0.08])
        self.check_modelos = mwidgets.CheckButtons(ax_check_modelos, modelo_labels, [True] * len(modelo_labels))
        self.check_modelos.on_clicked(self.atualizar_grafico)

        # Tentar ajustar tamanho dos checkboxes de modelos (se os atributos existirem)
        try:
            for rect in self.check_modelos.rectangles:
                rect.set_width(0.15)
                rect.set_height(0.15)
            for line in self.check_modelos.lines:
                for l in line:
                    l.set_linewidth(3)
        except (AttributeError, TypeError):
            pass

        # Criar checkboxes para Treino/Teste
        ax_check_tipo = plt.axes([0.75, 0.12, 0.2, 0.08])
        self.check_tipo = mwidgets.CheckButtons(ax_check_tipo, ['Treino', 'Teste'], [True, True])
        self.check_tipo.on_clicked(self.atualizar_grafico)

        try:
            for rect in self.check_tipo.rectangles:
                rect.set_width(0.15)
                rect.set_height(0.15)
            for line in self.check_tipo.lines:
                for l in line:
                    l.set_linewidth(3)
        except (AttributeError, TypeError):
            pass

        # Adicionar labels para cada seção
        self.fig.text(0.15, 0.22, 'Métricas:', fontsize=12, fontweight='bold')
        self.fig.text(0.4, 0.22, 'Modelos:', fontsize=12, fontweight='bold')
        self.fig.text(0.75, 0.22, 'Conjuntos:', fontsize=12, fontweight='bold')

        # Plotar gráfico inicial
        self.atualizar_grafico(None)
        plt.show()

    def mudar_metrica(self, label):
        self.metrica_atual = label
        self.atualizar_grafico(None)
        plt.draw()

    def atualizar_grafico(self, label):
        self.ax.clear()
        self.ax.set_title(f'Comparação de Modelos - Métrica: {self.metrica_atual.capitalize()}',
                         fontsize=16, fontweight='bold', pad=20)
        self.ax.set_xlabel('Proporção de Dados de Treino (%)', fontsize=12)
        self.ax.set_ylabel(f'Score ({self.metrica_atual.capitalize()})', fontsize=12)
        self.ax.grid(True, alpha=0.3)

        modelo_labels = list(self.resultados.keys())
        modelos_ativos = [modelo for i, modelo in enumerate(modelo_labels)
                          if self.check_modelos.get_status()[i]]

        tipos_ativos = []
        status_tipos = self.check_tipo.get_status()
        if status_tipos[0]:
            tipos_ativos.append('treino')
        if status_tipos[1]:
            tipos_ativos.append('teste')

        for modelo in modelos_ativos:
            scores_treino = []
            scores_teste = []
            for proporcao in self.proporcoes:
                metrica_dict = self.resultados[modelo][proporcao].get(self.metrica_atual, {})
                scores_treino.append(metrica_dict.get('score_treino', 0))
                scores_teste.append(metrica_dict.get('score_teste', 0))
            cor = self.cores.get(modelo, np.random.rand(3,))
            if 'treino' in tipos_ativos:
                self.ax.plot(self.x_values, scores_treino, 'o-', color=cor, linewidth=2.5,
                             markersize=7, label=f'{modelo} (Treino)', alpha=0.9)
            if 'teste' in tipos_ativos:
                self.ax.plot(self.x_values, scores_teste, 's--', color=cor, linewidth=2.5,
                             markersize=7, label=f'{modelo} (Teste)', alpha=0.7)

        self.ax.set_xlim(min(self.x_values)-2, max(self.x_values)+2)
        if modelos_ativos and tipos_ativos:
            todos_scores = []
            for modelo in modelos_ativos:
                for proporcao in self.proporcoes:
                    metrica_dict = self.resultados[modelo][proporcao].get(self.metrica_atual, {})
                    if 'treino' in tipos_ativos:
                        todos_scores.append(metrica_dict.get('score_treino', 0))
                    if 'teste' in tipos_ativos:
                        todos_scores.append(metrica_dict.get('score_teste', 0))
            if todos_scores:
                min_score = min(todos_scores)
                max_score = max(todos_scores)
                y_min = max(0.8, min_score - 0.02)
                y_max = min(1.0, max_score + 0.02)
                if y_max - y_min < 0.1:
                    centro = (y_min + y_max) / 2
                    y_min = max(0.0, centro - 0.05)
                    y_max = min(1.0, centro + 0.05)
                self.ax.set_ylim(y_min, y_max)
            else:
                self.ax.set_ylim(0.8, 1.0)
        else:
            self.ax.set_ylim(0.8, 1.0)
        self.ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        if modelos_ativos and tipos_ativos:
            self.adicionar_estatisticas(modelos_ativos, tipos_ativos)
        plt.draw()

    def adicionar_estatisticas(self, modelos_ativos, tipos_ativos):
        y_pos = 0.95
        for modelo in modelos_ativos:
            scores_treino = []
            scores_teste = []
            for proporcao in self.proporcoes:
                metrica_dict = self.resultados[modelo][proporcao].get(self.metrica_atual, {})
                scores_treino.append(metrica_dict.get('score_treino', 0))
                scores_teste.append(metrica_dict.get('score_teste', 0))
            texto_stats = f'{modelo}: '
            if 'treino' in tipos_ativos and scores_treino:
                media_treino = np.mean(scores_treino)
                std_treino = np.std(scores_treino)
                texto_stats += f'Treino(μ={media_treino:.3f}, σ={std_treino:.3f}) '
            if 'teste' in tipos_ativos and scores_teste:
                media_teste = np.mean(scores_teste)
                std_teste = np.std(scores_teste)
                texto_stats += f'Teste(μ={media_teste:.3f}, σ={std_teste:.3f})'
            if len(texto_stats) > len(f'{modelo}: '):
                self.ax.text(0.02, y_pos, texto_stats.strip(),
                             transform=self.ax.transAxes, fontsize=9,
                             bbox=dict(boxstyle="round,pad=0.3", facecolor=self.cores.get(modelo, 'gray'), alpha=0.3))
                y_pos -= 0.06

def print_resumo_resultados(resultados):
    print("\nResumo dos resultados:")
    for nome_modelo, res_mod in resultados.items():
        print(f"\nModelo: {nome_modelo}")
        for proporcao, res_prop in res_mod.items():
            print(f"  Proporção treino/teste: {proporcao}")
            for _, res in res_prop.items():
                print(f"Treino={res['score_treino']:.4f}, Teste={res['score_teste']:.4f}")

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

    # print_resumo_resultados(resultados)
                      
    
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