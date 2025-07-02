import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import numpy as np

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
