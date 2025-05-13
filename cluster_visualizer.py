import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QPushButton, QSlider,
                             QLabel, QFileDialog, QWidget, QComboBox, QRadioButton, QHBoxLayout, QApplication)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import sys

class ClusterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Кластеризация данных и визуализация")
        self.setGeometry(200, 200, 800, 600)
        
        # Основные данные
        self.data = None
        self.n_clusters = 3  # Начальное количество кластеров
        self.field_selection = None  # Переменная для хранения выбранного поля
        self.visualization_type = 'heatmap'  # Тип визуализации
        self.coordinate_weight = 1.0  # Начальный вес для координат

        # Инициализация интерфейса
        self.initUI()

    def initUI(self):
        # Кнопка для загрузки файла
        load_button = QPushButton("Загрузить файл", self)
        load_button.clicked.connect(self.load_file)

        # Кнопка для визуализации
        visualize_button = QPushButton("Визуализировать данные", self)
        visualize_button.clicked.connect(self.visualize_data)

        # Выпадающий список для выбора поля
        self.field_combo = QComboBox(self)
        self.field_combo.addItems(["Поле 1", "Поле 2", "Поле 3", "Поле 4", 
                                   "Поле 5", "Поле 6", "Поле 7", "Поле 8"])
        self.field_combo.currentIndexChanged.connect(self.update_field_selection)

        # Слайдер для настройки количества кластеров
        self.slider = QSlider(self)
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setMinimum(2)
        self.slider.setMaximum(10)
        self.slider.setValue(self.n_clusters)
        self.slider.valueChanged.connect(self.update_clusters)

        # Слайдер для настройки веса координат
        self.coordinate_slider = QSlider(self)
        self.coordinate_slider.setOrientation(Qt.Horizontal)
        self.coordinate_slider.setMinimum(0)
        self.coordinate_slider.setMaximum(100)
        self.coordinate_slider.setValue(int(self.coordinate_weight * 50))  # Начальное значение
        self.coordinate_slider.valueChanged.connect(self.update_coordinate_weight)

        # Радиокнопки для выбора типа карты
        heatmap_radio = QRadioButton("Тепловая карта")
        heatmap_radio.setChecked(True)
        heatmap_radio.toggled.connect(lambda: self.set_visualization_type('heatmap'))
        contour_radio = QRadioButton("Контурная карта")
        contour_radio.toggled.connect(lambda: self.set_visualization_type('contour'))

        # Кнопка для кластеризации
        cluster_button = QPushButton("Кластеризовать данные", self)
        cluster_button.clicked.connect(self.cluster_data)

        # Кнопка для экспорта данных кластеризации
        export_button = QPushButton("Экспорт кластерных данных", self)
        export_button.clicked.connect(self.export_clusters)

        # Место для визуализации графиков
        self.canvas = FigureCanvas(plt.Figure())

        # Размещение элементов интерфейса
        layout = QVBoxLayout()
        layout.addWidget(load_button)
        layout.addWidget(self.field_combo)
        layout.addWidget(visualize_button)

        # Радиокнопки
        radio_layout = QHBoxLayout()
        radio_layout.addWidget(heatmap_radio)
        radio_layout.addWidget(contour_radio)
        layout.addLayout(radio_layout)

        # Настройка количества кластеров
        layout.addWidget(QLabel("Число кластеров"))
        layout.addWidget(self.slider)

        # Настройка веса координат
        layout.addWidget(QLabel("Вес координат"))
        layout.addWidget(self.coordinate_slider)

        layout.addWidget(cluster_button)
        layout.addWidget(export_button)
        layout.addWidget(self.canvas)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите текстовый файл", "", "Data Files (*.dat *.txt *.csv)")
        if file_path:
            try:
                self.data = pd.read_csv(file_path, delim_whitespace=True, header=None)
                print("Файл загружен:", file_path)
            except Exception as e:
                print("Ошибка при загрузке файла:", e)

    def update_field_selection(self):
        self.field_selection = self.field_combo.currentIndex() + 2

    def set_visualization_type(self, vtype):
        self.visualization_type = vtype

    def visualize_data(self):
        if self.data is not None and self.field_selection is not None:
            try:
                self.canvas.figure.clf()
                ax = self.canvas.figure.add_subplot(111)

                x = self.data.iloc[:, 0].values
                y = self.data.iloc[:, 1].values
                z = self.data.iloc[:, self.field_selection].values

                grid_x, grid_y = np.meshgrid(
                    np.linspace(x.min(), x.max(), 100),
                    np.linspace(y.min(), y.max(), 100)
                )
                grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

                if self.visualization_type == 'heatmap':
                    cax = ax.imshow(grid_z, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='hot')
                else:
                    cax = ax.contourf(grid_x, grid_y, grid_z, cmap='hot')

                self.canvas.figure.colorbar(cax)
                self.canvas.draw()
                print(f"Визуализация для {self.field_combo.currentText()} завершена.")
            except Exception as e:
                print("Ошибка при визуализации данных:", e)
        else:
            print("Нет данных для визуализации или не выбрано поле.")

    def update_clusters(self):
        self.n_clusters = self.slider.value()

    def update_coordinate_weight(self):
        self.coordinate_weight = self.coordinate_slider.value() / 50

    def cluster_data(self):
        if self.data is not None:
            try:
                # Масштабируем координаты и значения полей
                scaler = MinMaxScaler()
                coordinates = self.data.iloc[:, :2].values * self.coordinate_weight
                fields = self.data.iloc[:, 2:].values
                
                scaled_data = np.hstack((scaler.fit_transform(coordinates), scaler.fit_transform(fields)))

                kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
                self.data['cluster'] = kmeans.fit_predict(scaled_data)

                self.canvas.figure.clf()
                ax = self.canvas.figure.add_subplot(111)
                scatter = ax.scatter(self.data.iloc[:, 0], self.data.iloc[:, 1], c=self.data['cluster'], cmap='viridis')
                self.canvas.figure.colorbar(scatter, ax=ax)
                self.canvas.draw()
                
                print(f"Кластеризация по полям с {self.n_clusters} кластерами завершена.")
            except Exception as e:
                print("Ошибка при кластеризации данных:", e)
        else:
            print("Нет данных для кластеризации.")

    def export_clusters(self):
        if self.data is not None and 'cluster' in self.data.columns:
            file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить файл", "", "Text Files (*.txt)")
            if file_path:
                self.data[['cluster']].to_csv(file_path, index=False)
                print("Данные кластеризации успешно сохранены:", file_path)
        else:
            print("Нет данных для сохранения кластеров.")

# Запуск приложения
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ClusterApp()
    window.show()
    sys.exit(app.exec_())
