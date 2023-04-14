import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QTableView, QVBoxLayout, QWidget, QHeaderView, QComboBox, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt, QSortFilterProxyModel
from PyQt5.QtGui import QStandardItemModel, QStandardItem

class DataDisplayApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Data Display")
        self.resize(1200, 600)

        layout = QVBoxLayout()

        data = pd.read_csv("/home/akagi/Documents/Projects/retro-analytics/data/outputs/results.csv")
        model = self.create_model(data)

        self.proxy_model = QSortFilterProxyModel()
        self.proxy_model.setSourceModel(model)
        self.proxy_model.setFilterKeyColumn(-1)  # Filter all columns

        table_view = QTableView()
        table_view.setModel(self.proxy_model)
        table_view.setSortingEnabled(True)
        table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        filter_label = QLabel("Filter by console:")
        self.filter_box = QComboBox()
        self.filter_box.addItem("All")
        self.filter_box.addItems(data["console"].unique())
        self.filter_box.currentTextChanged.connect(self.filter_changed)

        filter_layout = QHBoxLayout()
        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.filter_box)
        filter_layout.addStretch()

        layout.addLayout(filter_layout)
        layout.addWidget(table_view)
        self.setLayout(layout)

    def create_model(self, data):
        model = QStandardItemModel()

        model.setHorizontalHeaderLabels(
            ["Console", "Game", "Comments Metric", "Posts Metric", "Slope", "Percent Change", "Combined Metric"]
        )

        for index, row in data.iterrows():
            row_items = [
                QStandardItem(row["console"]),
                QStandardItem(row["game"]),
                QStandardItem(),
                QStandardItem(),
                QStandardItem(),
                QStandardItem(),
                QStandardItem(),
            ]

            for i, value in enumerate(row[2:], start=2):
                row_items[i].setData(value, Qt.DisplayRole)

            model.appendRow(row_items)

        return model

    def filter_changed(self, text):
        if text == "All":
            self.proxy_model.setFilterRegExp('')
        else:
            self.proxy_model.setFilterFixedString(text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = DataDisplayApp()
    mainWin.show()
    sys.exit(app.exec_())
