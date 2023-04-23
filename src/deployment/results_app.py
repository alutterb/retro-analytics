import sys
import pandas as pd
import json
import openai
import os

from PyQt5.QtWidgets import QApplication, QTableView, QVBoxLayout, QWidget, QHeaderView, QComboBox, QLabel, QHBoxLayout, QTextEdit, QPushButton
from PyQt5.QtCore import Qt, QSortFilterProxyModel
from PyQt5.QtGui import QStandardItemModel, QStandardItem

import json

def load_credentials(file_path):
    with open(file_path, 'r') as file:
        credentials = json.load(file)
    return credentials


script_dir = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(script_dir, '../../data/outputs/results.csv')
credentials_path = os.path.join(script_dir, '../../credentials/openai_credentials.json')

credentials = load_credentials(credentials_path)
openai.api_key = credentials['api_key']

def summarize_data(data):
    prompt = f"Please summarize the following data: {json.dumps(data)}"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = [
        {'role' : 'user', "content" : prompt}
        ]
    )

    summary = response['choices'][0]['message']['content']
    print(response)
    return summary

class DataDisplayApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Data Display")
        self.resize(1200, 600)

        layout = QVBoxLayout()

        data = pd.read_csv(data_path) 
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

        # Add GPT-4 summary textbox and button
        self.summary_textbox = QTextEdit()
        self.summary_textbox.setReadOnly(True)
        self.summary_textbox.setPlaceholderText("GPT-4 summary will appear here...")

        summary_button = QPushButton("Get GPT-4 Summary")
        summary_button.clicked.connect(self.get_gpt_summary)

        # Add widgets to the layout
        layout.addLayout(filter_layout)
        layout.addWidget(table_view)
        layout.addWidget(self.summary_textbox)
        layout.addWidget(summary_button)
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

    def get_gpt_summary(self):
        # Convert your data to a suitable format for GPT-4 summarization.
        # This example assumes that you want to summarize the entire dataset.
                # You can modify this as needed.
        data = {
            "data": self.proxy_model.sourceModel().rowCount(),
            "consoles": self.filter_box.count() - 1,
        }

        summary = summarize_data(data)
        self.summary_textbox.setPlainText(summary)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = DataDisplayApp()
    mainWin.show()
    sys.exit(app.exec_())

