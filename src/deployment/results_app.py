import tkinter as tk
import pandas as pd
from tkinter import ttk

class ResultsApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Game Results")
        self.geometry("1200x600")

        self.create_widgets()
        self.load_data()

    def create_widgets(self):
        self.tree = ttk.Treeview(self, columns=("Console", "Game", "Comments Metric", "Posts Metric", "Slope", "Percent Change", "Combined Metric"), show="headings")
        self.tree.heading("Console", text="Console")
        self.tree.heading("Game", text="Game")
        self.tree.heading("Comments Metric", text="Comments Metric")
        self.tree.heading("Posts Metric", text="Posts Metric")
        self.tree.heading("Slope", text="Slope")
        self.tree.heading("Percent Change", text="Percent Change")
        self.tree.heading("Combined Metric", text="Combined Metric")

        self.tree.column("Console", width=120)
        self.tree.column("Game", width=300)
        self.tree.column("Comments Metric", width=100)
        self.tree.column("Posts Metric", width=100)
        self.tree.column("Slope", width=100)
        self.tree.column("Percent Change", width=100)
        self.tree.column("Combined Metric", width=120)

        self.tree.pack(expand=True, fill=tk.BOTH)

        self.sort_console_button = tk.Button(self, text="Sort by Console", command=self.sort_by_console)
        self.sort_console_button.pack()

    def load_data(self):
        file_path = "/home/akagi/Documents/Projects/retro-analytics/data/outputs/results.csv" 
        self.df = pd.read_csv(file_path)
        self.df.sort_values("combined_metric", ascending=False, inplace=True)
        self.populate_tree()

    def populate_tree(self):
        self.tree.delete(*self.tree.get_children())  # Clear the treeview
        for index, row in self.df.iterrows():
            self.tree.insert("", tk.END, values=(row["console"], row["game"], row["comments metric"], row["posts metric"], row["slope"], row["percent_change"], row["combined_metric"]))

    def sort_by_console(self):
        self.df.sort_values(["console", "combined_metric"], ascending=(True, False), inplace=True)
        self.populate_tree()

if __name__ == "__main__":
    app = ResultsApp()
    app.mainloop()
