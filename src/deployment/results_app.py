import tkinter as tk
from ttkthemes import ThemedTk
import pandas as pd

class ResultsApp(tk.Tk):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.filtered_data = data.copy()
        self.console_var = tk.StringVar(self)
        self.console_var.set("All Consoles")

        self.title("Game Recommendations")
        self.geometry("800x600")

        self.create_widgets()

    def create_widgets(self):
        console_label = tk.Label(self, text="Console:")
        console_label.pack(side="top")

        consoles = sorted(list(set(self.data["console"])))
        consoles.insert(0, "All Consoles")
        console_option_menu = tk.OptionMenu(self, self.console_var, *consoles, command=self.filter_by_console)
        console_option_menu.pack(side="top")

        search_label = tk.Label(self, text="Search:")
        search_label.pack(side="top")
        
        self.search_entry = tk.Entry(self)
        self.search_entry.pack(side="top")

        search_button = tk.Button(self, text="Search", command=self.search_game)
        search_button.pack(side="top")

        self.results_listbox = tk.Listbox(self, width=150, height=20)
        self.results_listbox.pack(side="top", fill="both", expand=True)

        self.update_listbox()

    def update_listbox(self):
        self.results_listbox.delete(0, tk.END)
        for _, row in self.filtered_data.iterrows():
            item = f"{row['console']} - {row['game']} - Combined metric: {row['combined_metric']:.2f}"
            self.results_listbox.insert(tk.END, item)

    def filter_by_console(self, console):
        if console == "All Consoles":
            self.filtered_data = self.data.copy()
        else:
            self.filtered_data = self.data[self.data['console'] == console]
        self.update_listbox()

    def search_game(self):
        search_term = self.search_entry.get().lower()
        if search_term:
            self.filtered_data = self.data[self.data['game'].str.lower().str.contains(search_term)]
            self.update_listbox()
        else:
            self.filtered_data = self.data.copy()
            self.update_listbox()

if __name__ == "__main__":
    data = pd.read_csv(r'/home/akagi/Documents/Projects/retro-analytics/data/outputs/results.csv')

    root = ThemedTk(theme="arc")
    app = ResultsApp(data)
    app.mainloop()
