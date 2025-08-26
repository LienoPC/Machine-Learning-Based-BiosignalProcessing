import tkinter as tk

class SliderWindow:
    def __init__(self,min_val: float = 0.0,max_val: float = 1.0,resolution: float = 0.01,update_interval: int = 100):
        self.root = tk.Tk()
        self.root.title("Random center distribution for mock predictions")

        self.shared_var = tk.DoubleVar(value=min_val)

        self.slider = tk.Scale(
            self.root,
            from_=min_val,
            to=max_val,
            resolution=resolution,
            orient=tk.HORIZONTAL,
            label="Select value:",
            length=300,
            variable=self.shared_var
        )
        self.slider.pack(padx=10, pady=10)

        self.value_label = tk.Label(
            self.root,
            textvariable=self.shared_var,
            font=("Arial", 14)
        )
        self.value_label.pack(pady=(0, 10))

        self.update_interval = update_interval


    def update_task(self):
        self.root.after(self.update_interval, self.update_task)

    def tick(self):
        self.root.update()

    def run(self):
        self.root.mainloop()