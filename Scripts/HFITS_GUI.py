import sys
import os
import tkinter as tk
from tkinter import ttk
from Image_processing import IPA
from IHT import setup_second_tab

def main():
    root = tk.Tk()
    root.title("HFITS -- Heat Flux measurements using Infrared Thermography and a plate Sensor")
    root.geometry("800x1024")

    tabControl = ttk.Notebook(root)

    # First tab
    tab1 = ttk.Frame(tabControl)
    tabControl.add(tab1, text='Image Processing')
    IPA(tab1)

    # Second tab
    tab2 = ttk.Frame(tabControl)
    tabControl.add(tab2, text='Inverse Heat Transfer')
    setup_second_tab(tab2)

    tabControl.pack(expand=1, fill="both")
    root.mainloop()

if __name__ == "__main__":
    main()