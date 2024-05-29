import tkinter as tk
from tkinter import ttk, messagebox

import convergence
import performance


def check_convergence():

    ssize = int(convergence_sample_size.get())
    bsize = int(convergence_batch_size.get())
    epo = int(convergence_epoch.get())
    rseed = int(convergence_random_seed.get())

    convergence.exp_convergence_sequential_minibatch(rseed, ssize, epo, bsize)


def check_performance(selection):

    ssize = int(performance_sample_size.get())
    bsize = int(performance_batch_size.get())
    epo = int(performance_epoch.get())
    rseed = int(performance_random_seed.get())

    if selection == "Single-processing BGD vs. Single-processing MGD":
        performance.exp_performance_sequential_minibatch(rseed, ssize, epo, bsize)
    elif selection == "Single-processing BGD vs. Multi-processing MGD":
        performance.exp_performance_sequential_minibatchOnGPU(rseed, ssize, epo, bsize)
    elif selection == "Single-processing MGD vs. Multi-processing MGD":
        performance.exp_performance_minibatch_minibatchOnGPU(rseed, ssize, epo, bsize)
    elif selection == "All":
        performance.exp_performance_all(rseed, ssize, epo, bsize)


def validate_sample_size(value):
    if value == "" or value.isdigit():
        return True
    else:
        return False

def validate_batch_size(value):
    if value == "" or value.isdigit():
        return True
    else:
        return False

def validate_epoch(value):
    if value == "" or value.isdigit():
        return True
    else:
        return False

def validate_random_seed(value):
    if value == "" or value.isdigit():
        return True
    else:
        return False


root = tk.Tk()
root.title("Trainer")
root.geometry("400x300")

# Create a notebook (tabbed window)
notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True)


# First tab
tab1 = ttk.Frame(notebook)
notebook.add(tab1, text='Convergence')

ttk.Label(tab1, text="Convergence of BGD vs. MGD").pack(pady=10)

# Form for tab 1
ttk.Label(tab1, text="Sample size (500-100000):").pack()
convergence_sample_size = ttk.Entry(tab1, width=22)
convergence_sample_size.pack()
validate_sample_size_tab1 = tab1.register(validate_sample_size)
convergence_sample_size.config(validate="key", validatecommand=(validate_sample_size_tab1, '%P'))

ttk.Label(tab1, text="Batch size (32-500):           ").pack()
convergence_batch_size = ttk.Entry(tab1, width=22)
convergence_batch_size.pack()
validate_batch_size_tab1 = convergence_batch_size.register(validate_batch_size)
convergence_batch_size.config(validate="key", validatecommand=(validate_sample_size_tab1, '%P'))

ttk.Label(tab1, text="Epoch (1-1000):                 ").pack()
convergence_epoch = ttk.Entry(tab1, width=22)
convergence_epoch.pack()
validate_epoch_tab1 = tab1.register(validate_epoch)
convergence_epoch.config(validate="key", validatecommand=(validate_epoch_tab1, '%P'))

ttk.Label(tab1, text="Random seed (0-9):          ").pack()
convergence_random_seed = ttk.Entry(tab1, width=22)
convergence_random_seed.pack()
validate_random_seed_tab1 = tab1.register(validate_random_seed)
convergence_random_seed.config(validate="key", validatecommand=(validate_random_seed_tab1, '%P'))

def submit_form_tab1():
    ssize = convergence_sample_size.get()
    bsize = convergence_batch_size.get()
    epo = convergence_epoch.get()
    rseed = convergence_random_seed.get()
    if ((ssize.strip() and int(ssize) in range(500, 100001)) and (bsize.strip() and int(bsize) in range(32, 501))
            and (epo.strip() and int(epo) in range(1, 1001)) and (rseed.strip() and int(rseed) in range(0, 10))):
        button_tab1.config(state="disabled", text="Training")
        tab1.update_idletasks()
        check_convergence()
        button_tab1.config(state="normal", text="Train")
    else:
        messagebox.showinfo("Message", message="Invalid input!")


button_tab1 = ttk.Button(tab1, text="Train", command=submit_form_tab1)
button_tab1.pack(pady=20)


# Second tab
tab2 = ttk.Frame(notebook)
notebook.add(tab2, text='Performance')
#ttk.Label(tab2).pack()

# Dropdown list for tab 2
options_tab2 = ["Single-processing BGD vs. Single-processing MGD", "Single-processing BGD vs. Multi-processing MGD",
                "Single-processing MGD vs. Multi-processing MGD", "All"]
selected_option_tab2 = tk.StringVar(root)
selected_option_tab2.set(options_tab2[0])  # Default value
dropdown_tab2 = ttk.OptionMenu(tab2, selected_option_tab2, *options_tab2)
dropdown_tab2.pack(pady=10)

# Form for tab 2
ttk.Label(tab2, text="Sample size (500-100000):").pack()
performance_sample_size = ttk.Entry(tab2, width=22)
performance_sample_size.pack()
validate_sample_size_tab2 = tab2.register(validate_sample_size)
performance_sample_size.config(validate="key", validatecommand=(validate_sample_size_tab2, '%P'))

ttk.Label(tab2, text="Batch size (32-500):           ").pack()
performance_batch_size = ttk.Entry(tab2, width=22)
performance_batch_size.pack()
validate_batch_size_tab2 = tab2.register(validate_batch_size)
performance_batch_size.config(validate="key", validatecommand=(validate_batch_size_tab2, '%P'))

ttk.Label(tab2, text="Epoch (1-1000):                 ").pack()
performance_epoch = ttk.Entry(tab2, width=22)
performance_epoch.pack()
validate_epoch_tab2 = tab2.register(validate_epoch)
performance_epoch.config(validate="key", validatecommand=(validate_epoch_tab2, '%P'))

ttk.Label(tab2, text="Random seed (0-9):          ").pack()
performance_random_seed = ttk.Entry(tab2, width=22)
performance_random_seed.pack()
validate_random_seed_tab2 = tab2.register(validate_random_seed)
performance_random_seed.config(validate="key", validatecommand=(validate_random_seed_tab2, '%P'))


def submit_form_tab2():
    ssize = performance_sample_size.get()
    bsize = performance_batch_size.get()
    epo = performance_epoch.get()
    rseed = performance_random_seed.get()
    if ((ssize.strip() and int(ssize) in range(500, 100001)) and (bsize.strip() and int(bsize) in range(32, 501))
            and (epo.strip() and int(epo) in range(1, 1001)) and (rseed.strip() and int(rseed) in range(0, 10))):
        button_tab2.config(state="disabled", text="Training")
        tab2.update_idletasks()
        check_performance(selected_option_tab2.get())
        button_tab2.config(state="normal", text="Train")
    else:
        messagebox.showinfo("Message", message="Invalid input!")

button_tab2 = ttk.Button(tab2, text="Train", command=submit_form_tab2)
button_tab2.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()