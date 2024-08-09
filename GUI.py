import tkinter as tk
from tkinter import messagebox
import subprocess

def run_sign_language_recognition():
    try:
        subprocess.Popen(["python", "inference.py"])
    except FileNotFoundError:
        messagebox.showerror("Error", "Sign language recognition script not found!")

def run_gesture_mouse():
    try:
        subprocess.Popen(["python", "AiVirtualMouse.py"])
    except FileNotFoundError:
        messagebox.showerror("Error", "Gesture mouse script not found!")

# Create the main application window
root = tk.Tk()
root.title("Advanced Gesture Recognition")
root.geometry("640x480")  # Set window size to 640x480 pixels
root.configure(bg="#222")

# Welcome message
welcome_label = tk.Label(root, text="Welcome!", font=("manolo", 40, "bold"), fg="white", bg="#222")
welcome_label.pack(pady=20)

# Features information
features_label = tk.Label(root, text="Features:", font=("aeronaut", 20, "bold"), fg="white", bg="#222")
features_label.pack()

features_info = tk.Label(root, text="1. Sign Language Detection Using Machine Learning\n2. Gesture Controlled Mouse", font=("aeronaut", 16), fg="white", bg="#222")
features_info.pack(pady=10)

# Create a frame to contain the buttons
button_frame = tk.Frame(root, bg="#222")
button_frame.pack()

# Create a button for sign language recognition
sign_language_button = tk.Button(button_frame, text="Sign Language Recognition", command=run_sign_language_recognition, bg="#3b5998", fg="white", font=("Arial", 16, "bold"), padx=20, pady=10, bd=0, relief=tk.FLAT, activebackground="#4a6ea9", activeforeground="white")
sign_language_button.pack(side=tk.LEFT, padx=20, pady=20)

# Create a button for gesture mouse
gesture_mouse_button = tk.Button(button_frame, text="Gesture Mouse", command=run_gesture_mouse, bg="#3b5998", fg="white", font=("Arial", 16, "bold"), padx=20, pady=10, bd=0, relief=tk.FLAT, activebackground="#4a6ea9", activeforeground="white")
gesture_mouse_button.pack(side=tk.LEFT, padx=20, pady=20)

# Start the Tkinter event loop
root.mainloop()


# 