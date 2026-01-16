from tkinter import *
from tkinter import ttk

root = Tk()
window = Canvas(root, width = 400, height = 600)
window.pack()
canvas_height = 600
canvas_width = 400

Cord_label = window.create_text(200, 50, text="Corde Value", font=("Arial", 24))
Hitz_label = window.create_text(200, 325, text="Hitz Value", font=("Arial", 24))

window.create_line(0, 100, 400, 100, fill="black")
window.create_line(200, 150, 200, 500, fill="black", width=3)
window.create_arc(20, 150, 380, 230, start=180, extent=-180, fill="", style=ARC, width=2)
window.create_oval(190, 490, 210, 510, fill="black")


# frm = ttk.Frame (root, padding=10)
# frm.grid()
# ttk.Label(frm, text="Hitz Value").grid(column=0, row=0)

root.mainloop()
