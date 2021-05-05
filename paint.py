from tkinter import *

class Paint:
    def __init__(self, onchange):
        root = Tk()
        root.geometry("280x280")
        self.w = Canvas(root)
        self.w.pack(fill=BOTH)
        self.w.bind('<B1-Motion>', self.paint)
        self.w.bind('<ButtonPress-1>', self.downpen)
        self.w.bind('<ButtonRelease-1>', onchange)
        self.w.bind('<Button-2>', self.clear)
        root.mainloop()

    def paint(self, event):
        x, y = event.x, event.y
        self.w.create_line((self.lastx, self.lasty, x, y), fill='black', width=5)
        self.lastx, self.lasty = event.x, event.y

    def downpen(self, event):
        self.lastx, self.lasty = event.x, event.y

    def clear(self, event):
        self.w.delete("all")
