import sys
from tkinter import *
from tkinter import filedialog
root = Tk("Text Editor")
text = Text(root)
text.grid()
def save():
    global text
    t = text.get("1.0", "end-1c")
    loc = filedialog.asksaveasfilename()
    f1 = open(loc, "w+")
    f1.write(t)
    f1.close()
def fop():
    f = filedialog.askopenfilename()
    text.insert(0, f)
b = Button(root, text="Save", command=save) 
bb = Button(root, text="Open", command=fop) 
b.grid()
bb.grid()
root.mainloop()