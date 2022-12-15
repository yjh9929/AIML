from tkinter import *
from PIL import ImageTk,Image
 
def hello():
    img = ImageTk.PhotoImage(Image.open('Part5/apple.jpg'))
    label = Label(image=img)
    label.pack() 
