# Main game

import tkinter as tk
from PIL import ImageTk,Image
import PIL
import appleCalculator_game

def startGame():
    newWindow = tk.Toplevel(window)
    newWindow.title("New page")
    newWindow.geometry("700x400")

    img = Image.open('Part5/apple.jpg')
    image2=img.resize((100,50),Image.ANTIALIAS)
    resize_img=ImageTk.PhotoImage(image2)
    label = tk.Label(newWindow, image=resize_img)
    label.pack() 

    newWindow.mainloop()

window = tk.Tk()

# 텍스트 레이블
widget1 = tk.Label(
    window,
    text="Apple Calculator",
    fg="white",
    bg="#34A2FE",
    width=40,
    height=5
)
widget1.pack()

widget2 = tk.Label(
    window,
    text="게임을 진행하시려면 Start 버튼을 눌러주세요",
    fg="white",
    bg="#34A2FE",
    width=40,
    height=5
)
widget2.pack()

button1 = tk.Button(window, text="Start", command = startGame)

button1.pack()

window.mainloop()