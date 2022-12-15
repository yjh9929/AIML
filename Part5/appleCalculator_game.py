import tkinter as tk
from PIL import ImageTk

window = tk.Tk()

# 텍스트 레이블
widget1 = tk.Label(
    window,
    text="게임을 진행해봅시다.",
    fg="white",
    bg="#34A2FE",
    width=40,
    height=5
)
widget1.pack()

# 이미지 레이블
img = ImageTk.PhotoImage(file="Part5/apple.jpg")
widget2 = tk.Label(window, image=img)
widget2.pack()

window.mainloop()