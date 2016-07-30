from maxdiv.gui import MDIGUI

app = MDIGUI()
app.mainloop()
try:
    app.destroy()
except:
    pass