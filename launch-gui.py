import sys
from maxdiv.gui import MDIGUI

app = MDIGUI(sys.argv[1] if len(sys.argv) == 2 else None, False)
app.mainloop()
try:
    app.destroy()
except:
    pass