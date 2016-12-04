import platform, os, sys

# Check if OS is Windows
if platform.system() != 'Windows':
    print('This launcher is for Microsoft Windows systems only.')
    print('Try launch-gui.py in the parent directory or consult the installation guide included in that directory.')
    exit()

# Check architecture
bits = platform.architecture()[0]
if bits == '32bit':
    binary_dir = 'x86'
elif bits == '64bit':
    binary_dir = 'x64'
else:
    print('Unsupported architecture: {}'.format(bits))
    exit()

# Switch to adequate binary directory
os.chdir(os.path.join(os.path.dirname(__file__), binary_dir))

# Add root directory to Python search path
sys.path.append(os.path.join('..', '..'))

# Load libmaxdiv and GUI
from maxdiv.gui import MDIGUI

# Launch GUI
app = MDIGUI()
app.mainloop()
try:
    app.destroy()
except:
    pass