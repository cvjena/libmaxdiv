# coding=utf-8

"""Graphical User Interface to the MDI algorithm.

Provides a user interface for loading multivariate time-series from CSV files,
setting the MDI parameters, running the algorithm and inspecting the detected intervals.

Only non-spatial data is supported at the moment.
"""

try:
    # Python 3
    import tkinter
    from tkinter import N, E, S, W
    from tkinter import ttk, messagebox, filedialog, simpledialog
except:
    # Python 2
    import Tkinter as tkinter
    from Tkinter import N, E, S, W
    import ttk
    import tkMessageBox as messagebox
    import tkFileDialog as filedialog
    import tkSimpleDialog as simpledialog

import math, re, datetime, csv, os
import numpy as np
import matplotlib
matplotlib.use('Agg')

try:
    from PIL import Image, ImageTk
except:
    import Image, ImageTk

from .maxdiv import maxdiv
from . import preproc, libmaxdiv_wrapper, wckToolTips



PADDING = 8

MIN_DET_ZOOM_SIZE = 2000

MODES = [
    ('TS', 'Kullback-Leibler (prefer larger intervals)'),
    ('I_OMEGA', 'Kullback-Leibler (prefer smaller intervals)'),
    ('CROSSENT_TS', 'Cross Entropy (prefer larger intervals)'),
    ('CROSSENT', 'Cross Entropy (prefer smaller intervals)'),
    ('JSD', 'Jensen-Shannon')
]

ESTIMATORS = [
    ('gaussian_cov', 'Gaussian (Full Covariance)'),
    ('gaussian_global_cov', 'Gaussian (Global Covariance)'),
    ('gaussian_id_cov', 'Gaussian (Identity Covariance)'),
    ('parzen', 'Kernel Density Estimation'),
    ('erph', 'Ensemble of Random Projection Histograms')
]

SEARCH_STRATEGIES = [
    ('dense', 'Full Scan (accurate)'),
    ('pointwise', 'Interval Proposals (fast)')
]

PROPOSAL_METHODS = [
    ('hotellings_t', u'Hotelling\'s T²'),
    ('kde', 'Pointwise Kernel Density Estimation')
]

DETRENDING_METHODS = [
    ('', 'None'),
    ('deseasonalize_z', 'Z-Score'),
    ('deseasonalize_ols', 'Least Squares'),
    ('deseasonalize_ft', 'Fourier Transform')
]



def readDataFromCSV(csvFile):
    
    # Detect delimiter and datelines
    with open(csvFile) as f:
        firstLine = f.readline().strip()
        secondLine = f.readline().strip()
    
    delims = [',', ';', '\t', ' ']
    delim = None
    for d in delims:
        if re.match('^"?\\s*([+-]?[0-9]+(\\.[0-9]*)?(e[+-]?[0-9]+)?|nan)\\s*"?(' + d + '"?\\s*([+-]?[0-9]+(\\.[0-9]*)?(e[+-]?[0-9]+)?|nan)\\s*"?)*$', secondLine):
            delim = d
            hasDatelines = False
            break
        elif re.match('^"?\\s*[0-9]{4}-[01][0-9]-[0-3][0-9] [012][0-9]:[0-5][0-9]:[0-5][0-9]\\s*"?(' + d + '"?\\s*([+-]?[0-9]+(\\.[0-9]*)?(e[+-]?[0-9]+)?|nan)\\s*"?)*$', secondLine):
            delim = d
            hasDatelines = 'datetime'
            break
        elif re.match('^"?\\s*[0-9]{4}-[01][0-9]-[0-3][0-9]\\s*"?(' + d + '"?\\s*([+-]?[0-9]+(\\.[0-9]*)?(e[+-]?[0-9]+)?|nan)\\s*"?)*$', secondLine):
            delim = d
            hasDatelines = 'date'
            break
    
    if delim is None:
        raise RuntimeError('File format is not understood.')
    
    # Read data
    varnames = []
    timesteps = []
    data = []
    with open(csvFile) as f:
        for i, line in enumerate(csv.reader(f, delimiter = delim)):
            try:
                if hasDatelines:
                    timesteps.append(datetime.datetime.strptime(line[0].strip(), '%Y-%m-%d' if hasDatelines == 'date' else '%Y-%m-%d %H:%M:%S'))
                    del line[0]
                data.append([float(x.strip()) for x in line])
            except ValueError:
                if i == 0: # read header information
                    varnames = [n.strip() for n in line]
                    if hasDatelines:
                        del varnames[0]
                else:
                    raise
    data = np.array(data).T
    
    # Check if first attribute is a continous sequence of time steps
    if not hasDatelines:
        diff = data[0, 1:] - data[0, :-1]
        if (diff > 0).all() and (np.abs(diff - diff[0]) / np.maximum(diff, diff[0]) < 1e-4).all():
            timesteps = data[0,:]
            data = data[1:,:]
            if len(varnames) > 0:
                del varnames[0]
        else:
            timesteps = np.arange(data.shape[1])
    
    # Check for missing values
    if np.isnan(data).any():
        data = np.ma.masked_invalid(data)
    
    # Create artificial variable names if no header was available
    if len(varnames) == 0:
        varnames = ['VAR_{}'.format(i) for i in range(data.shape[0])]
    
    return data, timesteps, varnames
    


def figure2img(fig, w, h):
    """Converts a matplotlib.figure.Figure to PIL.Image.Image instance.
    
    fig - The matplotlib figure.
    w - The width of the resulting image.
    h - The height of the resulting image.
    """
    
    dpi = fig.get_dpi()
    fig.set_size_inches(float(w) / dpi, float(h) / dpi)
    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf_argb = fig.canvas.tostring_argb()
    return Image.frombytes('RGBA', (w, h), buf_argb, 'raw', 'ARGB')



class MDIGUI(tkinter.Tk):
    """Main window of the GUI to the MDI algorithm."""
    
    
    def __init__(self, dataFile = None, selectSubset = True):
        """Initializes the main window of the application.
        
        If a filename is given as parameter, that file will be loaded immediately.
        Otherwise, the user will be asked to select a file on start-up.
        If a file is given and `selectSubset` is set to `False`, it will be loaded
        without asking the user to select a subset of the data.
        """
    
        tkinter.Tk.__init__(self)
        self.title("Maximally Divergent Intervals for Anomaly Detection (MDI)")
        
        # Properties
        self.varnames = []
        self.timesteps = []
        self.data = np.ndarray((0, 0))
        self.preprocData = np.ndarray((0, 0))
        self.detections = []
        self.fig = matplotlib.pyplot.figure(facecolor = [x / 65535.0 for x in self.winfo_rgb(self['background'])])
        
        # Tk variables
        self.strFile = tkinter.StringVar(self)
        self.strKernelSigmaSq = tkinter.StringVar(self, value = '1.0')
        self.strPropTh = tkinter.StringVar(self, value = '1.5')
        self.strMinLen = tkinter.StringVar(self, value = '20')
        self.strMaxLen = tkinter.StringVar(self, value = '100')
        self.strOverlapTh = tkinter.StringVar(self, value = '0')
        self.strFigureNavMode = tkinter.StringVar(self, value = 'pan')
        self.intNumHist = tkinter.IntVar(self, value = 100)
        self.intNumBins = tkinter.IntVar(self, value = 0)
        self.intTdDim = tkinter.IntVar(self, value = 3)
        self.intTdLag = tkinter.IntVar(self, value = 1)
        self.intPeriodNum = tkinter.IntVar(self, value = 24)
        self.intPeriodLen = tkinter.IntVar(self, value = 1)
        self.intNumDetections = tkinter.IntVar(self, value = 10)
        self.boolNormalize = tkinter.BooleanVar(self, value = True)
        self.boolLinearTrend = tkinter.BooleanVar(self, value = True)
        self.boolPreviewPreproc = tkinter.BooleanVar(self, value = False)
        self.boolPropGrad = tkinter.BooleanVar(self, value = True)
        self.boolPropMAD = tkinter.BooleanVar(self, value = False)
        self.boolExpandPlot = tkinter.BooleanVar(self, value = False)
        
        self.strMinLen.trace('w', self._updateScaleValues)
        self.strMaxLen.trace('w', self._updateScaleValues)
        self.strOverlapTh.trace('w', self._updateScaleValues)
        self.strFigureNavMode.trace('w', self._changeNavMode)
        self.intPeriodNum.trace('w', self._onPreprocChange)
        self.intPeriodLen.trace('w', self._onPreprocChange)
        self.boolNormalize.trace('w', self._onPreprocChange)
        self.boolLinearTrend.trace('w', self._onPreprocChange)
        self.boolPreviewPreproc.trace('w', self._onPreviewPreprocChange)
        self.boolExpandPlot.trace('w', self._onExpandPlotChange)
        
        # Private variables
        self._destroyed = False
        self._resizeAfterId = None
        self._preprocAfterId = None
        self._lastPreprocParams = None
        self._detectionLabels = []
        self._lastFigSize = (0, 0)
        self._shownDetection = -1
        self._lastMousePosition = None
        
        # Configure ttk widgets to have the same background color as the window
        style = ttk.Style(self)
        style.configure('TButton', background = style.lookup('.','background'))
        style.configure('.', background = self['background'])
        
        # Create interface
        self.createWidgets()
        self.updatePlot()
        self._changeNavMode()
        self.bind('<Configure>', self._onResize)
        self.bind('<Destroy>', self._onDestroy, True)
        
        self.update()
        self._checkLibMaxDiv()
        if dataFile:
            self.loadData(dataFile, selectSubset = selectSubset)
        else:
            self.selectDataFile()
    
    
    def _onResize(self, evt):
        """Callback for re-drawing the plot whenever the window is resized."""
        
        if not self._resizeAfterId is None:
            self.after_cancel(self._resizeAfterId)
        if not self._destroyed:
            self._resizeAfterId = self.after(100, lambda: self.redrawPlot(False))
    
    
    def _onDestroy(self, evt):
        if (evt.widget is self):
            self._destroyed = True
            try:
                # Break reference cycles of TCL variables, because their
                # __del__ method prevents the garbage collector from freeing them:
                del self.strFile
                del self.strKernelSigmaSq
                del self.strPropTh
                del self.strMinLen
                del self.strMaxLen
                del self.strOverlapTh
                del self.strFigureNavMode
                del self.intNumHist
                del self.intNumBins
                del self.intTdDim
                del self.intTdLag
                del self.intPeriodNum
                del self.intPeriodLen
                del self.intNumDetections
                del self.boolNormalize
                del self.boolLinearTrend
                del self.boolPreviewPreproc
                del self.boolPropGrad
                del self.boolPropMAD
                del self.boolExpandPlot
            except:
                pass
    
    
    def createWidgets(self):
        
        # Data selection
        self.frmDataSel = ttk.Frame(self)
        self.lblDataFilePre = ttk.Label(self.frmDataSel, text = 'Loaded file:')
        self.lblDataFile = ttk.Label(self.frmDataSel, textvariable = self.strFile)
        self.btnSelectFile = ttk.Button(self.frmDataSel, text = '...', width = 3, command = self.selectDataFile)
        self.btnHelpDataFormat = ttk.Button(self.frmDataSel, text = '?', width = 2, command = self.showDataFormatHelp)
        self.lblDataFilePre.grid(row = 0, column = 0, padx = PADDING)
        self.lblDataFile.grid(row = 0, column = 1, sticky = (W, E))
        self.btnSelectFile.grid(row = 0, column = 2)
        self.btnHelpDataFormat.grid(row = 0, column = 3, padx = (0, PADDING))
        self.frmDataSel.columnconfigure(1, weight = 1)
        self.frmDataSel.grid(row = 0, column = 0, columnspan = 2, sticky = (W,E), pady = (PADDING, 0))
        
        # Visualization
        self.frmTS = ttk.Frame(self)
        self.lblTS = tkinter.Label(self.frmTS, borderwidth = 0, highlightthickness = 0)
        self.btnPan = ttk.Radiobutton(self.frmTS, text = 'P', variable = self.strFigureNavMode, value = 'pan')
        self.btnZoom = ttk.Radiobutton(self.frmTS, text = 'Z', variable = self.strFigureNavMode, value = 'zoom')
        self.btnReset = ttk.Button(self.frmTS, text = 'R', command = self.resetView)
        self.btnExpand = ttk.Checkbutton(self.frmTS, text = 'X', variable = self.boolExpandPlot)
        self.btnFirstDet = ttk.Button(self.frmTS, text = '1', command = lambda *args: self.goToDetection(0))
        self.btnNextDet = ttk.Button(self.frmTS, text = '->', command = self.goToNextDetection)
        self.btnPrevDet = ttk.Button(self.frmTS, text = '<-', command = self.goToPrevDetection)
        self.btnSaveFig = ttk.Button(self.frmTS, text = 'S', command = self.saveFigDlg)
        self.btnExportDetections = ttk.Button(self.frmTS, text = 'E', command = self.exportDetectionsDlg)
        self.lblTS.grid(row = 0, column = 0, rowspan = 6, sticky = (N,S,W,E))
        self.btnsVis = [[self.btnPan, self.btnZoom, self.btnReset, self.btnExpand, None, self.btnSaveFig],
                        [self.btnFirstDet, self.btnNextDet, self.btnPrevDet, None, None, self.btnExportDetections]]
        btnIcons = [['move.png', 'zoom-in.png', 'reload.png', 'expand.png', None, 'camera.png'],
                    ['medal.png', 'arrow-right.png', 'arrow-left.png', None, None, 'save.png']]
        btnHints = [['Pan figure', 'Zoom into figure with left click and out with right click', 'Reset figure', 'Expand figure and show variables in separate axes.', None, 'Save figure'],
                    ['Zoom in to first detection', 'Zoom in to next detection', 'Zoom in to previous detection', None, None, 'Export detections as CSV file']]
        for c, btns in enumerate(self.btnsVis):
            for r, btn in enumerate(btns):
                if btn is not None:
                    btn['style'] = 'Toolbutton'
                    btn['state'] = 'disabled'
                    try:
                        img = Image.open(os.path.join(os.path.dirname(__file__), '..', 'gfx', btnIcons[c][r]))
                        btn._img = ImageTk.PhotoImage(img)
                        btn['image'] = btn._img
                    except:
                        pass
                    wckToolTips.register(btn, btnHints[c][r])
                    btn.grid(row = r, column = c + 1, sticky = (N,S,W,E), padx = 0,
                             pady = (PADDING if r == 0 else 0, 30 if r == len(btns) - 1 else 0))
        self.frmTS.columnconfigure(0, weight = 1)
        self.frmTS.rowconfigure(4, weight = 1)
        self.frmTS.grid(row = 1, column = 0, columnspan = 2, sticky = (N,S,W,E), padx = PADDING)
        
        # Two-column layout for settings
        self.frmLeftCol = ttk.Frame(self)
        self.frmRightCol = ttk.Frame(self)
        self.frmLeftCol.grid(row = 2, column = 0, sticky = (N,W,E))
        self.frmRightCol.grid(row = 2, column = 1, sticky = (N,W,E))
        
        # General parameters
        self.frmGeneralParams = ttk.Labelframe(self.frmLeftCol, text = 'General Settings')
        self.lblDivergence = ttk.Label(self.frmGeneralParams, text = 'Divergence:')
        self.selDivergence = ttk.Combobox(self.frmGeneralParams, state = 'readonly', values = tuple(m[1] for m in MODES))
        self.selDivergence.current(0)
        self.lblEstimator = ttk.Label(self.frmGeneralParams, text = 'Distribution Model:')
        self.selEstimator = ttk.Combobox(self.frmGeneralParams, state = 'readonly', values = tuple(e[1] for e in ESTIMATORS))
        self.selEstimator.bind('<<ComboboxSelected>>', self._onEstimatorSelect)
        self.selEstimator.current(0)
        self.lblKernelSigmaSq = ttk.Label(self.frmGeneralParams, text = 'Kernel Variance:')
        self.txtKernelSigmaSq = tkinter.Spinbox(self.frmGeneralParams, textvariable = self.strKernelSigmaSq, width = 10, from_ = 0.001, to = 1000.0, increment = 0.1)
        self.lblNumHist = ttk.Label(self.frmGeneralParams, text = 'Histograms:')
        self.txtNumHist = tkinter.Spinbox(self.frmGeneralParams, textvariable = self.intNumHist, width = 10, from_ = 1, to = 1000, increment = 1)
        self.lblNumBins = ttk.Label(self.frmGeneralParams, text = 'Bins (0 = auto):')
        self.txtNumBins = tkinter.Spinbox(self.frmGeneralParams, textvariable = self.intNumBins, width = 10, from_ = 0, to = 1000, increment = 1)
        self.widgetsGeneralParams = [
            (self.lblDivergence, self.selDivergence),
            (self.lblEstimator, self.selEstimator),
            (self.lblKernelSigmaSq, self.txtKernelSigmaSq),
            (self.lblNumHist, self.txtNumHist),
            (self.lblNumBins, self.txtNumBins)
        ]
        for r, (label, widget) in enumerate(self.widgetsGeneralParams):
            label.grid(row = r, column = 0, sticky = W, padx = PADDING, pady = PADDING)
            widget.grid(row = r, column = 1, sticky = (W, E), padx = PADDING, pady = PADDING)
        self.frmGeneralParams.columnconfigure(1, weight = 1)
        self.frmGeneralParams.pack(side = 'top', fill = 'x', padx = PADDING, pady = (0, PADDING))
        
        # Pre-processing
        self.frmPreproc = ttk.Labelframe(self.frmLeftCol, text = 'Pre-processing')
        self.chkNormalize = ttk.Checkbutton(self.frmPreproc, text = 'Normalize data', variable = self.boolNormalize)
        self.lblTdDim = ttk.Label(self.frmPreproc, text = 'Time-Delay Embedding Dimension:')
        self.txtTdDim = tkinter.Spinbox(self.frmPreproc, textvariable = self.intTdDim, width = 5, from_ = 1, to = 50, increment = 1)
        self.lblTdLag = ttk.Label(self.frmPreproc, text = 'Time-Delay Lag:')
        self.txtTdLag = tkinter.Spinbox(self.frmPreproc, textvariable = self.intTdLag, width = 5, from_ = 1, to = 1000, increment = 1)
        self.btnTdAuto = ttk.Button(self.frmPreproc, text = 'auto', command = self.optimizeTdEmbedding)
        self.lblDetrending = ttk.Label(self.frmPreproc, text = 'Deseasonalization:')
        self.selDetrending = ttk.Combobox(self.frmPreproc, state = 'readonly', values = tuple(m[1] for m in DETRENDING_METHODS))
        self.selDetrending.bind('<<ComboboxSelected>>', self._onDetrendingSelect)
        self.selDetrending.current(0)
        self.lblPeriodNum = ttk.Label(self.frmPreproc, text = 'Number of seasons:')
        self.txtPeriodNum = tkinter.Spinbox(self.frmPreproc, textvariable = self.intPeriodNum, width = 6, from_ = 1, to = 10000, increment = 1)
        self.lblPeriodLen = ttk.Label(self.frmPreproc, text = 'Length of each season:')
        self.txtPeriodLen = tkinter.Spinbox(self.frmPreproc, textvariable = self.intPeriodLen, width = 6, from_ = 1, to = 10000, increment = 1)
        self.btnPeriodAuto = ttk.Button(self.frmPreproc, text = 'auto', command = self.detectSeasonality)
        self.chkLinearTrend = ttk.Checkbutton(self.frmPreproc, text = 'Remove global linear trend', variable = self.boolLinearTrend)
        self.chkPreviewPreproc = ttk.Checkbutton(self.frmPreproc, text = 'Plot pre-processed instead of original data', variable = self.boolPreviewPreproc)
        self.chkNormalize.grid(row = 0, column = 0, columnspan = 3, sticky = W, padx = PADDING, pady = PADDING)
        self.widgetsPreproc = [
            (self.lblTdDim, self.txtTdDim),
            (self.lblTdLag, self.txtTdLag),
            (self.lblDetrending, self.selDetrending),
            (self.lblPeriodNum, self.txtPeriodNum),
            (self.lblPeriodLen, self.txtPeriodLen),
        ]
        for r, (label, widget) in enumerate(self.widgetsPreproc):
            label.grid(row = r + 1, column = 0, sticky = W, padx = PADDING, pady = PADDING)
            widget.grid(row = r + 1, column = 1,
                        columnspan = 1 if widget in (self.txtTdDim, self.txtTdLag, self.txtPeriodNum) else 2,
                        sticky = (W, E), padx = PADDING, pady = PADDING)
            if widget is self.txtTdDim:
                self.btnTdAuto.grid(row = r + 1, column = 2, rowspan = 2, sticky = (W, E, N, S), padx = PADDING, pady = PADDING)
            elif widget is self.txtPeriodNum:
                self.btnPeriodAuto.grid(row = r + 1, column = 2, sticky = (W, E, N, S), padx = PADDING, pady = PADDING)
        self.chkLinearTrend.grid(row = len(self.widgetsPreproc) + 1, column = 0, columnspan = 3, sticky = W, padx = PADDING, pady = PADDING)
        self.chkPreviewPreproc.grid(row = len(self.widgetsPreproc) + 2, column = 0, columnspan = 3, sticky = W, padx = PADDING, pady = PADDING)
        self.frmPreproc.columnconfigure(1, weight = 1)
        self.frmPreproc.pack(side = 'top', fill = 'x', padx = PADDING, pady = PADDING)
        
        # Search strategy parameters
        self.frmSearchStrategy = ttk.Labelframe(self.frmRightCol, text = 'Search Strategy')
        self.lblSearchStrategy = ttk.Label(self.frmSearchStrategy, text = 'Search Strategy:')
        self.selSearchStrategy = ttk.Combobox(self.frmSearchStrategy, state = 'readonly', values = tuple(s[1] for s in SEARCH_STRATEGIES))
        self.selSearchStrategy.bind('<<ComboboxSelected>>', self._onSearchStrategySelect)
        self.selSearchStrategy.current(0)
        self.lblNumDetections = ttk.Label(self.frmSearchStrategy, text = 'Number of detections:')
        self.txtNumDetections = tkinter.Spinbox(self.frmSearchStrategy, textvariable = self.intNumDetections, from_ = 1, to = 1000, increment = 1)
        self.lblMinLen = ttk.Label(self.frmSearchStrategy, text = 'Minimum Interval Length')
        self.lblMinLenValue = ttk.Label(self.frmSearchStrategy)
        self.sclMinLen = ttk.Scale(self.frmSearchStrategy, orient = tkinter.HORIZONTAL,
                                   from_ = 10, to = 100, variable = self.strMinLen)
        self.lblMaxLen = ttk.Label(self.frmSearchStrategy, text = 'Maximum Interval Length')
        self.lblMaxLenValue = ttk.Label(self.frmSearchStrategy)
        self.sclMaxLen = ttk.Scale(self.frmSearchStrategy, orient = tkinter.HORIZONTAL,
                                   from_ = 10, to = 100, variable = self.strMaxLen)
        self.lblOverlapTh = ttk.Label(self.frmSearchStrategy, text = 'Overlap Threshold')
        self.lblOverlapThValue = ttk.Label(self.frmSearchStrategy)
        self.sclOverlapTh = ttk.Scale(self.frmSearchStrategy, orient = tkinter.HORIZONTAL,
                                      from_ = 0.0, to = 1.0, variable = self.strOverlapTh)
        self.lblSearchStrategy.grid(row = 0, column = 0, sticky = W, padx = PADDING, pady = PADDING)
        self.selSearchStrategy.grid(row = 0, column = 1, sticky = (W, E), padx = PADDING, pady = PADDING)
        self.lblNumDetections.grid(row = 1, column = 0, sticky = W, padx = PADDING, pady = PADDING)
        self.txtNumDetections.grid(row = 1, column = 1, sticky = (W, E), padx = PADDING, pady = PADDING)
        self.widgetsSearchStrategy = [
            (self.lblMinLen, self.lblMinLenValue, self.sclMinLen),
            (self.lblMaxLen, self.lblMaxLenValue, self.sclMaxLen),
            (self.lblOverlapTh, self.lblOverlapThValue, self.sclOverlapTh)
        ]
        for r, (label, valLabel, widget) in enumerate(self.widgetsSearchStrategy):
            label.grid(row = 2 * r + 2, column = 0, sticky = W, padx = PADDING, pady = (PADDING, 0))
            valLabel.grid(row = 2 * r + 2, column = 1, sticky = E, padx = PADDING, pady = (PADDING, 0))
            widget.grid(row = 2 * r + 3, column = 0, columnspan = 2, sticky = (W, E), padx = PADDING, pady = (0, PADDING))
        self.frmSearchStrategy.columnconfigure(1, weight = 1)
        self.frmSearchStrategy.pack(side = 'top', fill = 'x', padx = PADDING, pady = (0, PADDING))
        
        # Proposal Parameters
        self.frmProposals = ttk.Labelframe(self.frmRightCol, text = 'Interval Proposals')
        self.lblProposalMethod = ttk.Label(self.frmProposals, text = 'Pointwise Scoring Method:')
        self.selProposalMethod = ttk.Combobox(self.frmProposals, state = 'readonly', values = tuple(m[1] for m in PROPOSAL_METHODS))
        self.selProposalMethod.current(0)
        self.lblPropTh = ttk.Label(self.frmProposals, text = 'Proposal Threshold:')
        self.txtPropTh = tkinter.Spinbox(self.frmProposals, textvariable = self.strPropTh, width = 10, from_ = -3.0, to = 3.0, increment = 0.1)
        self.chkPropGrad = ttk.Checkbutton(self.frmProposals, text = 'Use gradient of point-wise scores', variable = self.boolPropGrad)
        self.chkPropMAD = ttk.Checkbutton(self.frmProposals, text = 'Use MAD for robust estimation of mean and variance', variable = self.boolPropMAD)
        self.lblProposalMethod.grid(row = 0, column = 0, sticky = W, padx = PADDING, pady = PADDING)
        self.selProposalMethod.grid(row = 0, column = 1, sticky = (W,E), padx = PADDING, pady = PADDING)
        self.lblPropTh.grid(row = 1, column = 0, sticky = W, padx = PADDING, pady = PADDING)
        self.txtPropTh.grid(row = 1, column = 1, sticky = (W,E), padx = PADDING, pady = PADDING)
        self.chkPropGrad.grid(row = 2, column = 0, columnspan = 2, sticky = W, padx = PADDING, pady = PADDING)
        self.chkPropMAD.grid(row = 3, column = 0, columnspan = 2, sticky = W, padx = PADDING, pady = PADDING)
        self.frmProposals.columnconfigure(1, weight = 1)
        self.frmProposals.pack(side = 'top', fill = 'x', padx = PADDING, pady = PADDING)
        
        # Run button
        ttk.Style(self).configure('Submit.TButton', font = 'TkDefaultFont 10 bold')
        self.btnRun = ttk.Button(self, text = 'Detect Anomalous Intervals', style = 'Submit.TButton',
                                 default = 'active', state = 'disabled', command = self.runDetector)
        self.btnRun.grid(row = 3, column = 0, columnspan = 2, padx = PADDING, pady = (0, PADDING))
        
        # Expand plot to take up any superfluous space
        self.rowconfigure(1, weight = 1, minsize = 180)
        self.columnconfigure((0,1), weight = 1, uniform = True)
        
        # Initialize dynamic widgets
        self._onEstimatorSelect()
        self._onDetrendingSelect()
        self._onSearchStrategySelect()
        self._updateScaleValues()
    
    
    def _checkLibMaxDiv(self):
        """Checks if libmaxdiv has been found and is up-to-date."""
        
        basedir = os.path.dirname(__file__)
        if libmaxdiv_wrapper.libmaxdiv is None:
            messagebox.showwarning(
                title = 'libmaxdiv not found',
                message = 'libmaxdiv could not be found.\n\n' \
                          'You can still use the full functionality of this tool, but the detection ' \
                          'of anomalous intervals will be extremely slow.\n' \
                          'Information on how to build the libmaxdiv library to gain a significant speed-up ' \
                          'can be found in maxdiv/libmaxdiv/README.md.'
            )
        elif os.path.exists(libmaxdiv_wrapper.libmaxdiv_path) \
                and os.path.exists(os.path.join(basedir, 'libmaxdiv_wrapper.py')) \
                and os.path.exists(os.path.join(basedir, 'libmaxdiv', 'libmaxdiv.h')) \
                and os.path.getmtime(libmaxdiv_wrapper.libmaxdiv_path) < os.path.getmtime(os.path.join(basedir, 'libmaxdiv_wrapper.py')) \
                and os.path.getmtime(libmaxdiv_wrapper.libmaxdiv_path) < os.path.getmtime(os.path.join(basedir, 'libmaxdiv', 'libmaxdiv.h')):
            messagebox.showerror(
                title = 'libmaxdiv is out-dated',
                message = 'Your build of libmaxdiv is older than the the source code.\n\n'
                          'Please re-build libmaxdiv in order to make sure that you\'re working with the latest version.'
            )
            self.destroy()
            exit()
    
    
    def _onEstimatorSelect(self, *args):
        """Shows and hides form fields specific to the selected distribution model."""
        
        estimator = ESTIMATORS[self.selEstimator.current()][0]
        if estimator == 'parzen':
            self.lblKernelSigmaSq.grid()
            self.txtKernelSigmaSq.grid()
        else:
            self.lblKernelSigmaSq.grid_remove()
            self.txtKernelSigmaSq.grid_remove()
        if estimator == 'erph':
            self.lblNumHist.grid()
            self.txtNumHist.grid()
            self.lblNumBins.grid()
            self.txtNumBins.grid()
        else:
            self.lblNumHist.grid_remove()
            self.txtNumHist.grid_remove()
            self.lblNumBins.grid_remove()
            self.txtNumBins.grid_remove()
        self._resizeWindow()
    
    
    def _onDetrendingSelect(self, *args):
        """Shows and hides form fields specific to the deseasonalization method."""
        
        method = DETRENDING_METHODS[self.selDetrending.current()][0]
        if (self.data.shape[1] > 0) and np.ma.isMaskedArray(self.data) and (method == 'deseasonalize_ft'):
            messagebox.showerror(
                title = 'Function not available',
                message = 'Fourier Transform based deseasonalization is not available for data with missing values.'
            )
            self.selDetrending.current(0)
            method = ''
        if method in ('deseasonalize_z', 'deseasonalize_ols'):
            self.lblPeriodNum.grid()
            self.txtPeriodNum.grid()
            self.btnPeriodAuto.grid()
        else:
            self.lblPeriodNum.grid_remove()
            self.txtPeriodNum.grid_remove()
            self.btnPeriodAuto.grid_remove()
        if method == 'deseasonalize_ols':
            self.lblPeriodLen.grid()
            self.txtPeriodLen.grid()
            self.chkLinearTrend.grid()
        else:
            self.lblPeriodLen.grid_remove()
            self.txtPeriodLen.grid_remove()
            self.chkLinearTrend.grid_remove()
        self._resizeWindow()
        self._onPreprocChange()
    
    
    def _onSearchStrategySelect(self, *args):
        """Shows and hides form fields specific to the selected search strategy."""
        
        strategy = SEARCH_STRATEGIES[self.selSearchStrategy.current()][0]
        if strategy == 'pointwise':
            self.frmProposals.pack(side = 'top', fill = 'x', padx = PADDING, pady = PADDING)
        else:
            self.frmProposals.pack_forget()
        self._resizeWindow()
    
    
    def _updateScaleValues(self, *args):
        """Callback for updating the scale value labels according to the corresponding scale."""
        
        minLen = str(int(round(float(self.strMinLen.get()))))
        maxLen = str(int(round(float(self.strMaxLen.get()))))
        overlapTh = int(round(float(self.strOverlapTh.get()) * 100))
        self.strMinLen.set(minLen)
        self.strMaxLen.set(maxLen)
        self.strOverlapTh.set(str(overlapTh / 100.0))
        self.lblMinLenValue['text'] = minLen
        self.lblMaxLenValue['text'] = maxLen
        self.lblOverlapThValue['text'] = '{} %'.format(overlapTh)
    
    
    def _resizeWindow(self):
        """Adjusts the size of the window to fit it's contents and sets the new size as the minimum size."""
        
        self.minsize(0, 0)
        self.geometry('')   # resize window to fit new contents
        self.update()       # process pending geometry management to update actual size
        self.minsize(self.winfo_width(), self.winfo_height())
    
    
    def _onPreprocChange(self, *args):
        """Callback to run pre-processing when the parameters have been changed."""
        
        if self.boolPreviewPreproc.get() and (self.data.shape[1] > 0):
            if not self._preprocAfterId is None:
                self.after_cancel(self._preprocAfterId)
            self._preprocAfterId = self.after(1000, self._onPreviewPreprocChange)
    
    
    def _onPreviewPreprocChange(self, *args):
        """Callback for switching between plots of the original and the pre-processed data."""
        
        if self.boolPreviewPreproc.get():
            self.updatePreprocData()
        self.updatePlot()
    
    
    def updatePreprocData(self, *args):
        """Run pre-processing."""
        
        if self.data.shape[1] > 0:
            
            detrending = DETRENDING_METHODS[self.selDetrending.current()][0]
            
            preprocParams = (self.boolNormalize.get(), detrending, self.intPeriodNum.get(), self.intPeriodLen.get(), self.boolLinearTrend.get())
            if preprocParams != self._lastPreprocParams:
                
                if np.ma.isMaskedArray(self.data):
                    self.preprocData = np.ma.mask_cols(self.data.copy())
                    # Set masked values to 0, because NaNs cause some strange problems, even if they're masked
                    mask = self.preprocData.mask.copy()                 # backup mask
                    self.preprocData[:,self.preprocData.mask[0,:]] = 0  # set masked values to 0
                    self.preprocData.mask = mask                        # restore mask
                else:
                    self.preprocData = self.data
                
                if self.boolNormalize.get():
                    self.preprocData = preproc.normalize_time_series(self.preprocData)
                
                if detrending == 'deseasonalize_z':
                    self.preprocData = preproc.deseasonalize_zscore(self.preprocData, self.intPeriodNum.get())
                elif detrending == 'deseasonalize_ols':
                    self.preprocData = preproc.detrend_ols(self.preprocData, [(self.intPeriodNum.get(), self.intPeriodLen.get())], self.boolLinearTrend.get(), False)
                elif detrending == 'deseasonalize_ft':
                    self.preprocData = preproc.deseasonalize_ft(self.preprocData)
                
                self.resetDetections()
                self._lastPreprocParams = preprocParams
    
    
    def selectDataFile(self, *args):
        """Shows a dialog to select a file and loads it."""
        
        filename = filedialog.askopenfilename(parent = self, title = 'Load Time-Series', defaultextension = '.csv',
                                              filetypes = [('Comma Separated File', '.csv'), ('All Files', '*')])
        if filename:
            self.loadData(filename)
    
    
    def loadData(self, filename, selectSubset = True):
        """Loads data from a CSV file."""
        
        try:
            
            # Load data
            self.data, self.timesteps, self.varnames = readDataFromCSV(filename)
            self.strFile.set(filename)
            
            # Reset dependent data
            if np.ma.isMaskedArray(self.data) and (DETRENDING_METHODS[self.selDetrending.current()][0] == 'deseasonalize_ft'):
                self.selDetrending.current(0)
            self.preprocData = np.ndarray((0, 0))
            self._lastPreprocParams = None
            self.resetDetections()
            self._xlim = None
            
            # Let the user select a subset of the data
            if selectSubset and (len(self.varnames) > 1):
                selectDataDlg = SelectSubDataDialog(self, self.varnames, self.data.shape[1], title = 'Select Variables')
                if len(selectDataDlg.selected) > 0:
                    firstRecord = selectDataDlg.firstRecord - 1 if selectDataDlg.firstRecord is not None else 0
                    lastRecord = selectDataDlg.lastRecord if selectDataDlg.lastRecord is not None else self.data.shape[1]
                    self.data = self.data[np.array(selectDataDlg.selected), firstRecord:lastRecord]
                    self.varnames = [n for i, n in enumerate(self.varnames) if i in selectDataDlg.selected]
                    self.timesteps = np.array(self.timesteps)[firstRecord:lastRecord]
                else:
                    self.varnames = []
                    self.data = np.ndarray((0, 0))
                    self.timesteps = []
            
            # Adjust interactive controls
            if self.data.shape[1] > 0:
                maxLen = self.data.shape[1] // 4
                self.sclMinLen['to'] = self.sclMaxLen['to'] = maxLen
                if float(self.strMaxLen.get()) > maxLen:
                    self.strMaxLen.set(str(maxLen))
                if float(self.strMinLen.get()) > maxLen:
                    self.strMinLen.set(str(maxLen))
                self.btnRun['state'] = 'normal'
                for btn in self.btnsVis[0]:
                    if btn is not None:
                        btn['state'] = 'normal'
            else:
                self.btnRun['state'] = 'disabled'
                for btn in self.btnsVis[0]:
                    if btn is not None:
                        btn['state'] = 'disabled'
            
            # Update plot
            self.updatePlot()
            
        except RuntimeError as e:
            messagebox.showerror(title = 'Could not read file', message = str(e))
    
    
    def showDataFormatHelp(self, *args):
        """Shows a help message describing the data format."""
        
        messagebox.showinfo(title = 'Data Format', message = 'Time-Series data is accepted as CSV file, where each row '\
                                'corresponds to a time step and lists the values of different attributes separated by commas.\n\n'\
                                'The first attribute may be a continuous list of time steps which may also be given as date and time '\
                                'in the following format:\nYYYY-MM-DD HH:MM:SS or YYYY-MM-DD\n\n'\
                                'The first line in the file may be a comma-separated list of column names.\n\n'
                                'Missing values in the data can be encoded as "nan" (without the quotes).')
    
    
    def saveFigDlg(self, *args):
        """Shows a dialog for selecting a filename to save the figure to."""
        
        filename = filedialog.asksaveasfilename(parent = self, title = 'Export Figure', defaultextension = '.png',
                                                filetypes = [('Portable Network Graphics', '.png'),
                                                             ('Scalable Vector Graphics', '.svg'),
                                                             ('Encapsulated PostScript', '.eps'),
                                                             ('Portable Document Format', '.pdf')])
        if filename:
            try:
                self.saveFig(filename)
            except Exception as e:
                messagebox.showerror(title = 'Error', message = 'Could not export figure: {!s}'.format(e))
    
    
    def saveFig(self, filename):
        """Exports the current figure to the given filename."""
        
        self.fig.savefig(filename)
    
    
    def exportDetectionsDlg(self, *args):
        """Shows a dialog for selecting a filename to save the detections to."""
        
        filename = filedialog.asksaveasfilename(parent = self, title = 'Export Detections', defaultextension = '.csv',
                                                filetypes = [('Comma Separated File', '.csv')])
        if filename:
            try:
                self.exportDetections(filename)
            except Exception as e:
                try:
                    isIOError = (isinstance(e, PermissionError) or isinstance(e, IOError))
                except:
                    isIOError = isinstance(e, IOError)
                if isIOError:
                    messagebox.showerror(title = 'Error', message = 'Could not open file for writing.')
                else:
                    messagebox.showerror(title = 'Error', message = 'Could not export detections: {!s}'.format(e))
    
    
    def exportDetections(self, filename):
        """Exports the detections to the given filename."""
        
        if len(self.detections) == 0:
            raise RuntimeError('The algorithm has not been run yet.')
        
        try:
            f = open(filename, 'w', newline = '')
        except:
            f = open(filename, 'wb')
        
        with f:
            writer = csv.DictWriter(f, ['Start', 'End', 'Score'], delimiter = ',', extrasaction = 'ignore')
            writer.writeheader()
            writer.writerows({
                'Start' : str(self.timesteps[a]),
                'End' : str(self.timesteps[b-1]),
                'Score' : '{:.6f}'.format(score)
            } for a, b, score in self.detections)
    
    
    def optimizeTdEmbedding(self, *args):
        """Heuristically determines parameters for Time-Delay Embedding."""
        
        if self.data.shape[1] > 0:
            self.updatePreprocData()
            k, T = preproc.td_params(self.preprocData, None, None)
            self.intTdDim.set(k)
            self.intTdLag.set(T)
    
    
    def detectSeasonality(self, *args):
        """Tries to automatically detect the length of the main seasonal period in the time-series."""
        
        if self.data.shape[1] > 0:
            if np.ma.isMaskedArray(self.data):
                messagebox.showerror(
                    title = 'Function not available',
                    message = 'Automatic detection of the seasonality is not available for data with missing values.'
                )
            else:
                periods, _ = preproc.detect_periods(self.data)
                if len(periods) > 0:
                    self.intPeriodNum.set(int(round(periods[0])))
                    self.intPeriodLen.set(1)
    
    
    def runDetector(self):
        """Runs the MDI algorithm on self.data and stores the result in self.detections."""
        
        if self.data.shape[1] > 0:
            oldBtnText = self.btnRun['text']
            self.btnRun['text'] = 'Running...'
            self.update_idletasks()
            
            self._shownDetection = -1
            self._xlim = None
            
            try:
                
                # Pre-process data
                # (the library could do this as well, but we need the pre-processed data
                # for visualization anyway)
                self.updatePreprocData()
                
                # Get parameters from the controls
                params = {
                    'method'        : ESTIMATORS[self.selEstimator.current()][0],
                    'mode'          : MODES[self.selDivergence.current()][0],
                    'kernelparameters'  : { 'kernel_sigma_sq' : float(self.strKernelSigmaSq.get()) },
                    'num_hist'      : self.intNumHist.get(),
                    'num_bins'      : self.intNumBins.get(),
                    'td_dim'        : self.intTdDim.get(),
                    'td_lag'        : self.intTdLag.get(),
                    'proposals'     : SEARCH_STRATEGIES[self.selSearchStrategy.current()][0],
                    'num_intervals' : self.intNumDetections.get(),
                    'extint_min_len': int(self.strMinLen.get()),
                    'extint_max_len': int(self.strMaxLen.get()),
                    'overlap_th'    : float(self.strOverlapTh.get()),
                    'proposalparameters' : {
                        'method'    : PROPOSAL_METHODS[self.selProposalMethod.current()][0],
                        'filter'    : [-1, 0, 1] if self.boolPropGrad.get() else None,
                        'useMAD'    : self.boolPropMAD.get(),
                        'sd_th'     : float(self.strPropTh.get())
                    }
                }
                if params['num_intervals'] < 1:
                    params['num_intervals'] = None
                
                # Run detector
                if libmaxdiv_wrapper.libmaxdiv is None:
                    self.detections = maxdiv(self.preprocData, **params)
                else:
                    try:
                        self.detections = maxdiv(self.preprocData, useLibMaxDiv = True, **params)
                    except Exception as e:
                        if messagebox.askquestion(
                            title = 'libmaxdiv failure',
                            message = 'For some reason, the data could not be processed using libmaxdiv.\n' \
                                      'We will fall back to the Python implementation, but this will take a lot more time.\n\n' \
                                      'Error information: {!s}'.format(e),
                            type = messagebox.OKCANCEL,
                            default = messagebox.OK,
                            icon = messagebox.WARNING
                        ) == messagebox.OK:
                            self.detections = maxdiv(self.preprocData, **params)
                
            except Exception as e:
                messagebox.showerror(title = 'Exception', message = 'An error occurred: {!s}'.format(e))
            
            self.btnRun['text'] = oldBtnText
            for btn in self.btnsVis[1]:
                if btn is not None:
                    btn['state'] = 'normal' if len(self.detections) > 0 else 'disabled'
            self.updatePlot()
    
    
    def resetDetections(self):
        """Clears the list of detections and disables related controls."""
        
        self.detections = []
        self._shownDetection = -1
        for btn in self.btnsVis[1]:
            if btn is not None:
                btn['state'] = 'disabled'
    
    
    def _changeNavMode(self, *args):
        """Callback for switching between pan and zoom mode."""
        
        self.lblTS.unbind('<Button>')
        self.lblTS.unbind('<B1-Motion>')
        self.lblTS.unbind('<ButtonRelease-1>')
        
        if self.strFigureNavMode.get() == 'zoom':
            self.lblTS['cursor'] = 'plus'
            self.lblTS.bind('<Button>', self._figClick)
        else:
            self.lblTS['cursor'] = 'sb_h_double_arrow'
            self.lblTS.bind('<B1-Motion>', self._figPan)
            self.lblTS.bind('<ButtonRelease-1>', self._figRelease)
    
    
    def _onExpandPlotChange(self, *args):
        """Hides the settings and expands the figure to full height if `self.boolExpandPlot` is `True`."""
        
        if self.boolExpandPlot.get():
            self.frmLeftCol.grid_remove()
            self.frmRightCol.grid_remove()
            self.btnRun.grid_remove()
            self.minsize(400, 240)
            self.geometry('{}x{}'.format(self.winfo_width(), max(240, min(960, self.data.shape[0] * 120))))
        else:
            self.minsize(0, 0)
            self.geometry('{}x240'.format(self.winfo_width()))
            self.update()
            self.redrawPlot()
            self.frmLeftCol.grid()
            self.frmRightCol.grid()
            self.btnRun.grid()
            self._resizeWindow()
        self.updatePlot()
    
    
    def resetView(self, *args):
        """Resets the figure to show all data."""
        
        if len(self.timesteps) > 0:
            self._shownDetection = -1
            self._xlim = None
            self.fig.gca().set_xlim(self.timesteps[0], self.timesteps[-1])
            self.redrawPlot()
    
    
    def goToDetection(self, det):
        """Zooms in to the first detection."""
        
        if det == -1:
            self.resetView()
        elif (len(self.timesteps) > 0) and (len(self.detections) > 0) and (det >= 0) and (det < len(self.detections)):
            minZoomSize = min(MIN_DET_ZOOM_SIZE, self.data.shape[1])
            detLen = self.detections[det][1] - self.detections[det][0]
            margin = int(0.4 * detLen)
            if 2 * margin + detLen < minZoomSize:
                margin = (minZoomSize - detLen) // 2
            left = max(0, self.detections[det][0] - margin)
            right = min(len(self.timesteps), self.detections[det][1] + margin) - 1
            self.fig.gca().set_xlim(self.timesteps[left], self.timesteps[right])
            self._shownDetection = det
            self._xlim = (left, right)
            self.redrawPlot()
    
    
    def goToNextDetection(self, *args):
        """Zooms in to the next detection."""
        
        self.goToDetection(self._shownDetection + 1)
    
    
    def goToPrevDetection(self, *args):
        """Zooms in to the previous detection."""
        
        self.goToDetection(self._shownDetection - 1)
    
    
    def _figPan(self, evt):
        """Callback when the mouse is moved over the figure with the left button held down."""
        
        if len(self.timesteps) > 0:
            mousePos = (evt.x, evt.y)
            if self._lastMousePosition is not None:
                dx = self._lastMousePosition[0] - mousePos[0]
                left, right = self._xlim if self._xlim is not None else (0, len(self.timesteps) - 1)
                rangeLen = right - left
                speed = max(1, rangeLen // 250)
                left = max(0, min(len(self.timesteps) - 1 - rangeLen, left + dx * speed))
                self._xlim = (left, left + rangeLen)
                self.fig.gca().set_xlim(self.timesteps[left], self.timesteps[left + rangeLen])
                self.redrawPlot()
            self._lastMousePosition = mousePos
    
    
    def _figRelease(self, evt):
        """Callback when the mouse button is released over the figure."""
        
        self._lastMousePosition = None
    
    
    def _figClick(self, evt):
        """Callback when the figure is clicked."""
        
        if (len(self.timesteps) > 0) and (evt.num in (1, 3)):
            x, _ = self.fig.gca().transAxes.inverted().transform((evt.x, evt.y))
            if (x >= 0.0) and (x <= 1.0):
                
                left, right = self._xlim if self._xlim is not None else (0, len(self.timesteps) - 1)
                rangeLen = right - left + 1
                
                mul = 1.0
                if (evt.num == 1) and (rangeLen > 1):
                    mul = 0.5
                elif (evt.num == 3) and ((left != 0) or (right != len(self.timesteps) - 1)):
                    mul = 2.0
                
                if mul != 1.0:
                    center = int(round(left + rangeLen * x))
                    left = max(0, int(center - (center - left) * mul))
                    right = min(len(self.timesteps) - 1, int(round(center + (right - center) * mul)))
                    self.fig.gca().set_xlim(self.timesteps[left], self.timesteps[right])
                    self._xlim = (left, right)
                    self.redrawPlot()
    
    
    def updatePlot(self):
        """Re-draws the plot."""
        
        self.fig.clear()
        self._detectionLabels = []
        
        # Create plot
        plt = self.fig.add_subplot(111)
        if self.data.shape[1] > 0:
            
            # Create multiple axes if figure is expanded
            if self.boolExpandPlot.get():
                self.fig.clear()
                plots = [self.fig.add_subplot(self.data.shape[0], 1, 1)]
                for i in range(1, self.data.shape[0]):
                    plots.append(self.fig.add_subplot(self.data.shape[0], 1, i + 1, sharex = plots[0]))
            else:
                plots = [plt]
            
            # Plot data
            if self.boolPreviewPreproc.get():
                self.updatePreprocData()
            for i in range(self.data.shape[0]):
                plots[min(i, len(plots) - 1)].plot(
                    self.timesteps,
                    self.preprocData[i,:].T if self.boolPreviewPreproc.get() else self.data[i,:].T,
                    label = self.varnames[i],
                    linewidth = 1
                )
                if len(plots) > 1:
                    plots[i].set_ylabel(self.varnames[i])
                    plots[i].yaxis.set_label_position('right')
            if self._xlim is None:
                plots[0].set_xlim(self.timesteps[0], self.timesteps[-1])
            else:
                plots[0].set_xlim(self.timesteps[self._xlim[0]], self.timesteps[self._xlim[1]])
            for plt in plots:
                plt.grid()
            
            # Plot detections
            if len(self.detections) > 0:
                maxScore = max(score for a, b, score in self.detections)
                minScore = min(score for a, b, score in self.detections)
                for pltInd, plt in enumerate(plots):
                    ymin, ymax = plt.get_ylim()
                    plt.set_ylim(ymin, ymax)
                    for detInd, (a, b, score) in enumerate(self.detections):
                        ta, tb = self.timesteps[a], self.timesteps[b - 1]
                        if isinstance(ta, datetime.datetime):
                            ta = matplotlib.dates.date2num(ta)
                        if isinstance(tb, datetime.datetime):
                            tb = matplotlib.dates.date2num(tb)
                        intensity = float(score - minScore) / (maxScore - minScore) if minScore < maxScore else 1.0
                        col = (1.0, 0.8 - intensity * 0.8, 0.8 - intensity * 0.8)
                        plt.fill([ta, ta, tb, tb], [ymin, ymax, ymax, ymin], color = col, alpha = 0.3)
                        if (pltInd == 0) and (detInd < 9) and (len(self.detections) > 1):
                            self._detectionLabels.append(plt.text(
                                ta + min((tb - ta) / 4, 10),
                                ymax - (ymax - ymin) / 20,
                                str(detInd + 1),
                                color = 'r',
                                verticalalignment = 'top'
                            ))
            
            # Show legend
            if (not self.boolExpandPlot.get()) and (self.data.shape[0] > 1) and (self.data.shape[0] <= 10):
                plots[0].legend(bbox_to_anchor = (0., 1., 1., .1), loc = 3, ncol = self.data.shape[0], mode = "expand", borderaxespad = 0., fontsize = 9)
            
        else:
            plt.set_xlim(0.0, 1.0)
            plt.set_ylim(0.0, 1.0)
            plt.text(0.5, 0.5, 'No data loaded.', horizontalalignment = 'center', verticalalignment = 'center', transform = plt.transAxes)
        
        # Display plot
        self._lastFigSize = (0, 0)
        self.redrawPlot()
    
    
    def redrawPlot(self, force = True):
        """Displays the figure in the label with appropriate size."""
        
        if not self._destroyed:
            w, h = self.lblTS.winfo_width(), self.lblTS.winfo_height()
            if (w > 20) and (h > 20) and (force or (abs(w - self._lastFigSize[0]) >= 10) or (abs(h - self._lastFigSize[1]) >= 10)):
            
                # Show/hide detection labels depending on the display size of the detected interval
                figwidth = self.fig.get_dpi() * self.fig.get_figwidth()
                for det, lbl in zip(self.detections, self._detectionLabels):
                    ta, tb = self.timesteps[det[0]], self.timesteps[det[1] - 1]
                    pix_a, _ = self.fig.gca().transData.transform((matplotlib.dates.date2num(ta) if isinstance(ta, datetime.datetime) else ta, 0))
                    pix_b, _ = self.fig.gca().transData.transform((matplotlib.dates.date2num(tb) if isinstance(tb, datetime.datetime) else tb, 0))
                    lbl.set_visible(pix_b - pix_a > 10 and pix_a >= 0 and pix_b < figwidth)
                
                # Draw image
                self.lblTS._img = ImageTk.PhotoImage(figure2img(self.fig, w, h))
                self.lblTS['image'] = self.lblTS._img
                self._lastFigSize = (w, h)



class SelectSubDataDialog(simpledialog.Dialog):
    """ A simple dialog with controls to select the variables and the range of records to read from a data set.
    
    On completion, a list of indices of selected variables will be stored in the `selected` attribute.
    The indices of the first and the last record to be read will be stored in `firstRecord` and `lastRecord`, respectively.
    """
    
    
    def __init__(self, parent, options, numRecords, **kwargs):
        
        self.options = options
        self.numRecords = numRecords
        self.selected = []
        self.firstRecord = None
        self.lastRecord = None
        simpledialog.Dialog.__init__(self, parent, **kwargs)
    
    
    def body(self, master):
        
        # Checkbuttons for variable selection
        nv = len(self.options)
        rows = max(1, int(math.ceil(math.sqrt(2.0 * nv))))
        columns = ((nv-1) // rows) + 1
        
        self._intro = ttk.Label(master, text = 'Please select the variables to be loaded:')
        self._vars = [tkinter.BooleanVar(self, value = True) for i in range(nv)]
        self._btns = [ttk.Checkbutton(master, text = self.options[i], variable = self._vars[i]) for i in range(nv)]
        
        self._intro.grid(row = 0, column = 0, columnspan = columns, sticky = W, padx = PADDING, pady = PADDING)
        for i in range(nv):
            self._btns[i].grid(row = i % rows + 1, column = i // rows, sticky = W, padx = PADDING, pady = PADDING)
        
        # Record range
        if self.numRecords > 1:
        
            self._varFirstRecord = tkinter.IntVar(self, value = 1)
            self._varLastRecord = tkinter.IntVar(self, value = self.numRecords)
            
            self._frmRecRange = ttk.Frame(master)
            self._lblRecRange1 = ttk.Label(self._frmRecRange, text = 'Load records')
            self._lblRecRange2 = ttk.Label(self._frmRecRange, text = 'to')
            self._txtFirstRecord = tkinter.Spinbox(self._frmRecRange, textvariable = self._varFirstRecord,
                                                   width = 8, from_ = 1, to = self.numRecords, increment = 1)
            self._txtLastRecord = tkinter.Spinbox(self._frmRecRange, textvariable = self._varLastRecord,
                                                  width = 8, from_ = 1, to = self.numRecords, increment = 1)
            
            self._lblRecRange1.pack(side = 'left')
            self._txtFirstRecord.pack(side = 'left', padx = 2)
            self._lblRecRange2.pack(side = 'left')
            self._txtLastRecord.pack(side = 'left', padx = (2, 0))
            self._frmRecRange.grid(row = rows + 2, column = 0, columnspan = columns, sticky = W, padx = PADDING, pady = PADDING)
        
        self.resizable(False, False)
        return None
    
    
    def apply(self):
        
        self.selected = []
        for i, var in enumerate(self._vars):
            if var.get():
                self.selected.append(i)
        
        if self.numRecords > 1:
            self.firstRecord = self._varFirstRecord.get()
            self.lastRecord = self._varLastRecord.get()
    
    
    def validate(self):
        
        if self.numRecords > 1:
            try:
                fr = self._varFirstRecord.get()
                lr = self._varLastRecord.get()
            except:
                return 0
            if lr < fr:
                return 0
            
        return 1