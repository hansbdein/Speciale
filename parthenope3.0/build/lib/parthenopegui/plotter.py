import ast
import datetime
import os
import pickle
import sys
from collections import OrderedDict

import matplotlib
import numpy as np
import six

matplotlib.use("Qt5Agg")
os.environ["QT_API"] = "pyside2"
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PySide2.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt, Signal
from PySide2.QtGui import QGuiApplication, QPixmap
from PySide2.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QRadioButton,
    QStackedWidget,
    QTableView,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# PArthENoPE
try:
    import parthenopegui.resourcesPySide2
    from parthenopegui import FileNotFoundErrorClass
    from parthenopegui.configuration import (
        Configuration,
        Nuclides,
        Parameters,
        PGFont,
        PGLabel,
        PGLabelButton,
        PGPushButton,
        Reactions,
        UnicodeSymbols,
        askDirName,
        askSaveFileName,
        askYesNo,
        paramsRealPath,
    )
    from parthenopegui.errorManager import mainlogger, pErrorManager, pGUIErrorManager
    from parthenopegui.plotUtils import (
        PGContour,
        PGLine,
        chi2func,
        chi2labels,
        cmaps,
        defaultCmap,
        extendOptions,
        markerOptions,
        styleOptions,
    )
    from parthenopegui.runner import RunPArthENoPE
    from parthenopegui.texts import PGText
except ImportError:
    print("[plotter] Necessary parthenopegui submodules not found!")
    raise


class CurrentPlotContent(object):
    """Contains the information on the various lines
    that are present in the current plot
    """

    def __init__(self, plotSettings):
        """Save properties and clean everything

        Parameter:
            plotSettings: an instance of PlotSettings from which
                the plot properties will be read
        """
        self.pSett = plotSettings
        self.clearCurrent()

    def clearCurrent(self):
        """Clean the current selection
        and restores everything to the initial empty values
        """
        if hasattr(self, "lines") and isinstance(self.lines, list):
            for l in self.lines:
                del l
        if hasattr(self, "contours") and isinstance(self.contours, list):
            for c in self.contours:
                del c
        self.axesTextSize = PlotSettings._possibleTextSizesDefault
        self.lines = []
        self.contours = []
        self.title = ""
        self.tight = True
        self.legend = True
        self.legendLoc = PlotSettings._possibleLegendLoc[0]
        self.legendNcols = 1
        self.legendTextSize = PlotSettings._possibleTextSizesDefault
        self.figsize = None
        self.pSett.clearCurrent()

    def readSettings(self):
        """Read all the properties defined in the PlotSettings panel"""
        self.axesTextSize = self.pSett.axesTextSize
        self.figsize = self.pSett.figsize
        self.legend = self.pSett.legend
        self.legendLoc = self.pSett.legendLoc
        self.legendNcols = self.pSett.legendNcols
        self.legendTextSize = self.pSett.legendTextSize
        self.tight = self.pSett.tight
        self.title = self.pSett.title

    def doPlot(self, savefig=None):
        """Use the information to produce a plot

        Parameter:
            savefig (default None): if it is a string, assume it is
                the filename where to save the plot
        """
        self.readSettings()
        fig = plt.figure(figsize=self.figsize)
        plt.plot(np.nan, np.nan)
        for l in self.lines:
            x = np.array(l.x)
            y = np.array(l.y)
            try:
                for ix in (0, -1):
                    while np.isnan(y[ix]):
                        x = np.delete(x, ix)
                        y = np.delete(y, ix)
                if len(y) > 2:
                    for ie, e in enumerate(np.array(y)[1:-1]):
                        if np.isnan(e):
                            foll = 2
                            while np.isnan(y[ie + foll]):
                                foll += 1
                            y[ie + 1] = y[ie] + (y[ie + foll] - y[ie]) / (
                                x[ie + foll] - x[ie]
                            ) * (x[ie + 1] - x[ie])
                if len(x) != len(l.x) or not np.allclose(y, l.y, equal_nan=True):
                    pGUIErrorManager.warning(
                        PGText.interpolationPerformedLine
                        % ((np.array(l.x), np.array(l.y)), (x, y))
                    )
            except (IndexError, TypeError):
                pass
            plt.plot(
                x,
                y,
                label=l.label,
                color=l.color,
                ls=l.style,
                marker=l.marker,
                lw=l.width,
            )
        for c in self.contours:
            if c.filled:
                func = plt.contourf
            else:
                func = plt.contour
            options = {}
            if c.levels is not None:
                options["levels"] = c.levels
            if c.extend is not None:
                options["extend"] = c.extend
            options["cmap"] = matplotlib.cm.get_cmap(c.cmap)
            z = np.array(c.z)
            try:
                # corner is problematic
                for cy, cx in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
                    sx = 1 if cx == 0 else -1
                    sy = 1 if cy == 0 else -1
                    if np.isnan(z[cy, cx]):
                        ix1 = cx
                        while np.isnan(z[cy, ix1]):
                            ix1 += sx
                        ix2 = ix1 + sx
                        while np.isnan(z[cy, ix2]):
                            ix2 += sx
                        iy1 = cy
                        while np.isnan(z[iy1, cx]):
                            iy1 += sy
                        iy2 = iy1 + sy
                        while np.isnan(z[iy2, cx]):
                            iy2 += sy
                        z[cy, cx] = 0.5 * (
                            (
                                z[cy, ix1]
                                + (z[cy, ix2] - z[cy, ix1])
                                / (c.x[ix2] - c.x[ix1])
                                * (c.x[cx] - c.x[ix1])
                            )
                            + (
                                z[iy1, cx]
                                + (z[iy2, cx] - z[iy1, cx])
                                / (c.y[iy2] - c.y[iy1])
                                * (c.y[cy] - c.y[iy1])
                            )
                        )
                # some border is problematic
                for cy in (0, -1):
                    for cx, xv in enumerate(c.x):
                        if np.isnan(z[cy, cx]):
                            ix1 = cx
                            while np.isnan(z[cy, ix1]):
                                ix1 -= 1
                            ix2 = cx
                            while np.isnan(z[cy, ix2]):
                                ix2 += 1
                            z[cy, cx] = z[cy, ix1] + (z[cy, ix2] - z[cy, ix1]) / (
                                c.x[ix2] - c.x[ix1]
                            ) * (xv - c.x[ix1])
                for cx in (0, -1):
                    for cy, yv in enumerate(c.y):
                        if np.isnan(z[cy, cx]):
                            iy1 = cy
                            while np.isnan(z[iy1, cx]):
                                iy1 -= 1
                            iy2 = cy
                            while np.isnan(z[iy2, cx]):
                                iy2 += 1
                            z[cy, cx] = z[iy1, cx] + (z[iy2, cx] - z[iy1, cx]) / (
                                c.y[iy2] - c.y[iy1]
                            ) * (yv - c.y[iy1])
                # some central point is problematic
                for cx, xv in enumerate(c.x):
                    for cy, yv in enumerate(c.y):
                        if np.isnan(z[cy, cx]):
                            ix1 = cx
                            while np.isnan(z[cy, ix1]):
                                ix1 -= 1
                            ix2 = cx
                            while np.isnan(z[cy, ix2]):
                                ix2 += 1
                            iy1 = cy
                            while np.isnan(z[iy1, cx]):
                                iy1 -= 1
                            iy2 = cy
                            while np.isnan(z[iy2, cx]):
                                iy2 += 1
                            z[cy, cx] = 0.5 * (
                                (
                                    z[cy, ix1]
                                    + (z[cy, ix2] - z[cy, ix1])
                                    / (c.x[ix2] - c.x[ix1])
                                    * (xv - c.x[ix1])
                                )
                                + (
                                    z[iy1, cx]
                                    + (z[iy2, cx] - z[iy1, cx])
                                    / (c.y[iy2] - c.y[iy1])
                                    * (yv - c.y[iy1])
                                )
                            )
                if not np.allclose(z, c.z, equal_nan=True):
                    pGUIErrorManager.warning(
                        PGText.interpolationPerformedContour
                        % (np.array(c.z), np.array(z))
                    )
            except (IndexError, TypeError):
                pass
            CS = func(c.x, c.y, z, **options)
            if c.hascbar:
                cbar = plt.colorbar(CS)
                cbar.ax.set_ylabel(
                    c.zlabel if self.pSett.zlabel == "" else self.pSett.zlabel,
                    fontsize=self.pSett.axesTextSize,
                )
        plt.xlabel(self.pSett.xlabel, fontsize=self.pSett.axesTextSize)
        plt.ylabel(self.pSett.ylabel, fontsize=self.pSett.axesTextSize)
        plt.xscale(self.pSett.xscale)
        plt.yscale(self.pSett.yscale)
        plt.xlim(self.pSett.xlims)
        plt.ylim(self.pSett.ylims)
        if isinstance(self.title, six.string_types) and self.title.strip() != "":
            plt.title(self.title)
        if self.tight:
            plt.tight_layout()
        if self.legend:
            plt.legend(
                loc=self.legendLoc, ncol=self.legendNcols, fontsize=self.legendTextSize
            )
        plt.text(
            PGText.PArthENoPEWatermark["x"],
            PGText.PArthENoPEWatermark["y"],
            PGText.PArthENoPEWatermark["text"],
            color=PGText.PArthENoPEWatermark["more"]["color"],
            fontsize=PGText.PArthENoPEWatermark["more"]["fontsize"],
            ha=PGText.PArthENoPEWatermark["more"]["ha"],
            rotation=PGText.PArthENoPEWatermark["more"]["rotation"],
            transform=plt.gca().transAxes,
            va=PGText.PArthENoPEWatermark["more"]["va"],
        )
        if savefig and isinstance(savefig, six.string_types):
            plt.savefig(savefig)
        plt.close()
        return fig


######### classes to show plots in a window, button to save the to file
class GridLoader(QFrame):
    """Button and functions to load a grid for plotting"""

    reloadedGrid = Signal()

    def __init__(self, parent=None, mainW=None):
        """Extension of `QFrame.__init__`, also adds a layout
        and the relevant buttons/labels

        Parameter:
            parent: the parent widget, should be a MWPlotPanel instance
            mainW: the main window instance
        """
        super(GridLoader, self).__init__(parent)
        self.mainW = mainW
        self.runner = None

        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(PGLabel(PGText.plotSelectGridText))
        self.openGrid = PGLabelButton(text=PGText.noGrid)
        self.openGrid.setToolTip(PGText.selectGridToolTip)
        self.openGrid.setMaximumWidth(
            QGuiApplication.primaryScreen().availableGeometry().width() * 0.45
        )
        self.openGrid.clicked.connect(self.loadGrid)
        layout.addWidget(self.openGrid)
        layout.addWidget(PGLabel(""))

        self.groupBox = QGroupBox(PGText.plotSelectPlotType, self)
        self.groupBox.setFlat(True)
        self.groupBox.setFont(PGFont())
        layout.addWidget(self.groupBox)
        boxlayout = QHBoxLayout()
        self.groupBox.setLayout(boxlayout)
        for attr, [title, desc] in PGText.plotTypeDescription.items():
            setattr(self, attr, QRadioButton(title, self))
            getattr(self, attr).setToolTip(desc)
            boxlayout.addWidget(getattr(self, attr))
            getattr(self, attr).toggled.connect(parent.updatePlotTypePanel)

    def loadGrid(self):
        """Ask the directory you want to open,
        read all the files it contains and load them
        in a RunPArthENoPE instance
        """
        dirname = askDirName(self, title=PGText.loadGridAsk)
        filename = os.path.join(dirname, Configuration.runSettingsObj)
        if dirname.strip() != "" and os.path.isfile(filename):
            try:
                with open(filename, "rb") as _f:
                    commonParams, gridPoints = pickle.load(_f)
            except (EOFError, FileNotFoundErrorClass, UnicodeDecodeError, ValueError):
                pGUIErrorManager.error(PGText.errorCannotLoadGrid)
            else:
                commonParams = paramsRealPath(commonParams, dirname)
                self.openGrid.setText(dirname)
                self.runner = RunPArthENoPE(commonParams, gridPoints, parent=self.mainW)
                self.runner.readAllResults()
                self.reloadedGrid.emit()
        else:
            pGUIErrorManager.error(PGText.errorCannotFindGrid)


class ShowPlotWidget(QFrame):
    """Panel where the plot will be shown"""

    def __init__(self, parent=None, mainW=None):
        """Builds the layout for the panel where the plot will be shown

        Parameters:
            parent (default None): the parent object,
                which should be a MWPlotPanel instance
            mainW: a MainWindow instance
        """
        super(ShowPlotWidget, self).__init__(parent)
        self.mainW = mainW
        self.fig = None
        self.canvas = None
        layout = QGridLayout(self)
        layout.setSpacing(1)
        self.setLayout(layout)
        self.updatePlots()

    def updatePlots(self, fig=None):
        """Reset the dialog window removing all the previous items
        and create a new canvas for the new figure.

        Parameters:
            fig: the figure to be shown
        """
        if fig is None:
            fig = plt.figure()
            plt.plot(np.nan, np.nan)
            plt.close()
        self.fig = fig
        while True:
            item = self.layout().takeAt(0)
            if item is None:
                break
            del item
        if hasattr(self, "canvas"):
            del self.canvas
        self.canvas = FigureCanvas(self.fig)
        self.layout().addWidget(self.canvas, 0, 0, 1, 2)
        self.canvas.draw()

    def saveAction(self):
        """Save the plot into a file,
        after asking the directory where to save them
        """
        savePath = askSaveFileName(self, PGText.plotWhereToSave)
        if savePath.strip() != "":
            try:
                self.updatePlots(
                    fig=self.mainW.plotPanel.currentPlotContent.doPlot(savefig=savePath)
                )
            except AttributeError:
                pGUIErrorManager.warning("", exc_info=True)
            else:
                pGUIErrorManager.info(PGText.plotSaved)


class PlotSettings(QFrame):
    """Panel where one can edit plot settings and save the plot"""

    _scales = ["linear", "log"]
    _possibleLegendLoc = [
        "best",
        "upper left",
        "upper center",
        "upper right",
        "center left",
        "center",
        "center right",
        "lower left",
        "lower center",
        "lower right",
    ]
    _possibleLegendMaxNcols = 5
    _possibleTextSizes = [
        "xx-small",
        "x-small",
        "small",
        "medium",
        "large",
        "x-large",
        "xx-large",
    ]
    _possibleTextSizesDefault = "medium"
    _default_xlimsd = 0.0
    _default_xlimsu = 1.0
    _default_ylimsd = 0.0
    _default_ylimsu = 1.0
    _default_figsizex = 5.0
    _default_figsizey = 5.0

    def __init__(self, parent=None):
        """Builds the layout for the panel where one can configure the plot settings

        Parameters:
            parent (default None): the parent object
        """
        super(PlotSettings, self).__init__(parent)
        self.automaticSettings = {}

        layout = QVBoxLayout(self)
        self.setLayout(layout)

        self.tabWidget = QTabWidget(self)
        self.tabWidget.setFont(PGFont())
        self.tabWidget.setTabsClosable(False)

        self.layout().addWidget(self.tabWidget)

        self._doAxesTab()
        self._doLegendTab()
        self._doFigureTab()
        self._doButtons()

    def _doAxesTab(self):
        """Create the inputs and fill the tab for axes settings"""
        self.axesTab = QWidget(self)
        layout = QGridLayout(self.axesTab)
        self.axesTab.setLayout(layout)
        self.axesTab.setFont(PGFont())

        # labels
        r = 0
        layout.addWidget(PGLabel(PGText.plotSettXLab), r, 0)
        self.xlabelEd = QLineEdit("")
        self.xlabelEd.setToolTip(PGText.plotSettXLabToolTip)
        layout.addWidget(self.xlabelEd, r, 1, 1, 3)
        layout.addWidget(PGLabel(PGText.plotSettYLab), r, 4)
        self.ylabelEd = QLineEdit("")
        self.ylabelEd.setToolTip(PGText.plotSettYLabToolTip)
        layout.addWidget(self.ylabelEd, r, 5, 1, 3)
        r += 1
        layout.addWidget(PGLabel(PGText.plotSettZLab), r, 0)
        self.zlabelEd = QLineEdit("")
        self.zlabelEd.setToolTip(PGText.plotSettZLabToolTip)
        layout.addWidget(self.zlabelEd, r, 1, 1, 3)

        # scales
        r += 1
        layout.addWidget(PGLabel(PGText.plotSettXScale), r, 0)
        layout.addWidget(PGLabel(PGText.plotSettYScale), r, 4)
        self.xscaleCombo = QComboBox(self)
        self.yscaleCombo = QComboBox(self)
        self.xscaleCombo.setToolTip(PGText.plotSettXScaleToolTip)
        self.yscaleCombo.setToolTip(PGText.plotSettYScaleToolTip)
        for f in self._scales:
            for c in (self.xscaleCombo, self.yscaleCombo):
                c.addItem("%s" % f)
        layout.addWidget(self.xscaleCombo, r, 1, 1, 3)
        layout.addWidget(self.yscaleCombo, r, 5, 1, 3)

        # limits
        r += 1
        layout.addWidget(PGLabel(PGText.plotSettXLims), r, 0)
        layout.addWidget(PGLabel(PGText.smallAsciiArrow), r, 2)
        layout.addWidget(PGLabel(PGText.plotSettYLims), r, 4)
        layout.addWidget(PGLabel(PGText.smallAsciiArrow), r, 6)
        self.xlimsLow = QLineEdit("")
        self.xlimsUpp = QLineEdit("")
        self.ylimsLow = QLineEdit("")
        self.ylimsUpp = QLineEdit("")
        self.xlimsLow.setToolTip(PGText.plotSettXLimsToolTipLow)
        self.xlimsUpp.setToolTip(PGText.plotSettXLimsToolTipUpp)
        self.ylimsLow.setToolTip(PGText.plotSettYLimsToolTipLow)
        self.ylimsUpp.setToolTip(PGText.plotSettYLimsToolTipUpp)
        layout.addWidget(self.xlimsLow, r, 1)
        layout.addWidget(self.xlimsUpp, r, 3)
        layout.addWidget(self.ylimsLow, r, 5)
        layout.addWidget(self.ylimsUpp, r, 7)

        # text size
        r += 1
        layout.addWidget(PGLabel(PGText.plotSettAxesTextSize), r, 0)
        self.axesTextSizeCombo = QComboBox(self)
        self.axesTextSizeCombo.setToolTip(PGText.plotSettAxesTextSizeToolTip)
        for p in self._possibleTextSizes:
            self.axesTextSizeCombo.addItem(p)
        self.axesTextSizeCombo.setCurrentText(self._possibleTextSizesDefault)
        layout.addWidget(self.axesTextSizeCombo, r, 1)

        self.tabWidget.addTab(self.axesTab, PGText.plotSettTabTitleAxes)

    def _doButtons(self):
        """Create and insert the buttons below the tabs"""
        self.refreshButton = PGPushButton(PGText.plotSettRefreshImage)
        self.refreshButton.setToolTip(PGText.plotSettRefreshImageToolTip)
        self.revertButton = PGPushButton(PGText.plotSettRevertImage)
        self.revertButton.setToolTip(PGText.plotSettRevertImageToolTip)
        self.resetButton = PGPushButton(PGText.plotSettResetImage)
        self.resetButton.setToolTip(PGText.plotSettResetImageToolTip)
        self.saveButton = PGPushButton(PGText.plotSettSaveImage)
        self.saveButton.setToolTip(PGText.plotSettSaveImageToolTip)
        self.exportButton = PGPushButton(PGText.plotSettExportImage)
        self.exportButton.setToolTip(PGText.plotSettExportImageToolTip)
        self._buttonsA = QWidget(self)
        self._buttonsALayout = QHBoxLayout(self._buttonsA)
        self._buttonsA.setLayout(self._buttonsALayout)
        self._buttonsA.layout().addWidget(self.refreshButton)
        self._buttonsA.layout().addWidget(self.revertButton)
        self._buttonsA.layout().addWidget(self.resetButton)
        self._buttonsB = QWidget(self)
        self._buttonsBLayout = QHBoxLayout(self._buttonsB)
        self._buttonsB.setLayout(self._buttonsBLayout)
        self._buttonsB.layout().addWidget(self.saveButton)
        self._buttonsB.layout().addWidget(self.exportButton)
        self.layout().addWidget(self._buttonsA)
        self.layout().addWidget(self._buttonsB)

    def _doFigureTab(self):
        """Fill the content of the tab where the user can define
        some general settings for the figure
        """
        self.figureTab = QWidget(self)
        layout = QGridLayout(self.figureTab)
        self.figureTab.setLayout(layout)
        self.figureTab.setFont(PGFont())

        # title
        r = 0
        layout.addWidget(PGLabel(PGText.plotSettTitleLab), r, 0, 1, 3)
        self.titleEd = QLineEdit("")
        self.titleEd.setToolTip(PGText.plotSettTitleLabToolTip)
        layout.addWidget(self.titleEd, r + 1, 0, 1, 3)

        # figure size and warning
        r += 2
        layout.addWidget(PGLabel(PGText.plotSettFigSize), r, 0, 1, 3)
        r += 1
        self.figsizex = QLineEdit("")
        self.figsizey = QLineEdit("")
        self.figsizex.setToolTip(PGText.plotSettFigSizeToolTipH)
        self.figsizey.setToolTip(PGText.plotSettFigSizeToolTipV)
        layout.addWidget(self.figsizex, r, 0)
        layout.addWidget(PGLabel(PGText.comma), r, 1)
        layout.addWidget(self.figsizey, r, 2)
        r += 1
        layout.addWidget(PGLabel(PGText.plotSettFigSizeWarning), r, 0, 1, 3)

        # tight layout
        r += 1
        self.tightCheck = QCheckBox(PGText.plotSettTight, self)
        self.tightCheck.setToolTip(PGText.plotSettTightToolTip)
        self.tightCheck.setChecked(True)
        layout.addWidget(self.tightCheck, r, 0, 1, 3)

        self.tabWidget.addTab(self.figureTab, PGText.plotSettTabTitleFigure)

    def _doLegendTab(self):
        """Fill the content of the tab where the user can define
        some legend settings"""
        self.legendTab = QWidget(self)
        layout = QGridLayout(self.legendTab)
        self.legendTab.setLayout(layout)
        self.legendTab.setFont(PGFont())

        # activate legend
        r = 0
        self.legendCheck = QCheckBox(PGText.plotSettLegend, self)
        self.legendCheck.setToolTip(PGText.plotSettLegendToolTip)
        self.legendCheck.setChecked(True)
        layout.addWidget(self.legendCheck, r, 0)

        # legend location
        r += 1
        layout.addWidget(PGLabel(PGText.plotSettLegendLoc), r, 0)
        self.legendLocCombo = QComboBox(self)
        self.legendLocCombo.setToolTip(PGText.plotSettLegendLocToolTip)
        for p in self._possibleLegendLoc:
            self.legendLocCombo.addItem(p)
        layout.addWidget(self.legendLocCombo, r, 1)

        # number of columns
        r += 1
        layout.addWidget(PGLabel(PGText.plotSettLegendNumCols), r, 0)
        self.legendNumColsCombo = QComboBox(self)
        self.legendNumColsCombo.setToolTip(PGText.plotSettLegendNumColsToolTip)
        for n in range(self._possibleLegendMaxNcols):
            self.legendNumColsCombo.addItem("%d" % (n + 1))
        layout.addWidget(self.legendNumColsCombo, r, 1)

        # text size
        r += 1
        layout.addWidget(PGLabel(PGText.plotSettLegendTextSize), r, 0)
        self.legendTextSizeCombo = QComboBox(self)
        self.legendTextSizeCombo.setToolTip(PGText.plotSettLegendTextSizeToolTip)
        for p in self._possibleTextSizes:
            self.legendTextSizeCombo.addItem(p)
        self.legendTextSizeCombo.setCurrentText(self._possibleTextSizesDefault)
        layout.addWidget(self.legendTextSizeCombo, r, 1)

        self.tabWidget.addTab(self.legendTab, PGText.plotSettTabTitleLegend)

    def clearCurrent(self):
        """Reset the plot settings to the default values"""
        self.axesTextSize = PlotSettings._possibleTextSizesDefault
        self.figsize = [self._default_figsizex, self._default_figsizey]
        self.legend = True
        self.legendLoc = self._possibleLegendLoc[0]
        self.legendNcols = 1
        self.legendTextSize = PlotSettings._possibleTextSizesDefault
        self.tight = True
        self.title = ""
        self.updateAutomaticSettings(
            {
                "xlabel": "",
                "xlims": (self._default_xlimsd, self._default_xlimsu),
                "xscale": self._scales[0],
                "ylabel": "",
                "ylims": (self._default_ylimsd, self._default_ylimsu),
                "yscale": self._scales[0],
                "zlabel": "",
            },
            force=True,
        )

    def restoreAutomaticSettings(self):
        """Restore the plot properties
        to the most recent available automatic settings
        """
        for k, v in self.automaticSettings.items():
            setattr(self, k, v)

    def updateAutomaticSettings(self, new, force=False):
        """Read a dictionary and use it to update automaticSettings
        and the corresponding PlotSettings properties

        Parameters:
            new: the dictionary containing the new automatic values
            force (default False): if True, overwrite the current settings
                regardless of their values
        """
        for k, v in new.items():
            if (
                force
                or k not in self.automaticSettings.keys()
                or self.automaticSettings[k] == getattr(self, k)
            ):
                setattr(self, k, v)
                if v == getattr(self, k):
                    self.automaticSettings[k] = v
                else:
                    pErrorManager.logger.debug("Value not accepted for %s: %s" % (k, v))
                    self.automaticSettings[k] = getattr(self, k)
            else:
                self.automaticSettings[k] = v

    @property
    def axesTextSize(self):
        """Get the axesTextSize property"""
        return self.axesTextSizeCombo.currentText()

    @axesTextSize.setter
    def axesTextSize(self, val):
        """Set the axesTextSize property"""
        if val in self._possibleTextSizes:
            self.axesTextSizeCombo.setCurrentText("%s" % val)

    @property
    def figsize(self):
        """Get the figsize property (a tuple),
        after checking that the corresponding fields contain float
        """
        x = self.figsizex.text()
        y = self.figsizey.text()
        try:
            xs = float(x)
        except ValueError:
            mainlogger.error(
                "Invalid figsize horizontal component: %s. Use %s"
                % (x, self._default_figsizex)
            )
            xs = self._default_figsizex
            self.figsizex.setText("%s" % xs)
        try:
            ys = float(y)
        except ValueError:
            mainlogger.error(
                "Invalid figsize vertical component: %s. Use %s"
                % (y, self._default_figsizey)
            )
            ys = self._default_figsizey
            self.figsizey.setText("%s" % ys)
        return (xs, ys)

    @figsize.setter
    def figsize(self, val):
        """Set the figsize property, after checking that
        the argument is a list/tuple and has two positive float values
        """
        if not (isinstance(val, (list, tuple))) or len(val) != 2:
            mainlogger.error(
                "Invalid length of figsize: %s. Use (%s, %s)"
                % (val, self._default_figsizex, self._default_figsizey)
            )
            val = [self._default_figsizex, self._default_figsizey]
        val = list(val)
        try:
            assert float(val[0]) > 0
        except (AssertionError, ValueError):
            mainlogger.error(
                "Invalid figsize horizontal component: %s. Use %s"
                % (val[0], self._default_figsizey)
            )
            val[0] = self._default_figsizex
        try:
            assert float(val[1]) > 0
        except (AssertionError, ValueError):
            mainlogger.error(
                "Invalid figsize vertical component: %s. Use %s"
                % (val[1], self._default_figsizey)
            )
            val[1] = self._default_figsizey
        self.figsizex.setText("%s" % val[0])
        self.figsizey.setText("%s" % val[1])

    @property
    def legend(self):
        """Get the legend property"""
        return self.legendCheck.isChecked()

    @legend.setter
    def legend(self, val):
        """Set the legend property"""
        if val:
            self.legendCheck.setChecked(True)
        else:
            self.legendCheck.setChecked(False)

    @property
    def legendLoc(self):
        """Get the legendLoc property"""
        return self.legendLocCombo.currentText()

    @legendLoc.setter
    def legendLoc(self, val):
        """Set the legendLoc property"""
        if val in self._possibleLegendLoc:
            self.legendLocCombo.setCurrentText("%s" % val)

    @property
    def legendNcols(self):
        """Get the legendNcols property"""
        return int(self.legendNumColsCombo.currentText())

    @legendNcols.setter
    def legendNcols(self, val):
        """Set the legendNcols property"""
        try:
            int(val)
        except ValueError:
            return
        if int(val) in [x + 1 for x in range(self._possibleLegendMaxNcols)]:
            self.legendNumColsCombo.setCurrentText("%s" % val)

    @property
    def legendTextSize(self):
        """Get the legendTextSize property"""
        return self.legendTextSizeCombo.currentText()

    @legendTextSize.setter
    def legendTextSize(self, val):
        """Set the legendTextSize property"""
        if val in self._possibleTextSizes:
            self.legendTextSizeCombo.setCurrentText("%s" % val)

    @property
    def tight(self):
        """Get the tight property"""
        return self.tightCheck.isChecked()

    @tight.setter
    def tight(self, val):
        """Set the tight property"""
        if val:
            self.tightCheck.setChecked(True)
        else:
            self.tightCheck.setChecked(False)

    @property
    def title(self):
        """Get the title property"""
        return self.titleEd.text()

    @title.setter
    def title(self, val):
        """Set the title property"""
        self.titleEd.setText("%s" % val)

    @property
    def xlabel(self):
        """Get the xlabel property"""
        return self.xlabelEd.text()

    @xlabel.setter
    def xlabel(self, val):
        """Set the xlabel property"""
        self.xlabelEd.setText("%s" % val)

    @property
    def xlims(self):
        """Get the xlims property, after checking
        that the corresponding fields have float values
        """
        x = self.xlimsLow.text()
        y = self.xlimsUpp.text()
        try:
            xs = float(x)
        except ValueError:
            mainlogger.error(
                "Invalid lower xlims: %s. Use %s" % (x, self._default_xlimsd)
            )
            xs = self._default_xlimsd
            self.xlimsLow.setText("%s" % xs)
        try:
            ys = float(y)
        except ValueError:
            mainlogger.error(
                "Invalid upper xlims: %s. Use %s" % (y, self._default_xlimsu)
            )
            ys = self._default_xlimsu
            self.xlimsUpp.setText("%s" % ys)
        return (xs, ys)

    @xlims.setter
    def xlims(self, val):
        """Set the xlims property, after checking that the argument
        is a tuple/list of two float values
        """
        if not (isinstance(val, (list, tuple))) or len(val) != 2:
            mainlogger.error("Invalid length of xlims: %s" % val)
            val = (self._default_xlimsd, self._default_xlimsu)
        try:
            lo = float(val[0])
        except ValueError:
            mainlogger.error("Invalid lower xlims: %s. Use 0" % val[0])
            lo = 0.0
        try:
            up = float(val[1])
        except ValueError:
            mainlogger.error("Invalid upper xlims: %s. Use 1" % val[1])
            up = 1.0
        self.xlimsLow.setText("%s" % lo)
        self.xlimsUpp.setText("%s" % up)

    @property
    def xscale(self):
        """Get the xscale property"""
        return self.xscaleCombo.currentText()

    @xscale.setter
    def xscale(self, val):
        """Set the xscale property,
        after checking that the passed value is valid
        """
        if val not in self._scales:
            mainlogger.error("Invalid scale: %s. Use '%s'" % (val, self._scales[0]))
            val = self._scales[0]
        self.xscaleCombo.setCurrentText(val)

    @property
    def ylabel(self):
        """Get the ylabel property"""
        return self.ylabelEd.text()

    @ylabel.setter
    def ylabel(self, val):
        """Set the ylabel property"""
        self.ylabelEd.setText("%s" % val)

    @property
    def ylims(self):
        """Get the ylims property, after checking
        that the corresponding fields have float values
        """
        x = self.ylimsLow.text()
        y = self.ylimsUpp.text()
        try:
            xs = float(x)
        except ValueError:
            mainlogger.error(
                "Invalid lower ylims: %s. Use %s" % (x, self._default_ylimsd)
            )
            xs = self._default_ylimsd
            self.ylimsLow.setText("%s" % xs)
        try:
            ys = float(y)
        except ValueError:
            mainlogger.error(
                "Invalid upper ylims: %s. Use %s" % (y, self._default_ylimsu)
            )
            ys = self._default_ylimsu
            self.ylimsUpp.setText("%s" % ys)
        return (xs, ys)

    @ylims.setter
    def ylims(self, val):
        """Set the ylims property, after checking that the argument
        is a tuple/list of two float values
        """
        if not (isinstance(val, (list, tuple))) or len(val) != 2:
            mainlogger.error("Invalid length of ylims: %s" % val)
            val = (self._default_ylimsd, self._default_ylimsu)
        try:
            lo = float(val[0])
        except ValueError:
            mainlogger.error("Invalid lower ylims: %s. Use 0" % val[0])
            lo = 0.0
        try:
            up = float(val[1])
        except ValueError:
            mainlogger.error("Invalid upper ylims: %s. Use 1" % val[1])
            up = 1.0
        self.ylimsLow.setText("%s" % lo)
        self.ylimsUpp.setText("%s" % up)

    @property
    def yscale(self):
        """Get the yscale property"""
        return self.yscaleCombo.currentText()

    @yscale.setter
    def yscale(self, val):
        """Set the yscale property,
        after checking that the passed value is valid
        """
        if val not in self._scales:
            mainlogger.error("Invalid scale: %s. Use '%s'" % (val, self._scales[0]))
            val = self._scales[0]
        self.yscaleCombo.setCurrentText(val)

    @property
    def zlabel(self):
        """Get the zlabel property"""
        return self.zlabelEd.text()

    @zlabel.setter
    def zlabel(self, val):
        """Set the zlabel property"""
        self.zlabelEd.setText("%s" % val)


class PlotPlaceholderPanel(QFrame):
    """Panel which is only shown before a plot type is selected"""

    def __init__(self, parent=None, mainW=None):
        """Extension of `QFrame.__init__`, adds a layout and
        a QTextEdit to the placeholder frame

        Parameter:
            parent (default None): the parent widget
            mainW (default None): a MainWindow instance
        """
        super(PlotPlaceholderPanel, self).__init__(parent)
        self.mainW = mainW

        layout = QHBoxLayout()
        self.setLayout(layout)

        self.text = QTextEdit("")
        self.text.setFont(PGFont())
        self.text.setReadOnly(True)

        layout.addWidget(self.text)
        self.text.setText(PGText.plotPlaceholderText)

    def clearCurrent(self):
        """Will do nothing"""
        pass

    def reloadModel(self):
        """Will do nothing"""
        pass

    def updatePlotProperties(self):
        """Will do nothing"""
        pass


class PGEmptyTableModel(QAbstractTableModel):
    """Extension of `QAbstractTableModel` with basic definitions used
    in other table models, must be extended to be used!
    """

    def columnCount(self, parent=None):
        """Count the columns of the given model based on the header

        Parameter:
            parent: `QModelIndex` (required by the parent signature)

        Output:
            the number of columns or zero (if error occurred)
        """
        try:
            return len(self.header)
        except TypeError:
            return 0

    def flags(self, index):
        """Determine the flags of a given item

        Parameter:
            index: a `QModelIndex`

        Output:
            None if the index is not valid
            if `self.ask` and first column, show checkboxes:
            Qt.ItemIsUserCheckable | Qt.ItemIsEditable |
                Qt.ItemIsEnabled | Qt.ItemIsSelectable
            all the other cases: Qt.ItemIsEnabled | Qt.ItemIsSelectable
        """
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def rowCount(self, parent=None):
        """Count the rows of the given model based on the header

        Parameter:
            parent: `QModelIndex` (required by the parent signature)

        Output:
            the number od rows or zero (if error occurred)
        """
        try:
            return len(self.dataList)
        except TypeError:
            return 0


class PGListLinesModel(PGEmptyTableModel):
    """Extension of `PGEmptyTableModel`, used for lists of grid points"""

    def __init__(self, dataList, parent, *args):
        """Constructor, based on `PGEmptyTableModel.__init__`

        Parameters:
            datalist: the list of data which will enter the table
            parent: the parent widget
        """
        self.dataList = dataList
        self.header = [
            "description",
            "label",
            "color",
            "style",
            "marker",
            "width",
            "Delete",
            "Move up",
            "Move down",
        ]
        self.headerToolTips = [
            "Description of the line, including the list of "
            + "physical parameters and the considered one in the current plot",
            "Label of the line in the legend",
            "Line color to use in the plot",
            "Line style to use in the plot",
            "Point marker style to use in the plot",
            "Line width to use in the plot",
            "Double click this cell to remove the line from the current plot",
            "Double click this cell to move the line up in the order",
            "Double click this cell to move the line down in the order",
        ]
        PGEmptyTableModel.__init__(self, parent, *args)

    def addLineToData(self, line):
        """Add one line to the data list and update the layout

        Parameters:
            line: the new line to append to the dataList
        """
        self.layoutAboutToBeChanged.emit()
        self.dataList.append(line)
        self.layoutChanged.emit()

    def data(self, index, role):
        """Return the cell data for the given index and role

        Parameters:
            index: the `QModelIndex` for which the data are required
            role: the desired Qt role for the data

        Output:
            None if the index or the role are not valid,
            the cell content or properties otherwise
        """
        if not index.isValid():
            return None
        row = index.row()
        column = index.column()
        try:
            self.header[column]
        except IndexError:
            mainlogger.exception(PGText.errorCannotFindIndex)
            return None
        if self.header[column] == "Delete":
            if role == Qt.DecorationRole:
                return QPixmap(":/images/delete.png").scaledToHeight(
                    Configuration.rowImageHeight
                )
            elif role == Qt.DisplayRole:
                return ""
            else:
                return None
        elif self.header[column] == "Move down":
            if role == Qt.DecorationRole and row != self.rowCount() - 1:
                return QPixmap(":/images/arrow-down.png").scaledToHeight(
                    Configuration.rowImageHeight
                )
            elif role == Qt.DisplayRole:
                return ""
            else:
                return None
        elif self.header[column] == "Move up":
            if role == Qt.DecorationRole and row != 0:
                return QPixmap(":/images/arrow-up.png").scaledToHeight(
                    Configuration.rowImageHeight
                )
            elif role == Qt.DisplayRole:
                return ""
            else:
                return None
        try:
            value = "%s" % getattr(self.dataList[row], self.header[column])
        except (IndexError, AttributeError):
            mainlogger.exception("")
            return None
        if role == Qt.DisplayRole:
            return value
        return None

    def deleteLine(self, ix):
        """Delete one line from the data list and update the layout

        Parameters:
            ix: the index of the line to delete
        """
        self.layoutAboutToBeChanged.emit()
        try:
            del self.dataList[ix]
        except IndexError:
            mainlogger.debug(PGText.warningMissingLine)
        self.layoutChanged.emit()

    def headerData(self, col, orientation, role):
        """Obtain column name if correctly asked

        Parameters:
            col: the column index in `self.header`
            orientation: int from `Qt.Orientation`
            role: int from `Qt.ItemDataRole`

        Output:
            the column name or `None`
        """
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.header[col]
        elif orientation == Qt.Horizontal and role == Qt.ToolTipRole:
            return self.headerToolTips[col]
        return None

    def replaceLine(self, line, ix):
        """Replace one line in the data list and update the layout

        Parameters:
            line: the new line to replace in the dataList
            ix: the index of the line to delete
        """
        self.layoutAboutToBeChanged.emit()
        try:
            self.dataList[ix] = line
        except IndexError:
            mainlogger.warning(PGText.warningMissingLine)
        self.layoutChanged.emit()


class Chi2ParamConfig(object):
    """Class that contains the six objects used to construct the panel
    where the chi2 is configured
    """

    def __init__(self, paramCombo, chi2Mean, chi2Std):
        """Store the input fields and create the associated labels"""
        if isinstance(paramCombo, QComboBox):
            self.paramCombo = paramCombo
        else:
            raise TypeError("%s is not QComboBox" % type(paramCombo))
        if isinstance(chi2Mean, QLineEdit):
            self.chi2Mean = chi2Mean
        else:
            raise TypeError("%s is not QLineEdit" % type(chi2Mean))
        if isinstance(chi2Std, QLineEdit):
            self.chi2Std = chi2Std
        else:
            raise TypeError("%s is not QLineEdit" % type(chi2Std))
        self.paramLabel = PGLabel(PGText.plotSelectNuclide)
        self.meanLabel = PGLabel(PGText.mean)
        self.stdLabel = PGLabel(PGText.standardDeviation)
        self.paramLabel.setWordWrap(False)
        self.meanLabel.setWordWrap(False)
        self.stdLabel.setWordWrap(False)


class AddFromPoint(QDialog):
    """Some common methods that are used for adding a QFrame.HLine
    and the inputs for configuring the chi2 in the relevant dialogs
    """

    def _addChi2Line(self):
        """Clean the current layout, create a new set of chi2 inputs
        and generate again the layout of the chi2 stacked widget
        """
        while True:
            item = self.chi2Widget.layout().takeAt(0)
            if item is None:
                break
            del item
        self._newChi2Widget()
        self._fillChi2Widget()

    def _addHLine(self, r):
        """Add an horizontal line in the given position (row 'r')

        Parameter:
            r: the row index
        """
        hline = QFrame()
        hline.setMinimumHeight(2)
        hline.setFrameShape(QFrame.HLine)
        self.layout().addWidget(hline, r, 0, 1, 4)

    def _doAddChi2Button(self):
        """Generate the PushButton for adding new chi2 lines"""
        self.addChi2Line = PGPushButton(PGText.plotSelectChi2AddLine)
        self.addChi2Line.setToolTip(PGText.plotSelectChi2AddLineToolTip)
        self.addChi2Line.clicked.connect(self._addChi2Line)

    def _doGroupBox(self):
        """Generate the groupBox and the radio buttons"""
        self.groupBox = QGroupBox(PGText.plotSelectChi2Type, self)
        self.groupBox.setFlat(True)
        self.groupBox.setFont(PGFont())
        self.groupBox.setLayout(QHBoxLayout())
        for attr, [title, desc] in PGText.plotSelectChi2TypeContents.items():
            setattr(self, attr, QRadioButton(title, self))
            getattr(self, attr).setToolTip(desc)
            self.groupBox.layout().addWidget(getattr(self, attr))
        self.chi2.setChecked(True)

    def _fillChi2Widget(self):
        """Fill the widget layout with the fields for configuring the chi2"""
        r = 0
        for r, o in enumerate(self.chi2ParamConfigs):
            for i, f in enumerate(
                (
                    "paramLabel",
                    "paramCombo",
                    "meanLabel",
                    "chi2Mean",
                    "stdLabel",
                    "chi2Std",
                )
            ):
                self.chi2Widget.layout().addWidget(getattr(o, f), r, i)
        if not hasattr(self, "addChi2Line"):
            self._doAddChi2Button()
        self.chi2Widget.layout().addWidget(self.addChi2Line, r + 1, 0, 1, 2)
        if not hasattr(self, "groupBox"):
            self._doGroupBox()
        self.chi2Widget.layout().addWidget(self.groupBox, r + 2, 0, 1, 6)

    def _getChi2Conv(self, chi2):
        """Convert the chi2 to likelihood or similar, depending on which
        radio button is currently checked in the dialog window

        Parameter:
            chi2: the list of chi2 values to be eventually converted
        """
        if self.lh.isChecked():
            return np.exp(-chi2 / 2.0)
        elif self.nllh.isChecked():
            return chi2 / 2.0
        elif self.pllh.isChecked():
            return -chi2 / 2.0
        return chi2

    def _getChi2Type(self, chi2):
        """Return the type of chi2 that will be stored
        in order to determine the axes label, depending on which
        radio button is currently checked in the dialog window

        Parameter:
            chi2: whether asking the chi2 is enabled or not in the field
        """
        if not chi2:
            return False
        if self.chi2.isChecked():
            return "chi2"
        if self.lh.isChecked():
            return "lh"
        elif self.nllh.isChecked():
            return "nllh"
        elif self.pllh.isChecked():
            return "pllh"
        return False

    def _getDescLineParam(self):
        """Return a description of the chi2 type depending on which
        radio button is currently checked in the dialog window
        """
        if self.lh.isChecked():
            return "lh(%s) "
        elif self.nllh.isChecked():
            return "-llh(%s) "
        elif self.pllh.isChecked():
            return "+llh(%s) "
        return "chi2(%s) "

    def _newChi2Widget(self):
        """Generate a new set of inputs for configuring the chi2"""
        c = Chi2ParamConfig(self._nuclideCombo(), QLineEdit(""), QLineEdit(""))
        c.chi2Mean.setToolTip(PGText.plotChi2AskMeanToolTip)
        c.chi2Std.setToolTip(PGText.plotChi2AskStdToolTip)
        self.chi2ParamConfigs.append(c)

    def _updateStack(self, state):
        """When the chi2 QCheckBox changes state,
        switch the visible content of the QStackedWidget
        """
        if state == Qt.Checked:
            self.stackWidget.setCurrentIndex(1)
        else:
            self.stackWidget.setCurrentIndex(0)


class AddLineFromPoint(AddFromPoint):
    """Dialog that asks which nuclide to use and various plot options"""

    def __init__(self, mainW, ptrow, desc, paramix, header, output, line=None):
        """Prepare a dialog to ask for line properties before
        adding it to the plot

        Parameters:
            mainW: a MainWindow instance
            ptrow: index of the line in the underlying model
            desc: description of the line
            paramix: index of the required parameter in the x axis
            header: header of the data, for detecting
                the available parameters for the plot
            output: list of data which will be used for the plot
            line (default None): a PGLine for editing its properties
        """
        if not hasattr(self, "headerCut"):
            self.headerCut = 0
        if not hasattr(self, "askChi2"):
            self.askChi2 = False
        self.mainW = mainW
        self.pointRow = ptrow
        self.desc = desc if line is None else line.description
        self.paramix = paramix
        self.header = header
        self.output = output
        self.line = line
        AddFromPoint.__init__(self, mainW)
        self.chi2ParamConfigs = []

        # create the stacked widgets
        self.stackWidget = QStackedWidget(self)
        self.stackWidget.setFont(PGFont())
        self.paramWidget = QWidget(self)
        self.paramWidget.setLayout(QHBoxLayout(self.paramWidget))
        self.stackWidget.addWidget(self.paramWidget)
        self.chi2Widget = QWidget(self)
        self.chi2Widget.setLayout(QGridLayout(self.chi2Widget))
        self.stackWidget.addWidget(self.chi2Widget)

        layout = QGridLayout()
        self.setLayout(layout)
        self.setWindowTitle(PGText.plotSelectNuclideLine)

        r = 0
        layout.addWidget(PGLabel(self.desc), r, 0, 1, 4)

        self.chi2Check = QCheckBox("", self)
        self.chi2Check.setToolTip(PGText.plotChi2AskCheckToolTip)
        self.chi2Check.setChecked(False if line is None else bool(line.chi2))
        if line is None:
            r += 1
            self._addHLine(r)

            l = PGLabel(PGText.plotSelectNuclide)
            l.setWordWrap(False)
            self.paramWidget.layout().addWidget(l)
            self.nuclideCombo = self._nuclideCombo()
            self.paramWidget.layout().addWidget(self.nuclideCombo)

            if self.askChi2:
                self._newChi2Widget()
                # this checkbox is connected to a function that switches between the two stacked widgets
                r += 1
                layout.addWidget(PGLabel(PGText.plotChi2AskLabel), r, 0, 1, 3)
                self.chi2Check.stateChanged.connect(self._updateStack)
                layout.addWidget(self.chi2Check, r, 3)

                self._fillChi2Widget()
            else:
                self.chi2Check.hide()
        else:
            self.chi2Check.hide()

        r += 1
        layout.addWidget(self.stackWidget, r, 0, 1, 4)

        r += 1
        self._addHLine(r)

        r += 1
        layout.addWidget(PGLabel(PGText.plotLineLabel), r, 0)
        self.labelInput = QLineEdit("" if line is None else line.label)
        self.labelInput.setToolTip(PGText.plotLineLabelToolTip)
        layout.addWidget(self.labelInput, r, 1, 1, 3)

        r += 1
        layout.addWidget(PGLabel(PGText.plotLineColor), r, 0)
        self.colorInput = QComboBox(self)
        if line:
            self.colorInput.addItem(line.color)
        for k in ("k", "r", "b", "g", "m", "c", "y", "other (edit)"):
            if not (line and k == line.color):
                self.colorInput.addItem(k)
        self.colorInput.setEditable(True)
        self.colorInput.setToolTip(PGText.plotLineColorToolTip)
        layout.addWidget(self.colorInput, r, 1)
        layout.addWidget(PGLabel(PGText.plotLineStyle), r, 2)
        self.styleCombo = QComboBox(self)
        self.styleCombo.setToolTip(PGText.plotLineStyleToolTip)
        for f in styleOptions:
            self.styleCombo.addItem(f)
        if line is not None:
            self.styleCombo.setCurrentText(line.style)
        layout.addWidget(self.styleCombo, r, 3)

        r += 1
        layout.addWidget(PGLabel(PGText.plotLineWidth), r, 0)
        self.widthInput = QLineEdit("1" if line is None else "%s" % line.width)
        self.widthInput.setToolTip(PGText.plotLineWidthToolTip)
        layout.addWidget(self.widthInput, r, 1)
        layout.addWidget(PGLabel(PGText.plotLineMarker), r, 2)
        self.markerCombo = QComboBox(self)
        self.markerCombo.setToolTip(PGText.plotLineMarkerToolTip)
        for f in markerOptions:
            self.markerCombo.addItem(f)
        if line is not None:
            self.markerCombo.setCurrentText(line.marker)
        layout.addWidget(self.markerCombo, r, 3)

        r += 1
        self.acceptButton = PGPushButton(PGText.buttonAccept)
        self.acceptButton.clicked.connect(self.testAccept)
        layout.addWidget(self.acceptButton, r, 0, 1, 2)
        self.cancelButton = PGPushButton(PGText.buttonCancel)
        self.cancelButton.clicked.connect(self.reject)
        layout.addWidget(self.cancelButton, r, 2, 1, 2)

    def _nuclideCombo(self):
        """Create a QComboBox with all the possible quantity options
        and the correct tooltip
        """
        combo = QComboBox(self)
        combo.setToolTip(PGText.plotSelectNuclideToolTip)
        for f in self.header[self.headerCut :]:
            combo.addItem("%s" % f)
        return combo

    def getLine(self):
        """Create a PGLine with the information from the form
        and the previous line (if there is one)

        Output:
            a PGLine
        """
        if self.line is None:
            if self.askChi2 and self.chi2Check.isChecked():
                chi2 = True
                values = []
                params = ""
                c2 = 0.0
                for r, o in enumerate(self.chi2ParamConfigs):
                    params += self._getDescLineParam() % o.paramCombo.currentText()
                    c2 += chi2func(
                        self.output[:, self.headerCut + o.paramCombo.currentIndex()],
                        float(o.chi2Mean.text()),
                        float(o.chi2Std.text()),
                    )
                out = self._getChi2Conv(c2)
                desc = self.desc + ": %s" % params
            else:
                chi2 = False
                out = self.output[:, self.headerCut + self.nuclideCombo.currentIndex()]
                desc = self.desc + ": %s" % self.nuclideCombo.currentText()
            line = PGLine(
                self.output[:, self.paramix],
                out,
                self.pointRow,
                d=desc,
                l=self.labelInput.text(),
                c=self.colorInput.currentText(),
                s=self.styleCombo.currentText(),
                m=self.markerCombo.currentText(),
                w=float(self.widthInput.text()),
                xl=self.header[self.paramix],
                yl=self.header[self.headerCut + self.nuclideCombo.currentIndex()],
                c2=self._getChi2Type(chi2),
            )
        else:
            line = PGLine(
                self.line.x,
                self.line.y,
                self.line.pointRow,
                d=self.line.description,
                l=self.labelInput.text(),
                c=self.colorInput.currentText(),
                s=self.styleCombo.currentText(),
                m=self.markerCombo.currentText(),
                w=float(self.widthInput.text()),
                xl=self.line.xlabel,
                yl=self.line.ylabel,
                c2=self.line.chi2,
            )
        return line

    def testAccept(self):
        """Verify that the form contains valid line properties
        (line width, chi2 parameters, color)
        """
        try:
            assert float(self.widthInput.text()) > 0
        except (AssertionError, ValueError):
            pGUIErrorManager.warning(PGText.plotInvalidWidth % self.widthInput.text())
            return
        if self.askChi2 and self.chi2Check.isChecked():
            for r, o in enumerate(self.chi2ParamConfigs):
                try:
                    float(o.chi2Mean.text())
                except ValueError:
                    pGUIErrorManager.warning(PGText.plotInvalidMean % o.chi2Mean.text())
                    return
                try:
                    assert float(o.chi2Std.text()) > 0
                except (AssertionError, ValueError):
                    pGUIErrorManager.warning(
                        PGText.plotInvalidStddev % o.chi2Std.text()
                    )
                    return
        if not matplotlib.colors.is_color_like(self.colorInput.currentText()):
            pGUIErrorManager.warning(
                PGText.plotInvalidColor % self.colorInput.currentText()
            )
            return
        self.accept()


class PGListPointsModel(PGEmptyTableModel):
    """Extension of `PGEmptyTableModel`, used for lists of grid points"""

    def __init__(self, dataList, parent, mainW, *args):
        """Define header and other attributes

        Parameters:
            datalist: the list of points
            parent: the parent widget
            mainW: a MainWindow instance
        """
        self.mainW = mainW
        self.dataList = dataList
        self.inUse = [[] for x in self.dataList]
        self.header = ["Used"] + list([p for p in Parameters.paramOrder])
        self.headerToolTips = [
            "If marked, one of the outputs of the corresponding line "
            + "is used in the current plot"
        ] + list([p for p in Parameters.paramOrder])
        PGEmptyTableModel.__init__(self, parent, *args)

    def data(self, index, role):
        """Return the cell data for the given index and role

        Parameters:
            index: the `QModelIndex` for which the data are required
            role: the desired Qt role for the data

        Output:
            None if the index or the role are not valid,
            the cell content or properties otherwise
        """
        if not index.isValid():
            return None
        row = index.row()
        column = index.column()
        hasImg = False
        value = ""
        try:
            self.inUse[row]
        except IndexError:
            mainlogger.exception(PGText.errorCannotFindIndex)
            return None
        if column == 0:
            if self.inUse[row]:
                hasImg = True
                value = QPixmap(":/images/dialog-ok-apply.png").scaledToHeight(
                    Configuration.rowImageHeight
                )
        else:
            try:
                item = self.dataList[row, column - 1]
            except IndexError:
                mainlogger.exception(PGText.errorCannotFindIndex)
                return None
            except TypeError:
                mainlogger.exception(
                    PGText.warningWrongType % ("self.dataList", "np.ndarray")
                )
                return None
            try:
                value = Configuration.formatParam % item
            except TypeError:
                value = "%s" % item
        if role == Qt.DecorationRole and hasImg:
            return value
        elif role == Qt.DisplayRole and not hasImg:
            return value
        return None

    def headerData(self, col, orientation, role):
        """Obtain column name if correctly asked

        Parameters:
            col: the column index in `self.header`
            orientation: int from `Qt.Orientation`
            role: int from `Qt.ItemDataRole`

        Output:
            the column name or `None`
        """
        if orientation == Qt.Horizontal:
            if col != 0 and role == Qt.DecorationRole:
                return self.mainW.parameters.all[self.header[col]].fig
            elif col == 0 and role == Qt.DisplayRole:
                return self.header[col]
            if role == Qt.ToolTipRole:
                return self.headerToolTips[col]
        return None

    def replaceDataList(self, dataList):
        """Replace the data list and update the layout"""
        self.layoutAboutToBeChanged.emit()
        self.dataList = dataList
        self.inUse = [[] for x in self.dataList]
        self.layoutChanged.emit()


class AbundancesGenericPanel(QFrame):
    """Generic class that shows two tables
    (one for available and one for selected plot objects).
    Must be extended before use
    """

    def __init__(self, parent=None, mainW=None):
        """Define some attributes, create the two tables
        and their models and connect double click signals

        Parameter:
            parent (default None): the parent widget
            mainW (default None): a MainWindow instance
        """
        super(AbundancesGenericPanel, self).__init__(parent)
        self.mainW = mainW
        self.runner = None
        self.dataList = []

        layout = QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(PGLabel(PGText.plotAvailablePoints))

        self.tableViewPts = QTableView(self)
        self.tableViewPts.setFont(PGFont())
        self.tableViewPts.doubleClicked.connect(self.cellDoubleClickPoints)
        layout.addWidget(self.tableViewPts)
        self.pointsModel = PGListPointsModel(self.dataList, self, self.mainW)
        self.tableViewPts.setModel(self.pointsModel)

        self.tableViewObjects = QTableView(self)
        self.tableViewObjects.setFont(PGFont())
        self.tableViewObjects.doubleClicked.connect(self.cellDoubleClickObjects)
        if self.hasLines:
            layout.addWidget(PGLabel(PGText.plotLinesInUse))
            self.objectsModel = PGListLinesModel(self.dataList, self)
        else:
            layout.addWidget(PGLabel(PGText.plotContoursInUse))
            self.objectsModel = PGListContoursModel(self.dataList, self)
        layout.addWidget(self.tableViewObjects)
        self.tableViewObjects.setModel(self.objectsModel)

    def addPlotObject(self, obj, ix=None):
        """Add a new PGPlotObject to self.objectsModel, or replace one.
        Also refresh table content and plot settings

        Parameters:
            obj: a PGLine or a PGContour, depending on the class
            ix (default None):
                if not None, the index of the object to replace
        """
        if ix is None:
            self.objectsModel.addLineToData(obj)
        else:
            self.objectsModel.replaceLine(obj, ix)
        self.tableViewObjects.resizeColumnsToContents()
        self.tableViewObjects.resizeRowsToContents()
        self.optimizePlotSettings()

    def clearCurrent(self):
        """Remove all the lines from self.objectsModel
        and mark them as not in use in the self.pointsModel
        """
        while self.objectsModel.rowCount() > 0:
            try:
                del self.pointsModel.inUse[self.objectsModel.dataList[0].pointRow][-1]
            except IndexError:
                pass
            self.objectsModel.deleteLine(0)

    def optimizePlotSettings(self):
        """Use the information on lines and contours in the plot
        to update xlims/ylims and xlabel/ylabel/zlabel
        in the plotSettings
        """
        if len(self.objectsModel.dataList) == 0:
            return
        newValues = {}
        xmin = 1e300
        xmax = -1e300
        ymin = 1e300
        ymax = -1e300
        for l in self.objectsModel.dataList:
            if min(l.x) < xmin:
                xmin = min(l.x)
            if min(l.y) < ymin:
                ymin = min(l.y)
            if max(l.x) > xmax:
                xmax = max(l.x)
            if max(l.y) > ymax:
                ymax = max(l.y)
        if self.invert:
            newValues["xlims"] = (xmax, xmin)
        else:
            newValues["xlims"] = (xmin, xmax)
        newValues["ylims"] = (ymin, ymax)
        if len(self.objectsModel.dataList) > 0:
            xl = self.objectsModel.dataList[0].xlabel
            newValues["xlabel"] = xl
            yl = self.objectsModel.dataList[0].getYlabel()
            newValues["ylabel"] = yl
            zl = self.objectsModel.dataList[0].getZlabel()
            newValues["zlabel"] = zl
        for l in self.objectsModel.dataList:
            if l.xlabel != xl:
                newValues["xlabel"] = ""
            if l.getYlabel() != yl:
                newValues["ylabel"] = ""
            if l.getZlabel() != zl:
                newValues["zlabel"] = ""
        self.mainW.plotPanel.plotSettings.updateAutomaticSettings(newValues)

    def cellDoubleClickObjects(self, index):
        """When double clicking a line with a plot object,
        perform an action (edit, delete, switch rows...)"""
        if not index.isValid():
            return
        row = index.row()
        col = index.column()
        if self.type in ("evo", "1d"):
            objName = "lines"
        elif self.type == "2d":
            objName = "contours"
        else:
            raise TypeError(PGText.plotAbundancesTypeError)
        obj = self.objectsModel.dataList[row]
        oldix = getattr(self.mainW.plotPanel.currentPlotContent, objName).index(obj)
        if self.objectsModel.header[col] == "Delete":
            if askYesNo(PGText.plotAskDeleteLine):
                del getattr(self.mainW.plotPanel.currentPlotContent, objName)[oldix]
                try:
                    del self.pointsModel.inUse[obj.pointRow][-1]
                except IndexError:
                    pass
                self.objectsModel.deleteLine(row)
                self.optimizePlotSettings()
                self.mainW.plotPanel.refreshPlot()
        elif self.objectsModel.header[col] == "Move down":
            try:
                olddown = self.objectsModel.dataList[row + 1]
            except IndexError:
                return
            oldixdown = getattr(self.mainW.plotPanel.currentPlotContent, objName).index(
                olddown
            )
            getattr(self.mainW.plotPanel.currentPlotContent, objName)[oldix] = olddown
            getattr(self.mainW.plotPanel.currentPlotContent, objName)[oldixdown] = obj
            self.addPlotObject(olddown, ix=row)
            self.addPlotObject(obj, ix=row + 1)
            self.mainW.plotPanel.refreshPlot()
        elif self.objectsModel.header[col] == "Move up":
            if row - 1 >= 0:
                oldup = self.objectsModel.dataList[row - 1]
            else:
                return
            oldixup = getattr(self.mainW.plotPanel.currentPlotContent, objName).index(
                oldup
            )
            getattr(self.mainW.plotPanel.currentPlotContent, objName)[oldix] = oldup
            getattr(self.mainW.plotPanel.currentPlotContent, objName)[oldixup] = obj
            self.addPlotObject(oldup, ix=row)
            self.addPlotObject(obj, ix=row - 1)
            self.mainW.plotPanel.refreshPlot()
        else:
            if self.type == "evo":
                afp = AddEvoLineFromPoint(self.mainW, None, "", [], [], line=obj)
            elif self.type == "1d":
                afp = Add1DLineFromPoint(self.mainW, None, "", None, [], [], line=obj)
            elif self.type == "2d":
                afp = Add2DContourFromPoint(
                    self.mainW, None, "", None, None, [], [], cnt=obj
                )
            afp.exec_()
            if afp.result():
                l = getattr(afp, "get" + objName[0].upper() + objName[1:-1])()
                getattr(self.mainW.plotPanel.currentPlotContent, objName)[oldix] = l
                self.addPlotObject(l, ix=row)
                self.mainW.plotPanel.refreshPlot()

    def cellDoubleClickPoints(self, index):
        """Not Implemented. Must be subclassed!"""
        raise NotImplementedError

    def pointString(self, pt):
        """Create a string that reports all the values in the list,
        comma separated and with the same format if possible

        Output:
            a string
        """
        l = []
        for a in pt:
            try:
                l.append("%.12e" % a)
            except TypeError:
                l.append("%s" % a)
        return ", ".join(l)

    def gridPointsString(self, pts):
        """For each element in the list, call self.pointString,
        and return the list of the results

        Output:
            a list of strings
        """
        return [self.pointString(p) for p in pts]

    def reloadModel(self):
        """Not Implemented. Must be subclassed!"""
        raise NotImplementedError

    def updatePlotProperties(self):
        """Reset the plot settings when the plot type is changed"""
        self.mainW.plotPanel.plotSettings.updateAutomaticSettings(
            {
                "xlabel": "",
                "ylabel": "",
                "zlabel": "",
                "xscale": "linear",
                "yscale": "linear",
            },
            force=True,
        )


# objects for plotting the evolution of nuclide abundances with time
class AddEvoLineFromPoint(AddLineFromPoint):
    """Dialog to manage the line to add from a grid point:
    ask which nuclide to use and various plot options.
    Based on AddLineFromPoint, with only two custom properties.
    """

    def __init__(self, mainW, ptrow, desc, header, output, line=None):
        """Set two custom attributes and call AddLineFromPoint.__init__"""
        self.headerCut = 1
        self.askChi2 = False
        AddLineFromPoint.__init__(
            self, mainW, ptrow, desc, 0, header, output, line=line
        )


class AbundancesEvolutionPanel(AbundancesGenericPanel):
    """Panel used when plotting the evolution of some abundances"""

    def __init__(self, parent=None, mainW=None):
        """Define required properties before calling
        AbundancesGenericPanel.__init__

        Parameter:
            parent (default None): the parent widget
            mainW (default None): a MainWindow instance
        """
        self.invert = True
        self.hasLines = True
        self.type = "evo"
        super(AbundancesEvolutionPanel, self).__init__(parent, mainW)

    def cellDoubleClickPoints(self, index):
        """When double clicking a row,
        show a dialog to customize the line settings
        and eventually add the prepared line to the plot
        """
        if index.isValid():
            row = index.row()
        else:
            return
        desc = ", ".join(
            [
                "%s = " % p
                + Configuration.formatParam % self.pointsModel.dataList[row, i]
                for i, p in enumerate(Parameters.paramOrder)
            ]
        )
        # self.mainW.plotPanel.refreshPlot()
        afp = AddEvoLineFromPoint(
            self.mainW,
            row,
            desc,
            self.runner.nuclidesHeader,
            self.runner.nuclidesEvolution[row],
        )
        afp.exec_()
        if afp.result():
            l = afp.getLine()
            self.mainW.plotPanel.currentPlotContent.lines.append(l)
            self.addPlotObject(l)
            self.pointsModel.inUse[row].append(True)
            self.mainW.plotPanel.refreshPlot()

    def reloadModel(self):
        """Replace the self.pointsModel when a new grid is loaded"""
        self.runner = self.mainW.plotPanel.gridLoader.runner
        self.pointsModel.replaceDataList(self.runner.getGridPointsList())

    def updatePlotProperties(self):
        """Reset the plot settings when the plot type is changed"""
        self.mainW.plotPanel.plotSettings.updateAutomaticSettings(
            {
                "xlabel": "$T$ [MeV]",
                "ylabel": "",
                "zlabel": "",
                "xscale": "log",
                "yscale": "log",
            },
            force=True,
        )


# plot 1D dependence of final abundances on physical params
class Add1DLineFromPoint(AddLineFromPoint):
    """Dialog to manage the line to add from a list of grid points:
    ask which nuclide to use and various plot options.
    Based on AddLineFromPoint, with only two custom properties.
    """

    def __init__(self, mainW, ptrow, desc, paramix, header, output, line=None):
        """Set two custom attributes and call AddLineFromPoint.__init__"""
        self.headerCut = 6
        self.askChi2 = True
        AddLineFromPoint.__init__(
            self, mainW, ptrow, desc, paramix, header, output, line=line
        )


class AbundancesParams1DPanel(AbundancesGenericPanel):
    """Panel where one can select the lines to include in the plot
    of final abundances as a function of one physical parameter
    """

    def __init__(self, parent=None, mainW=None):
        """Set basic properties and call AbundancesGenericPanel.__init__

        Parameter:
            parent (default None): the parent widget
            mainW (default None): a MainWindow instance
        """
        self.invert = False
        self.hasLines = True
        self.type = "1d"
        super(AbundancesParams1DPanel, self).__init__(parent, mainW)

    def cellDoubleClickPoints(self, index):
        """When double clicking an available point,
        open a dialog for editing the line settings
        and add it to the available plot lines if accepted

        Parameter:
            index: the QModelIndex of the clicked cell
        """
        if index.isValid():
            row = index.row()
        else:
            return
        descfixed = []
        for i, p in enumerate(Parameters.paramOrder):
            try:
                # test if the parameter is varying or not:
                len(self.pointsModel.dataList[row, i])
            except TypeError:
                # if not varying, add its value to the description
                descfixed.append(
                    "%s = " % p
                    + Configuration.formatParam % self.pointsModel.dataList[row, i]
                )
        desc = (
            ", ".join(descfixed)
            + ": %s" % Parameters.paramOrderParth[self.pointsModel.dataList[row, -2]]
        )
        afp = Add1DLineFromPoint(
            self.mainW,
            row,
            desc,
            self.pointsModel.dataList[row, -2],
            self.runner.parthenopeHeader,
            self.pointsModel.dataList[row, -1],
        )
        afp.exec_()
        if afp.result():
            l = afp.getLine()
            self.mainW.plotPanel.currentPlotContent.lines.append(l)
            self.addPlotObject(l)
            self.pointsModel.inUse[row].append(True)
            self.mainW.plotPanel.refreshPlot()

    def reloadModel(self):
        """Use the information related to the currently selected grid
        to fill the list of available points for the plots.
        They will be saved in the self.pointsModel.
        """
        self.runner = self.mainW.plotPanel.gridLoader.runner
        gridPoints = self.gridPointsString(self.runner.getGridPointsList())
        allparams = []
        allpoints = []
        allpointsstr = []
        for g in self.runner.gridPoints:
            params = [[] for i in range(6)]
            for pt in g:
                for i in range(6):
                    if pt[i] not in params[i]:
                        params[i].append(pt[i])
            allparams.append(params)
            for i in range(6):
                if len(params[i]) > 1:
                    for pt in g:
                        if pt[i] in params[i]:
                            n = [pt[j] if j != i else params[i] for j in range(6)]
                            ns = self.pointString(n)
                            if ns not in allpointsstr:
                                allpointsstr.append(ns)
                                parth = []
                                for j in params[i]:
                                    c = self.pointString(
                                        [pt[k] if k != i else j for k in range(6)]
                                    )
                                    ix = gridPoints.index(c)
                                    parth.append(self.runner.parthenopeOutPoints[ix])
                                n.extend(
                                    (
                                        Parameters.paramOrderParth.index(
                                            Parameters.paramOrder[i]
                                        ),
                                        np.asarray(parth),
                                    )
                                )
                                allpoints.append(n)
        self.pointsModel.replaceDataList(np.asarray(allpoints, dtype=object))


# plot 2D dependence of final abundances on physical params
class PGListContoursModel(PGListLinesModel):
    """Extension of `PGListLinesModel`, used for list of contours"""

    def __init__(self, dataList, parent, *args):
        """Constructor, based on `PGListLinesModel.__init__`,
        just with a different header

        Parameters:
            datalist:
            parent: the parent widget
        """
        PGListLinesModel.__init__(self, dataList, parent, *args)
        self.header = [
            "description",
            "label",
            "filled",
            "cmap",
            "levels",
            "extend",
            "Delete",
        ]
        self.headerToolTips = [
            "Description of the line, including the list of "
            + "physical parameters and the considered one in the current plot",
            "Label of the line in the legend",
            "Use filled contours or not",
            "Name of the colormap to use for the 2D plot",
            "Levels for the colormap",
            "Settings for extending the colormap above or below its current range"
            + " (only if levels are defined)",
            "Double click this cell to remove the line from the current plot",
        ]


class CMapItems(QObject):
    """Item with two connected QComboBox for the cmap category and name"""

    def __init__(self, parent=None, curr=None):
        """Construct the element defining some basic properties

        Parameter:
            parent: the parent widget
        """
        QObject.__init__(self, parent)
        self._cat = QComboBox(parent)
        self._cmap = QComboBox(parent)
        for i, c in enumerate(cmaps.keys()):
            self._cat.addItem(c)
            if (curr in cmaps[c]) or (curr is None and i == 0):
                self._cat.setCurrentText(c)
                for m in cmaps[c]:
                    self._cmap.addItem(m)
                    if m == curr:
                        self._cmap.setCurrentText(m)
        self._cat.currentTextChanged.connect(self.reloadCmapCombo)

    @property
    def cmap(self):
        """Return the current cmap name from the combo box

        Output:
            the cmap name
        """
        return self._cmap.currentText()

    def reloadCmapCombo(self):
        """Reload the _cmap combo box items when the cmap category
        is changed in the _cat combo box
        """
        self._cmap.clear()
        self._cmap.addItems(cmaps[self._cat.currentText()])


class Add2DContourFromPoint(AddFromPoint):
    """Dialog that asks which output value to use and various plot options"""

    def __init__(
        self,
        mainW,
        ptrow,
        desc,
        paramix1,
        paramix2,
        parthenopeHeader,
        parthenopeOut,
        cnt=None,
    ):
        """Build a form where all the relevant settings
        for a contour to be added in the plot can be configured

        Parameters:
            mainW: a MainWindow instance
            ptrow: the row id of the point in the model
            desc: a short description
            paramix1: index of x parameter in parthenopeOut
            paramix2: index of the y parameter in parthenopeOut
            parthenopeHeader: list of parameter names
            parthenopeOut: list of points from which the plot data
                will be extracted
            cnt (default None): a PGContour for editing its properties
        """
        self.pointRow = ptrow
        self.desc = desc if cnt is None else cnt.description
        self.paramix1 = paramix1
        self.paramix2 = paramix2
        self.parthenopeHeader = parthenopeHeader
        self.parthenopeOut = np.asarray(parthenopeOut)
        self.cnt = cnt
        AddFromPoint.__init__(self, mainW)
        self.chi2ParamConfigs = []

        # create the stacked widgets
        self.stackWidget = QStackedWidget(self)
        self.stackWidget.setFont(PGFont())
        self.paramWidget = QWidget(self)
        self.paramWidget.setLayout(QHBoxLayout(self.paramWidget))
        self.stackWidget.addWidget(self.paramWidget)
        self.chi2Widget = QWidget(self)
        self.chi2Widget.setLayout(QGridLayout(self.chi2Widget))
        self.stackWidget.addWidget(self.chi2Widget)

        layout = QGridLayout()
        self.setLayout(layout)
        self.setWindowTitle(PGText.plotSelectNuclideContour)

        r = 0
        layout.addWidget(PGLabel(self.desc), r, 0, 1, 4)

        self.chi2Check = QCheckBox("", self)
        if cnt is None:
            r += 1
            self._addHLine(r)

            l = PGLabel(PGText.plotSelectNuclide)
            l.setWordWrap(False)
            self.paramWidget.layout().addWidget(l)
            self.nuclideCombo = self._nuclideCombo()
            self.nuclideCombo.currentIndexChanged.connect(self.updateZlabel)
            self.paramWidget.layout().addWidget(self.nuclideCombo)

            self._newChi2Widget()
            # this checkbox is connected to a function that switches between the two stacked widgets
            r += 1
            layout.addWidget(PGLabel(PGText.plotChi2AskLabel), r, 0, 1, 3)
            self.chi2Check.setToolTip(PGText.plotChi2AskCheckToolTip)
            self.chi2Check.stateChanged.connect(self._updateStack)
            self.chi2Check.stateChanged.connect(self.updateZlabel)
            layout.addWidget(self.chi2Check, r, 3)

            self._fillChi2Widget()
        else:
            self.chi2Check.setChecked(False if cnt is None else bool(cnt.chi2))
            self.chi2Check.hide()

        r += 1
        layout.addWidget(self.stackWidget, r, 0, 1, 4)

        r += 1
        self._addHLine(r)

        r += 1
        layout.addWidget(PGLabel(PGText.plotLineLabel), r, 0, 1, 2)
        self.labelInput = QLineEdit("" if cnt is None else cnt.label)
        self.labelInput.setToolTip(PGText.plotLineLabelToolTip)
        layout.addWidget(self.labelInput, r, 2, 1, 2)

        r += 1
        layout.addWidget(PGLabel(PGText.plotContourFilled), r, 0, 1, 2)
        self.filledCheck = QCheckBox("", self)
        self.filledCheck.setToolTip(PGText.plotContourFilledToolTip)
        self.filledCheck.setChecked(True if cnt is None else cnt.filled)
        layout.addWidget(self.filledCheck, r, 2)

        r += 1
        self._addHLine(r)

        r += 1
        layout.addWidget(PGLabel(PGText.plotContourCMap), r, 0, 1, 4)
        r += 1
        layout.addWidget(PGLabel(PGText.category), r, 0)
        layout.addWidget(PGLabel(PGText.current), r, 2)
        self.cmapItems = CMapItems(self, cnt.cmap if cnt is not None else defaultCmap)
        layout.addWidget(self.cmapItems._cat, r, 1)
        layout.addWidget(self.cmapItems._cmap, r, 3)

        r += 1
        layout.addWidget(PGLabel(PGText.plotContourHasCbar), r, 0, 1, 2)
        self.hasCbarCheck = QCheckBox("", self)
        self.hasCbarCheck.setToolTip(PGText.plotContourHasCbarToolTip)
        self.hasCbarCheck.setChecked(True if cnt is None else cnt.hascbar)
        layout.addWidget(self.hasCbarCheck, r, 2)

        r += 1
        layout.addWidget(PGLabel(PGText.plotContourZLabel), r, 0, 1, 2)
        self.zlabelInput = QLineEdit(
            cnt.zlabel if cnt is not None else self.nuclideCombo.currentText()
        )
        self.zlabelInput.setToolTip(PGText.plotContourZLabelToolTip)
        layout.addWidget(self.zlabelInput, r, 2, 1, 2)

        r += 1
        layout.addWidget(PGLabel(PGText.plotContourExtend), r, 0, 1, 2)
        self.extendCombo = QComboBox(self)
        self.extendCombo.setToolTip(PGText.plotContourExtendToolTip)
        for f in extendOptions:
            self.extendCombo.addItem(f)
        if cnt is not None:
            self.extendCombo.setCurrentText(cnt.extend)
        layout.addWidget(self.extendCombo, r, 2, 1, 2)

        r += 1
        layout.addWidget(PGLabel(PGText.plotContourLevelsLabel), r, 0, 1, 2)
        self.levelsInput = QComboBox(self)
        if cnt:
            self.levelsInput.addItem("%s" % cnt.levels)
        for k in (
            "",
            "%s" % [0.0, 1.0, 4.0, 9.0, 16.0, 25.0],
            "%s" % [0.0, 2.30, 6.18, 11.83, 19.33, 28.74],
            "other (edit)",
        ):
            if not (cnt and k == "%s" % cnt.levels):
                self.levelsInput.addItem(k)
        self.levelsInput.setEditable(True)
        self.levelsInput.setToolTip(PGText.plotContourLevelsToolTip)
        layout.addWidget(self.levelsInput, r, 2, 1, 2)

        r += 1
        self.acceptButton = PGPushButton(PGText.buttonAccept)
        self.acceptButton.clicked.connect(self.testAccept)
        layout.addWidget(self.acceptButton, r, 0, 1, 2)
        self.cancelButton = PGPushButton(PGText.buttonCancel)
        self.cancelButton.clicked.connect(self.reject)
        layout.addWidget(self.cancelButton, r, 2, 1, 2)

    def _nuclideCombo(self):
        """Create a QComboBox with all the possible quantity options
        and the correct tooltip
        """
        combo = QComboBox(self)
        combo.setToolTip(PGText.plotSelectNuclideToolTip)
        for f in self.parthenopeHeader[6:]:
            combo.addItem("%s" % f)
        return combo

    def getContour(self):
        """Prepare a PGContour given the previous PGContour (if any)
        and the content of the form.

        Output:
            a PGContour
        """
        if self.cnt is None:
            chi2 = self.chi2Check.isChecked()
            x = np.unique(self.parthenopeOut[:, self.paramix1])
            y = np.unique(self.parthenopeOut[:, self.paramix2])
            if chi2:
                values = []
                params = ""
                c2 = 0.0
                for r, o in enumerate(self.chi2ParamConfigs):
                    params += self._getDescLineParam() % o.paramCombo.currentText()
                    c2 += chi2func(
                        self.parthenopeOut[:, 6 + o.paramCombo.currentIndex()],
                        float(o.chi2Mean.text()),
                        float(o.chi2Std.text()),
                    )
                out = self._getChi2Conv(c2)
                desc = self.desc + " - %s" % params
            else:
                chi2 = False
                out = self.parthenopeOut[:, 6 + self.nuclideCombo.currentIndex()]
                desc = self.desc + " - %s" % self.nuclideCombo.currentText()
            cnt = PGContour(
                x,
                y,
                out.reshape(len(x), len(y)).T,
                self.pointRow,
                d=desc,
                l=self.labelInput.text(),
                f=self.filledCheck.isChecked(),
                hcb=self.hasCbarCheck.isChecked(),
                cm=self.cmapItems.cmap,
                lvs=self.levels,
                ex=self.extendCombo.currentText(),
                xl=self.parthenopeHeader[self.paramix1],
                yl=self.parthenopeHeader[self.paramix2],
                zl=self.zlabelInput.text(),
                c2=self._getChi2Type(chi2),
            )
        else:
            cnt = PGContour(
                self.cnt.x,
                self.cnt.y,
                self.cnt.z,
                self.cnt.pointRow,
                d=self.cnt.description,
                l=self.labelInput.text(),
                f=self.filledCheck.isChecked(),
                hcb=self.hasCbarCheck.isChecked(),
                cm=self.cmapItems.cmap,
                lvs=self.levels,
                ex=self.extendCombo.currentText(),
                xl=self.cnt.xlabel,
                yl=self.cnt.ylabel,
                zl=self.zlabelInput.text(),
                c2=self.cnt.chi2,
            )
        return cnt

    @property
    def levels(self):
        """Process the content of self.levelsInput in order to
        obtain a list of levels for plt.contour/plt.contourf
        (the levels must be float)

        Output:
            a list of float if the input is valid, None otherwise
        """
        text = self.levelsInput.currentText().strip().lower()
        if text in ("", "none"):
            return None
        try:
            lvs = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            pGUIErrorManager.info(PGText.plotContourLevelsInvalidSyntax % text)
            return None
        else:
            if isinstance(lvs, (list, tuple)):
                new = []
                for f in lvs:
                    try:
                        new.append(float(f))
                    except ValueError:
                        mainlogger.info(PGText.plotContourLevelsInvalidLevel % f)
                if len(new) < 2:
                    pGUIErrorManager.info(PGText.plotContourLevelsInvalidLen % new)
                    return None
                return new
            else:
                pGUIErrorManager.info(PGText.plotContourLevelsInvalidType % lvs)
                return None

    def testAccept(self):
        """If chi2 is requested, verify that mean and standard deviation
        are valid float numbers before accepting the dialog and
        close the window
        """
        if self.chi2Check.isChecked():
            for r, o in enumerate(self.chi2ParamConfigs):
                try:
                    float(o.chi2Mean.text())
                except ValueError:
                    pGUIErrorManager.warning(PGText.plotInvalidMean % o.chi2Mean.text())
                    return
                try:
                    assert float(o.chi2Std.text()) > 0
                except (AssertionError, ValueError):
                    pGUIErrorManager.warning(
                        PGText.plotInvalidStddev % o.chi2Std.text()
                    )
                    return
        self.accept()

    def updateZlabel(self):
        """Update the zlabel of the field when the input is changed,
        using the nuclide label or one chi2labels[] depending on
        the content of the other fields
        """
        if self.cnt is not None:
            return
        self.zlabelInput.setText(
            chi2labels[self._getChi2Type(True)]
            if self.chi2Check.isChecked()
            else self.parthenopeHeader[6 + self.nuclideCombo.currentIndex()]
        )


class AbundancesParams2DPanel(AbundancesGenericPanel):
    """Panel where one can select the contours to include in the plot
    of final abundances as a function of two physical parameters
    """

    def __init__(self, parent=None, mainW=None):
        """Set basic properties and call AbundancesGenericPanel.__init__

        Parameter:
            parent (default None): the parent widget
            mainW (default None): a MainWindow instance
        """
        self.invert = False
        self.hasLines = False
        self.type = "2d"
        super(AbundancesParams2DPanel, self).__init__(parent, mainW)

    def cellDoubleClickPoints(self, index):
        """When double clicking an available point,
        open a dialog for editing the contour settings
        and add it to the available plot contours if accepted

        Parameter:
            index: the QModelIndex of the clicked cell
        """
        if index.isValid():
            row = index.row()
        else:
            return
        descfixed = []
        for i, p in enumerate(Parameters.paramOrder):
            try:
                # test if the parameter is varying or not:
                len(self.pointsModel.dataList[row, i])
            except TypeError:
                # if not varying, add its value to the description
                descfixed.append(
                    "%s = " % p
                    + Configuration.formatParam % self.pointsModel.dataList[row, i]
                )
        desc = (
            ", ".join(descfixed)
            + ": "
            + "%s" % Parameters.paramOrderParth[self.pointsModel.dataList[row, -3]]
            + ", "
            + "%s" % Parameters.paramOrderParth[self.pointsModel.dataList[row, -2]]
        )
        afp = Add2DContourFromPoint(
            self.mainW,
            row,
            desc,
            self.pointsModel.dataList[row, -3],
            self.pointsModel.dataList[row, -2],
            self.runner.parthenopeHeader,
            self.pointsModel.dataList[row, -1],
        )
        afp.exec_()
        if afp.result():
            l = afp.getContour()
            self.mainW.plotPanel.currentPlotContent.contours.append(l)
            self.addPlotObject(l)
            self.pointsModel.inUse[row].append(True)
            self.mainW.plotPanel.refreshPlot()

    def reloadModel(self):
        """Use the information related to the currently selected grid
        to fill the list of available points for the plots.
        They will be saved in the self.pointsModel.
        """
        self.runner = self.mainW.plotPanel.gridLoader.runner
        gridPoints = self.gridPointsString(self.runner.getGridPointsList())
        allparams = []
        allpoints = []
        allpointsstr = []
        for g in self.runner.gridPoints:
            params = [[] for i0 in range(6)]
            for pt in g:
                for i0 in range(6):
                    if pt[i0] not in params[i0]:
                        params[i0].append(pt[i0])
            allparams.append(params)
            for i1 in range(6):
                for i2 in range(i1 + 1, 6):
                    if len(params[i1]) > 1 and len(params[i2]) > 1:
                        for pt in g:
                            if pt[i1] in params[i1] and pt[i2] in params[i2]:
                                n = [
                                    pt[j]
                                    if j != i1 and j != i2
                                    else (params[i1] if j == i1 else params[i2])
                                    for j in range(6)
                                ]
                                ns = self.pointString(n)
                                if ns not in allpointsstr:
                                    allpointsstr.append(ns)
                                    parth = []
                                    for j1 in params[i1]:
                                        for j2 in params[i2]:
                                            c = self.pointString(
                                                [
                                                    pt[k]
                                                    if k != i1 and k != i2
                                                    else j1
                                                    if k == i1
                                                    else j2
                                                    for k in range(6)
                                                ]
                                            )
                                            ix = gridPoints.index(c)
                                            parth.append(
                                                self.runner.parthenopeOutPoints[ix]
                                            )
                                    n.extend(
                                        (
                                            Parameters.paramOrderParth.index(
                                                Parameters.paramOrder[i1]
                                            ),
                                            Parameters.paramOrderParth.index(
                                                Parameters.paramOrder[i2]
                                            ),
                                            parth,
                                        )
                                    )
                                    allpoints.append(n)
        self.pointsModel.replaceDataList(np.asarray(allpoints, dtype=object))


# panel for main window
class MWPlotPanel(QFrame):
    """`QFrame` extension to create a panel
    with the plot settings and output
    """

    plotTypePanelClass = {
        "none": PlotPlaceholderPanel,
        "evolution": AbundancesEvolutionPanel,
        "1Ddependence": AbundancesParams1DPanel,
        "2Ddependence": AbundancesParams2DPanel,
    }

    def __init__(self, parent=None):
        """Extension of `QFrame.__init__`, also adds a layout,
        a QStackedWidget and some other stuff

        Parameter:
            parent (default None): the parent widget
                (a MainWindow instance)
        """
        super(MWPlotPanel, self).__init__(parent)

        layout = QGridLayout()
        self.setLayout(layout)

        self.gridLoader = GridLoader(parent=self, mainW=parent)
        self.showPlotWidget = ShowPlotWidget(parent=self, mainW=parent)
        self.plotSettings = PlotSettings(parent=self)
        self.plotSettings.refreshButton.clicked.connect(self.refreshPlot)
        self.plotSettings.revertButton.clicked.connect(self.revertPlotSettings)
        self.plotSettings.resetButton.clicked.connect(self.resetPlot)
        self.plotSettings.saveButton.clicked.connect(self.showPlotWidget.saveAction)
        self.plotSettings.exportButton.clicked.connect(self.exportScript)

        self.specificPlotPanelStack = QStackedWidget(self)
        self.stackWidgets = {}
        self.stackAttrs = sorted(self.plotTypePanelClass.keys())
        for attr in self.stackAttrs:
            self.stackWidgets[attr] = self.plotTypePanelClass[attr](
                parent=self, mainW=parent
            )
            self.specificPlotPanelStack.addWidget(self.stackWidgets[attr])
            self.gridLoader.reloadedGrid.connect(self.stackWidgets[attr].reloadModel)
        self.specificPlotPanelStack.setCurrentIndex(self.stackAttrs.index("none"))
        self.specificPlotPanelStack.currentChanged.connect(self.updatePlotProperties)

        self.currentPlotContent = CurrentPlotContent(self.plotSettings)

        layout.addWidget(self.gridLoader, 0, 0)
        layout.addWidget(self.specificPlotPanelStack, 1, 0, 2, 1)
        layout.addWidget(self.plotSettings, 0, 1, 2, 1)
        layout.addWidget(self.showPlotWidget, 2, 1)
        layout.setRowMinimumHeight(0, 200)
        layout.setRowMinimumHeight(1, 200)
        layout.setRowMinimumHeight(2, 200)

    def exportScript(self):
        """Create a python script that creates the current plot"""
        savePath = askSaveFileName(self, PGText.plotWhereToSaveScript, filter="*.py")
        if savePath.strip() != "":
            if not savePath.endswith(".py"):
                savePath = savePath.strip() + ".py"
            if os.path.exists(savePath) and not askYesNo(PGText.askReplace):
                return
            try:
                ExportPlotCode(self.currentPlotContent, savePath)
            except FileNotFoundErrorClass:
                pGUIErrorManager.exception(PGText.errorCannotWriteFile)
            else:
                pGUIErrorManager.info(PGText.plotScriptSaved % savePath)

    def refreshPlot(self):
        """Refresh the plot in the figure canvas"""
        self.showPlotWidget.updatePlots(fig=self.currentPlotContent.doPlot())

    def revertPlotSettings(self):
        """Revert the plot properties to the most recent automatic settings"""
        self.plotSettings.restoreAutomaticSettings()

    def resetPlot(self, force=False):
        """Clear the current settings and refresh the plot

        Parameter:
            force (default False): if True,
                do not ask if the user wants to reset the plot
        """
        if force or askYesNo(PGText.plotSettAskResetImage):
            for attr in self.stackAttrs:
                self.stackWidgets[attr].clearCurrent()
            self.currentPlotContent.clearCurrent()
            self.refreshPlot()

    def updatePlotProperties(self, index):
        """Use the information related to the type of plot
        defined by the current stacked widget

        Parameter:
            index (not used)
        """
        self.specificPlotPanelStack.currentWidget().updatePlotProperties()

    def updatePlotTypePanel(self, status):
        """Change the lower left panel according to the plot type

        Parameter:
            status (bool): the status of the radio button
        """
        if not status:
            return
        ty = "none"
        self.resetPlot(force=True)
        for attr in PGText.plotTypeDescription.keys():
            if getattr(self.gridLoader, attr).isChecked():
                ty = attr
        self.specificPlotPanelStack.setCurrentIndex(self.stackAttrs.index(ty))


class ExportPlotCode(object):
    """Class that creates a python script with the code
    for reading the appropriate files and producing the current figure.
    Provided for easier editing of the plot properties
    """

    def __init__(self, plotcontent, filename):
        """Create the instance and set some properties,
        then call the function that generates and stores the script

        Parameters:
            plotcontent: a CurrentPlotContent instance
            filename: the name of the output file where to save the code
        """
        if not isinstance(plotcontent, CurrentPlotContent):
            raise TypeError("Wrong argument type for plotcontent!")
        self.content = plotcontent
        self.content.readSettings()
        self.settings = plotcontent.pSett
        self.filename = filename
        self.filecontent = ""
        np.set_printoptions(threshold=sys.maxsize)

        self._createFile()

    def _createFile(self):
        """Function that calls other class methods to create the script"""
        self._plotInit()
        # at the moment, explicitely store all the plot objects
        # later, they may be read directly from the output files of the fortran code
        self._storeObjects()
        self._plotObjects()
        self._plotSettings()

        self._write()

    def _plotInit(self):
        """Function to write the part of code that initializes the plot
        (create figure and axes)
        """
        self.filecontent += '"""File generated by the PArthENoPE GUI"""\n'
        self.filecontent += "import numpy as np\n"
        self.filecontent += "import matplotlib\n"
        self.filecontent += "matplotlib.use('agg')\n"
        self.filecontent += "import matplotlib.pyplot as plt\n"
        self.filecontent += "from parthenopegui.plotUtils import PGLine, PGContour\n"
        self.filecontent += "\n"
        self.filecontent += "fig = plt.figure(figsize=%s)\n" % str(self.content.figsize)
        self.filecontent += "plt.plot(np.nan, np.nan)\n"

    def _plotObjects(self):
        """Add the lines that call plt.plot or plt.contour(f)
        to produce the plots of the requested contents
        """
        if len(self.content.lines) > 0:
            self.filecontent += "for l in lines:\n"
            self.filecontent += "    plt.plot(\n"
            self.filecontent += "        l.x,\n"
            self.filecontent += "        l.y,\n"
            self.filecontent += "        label=l.label,\n"
            self.filecontent += "        color=l.color,\n"
            self.filecontent += "        ls=l.style,\n"
            self.filecontent += "        marker=l.marker,\n"
            self.filecontent += "        lw=l.width,\n"
            self.filecontent += "    )\n"
        if len(self.content.contours) > 0:
            self.filecontent += "for c in contours:\n"
            self.filecontent += "    if c.filled:\n"
            self.filecontent += "        func = plt.contourf\n"
            self.filecontent += "    else:\n"
            self.filecontent += "        func = plt.contour\n"
            self.filecontent += "    options = {}\n"
            self.filecontent += "    if c.levels is not None:\n"
            self.filecontent += '        options["levels"] = c.levels\n'
            self.filecontent += "    if c.extend is not None:\n"
            self.filecontent += '        options["extend"] = c.extend\n'
            self.filecontent += '    options["cmap"] = matplotlib.cm.get_cmap(c.cmap)\n'
            self.filecontent += "    CS = func(c.x, c.y, c.z, **options)\n"
            self.filecontent += "    if c.hascbar:\n"
            self.filecontent += "        cbar = plt.colorbar(CS)\n"
            self.filecontent += "        cbar.ax.set_ylabel(%s, fontsize='%s')\n" % (
                "c.zlabel"
                if self.settings.zlabel == ""
                else "'%s'" % self.settings.zlabel,
                self.settings.axesTextSize,
            )

    def _plotSettings(self):
        """Function to write the part of code that customizes
        plot axes, legend, layout and more,
        and saves the figure to some file
        """
        self.filecontent += "\n"
        self.filecontent += "plt.xlabel('%s', fontsize='%s')\n" % (
            self.settings.xlabel,
            self.settings.axesTextSize,
        )
        self.filecontent += "plt.ylabel('%s', fontsize='%s')\n" % (
            self.settings.ylabel,
            self.settings.axesTextSize,
        )
        self.filecontent += "plt.xscale('%s')\n" % self.settings.xscale
        self.filecontent += "plt.yscale('%s')\n" % self.settings.yscale
        self.filecontent += "plt.xlim(%s)\n" % str(self.settings.xlims)
        self.filecontent += "plt.ylim(%s)\n" % str(self.settings.ylims)
        if (
            isinstance(self.content.title, six.string_types)
            and self.content.title.strip() != ""
        ):
            self.filecontent += "plt.title('%s')\n" % self.content.title.strip()
        if self.content.tight:
            self.filecontent += "plt.tight_layout()\n"
        if self.content.legend:
            self.filecontent += "plt.legend(loc='%s', ncol=%s, fontsize='%s')\n" % (
                self.settings.legendLoc,
                self.settings.legendNcols,
                self.settings.legendTextSize,
            )
        pwm = PGText.PArthENoPEWatermark
        self.filecontent += (
            "plt.text("
            + '{x:}, {y:}, "{text:}",'
            + ' color="{color:}", fontsize="{fontsize:}", ha="{ha:}",'
            + " rotation={rotation:},"
            + ' transform=plt.gca().transAxes, va="{va:}")\n'
        ).format(x=pwm["x"], y=pwm["y"], text=pwm["text"], **pwm["more"])
        self.filecontent += (
            "plt.savefig('fig_%s.pdf')\n"
            % datetime.datetime.today().strftime("%y%m%d_%H%M%S")
        )
        self.filecontent += "plt.close()\n"

    def _readNuclides(self):
        """Function to write the part of code that reads the files
        for a plot of nuclide abundances as functions of time
        """
        raise NotImplementedError

    def _readParthenope1d(self):
        """Function to write the part of code that reads the files
        and prepares the lists for 1D plots of nuclide abundances
        as functions of physical parameters
        """
        raise NotImplementedError

    def _readParthenope2d(self):
        """Function to write the part of code that reads the files
        and prepares the lists for 2D plots of nuclide abundances
        as functions of physical parameters
        """
        raise NotImplementedError

    def _storeObjects(self):
        """Function that stores the `PGLine`s and `PGContour`s
        in the output script file.
        The data are explicitely written in the file at the moment
        """
        self.filecontent += "\nlines = []\n"
        for l in self.content.lines:
            self.filecontent += "lines.append(\n"
            self.filecontent += "    PGLine(\n"
            self.filecontent += "        np.asarray(%s),\n" % str(list(l.x))
            self.filecontent += "        np.asarray(%s),\n" % str(list(l.y))
            self.filecontent += "        %s,\n" % l.pointRow
            self.filecontent += "        c='%s',\n" % l.color
            self.filecontent += "        c2=%s,\n" % (
                l.chi2 if isinstance(l.chi2, bool) else "'%s'" % l.chi2
            )
            self.filecontent += "        d='%s',\n" % l.description
            self.filecontent += "        l='%s',\n" % l.label
            self.filecontent += "        m='%s',\n" % l.marker
            self.filecontent += "        s='%s',\n" % l.style
            self.filecontent += "        w=%s,\n" % l.width
            self.filecontent += "        xl='%s',\n" % l.xlabel
            self.filecontent += "        yl='%s',\n" % l.ylabel
            self.filecontent += "    )\n"
            self.filecontent += ")\n"
        self.filecontent += "\ncontours = []\n"
        for c in self.content.contours:
            zmatstr = repr(c.z).replace("array", "np.asarray").replace("\n", " ")
            while "  " in zmatstr:
                zmatstr = zmatstr.replace("  ", " ")
            self.filecontent += "contours.append(\n"
            self.filecontent += "    PGContour(\n"
            self.filecontent += "        np.asarray(%s),\n" % str(list(c.x))
            self.filecontent += "        np.asarray(%s),\n" % str(list(c.y))
            self.filecontent += "        %s,\n" % zmatstr
            self.filecontent += "        %s,\n" % c.pointRow
            self.filecontent += "        c2=%s,\n" % (
                c.chi2 if isinstance(c.chi2, bool) else "'%s'" % c.chi2
            )
            self.filecontent += "        cm='%s',\n" % c.cmap
            self.filecontent += "        d='%s',\n" % c.description
            self.filecontent += "        ex=%s,\n" % (
                "None" if c.extend is None else "'%s'" % c.extend
            )
            self.filecontent += "        f=%s,\n" % c.filled
            self.filecontent += "        hcb=%s,\n" % c.hascbar
            self.filecontent += "        l='%s',\n" % c.label
            self.filecontent += "        lvs=%s,\n" % str(c.levels)
            self.filecontent += "        xl='%s',\n" % c.xlabel
            self.filecontent += "        yl='%s',\n" % c.ylabel
            self.filecontent += "        zl='%s',\n" % c.zlabel
            self.filecontent += "    )\n"
            self.filecontent += ")\n"
        self.filecontent += "\n"

    def _write(self):
        """Write the file content"""
        with open(self.filename, "w") as _f:
            _f.write(self.filecontent)
