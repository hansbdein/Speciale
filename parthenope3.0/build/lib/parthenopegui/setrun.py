"""PArthENoPE GUI module with functions related to setting
the parameters and running the fortran code
"""
import glob
import os
import pickle
from datetime import datetime
from multiprocessing import Pool, cpu_count

import numpy as np
import six
from PySide2.QtCore import Qt, Signal
from PySide2.QtGui import QGuiApplication, QIcon, QImage, QPalette, QPixmap
from PySide2.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QLineEdit,
    QProgressBar,
    QRadioButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    from parthenopegui import FileNotFoundErrorClass
    from parthenopegui.configuration import (
        Configuration,
        DDTableWidget,
        Nuclides,
        Parameters,
        PGFont,
        PGLabel,
        PGLabelButton,
        PGPushButton,
        Reactions,
        UnicodeSymbols,
        askDirName,
        askYesNo,
        paramsHidePath,
        paramsRealPath,
    )
    from parthenopegui.errorManager import mainlogger, pGUIErrorManager
    from parthenopegui.runner import RunPArthENoPE
    from parthenopegui.texts import PGText
except ImportError:
    print("[setrun] Necessary parthenopegui submodules not found!")
    raise


class EditReaction(QDialog):
    """Dialog where to edit a single reaction"""

    def __init__(self, rowData, parent=None):
        """Build the dialog with the labels and the inputs
        that are needed to edit a reaction

        Parameters:
            rowData: a dict with keys ["rea", "type", "corr", "val"]
                containing the current information on the reaction
            parent (default None): the parent widget
        """
        super(EditReaction, self).__init__(parent)
        self.setWindowTitle(PGText.editReactionTitle)
        layout = QGridLayout()
        self.setLayout(layout)

        layout.addWidget(PGLabel(PGText.editReactionFirstLabel), 0, 0, 1, 2)
        if isinstance(rowData["rea"], QImage):
            label = PGLabel("")
            label.setPixmap(QPixmap(rowData["rea"]))
            layout.addWidget(label, 0, 2)
        else:
            layout.addWidget(PGLabel("%s" % rowData["rea"]), 0, 2)
        layout.addWidget(PGLabel("   "), 0, 3)
        if isinstance(rowData["type"], QImage):
            label = PGLabel("")
            label.setPixmap(QPixmap(rowData["type"]))
            layout.addWidget(label, 0, 4)
        else:
            layout.addWidget(PGLabel("%s" % rowData["type"]), 0, 4)

        self.corrType = QComboBox(self)
        self.corrType.setToolTip(PGText.editReactionToolTip)
        layout.addWidget(self.corrType, 1, 0, 1, 2)
        for f in PGText.editReactionCombo:
            self.corrType.addItem("%s" % f)
        try:
            self.corrType.setCurrentText(rowData["corr"])
        except ValueError:
            pass
        self.corrType.currentIndexChanged.connect(self.updateFactor)

        self.corrValue = QLineEdit("%s" % rowData["val"])
        self.corrValue.setToolTip(PGText.editReactionValueToolTip)
        layout.addWidget(self.corrValue, 1, 2, 1, 3)
        if self.corrType.currentIndex() != 3:
            self.corrValue.setEnabled(False)

        self.acceptButton = PGPushButton(PGText.buttonAccept)
        self.acceptButton.clicked.connect(self.testAccept)
        layout.addWidget(self.acceptButton, 2, 1)

        self.cancelButton = PGPushButton(PGText.buttonCancel)
        self.cancelButton.clicked.connect(self.reject)
        layout.addWidget(self.cancelButton, 2, 2)

    def updateFactor(self, index):
        """When the index of self.corrType changes, update the content
        and the status (enable or disable field) of self.corrValue
        """
        if index == 3:
            self.corrValue.setEnabled(True)
            self.corrValue.setText("1.0")
        else:
            self.corrValue.setEnabled(False)
            if index == 0:
                self.corrValue.setText("1.0")
            elif index == 1:
                self.corrValue.setText("-" + UnicodeSymbols.l2u["sigma"])
            elif index == 2:
                self.corrValue.setText("+" + UnicodeSymbols.l2u["sigma"])

    def testAccept(self):
        """Test that the current value for the reaction, if needed,
        is a float between 0 and 1e5 before accepting the form content
        """
        if self.corrType.currentIndex() == 3:
            try:
                current = float(self.corrValue.text())
            except ValueError:
                pGUIErrorManager.warning(
                    PGText.errorInvalidField
                    % ("value", "reaction rate", self.corrValue.text())
                )
                return
            if current < 0:
                pGUIErrorManager.warning(
                    PGText.errorInvalidField
                    % ("negative value", "reaction rate", current)
                )
                return
            if current > 100000:
                pGUIErrorManager.warning(
                    PGText.warningLargeValue % ("the reaction rate", current)
                )
                return
        self.accept()


class ReactionsTableWidget(QTableWidget):
    """Table for showing the list of reactions"""

    def __init__(self, parent=None):
        """Set basic properties of the table
        and call the function that fills the rows

        Parameter:
            parent (default None): a MainWindow instance
        """
        self.mainW = parent
        super(ReactionsTableWidget, self).__init__(parent)
        self.setFont(PGFont())
        self.setColumnCount(5)
        self.setHorizontalHeaderLabels(PGText.reactionColumnHeaders)
        for i, t in enumerate(PGText.reactionColumnHeaderToolTips):
            self.horizontalHeaderItem(i).setToolTip(t)
        self.colKeys = ["rea", "type", "corr", "val"]
        self.reactionsData = {}
        self.fillReactions()
        self.cellDoubleClicked.connect(self.onCellDoubleClick)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)

    def fillReactions(self):
        """Fill the rows of the table using
        the current selection of reactions
        """
        self.reactionsData = {
            re: {"rea": uc[0], "type": uc[1], "corr": "default", "val": "1.0"}
            for re, uc in self.mainW.reactions.current.items()
        }
        self.clearContents()
        self.setRowCount(0)
        for i, r in enumerate(self.reactionsData.values()):
            self.insertRow(i)
            for c, k in enumerate(self.colKeys):
                if isinstance(r[k], QImage):
                    item = QTableWidgetItem("")
                    item.setData(Qt.DecorationRole, QPixmap(r[k]))
                else:
                    item = QTableWidgetItem(r[k])
                item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                self.setItem(i, c, item)
            item = QTableWidgetItem(QIcon(QPixmap(":/images/edit.png")), "")
            item.setToolTip(PGText.reactionEditToolTip)
            item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            self.setItem(i, 4, item)
        self.resizeColumnsToContents()
        self.resizeRowsToContents()

    def onCellDoubleClick(self, row, col):
        """Process event when mouse double clicks an item.
        Opens a link if some columns

        Parameters:
            index: a `QModelIndex` instance
        """
        try:
            rea = self.reactionsData[row + 1]
        except (KeyError, IndexError):
            pGUIErrorManager.debug(PGText.errorReadTable)
            return
        if col == 4:
            er = EditReaction(rea, self)
            if er.exec_():
                self.updateRow(row, er.corrType.currentText(), er.corrValue.text())
                self.resizeColumnsToContents()
                self.resizeRowsToContents()

    def updateRow(self, row, newCorr, newVal):
        """Update the a single row content with the new provided values,
        also edit the content of self.reactionsData

        Parameters:
            row: the row index
            newCorr: new type of correction
                (see PGText.editReactionCombo)
            newVal: the new numeric value for the correction factor
        """
        self.reactionsData[row + 1]["corr"] = newCorr
        self.reactionsData[row + 1]["val"] = "%s" % newVal
        self.item(row, 2).setText(newCorr)
        self.item(row, 3).setText("%s" % newVal)

    def readline(self, row):
        """Read the content of a single row in the reactions table

        Parameter:
            row: the row index
        """
        tyval = self.item(row, 2).text()
        ty = PGText.editReactionCombo.index(tyval)
        val = self.item(row, 3).text() if ty == 3 else "1.0"
        return [row + 1, ty, val]


class MWNetworkPanel(QFrame):
    """Panel where it is possible to select the network of reactions
    to be used in the fortran code
    """

    def __init__(self, parent=None):
        """Create a QGroupBox, a PGLabel and a ReactionsTableWidget
        for selecting and resuming the current reactions network

        Parameter:
            parent: the parent widget
        """
        super(MWNetworkPanel, self).__init__(parent)
        self.mainW = parent

        layout = QGridLayout()
        self.setLayout(layout)

        self.groupBox = QGroupBox(PGText.networkSelect, self)
        self.groupBox.setFlat(True)
        self.groupBox.setFont(PGFont())
        layout.addWidget(self.groupBox)
        boxlayout = QVBoxLayout()
        for attr, desc in PGText.networkDescription.items():
            setattr(self, attr, QRadioButton(desc, self))
            getattr(self, attr).toggled.connect(self.updateReactions)
            getattr(self, attr).setToolTip(PGText.networkToolTips[attr])
            boxlayout.addWidget(getattr(self, attr))
        self.smallNet.setChecked(True)
        self.groupBox.setLayout(boxlayout)

        layout.addWidget(PGLabel(PGText.networkCustomizeRate))
        self.reactionsTable = ReactionsTableWidget(self.mainW)
        self.mainW.reactions.updated.connect(self.reactionsTable.fillReactions)
        layout.addWidget(self.reactionsTable)

    def updateReactions(self):
        """When the reaction network is changed in the QGroupBox,
        update also the configuration (available nuclides and reactions)
        """
        net = None
        for attr in PGText.networkDescription.keys():
            if getattr(self, attr).isChecked():
                net = attr
        if net is None:
            pGUIErrorManager.error(PGText.errorNoReactionNetwork)
        new = [
            n
            for n in Nuclides.all.keys()
            if Nuclides.nuclideOrder[n] < Configuration.limitNuclides[net]
        ]
        self.mainW.nuclides.updateCurrent(new)
        new = [n for n in Reactions.all.keys() if n < Configuration.limitReactions[net]]
        self.mainW.reactions.updateCurrent(new)


class EditParameters(QDialog):
    """Dialog for editing or adding a new grid or single point"""

    def __init__(self, mainW=None, physPanel=None):
        """Fill few labels and a series of QComboBoxes and QLineEdits,
        for each param that can be edited

        Parameters:
            mainW (default None): a MainWindow instance
            physPanel (default None): a MWPhysParamsPanel instance,
                for emitting needsRefresh
        """
        self.mainW = mainW
        self.physPanel = physPanel
        super(EditParameters, self).__init__(mainW)
        self.setWindowTitle(PGText.editParametersTitle)
        layout = QGridLayout()
        self.setLayout(layout)
        self.inputs = {}

        layout.addWidget(PGLabel(PGText.editParametersAdd), 0, 0, 1, 5)
        for i, desc in enumerate(PGText.editParametersColumnLabels):
            layout.addWidget(PGLabel(desc), 1, i + 1)
        for i, p in enumerate(Parameters.paramOrder):
            pa = self.mainW.parameters.all[p]
            label = PGLabel("")
            label.setPixmap(pa.fig)
            label.setToolTip(PGText.parameterDescriptions[p])
            layout.addWidget(label, i + 2, 0)
            values = {}
            values["combo"] = QComboBox(self)
            for f in PGText.editParametersCombo:
                values["combo"].addItem("%s" % f)
            values["def"] = QLineEdit("%s" % pa.currentval)
            values["min"] = QLineEdit("%s" % pa.currentmin)
            values["max"] = QLineEdit("%s" % pa.currentmax)
            values["num"] = QLineEdit("%s" % pa.currentN)
            for f in ("combo", "def", "min", "max", "num"):
                values[f].setToolTip(PGText.editParametersToolTips[f])
            if pa.currenttype == "grid":
                values["combo"].setCurrentText("grid")
                values["def"].setEnabled(False)
            else:
                values["min"].setEnabled(False)
                values["max"].setEnabled(False)
                values["num"].setEnabled(False)
            layout.addWidget(values["combo"], i + 2, 1)
            layout.addWidget(values["def"], i + 2, 2)
            layout.addWidget(values["min"], i + 2, 3)
            layout.addWidget(values["max"], i + 2, 4)
            layout.addWidget(values["num"], i + 2, 5)
            values["combo"].currentTextChanged.connect(
                lambda x, y=p: self.updateFactor(x, y)
            )
            self.inputs[p] = values

        self.acceptButton = PGPushButton(PGText.buttonAccept)
        self.acceptButton.clicked.connect(self.testAccept)
        layout.addWidget(self.acceptButton, 8, 2)

        self.cancelButton = PGPushButton(PGText.buttonCancel)
        self.cancelButton.clicked.connect(self.reject)
        layout.addWidget(self.cancelButton, 8, 3)

    def testAccept(self):
        """Test that the inserted values are valid:
        check that the content of the fields is a number,
        that grid points have N>=2 and min != max
        """
        for i, p in enumerate(Parameters.paramOrder):
            if self.inputs[p]["combo"].currentText() != "grid":
                try:
                    float(self.inputs[p]["def"].text())
                except ValueError:
                    pGUIErrorManager.warning(
                        PGText.errorInvalidField
                        % ("value", p, self.inputs[p]["def"].text())
                    )
                    return
            else:
                try:
                    num = int(self.inputs[p]["num"].text())
                except ValueError:
                    pGUIErrorManager.warning(
                        PGText.errorInvalidField
                        % ("number of points", p, self.inputs[p]["num"].text())
                    )
                    return
                if num < 2:
                    pGUIErrorManager.warning(
                        PGText.errorInvalidField
                        % ("number of points (smaller than 2)", p, num)
                    )
                    return
                try:
                    minv = float(self.inputs[p]["min"].text())
                except ValueError:
                    pGUIErrorManager.warning(
                        PGText.errorInvalidField
                        % ("minimum", p, self.inputs[p]["min"].text())
                    )
                    return
                try:
                    maxv = float(self.inputs[p]["max"].text())
                except ValueError:
                    pGUIErrorManager.warning(
                        PGText.errorInvalidField
                        % ("maximum", p, self.inputs[p]["max"].text())
                    )
                    return
                if minv == maxv:
                    pGUIErrorManager.warning(
                        PGText.errorInvalidParamLimits % (p, minv, maxv)
                    )
                    return
        self.accept()

    def updateFactor(self, val, param):
        """Enable/Disable the relevant inputs when the user
        changes the type between single point and grid
        for a given parameter

        Parameters:
            val: the new currentText of self.inputs[param]["combo"]
            param: the parameter name
        """
        if val == "grid":
            self.inputs[param]["def"].setEnabled(False)
            self.inputs[param]["min"].setEnabled(True)
            self.inputs[param]["max"].setEnabled(True)
            self.inputs[param]["num"].setEnabled(True)
        else:
            self.inputs[param]["def"].setEnabled(True)
            self.inputs[param]["min"].setEnabled(False)
            self.inputs[param]["max"].setEnabled(False)
            self.inputs[param]["num"].setEnabled(False)

    def emitRefresh(self):
        """If possible, emit the needsRefresh signal of self.physPanel,
        else just log a debug
        """
        try:
            self.physPanel.needsRefresh.emit()
        except AttributeError:
            mainlogger.debug("", exc_info=True)

    def processOutput(self, newid=None):
        """Read the form content and add the values
        to the Parameters.gridsList and the single Parameter grids.
        If some errors occur, set values to their default

        Parameters:
            newid (default None): the optional index
                to use for the new grid
        """
        grid = {}
        if newid is None:
            try:
                newid = max(self.mainW.parameters.gridsList.keys()) + 1
            except AttributeError:
                mainlogger.debug(
                    PGText.warningWrongType % ("Parameters.gridsList", "dict")
                )
                self.mainW.parameters.gridsList = {}
                newid = 0
            except ValueError:
                mainlogger.debug(PGText.warningMaxEmptyList)
                newid = 0
        else:
            try:
                self.mainW.parameters.gridsList.keys()
            except AttributeError:
                mainlogger.debug(
                    PGText.warningWrongType % ("Parameters.gridsList", "dict")
                )
                self.mainW.parameters.gridsList = {}
            try:
                newid = int(newid)
            except ValueError:
                mainlogger.debug(
                    PGText.errorInvalidFieldSet % ("value", "newid", newid, 0)
                )
                newid = 0
        numpts = 1
        for p, v in self.inputs.items():
            grid[p] = {}
            if v["combo"].currentText() == "grid":
                grid[p]["type"] = "grid"
                try:
                    grid[p]["min"] = float(v["min"].text())
                except ValueError:
                    pGUIErrorManager.warning(
                        PGText.errorInvalidField % ("minimum", p, v["min"].text())
                    )
                    grid[p]["min"] = self.mainW.parameters.all[p].defaultmin
                try:
                    grid[p]["max"] = float(v["max"].text())
                except ValueError:
                    pGUIErrorManager.warning(
                        PGText.errorInvalidField % ("maximum", p, v["max"].text())
                    )
                    grid[p]["max"] = self.mainW.parameters.all[p].defaultmax
                try:
                    grid[p]["N"] = int(v["num"].text())
                except ValueError:
                    pGUIErrorManager.warning(
                        PGText.errorInvalidField
                        % ("number of points", p, v["num"].text())
                    )
                    grid[p]["N"] = self.mainW.parameters.all[p].defaultnum
                self.mainW.parameters.all[p].addGrid(
                    newid, grid[p]["N"], grid[p]["min"], grid[p]["max"]
                )
            else:
                grid[p]["type"] = "point"
                try:
                    grid[p]["val"] = float(v["def"].text())
                except ValueError:
                    pGUIErrorManager.warning(
                        PGText.errorInvalidField % ("value", p, v["def"].text())
                    )
                    grid[p]["val"] = self.mainW.parameters.all[p].defaultval
                grid[p]["N"] = 1
                self.mainW.parameters.all[p].addGrid(
                    newid, grid[p]["N"], grid[p]["val"], grid[p]["val"]
                )
            numpts *= grid[p]["N"]
        self.mainW.parameters.gridsList[newid] = numpts
        self.emitRefresh()


class ResumeGrid(QFrame):
    """Frame where a table resuming the content of a grid is shown.
    It uses a QTableWidget plus few buttons.
    """

    def __init__(self, idGrid, mainW=None, physPanel=None):
        """Build the layout that shows a new grid point
        with its related buttons for edit/delete actions

        Parameters:
            idGrid: the id of the grid
            mainW (default None): a MainWindow instance
            physPanel (default None): a MWPhysParamsPanel instance
        """
        self.idGrid = idGrid
        self.mainW = mainW
        self.physPanel = physPanel
        super(ResumeGrid, self).__init__(mainW)

        layout = QGridLayout()
        self.setLayout(layout)

        self.editGrid = PGPushButton(PGText.buttonEdit)
        self.editGrid.setToolTip(PGText.resumeGridEditToolTip)
        self.editGrid.clicked.connect(self.onEdit)
        layout.addWidget(self.editGrid, 1, 1)

        self.deleteGrid = PGPushButton(PGText.buttonDelete)
        self.deleteGrid.setToolTip(PGText.resumeGridDeleteToolTip)
        self.deleteGrid.clicked.connect(self.onDelete)
        layout.addWidget(self.deleteGrid, 2, 1)

        self.table = QTableWidget(self)
        layout.addWidget(self.table, 0, 0, 4, 1)
        self.table.setFont(PGFont())
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(PGText.resumeGridHeaders)
        for i, t in enumerate(PGText.resumeGridHeadersToolTip):
            self.table.horizontalHeaderItem(i).setToolTip(t)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        for i, p in enumerate(Parameters.paramOrder):
            pa = self.mainW.parameters.all[p]
            npts, grid = pa.getGridPoints(idGrid)
            self.table.insertRow(i)
            item = QTableWidgetItem(QIcon(pa.fig), "")
            item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            self.table.setItem(i, 0, item)
            item = QTableWidgetItem("%s" % npts)
            item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            self.table.setItem(i, 1, item)
            item = QTableWidgetItem("%s" % grid)
            item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            self.table.setItem(i, 2, item)
        self.setMinimumHeight(self.table.rowHeight(0) * 7.5)
        self.table.resizeColumnsToContents()

    def onEdit(self):
        """Action to perform when editing a grid/point.
        Open an instance for `EditParameters`.
        """
        self.mainW.parameters.getGrid(self.idGrid)
        ep = EditParameters(self.mainW, self.physPanel)
        if ep.exec_():
            ep.processOutput(newid=self.idGrid)

    def onDelete(self):
        """Action to perform when deleting a grid/point.
        Ask confirmation before performing any action.
        """
        if askYesNo(PGText.resumeGridAskDelete):
            if self.mainW.parameters.deleteGrid(self.idGrid):
                self.physPanel.needsRefresh.emit()
            else:
                pGUIErrorManager.warning(PGText.errorLoadingGrid)


class MWPhysParamsPanel(QScrollArea):
    """`QScrollArea` extension to create a scrollable panel
    where to show the list of points to be used in the Fortran code
    """

    needsRefresh = Signal()

    def __init__(self, parent=None):
        """Create the basic properties, connect the signal to slot,
        fill the current widget content

        Parameter:
            parent: the parent widget
        """
        self.mainW = parent
        if not hasattr(self.mainW, "parameters"):
            raise AttributeError(PGText.parentAttributeMissing)
        super(MWPhysParamsPanel, self).__init__(parent)
        self.needsRefresh.connect(self.refreshContent)

        self.inwidget = QWidget(self)
        layout = QVBoxLayout()
        self.inwidget.setLayout(layout)
        self.refreshContent()

    def cleanLayout(self):
        """Delete the previous widgets from the layout"""
        if self.inwidget.layout() is None:
            return
        while True:
            o = self.inwidget.layout().takeAt(0)
            if o is None:
                break
            o.widget().deleteLater()

    def refreshContent(self):
        """Use the information on the parameter grids"""
        self.cleanLayout()
        self.inwidget.layout().addWidget(PGLabel(PGText.physicalParamsDescription))
        self.addNew = PGPushButton(PGText.physicalParamsAdd)
        self.addNew.setToolTip(PGText.physicalParamsAddToolTip)
        self.addNew.clicked.connect(self.onAddPoints)
        self.inwidget.layout().addWidget(self.addNew)

        self.gridNumberInfo = PGLabel(
            PGText.physicalParamsNumberSummary
            % (
                len(self.mainW.parameters.gridsList.keys()),
                np.sum(list(self.mainW.parameters.gridsList.values())),
            )
            if len(self.mainW.parameters.gridsList.keys()) > 0
            else PGText.physicalParamsEmpty
        )
        self.inwidget.layout().addWidget(self.gridNumberInfo)

        self.tableList = {}
        for idg in sorted(self.mainW.parameters.gridsList.keys()):
            self.tableList[idg] = ResumeGrid(idg, self.mainW, self)
            self.inwidget.layout().addWidget(self.tableList[idg])
        self.setBackgroundRole(QPalette.Base)
        self.setWidget(self.inwidget)
        self.setWidgetResizable(True)

    def onAddPoints(self):
        """Ask a new parameter point/grid to be added to the list"""
        self.mainW.parameters.resetCurrent()
        ep = EditParameters(self.mainW, self)
        if ep.exec_():
            ep.processOutput()


class ConfigNuclidesOutput(QWidget):
    """Dialogs to create two drag&drop table widgets for the selection
    of nuclides to be added to the final output
    """

    def __init__(self, parent=None):
        """Set some initial values for the default properties

        Parameters:
            parent: the parent widget
        """
        self.mainW = parent
        super(ConfigNuclidesOutput, self).__init__(parent)
        self.items = []
        self.selected = []

        layout = QGridLayout()
        self.setLayout(layout)

        self.listAll = DDTableWidget(self, PGText.nuclidesOthersTitle)
        self.listAll.setToolTip(PGText.nuclidesOthersToolTip)
        self.listSel = DDTableWidget(self, PGText.nuclidesSelectedTitle)
        self.listSel.setToolTip(PGText.nuclidesSelectedToolTip)
        layout.addWidget(self.listSel, 0, 0, 4, 1)
        layout.addWidget(self.listAll, 0, 2, 4, 1)

        self.selAllButton = PGPushButton(PGText.nuclidesSelectAllText)
        self.selAllButton.clicked.connect(lambda x: self.fillNuclides(addall=True))
        self.selAllButton.setToolTip(PGText.nuclidesSelectAllToolTip)
        layout.addWidget(self.selAllButton, 1, 1)
        self.unselAllButton = PGPushButton(PGText.nuclidesUnselectAllText)
        self.unselAllButton.clicked.connect(self.unselectAll)
        self.unselAllButton.setToolTip(PGText.nuclidesUnselectAllToolTip)
        layout.addWidget(self.unselAllButton, 2, 1)

        self.fillNuclides()

    def readCurrent(self):
        """Read the current list and prepare `self.selected`"""
        self.selected = list(
            map(
                lambda r: self.listSel.item(r, 0).data(Qt.UserRole),
                range(self.listSel.rowCount()),
            )
        )

    def unselectAll(self):
        """Unselect all the nuclides (move them to listAll)"""
        self.selected = []
        self.fillNuclides(clear=False)

    def fillNuclides(self, clear=True, addall=False):
        """Add the various nuclides to the two drag&drop tables,
        depending if they are currently selected or not,
        creating for each one its QTableWidgetItem

        Parameter:
            clear (default True): if True, reset the current selection
                and add all the available nuclides to the selection
        """
        isel = 0
        iall = 0
        if clear:
            self.selected = [
                n
                for n in self.mainW.nuclides.current.keys()
                if (
                    addall
                    or Nuclides.nuclideOrder[n]
                    < Configuration.limitNuclides["smallNet"]
                )
            ]
        self.clearNuclides()
        for n in sorted(
            self.mainW.nuclides.current.keys(), key=lambda n: Nuclides.nuclideOrder[n]
        ):
            cont = Nuclides.all[n]
            if isinstance(cont, QImage):
                item = QTableWidgetItem("")
                item.setData(Qt.DecorationRole, QPixmap(cont))
            else:
                item = QTableWidgetItem(cont)
            item.setData(Qt.UserRole, n)
            item.setFlags(
                Qt.ItemIsSelectable
                | Qt.ItemIsEnabled
                | Qt.ItemIsDragEnabled
                | Qt.ItemIsDropEnabled
            )
            self.items.append(item)
            if n in self.selected:
                self.listSel.insertRow(isel)
                self.listSel.setItem(isel, 0, item)
                isel += 1
            else:
                self.listAll.insertRow(iall)
                self.listAll.setItem(iall, 0, item)
                iall += 1

    def clearNuclides(self):
        """Delete the current content of the two tables
        and set them to empty"""
        del self.items
        self.items = []
        self.listAll.clearContents()
        self.listSel.clearContents()
        self.listAll.setRowCount(0)
        self.listSel.setRowCount(0)


class MWOutputPanel(QFrame):
    """`QFrame` extension to create a panel where to select
    the nuclides that go in the output and the file/folder names
    """

    def __init__(self, parent=None):
        """Extension of `QFrame.__init__`, adds a layout and few items

        Parameter:
            parent: the parent widget (a MainWindow instance)
        """
        super(MWOutputPanel, self).__init__(parent)
        self.mainW = parent

        layout = QGridLayout()
        self.setLayout(layout)

        descLabel = PGLabel("")
        descLabel.setCenteredText(PGText.outputPanelSelectDescription)
        layout.addWidget(descLabel, 0, 0)
        self.nuclidesOutput = ConfigNuclidesOutput(self.mainW)
        self.mainW.nuclides.updated.connect(self.nuclidesOutput.fillNuclides)
        layout.addWidget(self.nuclidesOutput, 1, 0, 7, 1)

        layout.addWidget(PGLabel(PGText.outputPanelDirectoryAsk), 2, 1)
        self.outputFolderName = PGLabelButton(
            text=PGText.outputPanelDirectoryInitialTitle
        )
        self.outputFolderName.setToolTip(PGText.outputPanelDirectoryToolTip)
        self.outputFolderName.setMaximumWidth(
            QGuiApplication.primaryScreen().availableGeometry().width() * 0.2
        )
        self.outputFolderName.clicked.connect(self.askDirectoryName)
        layout.addWidget(self.outputFolderName, 3, 1)

        layout.addWidget(PGLabel(""), 4, 1)
        layout.addWidget(PGLabel(PGText.outputPanelNuclidesInOutput), 5, 1)
        self.nuclidesInOutput = QCheckBox("")
        self.nuclidesInOutput.setChecked(True)
        layout.addWidget(self.nuclidesInOutput, 6, 1)

    def askDirectoryName(self):
        """Ask the name of the new directory and save it in the button text"""
        directory = askDirName(
            self, PGText.outputPanelDirectoryDialogTitle, self.outputFolderName.text()
        )
        if directory.strip() != "":
            if " " in directory:
                pGUIErrorManager.error(PGText.runPanelSpaceInFolderName)
                return
            self.outputFolderName.setText(directory.strip())
            self.mainW.runSettingsTab.runPanel.saveDefault.setToolTip(
                PGText.runPanelSaveDefaultToolTip % directory.strip()
            )
            self.mainW.runSettingsTab.runPanel.saveCustom.setToolTip(
                PGText.runPanelSaveCustomToolTip % directory.strip()
            )


class MWRunPanel(QFrame):
    """`QFrame` extension to create a panel where the "run" buttons
    will be placed, together with some comment and the progress bar
    """

    def __init__(self, parent=None):
        """Extension of `QFrame.__init__`, adds a layout,
        some text, some buttons and a progress bar to the frame

        Parameter:
            parent: the parent widget (a MainWindow instance)
        """
        super(MWRunPanel, self).__init__(parent)
        self.mainW = parent

        layout = QGridLayout()
        self.setLayout(layout)

        self.text = PGLabel("")
        self.text.setCenteredText(PGText.runPanelDescription)
        layout.addWidget(self.text, 0, 0, 1, 4)

        self.runDefault = PGPushButton(PGText.runPanelDefaultTitle)
        self.runDefault.setToolTip(PGText.runPanelDefaultToolTip)
        self.runDefault.clicked.connect(self.startRunDefault)
        layout.addWidget(self.runDefault, 1, 1)
        self.saveDefault = PGPushButton(PGText.runPanelSaveDefaultTitle)
        self.saveDefault.setToolTip(PGText.runPanelSaveDefaultToolTip % "folder")
        self.saveDefault.clicked.connect(self.prepareRun)
        layout.addWidget(self.saveDefault, 1, 2)

        self.runCustom = PGPushButton(PGText.runPanelCustomTitle)
        self.runCustom.setToolTip(PGText.runPanelCustomToolTip)
        self.runCustom.clicked.connect(self.startRunCustom)
        layout.addWidget(self.runCustom, 2, 1)
        self.saveCustom = PGPushButton(PGText.runPanelSaveCustomTitle)
        self.saveCustom.setToolTip(PGText.runPanelSaveCustomToolTip % "folder")
        self.saveCustom.clicked.connect(lambda: self.prepareRun(default=False))
        layout.addWidget(self.saveCustom, 2, 2)

        self.statusLabel = PGLabel("")
        layout.addWidget(self.statusLabel, 3, 0, 1, 4)

        self.stopButton = PGPushButton(PGText.runPanelStopTitle)
        self.stopButton.setToolTip(PGText.runPanelStopToolTip)
        self.stopButton.clicked.connect(self.stopRun)
        layout.addWidget(self.stopButton, 4, 1, 1, 2)
        self.stopButton.setDisabled(True)

        self.progressBar = QProgressBar(self)
        self.progressBar.setToolTip(PGText.runPanelProgressBarToolTip)
        self.progressBar.setMinimum(0)
        layout.addWidget(self.progressBar, 5, 0, 1, 4)

    def startRunDefault(self):
        """Alias for self.prepareRun(default=True);self.startRun()
        The output is the one from self.prepareRun
        """
        try:
            cp, gp = self.prepareRun()
        except TypeError:
            return None, None
        else:
            self.startRun()
            return cp, gp

    def startRunCustom(self):
        """Alias for self.prepareRun(default=False);self.startRun()
        The output is the one from self.prepareRun
        """
        try:
            cp, gp = self.prepareRun(default=False)
        except TypeError:
            return None, None
        else:
            self.startRun()
            return cp, gp

    def startRunPickle(self, foldername):
        """Read the settings stored with pickle.dump,
        prepare a RunPArthENoPE instance and start the runs

        Parameter:
            foldername: the name of the folder
                where the settings.obj file is stored

        The output is the same as for self.prepareRun
        """
        filename = os.path.join(foldername, Configuration.runSettingsObj)
        if os.path.isfile(filename):
            mainlogger.info(PGText.tryToOpenFolder % foldername)
            try:
                with open(filename, "rb") as _f:
                    commonParams, gridPoints = pickle.load(_f)
            except (EOFError, FileNotFoundErrorClass, UnicodeDecodeError, ValueError):
                pass
            else:
                commonParams = paramsRealPath(commonParams, foldername)
                self.mainW.runner = RunPArthENoPE(
                    commonParams, gridPoints, parent=self.mainW
                )
                mainlogger.info(PGText.startRun)
                self.startRun()
                return commonParams, gridPoints
        pGUIErrorManager.error(PGText.errorCannotLoadGrid)

    def prepareRun(self, default=True):
        """Read the various settings and prepare
        the RunPArthENoPE instance for the parallel runs

        Parameter:
            default (default True): if True, use the default settings
                for the PArthENoPE runs instead of the custom ones

        Output:
            (commonParams, gridPoints):
                commonParams: a dictionary
                    with all the common run params
                gridPoints: a np.ndarray with the list of points
                    for the fortran code runs
        """
        now = datetime.today().strftime("%y%m%d_%H%M%S")
        outParams = self.mainW.runSettingsTab.outputParams
        outParams.nuclidesOutput.readCurrent()
        if (
            outParams.outputFolderName.text() == PGText.outputPanelDirectoryInitialTitle
            or outParams.outputFolderName.text().strip() == ""
        ):
            pGUIErrorManager.warning(PGText.errorNoOutputFolder)
            return
        netParams = self.mainW.runSettingsTab.networkParams
        if default:
            netParams.smallNet.setChecked(True)
            netParams.updateReactions()
        net = None
        for attr in PGText.networkDescription.keys():
            if getattr(netParams, attr).isChecked():
                net = attr
        if net is None:
            pGUIErrorManager.warning(PGText.errorNoReactionNetwork)
            return
        if not default and len(self.mainW.parameters.gridsList) == 0:
            pGUIErrorManager.warning(PGText.errorNoPhysicalParams)
            return
        outputFolder = outParams.outputFolderName.text()
        outParams.nuclidesOutput.readCurrent()
        outputNuclides = (
            [
                k
                for k in sorted(
                    Nuclides.all.keys(), key=lambda x: Nuclides.nuclideOrder[x]
                )
                if k in outParams.nuclidesOutput.selected
            ]
            if not default
            else sorted(
                list(self.mainW.nuclides.current.keys()),
                key=lambda x: Nuclides.nuclideOrder[x],
            )
        )
        changed_rates_list = (
            [
                netParams.reactionsTable.readline(row)
                for row in range(netParams.reactionsTable.rowCount())
            ]
            if not default
            else []
        )
        changed_rates_list = [a for a in changed_rates_list if a[1] > 0]
        commonParams = {
            "num_nuclides_net": Configuration.limitNuclides[net] - 1,
            "N_changed_rates": 0 if default else len(changed_rates_list),
            "changed_rates_list": changed_rates_list,
            "onScreenOutput": False,
            "N_stored_nuclides": len(outputNuclides)
            if not default
            else Configuration.limitNuclides["smallNet"] - 1,
            "stored_nuclides": outputNuclides,
            "output_overwrite": True,
            "output_save_nuclides": outParams.nuclidesInOutput.isChecked() or default,
            "output_folder": outputFolder,
            "output_file_parthenope": os.path.join(
                outputFolder,
                "%s_%s" % (Configuration.parthenopeFilename, now) + "_%d.out",
            ),
            "output_file_nuclides": os.path.join(
                outputFolder,
                "%s_%s" % (Configuration.nuclidesFilename, now) + "_%d.out",
            ),
            "output_file_info": os.path.join(
                outputFolder,
                "%s_%s" % (Configuration.infoFilename, now) + "_%d.out",
            ),
            "output_file_grid": os.path.join(
                outputFolder, "%s_%s.out" % (Configuration.parthenopeFilename, now)
            ),
            "output_file_run": os.path.join(outputFolder, "run_" + now + "_%d.out"),
            "output_file_log": os.path.join(
                outputFolder,
                "%s_" % Configuration.fortranOutputFilename + now + "_%d.out",
            ),
            "output_fortran_log": os.path.join(
                outputFolder, "fortran_output_%s.out" % now
            ),
            "inputcard_filename": os.path.join(
                outputFolder, "input_" + now + "_%d.card"
            ),
            "now": now,
        }
        gridPoints = np.asarray(
            [
                self.mainW.parameters.getGridPoints(gid)
                for gid in self.mainW.parameters.gridsList.keys()
            ]
            if not default
            else [
                [
                    [
                        self.mainW.parameters.all[p].defaultval
                        for p in Parameters.paramOrder
                    ]
                ]
            ]
        )
        self.mainW.runner = RunPArthENoPE(commonParams, gridPoints, parent=self.mainW)
        with open(os.path.join(outputFolder, Configuration.runSettingsObj), "wb") as _f:
            pickle.dump([paramsHidePath(commonParams, outputFolder), gridPoints], _f)
        return commonParams, gridPoints

    def startRun(self):
        """Create a number of objects, disable parts of the GUI
        and start the runs of the fortran code
        """
        self.progressBar.setValue(0)
        self.progressBar.setMaximum(self.mainW.runner.totalRuns)
        self.mainW.runner.updateStatus.connect(self.statusLabel.setCenteredText)
        self.mainW.runner.updateProgressBar.connect(self.progressBar.setValue)
        self.mainW.runner.runHasFinished.connect(self.mainW.runner.oneHasFinished)
        self.mainW.runner.poolHasFinished.connect(self.enableAll)
        self.mainW.runner.poolHasFinished.connect(self.mainW.runner.allHaveFinished)
        self.mainW.runSettingsTab.outputParams.setEnabled(False)
        self.mainW.runSettingsTab.networkParams.setEnabled(False)
        self.mainW.runSettingsTab.physicsParams.setEnabled(False)
        self.runDefault.setEnabled(False)
        self.saveDefault.setEnabled(False)
        self.runCustom.setEnabled(False)
        self.saveCustom.setEnabled(False)
        self.stopButton.setEnabled(True)
        self.mainW.running = True
        self.mainW.runner.run()

    def stopRun(self):
        """Stop the current run and enable again the relevant items"""
        self.mainW.runner.stop()
        self.enableAll()

    def enableAll(self):
        """Disable some buttons and enable again the settings panels"""
        self.mainW.running = False
        self.mainW.runSettingsTab.outputParams.setEnabled(True)
        self.mainW.runSettingsTab.networkParams.setEnabled(True)
        self.mainW.runSettingsTab.physicsParams.setEnabled(True)
        self.runDefault.setEnabled(True)
        self.saveDefault.setEnabled(True)
        self.runCustom.setEnabled(True)
        self.saveCustom.setEnabled(True)
        self.stopButton.setEnabled(False)


class MWRunSettingsPanel(QFrame):
    """`QFrame` extension to create the tab where the four panels
    related to the run customization and execution are shown
    """

    def __init__(self, parent=None):
        """Extension of `QFrame.__init__`, adds items to the layout

        Parameter:
            parent: the parent widget (a MainWindow instance)
        """
        super(MWRunSettingsPanel, self).__init__(parent)
        self.mainW = parent

        layout = QGridLayout()
        self.setLayout(layout)

        self.runPanel = MWRunPanel(self.mainW)
        self.outputParams = MWOutputPanel(self.mainW)
        self.networkParams = MWNetworkPanel(self.mainW)
        self.physicsParams = MWPhysParamsPanel(self.mainW)

        layout.addWidget(self.networkParams, 0, 0)
        layout.addWidget(self.physicsParams, 0, 1)
        layout.setRowStretch(0, 2)
        layout.addWidget(self.outputParams, 1, 0)
        layout.addWidget(self.runPanel, 1, 1)
        layout.setRowStretch(1, 1)
