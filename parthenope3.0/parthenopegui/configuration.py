"""Configuration for the GUI for the PArthENoPE BBN code"""
import os

import matplotlib
import numpy as np
import six

matplotlib.use("Qt5Agg")
os.environ["QT_API"] = "pyside2"
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PySide2.QtCore import QObject, QSignalBlocker, Qt, Signal
from PySide2.QtGui import QFont, QIcon, QImage, QPixmap
from PySide2.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
)

try:
    from parthenopegui.basic import (
        PGConfiguration,
        __nuclideOrder__,
        __paramOrder__,
        __paramOrderParth__,
        paramsHidePath,
        paramsRealPath,
    )
    from parthenopegui.errorManager import mainlogger, pGUIErrorManager
except ImportError:
    print("[configuration] Necessary parthenopegui submodules not found!")
    raise


class UnicodeSymbols(QObject):
    """Contains some unicode characters that may be used when showing
    mathematical formulas.
    """

    # latex to unicode
    l2u = {
        "leftrightarrow": "\u27f7",
        "rightarrow": "\u27f6",
        "mu": "\u03bc",
        "sigma": "\u03c3",
        "tau": "\u03c4",
        "Omega": "\u03a9",
        "Lambda": "\u039b",
        "pi": "\u03c0",
        "rho": "\u03c1",
        "csi": "\u03be",
        "nu": "\u03bd",
        "alpha": "\u03b1",
        "gamma": "\u03b3",
        "Delta": "\u0394",
        "^1": "\u00b9",
        "^2": "\u00b2",
        "^3": "\u00b3",
        "^4": "\u2074",
        "^6": "\u2076",
        "^7": "\u2077",
        "^8": "\u2078",
        "^9": "\u2079",
        "^{1}": "\u00b9",
        "^{2}": "\u00b2",
        "^{3}": "\u00b3",
        "^{4}": "\u2074",
        "^{6}": "\u2076",
        "^{7}": "\u2077",
        "^{8}": "\u2078",
        "^{9}": "\u2079",
        "^{10}": "\u00b9" "\u2070",
        "^{11}": "\u00b9" "\u00b9",
        "^{12}": "\u00b9" "\u00b2",
        "^{13}": "\u00b9" "\u00b3",
        "^{14}": "\u00b9" "\u2074",
        "^{15}": "\u00b9" "\u2075",
        "^{16}": "\u00b9" "\u2076",
        "^-": "\u207b",
        "^{-}": "\u207b",
        "^+": "\u207a",
        "^{+}": "\u207a",
        "bar": "\u0305",
        "barnue": "\u0305\u03bd\u2091",
        "nue": "\u03bd\u2091",
    }


class Configuration(QObject, PGConfiguration):
    """Contains common strings, lists, dictionaries and variables"""

    def __init__(self):
        """For the moment empty constructor"""
        super(Configuration, self).__init__()


def askDirName(parent=None, title="Directory to use:", dir=""):
    """Uses `QFileDialog` to ask the names of a single directory

    Parameters (all optional):
        parent (default None): the parent of the window
        title: the window title
        dir: the initial directory

    Output:
        return the directory name
        or an empty string depending if the user selected "Ok" or "Cancel"
    """
    return QFileDialog.getExistingDirectory(parent, caption=title, dir=dir)


def askFileName(parent=None, title="Filename to use:", filter="", dir=""):
    """Uses `QFileDialog` to ask the name of a single, existing file

    Parameters (all optional):
        parent (default None): the parent of the window
        title: the window title
        filter: the filter to be used when displaying files
        dir: the initial directory

    Output:
        return the filename
        or an empty string if the user selected "Cancel"
    """
    result = QFileDialog.getOpenFileName(
        parent,
        caption=title,
        dir=dir,
        filter=filter,
        options=QFileDialog.DontConfirmOverwrite,
    )
    try:
        return result[0]
    except (IndexError, TypeError):
        return ""


def askFileNames(parent=None, title="Filename to use:", filter="", dir=""):
    """Uses `QFileDialog` to ask the names of a set of existing files

    Parameters (all optional):
        parent (default None): the parent of the window
        title: the window title
        filter: the filter to be used when displaying files
        dir: the initial directory

    Output:
        return the filenames list
        or an empty list depending if the user selected "Ok" or "Cancel"
    """
    result = QFileDialog.getOpenFileNames(
        parent,
        caption=title,
        dir=dir,
        filter=filter,
        options=QFileDialog.DontConfirmOverwrite,
    )
    try:
        return result[0]
    except (IndexError, TypeError):
        return ""


def askGenericText(message, title, parent=None):
    """Uses `QInputDialog` to ask a text answer for a given question.

    Parameters:
        message: the question to be displayed
        title: the window title
        parent (optional, default None): the parent of the window

    Output:
        return a tuple containing the text as first element
        and True/False
            (depending if the user selected "Ok" or "Cancel")
            as the second element
    """
    dialog = QInputDialog(parent)
    dialog.setInputMode(QInputDialog.TextInput)
    dialog.setWindowTitle(title)
    dialog.setLabelText(message)
    if dialog.exec_():
        return dialog.textValue()
    return ""


def askSaveFileName(parent=None, title="Filename to use:", filter="", dir=""):
    """Uses `QFileDialog` to ask the names of a single file
    where something will be saved (the file may not exist)

    Parameters (all optional):
        parent (default None): the parent of the window
        title: the window title
        filter: the filter to be used when displaying files
        dir: the initial directory

    Output:
        return the filename
    """
    result = QFileDialog.getSaveFileName(
        parent,
        caption=title,
        dir=dir,
        filter=filter,
        options=QFileDialog.DontConfirmOverwrite,
    )
    try:
        return result[0]
    except (IndexError, TypeError):
        return ""


def askYesNo(message, title="Question"):
    """Uses `QMessageBox` to ask "Yes" or "No" for a given question.

    Parameters:
        message: the question to be displayed
        title: the window title

    Output:
        return True if the "Yes" button has been clicked,
            False otherwise
    """
    mbox = QMessageBox(QMessageBox.Question, title, message)
    yesButton = mbox.addButton(QMessageBox.Yes)
    noButton = mbox.addButton(QMessageBox.No)
    mbox.setDefaultButton(noButton)
    mbox.exec_()
    return mbox.clickedButton() == yesButton


class PGFont(QFont):
    """Extend QFont with a custom fontsize"""

    def __init__(self, *args, **kwargs):
        """Call constructor and set custom fontsize"""
        QFont.__init__(self, "Helvetica", *args, **kwargs)
        self.setPointSize(Configuration.fontsize)


class PGLabel(QLabel):
    """Extend QLabel with custom font"""

    def __init__(self, *args, **kwargs):
        """Call the constructor, set the font"""
        QLabel.__init__(self, *args, **kwargs)
        self.setFont(PGFont())
        self.setWordWrap(True)

    def setCenteredText(self, text):
        """Set a centered text in the label"""
        self.setText("<center>%s</center>" % text)


class PGPushButton(QPushButton):
    """Extend QPushButton with custom font"""

    def __init__(self, *args, **kwargs):
        """Call the constructor and set the font"""
        QPushButton.__init__(self, *args, **kwargs)
        self.setFont(PGFont())


# adapted from https://stackoverflow.com/a/62893567
class PGLabelButton(PGPushButton):
    """Extension of PGPushButton that contains a QLabel for word-wrapping
    the text. Includes auto-resizing of the button when updated
    """

    __m = 20
    __s = 2

    def __init__(self, parent=None, text=None):
        """Construct the PGPushButton, the PGLabel, add it to the layout
        and define all the properties of the widget
        """
        if parent is not None:
            super(PGLabelButton, self).__init__(parent)
        else:
            super(PGLabelButton, self).__init__()
        self.__lyt = QHBoxLayout()
        self.setLayout(self.__lyt)
        self.__lyt.setContentsMargins(self.__m, self.__m, self.__m, self.__m)
        self.__lyt.setSpacing(self.__s)
        self.__lbl = PGLabel(self)
        self.__lyt.addWidget(self.__lbl)
        if text is not None:
            self.__lbl.setText(text)
        self.__lbl.setAttribute(Qt.WA_TranslucentBackground)
        self.__lbl.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.__lbl.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

    def setMaximumWidth(self, w):
        """Pass the maximum width to the QLabel, apart from setting
        it to the QPushButton

        Parameter:
            w: the new maximum width
        """
        QPushButton.setMaximumWidth(self, w)
        self.__lbl.setMaximumWidth(w)

    def setText(self, text):
        """Replace the label text and update the geometry of the button

        Parameter:
            text: the new text for the label
        """
        self.__lbl.setText(text)
        self.updateGeometry()

    def sizeHint(self):
        """Determine the sizeHint for the QPushButton
        depending on the QLabel one.

        Output:
            a QSize instance
        """
        s = QPushButton.sizeHint(self)
        w = self.__lbl.sizeHint()
        s.setWidth(w.width() + 2 * self.__m)
        s.setHeight(w.height() + 2 * self.__m)
        return s

    def text(self):
        """Return the text from the QLabel
        instead of the one from the QPushButton

        Output:
            the text of the QLabel
        """
        return self.__lbl.text()


def imageFromTex(mathTex):
    """Use matplotlib to convert mathTex in a QImage

    Parameters:
        mathTex: the latex text to convert

    Output:
        a QImage
    """
    fig = plt.Figure()
    fig.patch.set_facecolor("w")
    fig.set_canvas(FigureCanvasAgg(fig))
    renderer = fig.canvas.get_renderer()

    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.patch.set_facecolor("w")
    t = ax.text(0, 0, mathTex, ha="left", va="bottom", fontsize=Configuration.fontsize)

    fwidth, fheight = fig.get_size_inches()
    fig_bbox = fig.get_window_extent(renderer)

    try:
        text_bbox = t.get_window_extent(renderer)
    except ValueError:
        mainlogger.exception("Error when converting latex to image")
        return None

    tight_fwidth = text_bbox.width * fwidth / fig_bbox.width
    tight_fheight = 1.05 * text_bbox.height * fheight / fig_bbox.height

    fig.set_size_inches(tight_fwidth, tight_fheight)

    buf, size = fig.canvas.print_to_buffer()
    qimage = QImage.rgbSwapped(QImage(buf, size[0], size[1], QImage.Format_ARGB32))
    return qimage


def cacheImageFromTex(string, name=""):
    """Read the string that describes a reaction and convert
    using unicode correspondence with latex-like coding
    """
    fname = os.path.join(Configuration.cachePath, "%s.png" % name)
    if name != "" and os.path.exists(fname):
        qimage = QImage()
        if qimage.load(fname):
            return qimage
    qimage = imageFromTex(string)
    if name != "":
        if not os.path.exists(Configuration.cachePath):
            os.makedirs(Configuration.cachePath)
        qimage.save(fname)
    return qimage


class DDTableWidget(QTableWidget):
    """Drag and drop extension of QTableWidget"""

    def __init__(self, parent, header):
        """Set some properties and settings.

        Parameters:
            header: the title of the column
        """
        super(DDTableWidget, self).__init__(parent)
        self.setColumnCount(1)
        self.setHorizontalHeaderLabels([header])
        self.horizontalHeader().setStretchLastSection(True)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setDragDropOverwriteMode(False)
        self.lastDropRow = None
        self.setFont(PGFont())
        self.cellChanged.connect(self.reorder)

    def reorder(self, row, col):
        """Read current table content and reorder the list of nuclides
        according to the default order defined in
        configuration.Nuclides.nuclideOrder
        """
        items = []
        for row in range(self.rowCount()):
            try:
                nuclide = self.item(row, 0).data(Qt.UserRole)
            except AttributeError:
                mainlogger.debug(
                    "probably drag&dropping several nuclides at each time. Reorder later"
                )
                return
            label = [k for k in Nuclides.all.keys() if k == nuclide]
            try:
                items.append(label[0])
            except IndexError:
                mainlogger.debug("missing nuclide? %s" % nuclide)
        sortedItems = sorted(items, key=lambda n: Nuclides.nuclideOrder[n])
        qsb = QSignalBlocker(self)
        for row in range(self.rowCount()):
            cont = Nuclides.all[sortedItems[row]]
            if isinstance(cont, QImage):
                self.item(row, 0).setText("")
                self.item(row, 0).setData(Qt.DecorationRole, cont)
            else:
                self.item(row, 0).setText(cont)
                self.item(row, 0).setData(Qt.DecorationRole, None)
            self.item(row, 0).setData(Qt.UserRole, sortedItems[row])

    def dropMimeData(self, row, col, mimeData, action):
        """Overridden method to get the row index for insertion

        Parameters:
            row: the index of the dropped row
            col, mimeData, action: not used params
                (see the signature of `QTableWidget.dropMimeData`)
        """
        self.lastDropRow = row
        return True

    def dropEvent(self, event):
        """Accept dropEvent and move the selected item
        to the new position

        Parameter:
            event: a `QDropEvent`
        """
        sender = event.source()
        super(DDTableWidget, self).dropEvent(event)
        dropRow = self.lastDropRow
        selectedRows = sender.getselectedRowsFast()
        for _ in selectedRows:
            self.insertRow(dropRow)
        sel_rows_offsets = [
            0 if self != sender or srow < dropRow else len(selectedRows)
            for srow in selectedRows
        ]
        selectedRows = [
            row + offset for row, offset in zip(selectedRows, sel_rows_offsets)
        ]
        for i, srow in enumerate(selectedRows):
            for j in range(self.columnCount()):
                item = sender.item(srow, j)
                if item:
                    source = QTableWidgetItem(item)
                    self.setItem(dropRow + i, j, source)
        for srow in reversed(selectedRows):
            sender.removeRow(srow)
        event.accept()

    def getselectedRowsFast(self):
        """Return the list of selected rows

        Output:
            a list with the row indexes of the selected rows
        """
        selectedRows = []
        for item in self.selectedItems():
            row = item.row()
            text = item.text()
            if row not in selectedRows:
                selectedRows.append(row)
        selectedRows.sort()
        return selectedRows


class Nuclides(QObject):
    """Definitions and properties related to the nuclides"""

    updated = Signal()

    all = {
        "n": cacheImageFromTex("$n$", "n"),
        "p": cacheImageFromTex("$p$", "p"),
        "2H": cacheImageFromTex("$^{2}$H", "2H"),
        "3H": cacheImageFromTex("$^{3}$H", "3H"),
        "3He": cacheImageFromTex("$^{3}$He", "3He"),
        "4He": cacheImageFromTex("$^{4}$He", "4He"),
        "6Li": cacheImageFromTex("$^{6}$Li", "6Li"),
        "7Li": cacheImageFromTex("$^{7}$Li", "7Li"),
        "7Be": cacheImageFromTex("$^{7}$Be", "7Be"),
        "8Li": cacheImageFromTex("$^{8}$Li", "8Li"),
        "8B": cacheImageFromTex("$^{8}$B", "8B"),
        "9Be": cacheImageFromTex("$^{9}$Be", "9Be"),
        "10B": cacheImageFromTex("$^{10}$B", "10B"),
        "11B": cacheImageFromTex("$^{11}$B", "11B"),
        "11C": cacheImageFromTex("$^{11}$C", "11C"),
        "12B": cacheImageFromTex("$^{12}$B", "12B"),
        "12C": cacheImageFromTex("$^{12}$C", "12C"),
        "12N": cacheImageFromTex("$^{12}$N", "12N"),
        "13C": cacheImageFromTex("$^{13}$C", "13C"),
        "13N": cacheImageFromTex("$^{13}$N", "13N"),
        "14C": cacheImageFromTex("$^{14}$C", "14C"),
        "14N": cacheImageFromTex("$^{14}$N", "14N"),
        "14O": cacheImageFromTex("$^{14}$O", "14O"),
        "15N": cacheImageFromTex("$^{15}$N", "15N"),
        "15O": cacheImageFromTex("$^{15}$O", "15O"),
        "16O": cacheImageFromTex("$^{16}$O", "16O"),
    }
    nuclideOrder = __nuclideOrder__
    current = {}

    def updateCurrent(self, new):
        """Update the list of currently available nuclides

        Parameter:
            new: list of keys of the new current nuclides
        """
        self.current = {n: self.all[n] for n in new if n in self.all.keys()}
        self.updated.emit()


class Reactions(QObject):
    """Definitions and properties related to the reactions"""

    updated = Signal()

    all = {
        1: (cacheImageFromTex(r"$n \leftrightarrow p$", "reaction1"), "weak"),
        2: (
            cacheImageFromTex(
                r"$^{3}{\rm H} \rightarrow \bar\nu_e  + e^{-} + ^{3}{\rm He}$",
                "reaction2",
            ),
            "weak",
        ),
        3: (
            cacheImageFromTex(
                r"$^{8}{\rm Li} \rightarrow \bar\nu_e + e^{-} + 2 ^{4}{\rm He}$",
                "reaction3",
            ),
            "weak",
        ),
        4: (
            cacheImageFromTex(
                r"$^{12}{\rm B} \rightarrow \bar\nu_e + e^{-} + ^{12}{\rm C}$",
                "reaction4",
            ),
            "weak",
        ),
        5: (
            cacheImageFromTex(
                r"$^{14}{\rm C} \rightarrow \bar\nu_e + e^{-} + ^{14}{\rm N}$",
                "reaction5",
            ),
            "weak",
        ),
        6: (
            cacheImageFromTex(
                r"$^{8}{\rm B} \rightarrow \nu_e + e^{+} + 2  ^{4}{\rm He}$",
                "reaction6",
            ),
            "weak",
        ),
        7: (
            cacheImageFromTex(
                r"$^{11}{\rm C} \rightarrow \nu_e + e^{+} + ^{11}{\rm B}$", "reaction7"
            ),
            "weak",
        ),
        8: (
            cacheImageFromTex(
                r"$^{12}{\rm N} \rightarrow \nu_e + e^{+} + ^{12}{\rm C}$", "reaction8"
            ),
            "weak",
        ),
        9: (
            cacheImageFromTex(
                r"$^{13}{\rm N} \rightarrow \nu_e + e^{+} + ^{13}{\rm C}$", "reaction9"
            ),
            "weak",
        ),
        10: (
            cacheImageFromTex(
                r"$^{14}{\rm O} \rightarrow \nu_e + e^{+} + ^{14}{\rm N}$", "reaction10"
            ),
            "weak",
        ),
        11: (
            cacheImageFromTex(
                r"$^{15}{\rm O} \rightarrow \nu_e + e^{+} + ^{15}{\rm N}$", "reaction11"
            ),
            "weak",
        ),
        12: (
            cacheImageFromTex(
                r"$p + n \leftrightarrow \gamma + ^{2}{\rm H}$", "reaction12"
            ),
            cacheImageFromTex(r"$(n, \gamma)$", "ngamma"),
        ),
        13: (
            cacheImageFromTex(
                r"$^{2}{\rm H} + n \leftrightarrow \gamma + ^{3}{\rm H}$", "reaction13"
            ),
            cacheImageFromTex(r"$(n, \gamma)$", "ngamma"),
        ),
        14: (
            cacheImageFromTex(
                r"$^{3}{\rm He} + n \leftrightarrow \gamma + ^{4}{\rm He}$",
                "reaction14",
            ),
            cacheImageFromTex(r"$(n, \gamma)$", "ngamma"),
        ),
        15: (
            cacheImageFromTex(
                r"$^{6}{\rm Li} + n \leftrightarrow \gamma + ^{7}{\rm Li}$",
                "reaction15",
            ),
            cacheImageFromTex(r"$(n, \gamma)$", "ngamma"),
        ),
        16: (
            cacheImageFromTex(
                r"$^{3}{\rm He} + n \leftrightarrow p + ^{3}{\rm H}$", "reaction16"
            ),
            "charge ex.",
        ),
        17: (
            cacheImageFromTex(
                r"$^{7}{\rm Be} + n \leftrightarrow p + ^{7}{\rm Li}$", "reaction17"
            ),
            "charge ex.",
        ),
        18: (
            cacheImageFromTex(
                r"$^{6}{\rm Li} + n \leftrightarrow ^{3}{\rm H} + ^{4}{\rm He}$",
                "reaction18",
            ),
            cacheImageFromTex(r"$^3$H Pickup", "3Hpick"),
        ),
        19: (
            cacheImageFromTex(
                r"$^{7}{\rm Be} + n \leftrightarrow ^{4}{\rm He} + ^{4}{\rm He}$",
                "reaction19",
            ),
            cacheImageFromTex(r"$^4$He Pickup", "4Hepick"),
        ),
        20: (
            cacheImageFromTex(
                r"$^{2}{\rm H} + p \leftrightarrow \gamma + ^{3}{\rm He}$", "reaction20"
            ),
            cacheImageFromTex(r"$(p, \gamma)$", "pgamma"),
        ),
        21: (
            cacheImageFromTex(
                r"$^{3}{\rm H} + p \leftrightarrow \gamma + ^{4}{\rm He}$", "reaction21"
            ),
            cacheImageFromTex(r"$(p, \gamma)$", "pgamma"),
        ),
        22: (
            cacheImageFromTex(
                r"$^{6}{\rm Li} + p \leftrightarrow \gamma + ^{7}{\rm Be}$",
                "reaction22",
            ),
            cacheImageFromTex(r"$(p, \gamma)$", "pgamma"),
        ),
        23: (
            cacheImageFromTex(
                r"$^{6}{\rm Li} + p \leftrightarrow ^{3}{\rm He} + ^{4}{\rm He}$",
                "reaction23",
            ),
            cacheImageFromTex(r"$^3$He Pickup", "3Hepick"),
        ),
        24: (
            cacheImageFromTex(
                r"$^{7}{\rm Li} + p \leftrightarrow ^{4}{\rm He} + ^{4}{\rm He}$",
                "reaction24",
            ),
            cacheImageFromTex(r"$^4$He Pickup", "4Hepick"),
        ),
        25: (
            cacheImageFromTex(
                r"$^{4}{\rm He} + ^{2}{\rm H} \leftrightarrow \gamma + ^{6}{\rm Li}$",
                "reaction25",
            ),
            cacheImageFromTex(r"$(^{2}{\rm H}, \gamma)$", "2Hgamma"),
        ),
        26: (
            cacheImageFromTex(
                r"$^{4}{\rm He} + ^{3}{\rm H} \leftrightarrow \gamma + ^{7}{\rm Li}$",
                "reaction26",
            ),
            cacheImageFromTex(r"$(^{3}{\rm H}, \gamma)$", "3Hgamma"),
        ),
        27: (
            cacheImageFromTex(
                r"$^{4}{\rm He} + ^{3}{\rm He} \leftrightarrow \gamma + ^{7}{\rm Be}$",
                "reaction27",
            ),
            cacheImageFromTex(r"$(^{3}{\rm He} \gamma)$", "3Hegamma"),
        ),
        28: (
            cacheImageFromTex(
                r"$^{2}{\rm H} + ^{2}{\rm H} \leftrightarrow n + ^{3}{\rm He}$",
                "reaction28",
            ),
            cacheImageFromTex(r"$^2$H Strip.", "2Hstrip"),
        ),
        29: (
            cacheImageFromTex(
                r"$^{2}{\rm H} + ^{2}{\rm H} \leftrightarrow p + ^{3}{\rm H}$",
                "reaction29",
            ),
            cacheImageFromTex(r"$^2$H Strip.", "2Hstrip"),
        ),
        30: (
            cacheImageFromTex(
                r"$^{3}{\rm H} + ^{2}{\rm H} \leftrightarrow n + ^{4}{\rm He}$",
                "reaction30",
            ),
            cacheImageFromTex(r"$^2$H Strip.", "2Hstrip"),
        ),
        31: (
            cacheImageFromTex(
                r"$^{3}{\rm He} + ^{2}{\rm H} \leftrightarrow p + ^{4}{\rm He}$",
                "reaction31",
            ),
            cacheImageFromTex(r"$^2$H Strip.", "2Hstrip"),
        ),
        32: (
            cacheImageFromTex(
                r"$^{3}{\rm He} + ^{3}{\rm He} \leftrightarrow p + p + ^{4}{\rm He}$",
                "reaction32",
            ),
            cacheImageFromTex(r"$(^3$He $2p)$", "3Hestrip"),
        ),
        33: (
            cacheImageFromTex(
                r"$^{7}{\rm Li} + ^{2}{\rm H} \leftrightarrow n + ^{4}{\rm He} + ^{4}{\rm He}$",
                "reaction33",
            ),
            cacheImageFromTex(r"$(^{2}{\rm H}, n \alpha)$", "2Hnalpha"),
        ),
        34: (
            cacheImageFromTex(
                r"$^{7}{\rm Be} + ^{2}{\rm H} \leftrightarrow p + ^{4}{\rm He} + ^{4}{\rm He}$",
                "reaction34",
            ),
            cacheImageFromTex(r"$(^{2}{\rm H}, n \alpha)$", "2Hnalpha"),
        ),
        35: (
            cacheImageFromTex(
                r"$^{3}{\rm He} + ^{3}{\rm H} \leftrightarrow \gamma + ^{6}{\rm Li}$",
                "reaction35",
            ),
            cacheImageFromTex(r"$(^{3}{\rm H}, \gamma)$", "3Hgamma"),
        ),
        36: (
            cacheImageFromTex(
                r"$^{6}{\rm Li} + ^{2}{\rm H} \leftrightarrow n + ^{7}{\rm Be}$",
                "reaction36",
            ),
            cacheImageFromTex(r"$^2$H Strip.", "2Hstrip"),
        ),
        37: (
            cacheImageFromTex(
                r"$^{6}{\rm Li} + ^{2}{\rm H} \leftrightarrow p + ^{7}{\rm Li}$",
                "reaction37",
            ),
            cacheImageFromTex(r"$^2$H Strip.", "2Hstrip"),
        ),
        38: (
            cacheImageFromTex(
                r"$^{3}{\rm He} + ^{3}{\rm H} \leftrightarrow ^{2}{\rm H} + ^{4}{\rm He}$",
                "reaction38",
            ),
            cacheImageFromTex(r"$(^{3}{\rm H},\,^{2}{\rm H})$", "3H2H"),
        ),
        39: (
            cacheImageFromTex(
                r"$^{3}{\rm H} + ^{3}{\rm H} \leftrightarrow n + n + ^{4}{\rm He}$",
                "reaction39",
            ),
            cacheImageFromTex(r"$(^{3}{\rm H}, n n)$", "3Hnn"),
        ),
        40: (
            cacheImageFromTex(
                r"$^{3}{\rm He} + ^{3}{\rm H} \leftrightarrow p + n + ^{4}{\rm He}$",
                "reaction40",
            ),
            cacheImageFromTex(r"$(^{3}{\rm H}, n p)$", "3Hnp"),
        ),
        41: (
            cacheImageFromTex(
                r"$^{7}{\rm Li} + ^{3}{\rm H} \leftrightarrow n + ^{9}{\rm Be}$",
                "reaction41",
            ),
            cacheImageFromTex(r"$^{3}{\rm H} Strip.$", "3Hstrip"),
        ),
        42: (
            cacheImageFromTex(
                r"$^{7}{\rm Be} + ^{3}{\rm H} \leftrightarrow p + ^{9}{\rm Be}$",
                "reaction42",
            ),
            cacheImageFromTex(r"$^{3}{\rm H} Strip.$", "3Hstrip"),
        ),
        43: (
            cacheImageFromTex(
                r"$^{7}{\rm Li} + ^{3}{\rm He} \leftrightarrow p + ^{9}{\rm Be}$",
                "reaction43",
            ),
            cacheImageFromTex(r"$^{3}{\rm He} Strip.$", "3Hestrip"),
        ),
        44: (
            cacheImageFromTex(
                r"$^{7}{\rm Li} + n \leftrightarrow \gamma + ^{8}{\rm Li}$",
                "reaction44",
            ),
            cacheImageFromTex(r"$(n, \gamma)$", "ngamma"),
        ),
        45: (
            cacheImageFromTex(
                r"$^{10}{\rm B} + n \leftrightarrow \gamma + ^{11}{\rm B}$",
                "reaction45",
            ),
            cacheImageFromTex(r"$(n, \gamma)$", "ngamma"),
        ),
        46: (
            cacheImageFromTex(
                r"$^{11}{\rm B} + n \leftrightarrow \gamma + ^{11}{\rm B}$",
                "reaction46",
            ),
            cacheImageFromTex(r"$(n, \gamma)$", "ngamma"),
        ),
        47: (
            cacheImageFromTex(
                r"$^{11}{\rm C} + n \leftrightarrow p + ^{11}{\rm B}$", "reaction47"
            ),
            cacheImageFromTex("(n, p)", "np"),
        ),
        48: (
            cacheImageFromTex(
                r"$^{10}{\rm B} + n \leftrightarrow ^{4}{\rm He} + ^{7}{\rm Li}$",
                "reaction48",
            ),
            cacheImageFromTex(r"$(n, \alpha)$", "nalpha"),
        ),
        49: (
            cacheImageFromTex(
                r"$^{7}{\rm Be} + p \leftrightarrow \gamma + ^{8}{\rm B}$", "reaction49"
            ),
            cacheImageFromTex(r"$(p, \gamma)$", "pgamma"),
        ),
        50: (
            cacheImageFromTex(
                r"$^{9}{\rm Be} + p \leftrightarrow \gamma + ^{10}{\rm B}$",
                "reaction50",
            ),
            cacheImageFromTex(r"$(p, \gamma)$", "pgamma"),
        ),
        51: (
            cacheImageFromTex(
                r"$^{10}{\rm B} + p \leftrightarrow \gamma + ^{11}{\rm C}$",
                "reaction51",
            ),
            cacheImageFromTex(r"$(p, \gamma)$", "pgamma"),
        ),
        52: (
            cacheImageFromTex(
                r"$^{11}{\rm B} + p \leftrightarrow \gamma + ^{12}{\rm C}$",
                "reaction52",
            ),
            cacheImageFromTex(r"$(p, \gamma)$", "pgamma"),
        ),
        53: (
            cacheImageFromTex(
                r"$^{11}{\rm C} + p \leftrightarrow \gamma + ^{12}{\rm N}$",
                "reaction53",
            ),
            cacheImageFromTex(r"$(p, \gamma)$", "pgamma"),
        ),
        54: (
            cacheImageFromTex(
                r"$^{12}{\rm B} + p \leftrightarrow n + ^{12}{\rm C}$", "reaction54"
            ),
            cacheImageFromTex("(p, n)", "pn"),
        ),
        55: (
            cacheImageFromTex(
                r"$^{9}{\rm Be} + p \leftrightarrow ^{4}{\rm He} + ^{6}{\rm Li}$",
                "reaction55",
            ),
            cacheImageFromTex(r"$(p, \alpha)$", "palpha"),
        ),
        56: (
            cacheImageFromTex(
                r"$^{10}{\rm B} + p \leftrightarrow ^{4}{\rm He} + ^{7}{\rm Be}$",
                "reaction56",
            ),
            cacheImageFromTex(r"$(p, \alpha)$", "palpha"),
        ),
        57: (
            cacheImageFromTex(
                r"$^{12}{\rm B} + p \leftrightarrow ^{4}{\rm He} + ^{9}{\rm Be}$",
                "reaction57",
            ),
            cacheImageFromTex(r"$(p, \alpha)$", "palpha"),
        ),
        58: (
            cacheImageFromTex(
                r"$^{6}{\rm Li} + ^{4}{\rm He} \leftrightarrow \gamma + ^{10}{\rm B}$",
                "reaction58",
            ),
            cacheImageFromTex(r"$(\alpha, \gamma)$", "alphagamma"),
        ),
        59: (
            cacheImageFromTex(
                r"$^{7}{\rm Li} + ^{4}{\rm He} \leftrightarrow \gamma + ^{11}{\rm B}$",
                "reaction59",
            ),
            cacheImageFromTex(r"$(\alpha, \gamma)$", "alphagamma"),
        ),
        60: (
            cacheImageFromTex(
                r"$^{7}{\rm Be} + ^{4}{\rm He} \leftrightarrow \gamma + ^{11}{\rm C}$",
                "reaction60",
            ),
            cacheImageFromTex(r"$(\alpha, \gamma)$", "alphagamma"),
        ),
        61: (
            cacheImageFromTex(
                r"$^{8}{\rm B} + ^{4}{\rm He} \leftrightarrow p + ^{11}{\rm C}$",
                "reaction61",
            ),
            cacheImageFromTex(r"$(\alpha, p)$", "alphap"),
        ),
        62: (
            cacheImageFromTex(
                r"$^{8}{\rm Li} + ^{4}{\rm He} \leftrightarrow n + ^{11}{\rm B}$",
                "reaction62",
            ),
            cacheImageFromTex(r"$(\alpha, n)$", "alphan"),
        ),
        63: (
            cacheImageFromTex(
                r"$^{9}{\rm Be} + ^{4}{\rm He} \leftrightarrow n + ^{12}{\rm C}$",
                "reaction63",
            ),
            cacheImageFromTex(r"$(\alpha, n)$", "alphan"),
        ),
        64: (
            cacheImageFromTex(
                r"$^{9}{\rm Be} + ^{2}{\rm H} \leftrightarrow n + ^{10}{\rm B}$",
                "reaction64",
            ),
            cacheImageFromTex(r"$(^{2}{\rm H}, n)$", "2Hn"),
        ),
        65: (
            cacheImageFromTex(
                r"$^{10}{\rm B} + ^{2}{\rm H} \leftrightarrow p + ^{11}{\rm B}$",
                "reaction65",
            ),
            cacheImageFromTex(r"$(^{2}{\rm H}, p)$", "2Hp"),
        ),
        66: (
            cacheImageFromTex(
                r"$^{11}{\rm B} + ^{2}{\rm H} \leftrightarrow n + ^{12}{\rm C}$",
                "reaction66",
            ),
            cacheImageFromTex(r"$(^{2}{\rm H}, n)$", "2Hn"),
        ),
        67: (
            cacheImageFromTex(
                r"$^{4}{\rm He} + ^{4}{\rm He} + n \leftrightarrow \gamma + ^{9}{\rm Be}$",
                "reaction67",
            ),
            cacheImageFromTex(r"$(\alpha n, \gamma)$", "alphagamma"),
        ),
        68: (
            cacheImageFromTex(
                r"$^{4}{\rm He} + ^{4}{\rm He} + ^{4}{\rm He} \leftrightarrow \gamma + ^{12}{\rm C}$",
                "reaction68",
            ),
            cacheImageFromTex(r"$(\alpha \alpha, \gamma)$", "alphaalphagamma"),
        ),
        69: (
            cacheImageFromTex(
                r"$^{8}{\rm Li} + p \leftrightarrow n + ^{4}{\rm He} + ^{4}{\rm He}$",
                "reaction69",
            ),
            cacheImageFromTex(r"$(p, n \alpha)$", "pnalpha"),
        ),
        70: (
            cacheImageFromTex(
                r"$^{8}{\rm B} + n \leftrightarrow p + ^{4}{\rm He} + ^{4}{\rm He}$",
                "reaction70",
            ),
            cacheImageFromTex(r"$(n, p \alpha)$", "npalpha"),
        ),
        71: (
            cacheImageFromTex(
                r"$^{9}{\rm Be} + p \leftrightarrow ^{2}{\rm H} + ^{4}{\rm He} + ^{4}{\rm He}$",
                "reaction71",
            ),
            cacheImageFromTex(r"$(n, p \alpha)$", "npalpha"),
        ),
        72: (
            cacheImageFromTex(
                r"$^{11}{\rm B} + p \leftrightarrow ^{4}{\rm He} + ^{4}{\rm He} + ^{4}{\rm He}$",
                "reaction72",
            ),
            cacheImageFromTex(r"$(p, \alpha \alpha)$", "palphaalpha"),
        ),
        73: (
            cacheImageFromTex(
                r"$^{11}{\rm C} + n \leftrightarrow ^{4}{\rm He} + ^{4}{\rm He} + ^{4}{\rm He}$",
                "reaction73",
            ),
            cacheImageFromTex(r"$(n, \alpha \alpha)$", "nalphaalpha"),
        ),
        74: (
            cacheImageFromTex(
                r"$^{12}{\rm C} + n \leftrightarrow \gamma + ^{13}{\rm C}$",
                "reaction74",
            ),
            cacheImageFromTex(r"$(n, \gamma)$", "ngamma"),
        ),
        75: (
            cacheImageFromTex(
                r"$^{13}{\rm C} + n \leftrightarrow \gamma + ^{14}{\rm C}$",
                "reaction75",
            ),
            cacheImageFromTex(r"$(n, \gamma)$", "ngamma"),
        ),
        76: (
            cacheImageFromTex(
                r"$^{14}{\rm N} + n \leftrightarrow \gamma + ^{15}{\rm N}$",
                "reaction76",
            ),
            cacheImageFromTex(r"$(n, \gamma)$", "ngamma"),
        ),
        77: (
            cacheImageFromTex(
                r"$^{13}{\rm N} + n \leftrightarrow p + ^{13}{\rm C}$", "reaction77"
            ),
            cacheImageFromTex("(n, p)", "np"),
        ),
        78: (
            cacheImageFromTex(
                r"$^{14}{\rm N} + n \leftrightarrow p + ^{14}{\rm C}$", "reaction78"
            ),
            cacheImageFromTex("(n, p)", "np"),
        ),
        79: (
            cacheImageFromTex(
                r"$^{15}{\rm O} + n \leftrightarrow p + ^{15}{\rm N}$", "reaction79"
            ),
            cacheImageFromTex("(n, p)", "np"),
        ),
        80: (
            cacheImageFromTex(
                r"$^{15}{\rm O} + n \leftrightarrow ^{4}{\rm He} + ^{12}{\rm C}$",
                "reaction80",
            ),
            cacheImageFromTex(r"$(n, \alpha)$", "nalpha"),
        ),
        81: (
            cacheImageFromTex(
                r"$^{12}{\rm C} + p \leftrightarrow \gamma + ^{13}{\rm N}$",
                "reaction81",
            ),
            cacheImageFromTex(r"$(p, \gamma)$", "pgamma"),
        ),
        82: (
            cacheImageFromTex(
                r"$^{13}{\rm C} + p \leftrightarrow \gamma + ^{14}{\rm N}$",
                "reaction82",
            ),
            cacheImageFromTex(r"$(p, \gamma)$", "pgamma"),
        ),
        83: (
            cacheImageFromTex(
                r"$^{14}{\rm C} + p \leftrightarrow \gamma + ^{15}{\rm N}$",
                "reaction83",
            ),
            cacheImageFromTex(r"$(p, \gamma)$", "pgamma"),
        ),
        84: (
            cacheImageFromTex(
                r"$^{13}{\rm N} + p \leftrightarrow \gamma + ^{14}{\rm O}$",
                "reaction84",
            ),
            cacheImageFromTex(r"$(p, \gamma)$", "pgamma"),
        ),
        85: (
            cacheImageFromTex(
                r"$^{14}{\rm N} + p \leftrightarrow \gamma + ^{15}{\rm O}$",
                "reaction85",
            ),
            cacheImageFromTex(r"$(p, \gamma)$", "pgamma"),
        ),
        86: (
            cacheImageFromTex(
                r"$^{15}{\rm N} + p \leftrightarrow \gamma + ^{16}{\rm O}$",
                "reaction86",
            ),
            cacheImageFromTex(r"$(p, \gamma)$", "pgamma"),
        ),
        87: (
            cacheImageFromTex(
                r"$^{15}{\rm N} + p \leftrightarrow ^{4}{\rm He} + ^{12}{\rm C}$",
                "reaction87",
            ),
            cacheImageFromTex(r"$(p, \alpha)$", "palpha"),
        ),
        88: (
            cacheImageFromTex(
                r"$^{12}{\rm C} + ^{4}{\rm He} \leftrightarrow  \gamma + ^{16}{\rm O}$",
                "reaction88",
            ),
            cacheImageFromTex(r"$(\alpha, \gamma)$", "alphagamma"),
        ),
        89: (
            cacheImageFromTex(
                r"$^{10}{\rm B} + ^{4}{\rm He} \leftrightarrow p + ^{13}{\rm C}$",
                "reaction89",
            ),
            cacheImageFromTex(r"$(\alpha, p)$", "alphap"),
        ),
        90: (
            cacheImageFromTex(
                r"$^{11}{\rm B} + ^{4}{\rm He} \leftrightarrow p + ^{14}{\rm C}$",
                "reaction90",
            ),
            cacheImageFromTex(r"$(\alpha, p)$", "alphap"),
        ),
        91: (
            cacheImageFromTex(
                r"$^{11}{\rm C} + ^{4}{\rm He} \leftrightarrow p + ^{14}{\rm N}$",
                "reaction91",
            ),
            cacheImageFromTex(r"$(\alpha, p)$", "alphap"),
        ),
        92: (
            cacheImageFromTex(
                r"$^{12}{\rm N} + ^{4}{\rm He} \leftrightarrow p + ^{15}{\rm O}$",
                "reaction92",
            ),
            cacheImageFromTex(r"$(\alpha, p)$", "alphap"),
        ),
        93: (
            cacheImageFromTex(
                r"$^{13}{\rm N} + ^{4}{\rm He} \leftrightarrow p + ^{16}{\rm O}$",
                "reaction93",
            ),
            cacheImageFromTex(r"$(\alpha, p)$", "alphap"),
        ),
        94: (
            cacheImageFromTex(
                r"$^{10}{\rm B} + ^{4}{\rm He} \leftrightarrow n + ^{13}{\rm N}$",
                "reaction94",
            ),
            cacheImageFromTex(r"$(\alpha, n)$", "alphan"),
        ),
        95: (
            cacheImageFromTex(
                r"$^{11}{\rm B} + ^{4}{\rm He} \leftrightarrow n + ^{14}{\rm N}$",
                "reaction95",
            ),
            cacheImageFromTex(r"$(\alpha, n)$", "alphan"),
        ),
        96: (
            cacheImageFromTex(
                r"$^{12}{\rm B} + ^{4}{\rm He} \leftrightarrow n + ^{15}{\rm N}$",
                "reaction96",
            ),
            cacheImageFromTex(r"$(\alpha, n)$", "alphan"),
        ),
        97: (
            cacheImageFromTex(
                r"$^{13}{\rm C} + ^{4}{\rm He} \leftrightarrow n + ^{16}{\rm O}$",
                "reaction97",
            ),
            cacheImageFromTex(r"$(\alpha, n)$", "alphan"),
        ),
        98: (
            cacheImageFromTex(
                r"$^{11}{\rm B} + ^{2}{\rm H} \leftrightarrow p + ^{12}{\rm B}$",
                "reaction98",
            ),
            cacheImageFromTex(r"$^2$H Strip.", "2Hstrip"),
        ),
        99: (
            cacheImageFromTex(
                r"$^{12}{\rm C} + ^{2}{\rm H} \leftrightarrow p + ^{13}{\rm C}$",
                "reaction99",
            ),
            cacheImageFromTex(r"$^2$H Strip.", "2Hstrip"),
        ),
        100: (
            cacheImageFromTex(
                r"$^{13}{\rm C} + ^{2}{\rm H} \leftrightarrow p + ^{14}{\rm C}$",
                "reaction100",
            ),
            cacheImageFromTex(r"$^2$H Strip.", "2Hstrip"),
        ),
    }
    current = {}

    def updateCurrent(self, new):
        """Update the list of currently available reactions

        Parameter:
            new: list of keys of the new current reactions
        """
        self.current = {n: self.all[n] for n in new if n in self.all.keys()}
        self.updated.emit()


class Parameter(QObject):
    """Class that contains the default parameter properties
    (ranges, default value, name) and the current values
    """

    def __init__(self, name, label, minimum, maximum, default):
        """Define parameter name, label, figure, default min/max/center
        and initial values for min/max/N/center

        Parameters:
            name: the parameter short name
            label: the parameter name in latex form
            minimum: the default min in a grid for the parameter
            maximum: the default max in a grid for the parameter
            default: the default value for a single point
        """
        self.name = name
        self.label = label
        self.defaultmin = minimum
        self.defaultmax = maximum
        self.defaultval = default
        self.currentmin = minimum
        self.currentmax = maximum
        self.currentval = default
        self.currentN = 2
        self.currenttype = "single"
        self.fig = QPixmap()
        self.fig.convertFromImage(self.paramLabel(label))
        self.__grids = {}

    def resetCurrent(self):
        """Reset the "current*" attributes to the default values"""
        self.currenttype = "single"
        self.currentmin = self.defaultmin
        self.currentmax = self.defaultmax
        self.currentval = self.defaultval
        self.currentN = 2

    def addGrid(self, gid, gN, gmin, gmax):
        """self.__grids will contain the grid values for the parameter.
        Each item of self.__grids (identified by different `gid`s)
        will contain the settings for a different grid

        Parameters:
            gid: the grid identifier
            gN: number of points in the current parameter
            gmin, gmax: minimum and maximum values to cover
        """
        if gN < 1:
            pGUIErrorManager.warning(
                "Invalid number of points for a grid: "
                + "N=%d for parameter %s" % (gN, self.name)
            )
            return
        if gN == 1:
            self.__grids[gid] = {
                "type": "single",
                "N": 1,
                "val": (gmin + gmax) / 2.0,
                "min": (gmin + gmax) / 2.0,
                "max": (gmin + gmax) / 2.0,
                "pts": np.asarray([(gmin + gmax) / 2.0]),
            }
        else:
            self.__grids[gid] = {
                "type": "grid",
                "N": gN,
                "min": gmin,
                "max": gmax,
                "val": None,
                "pts": np.linspace(gmin, gmax, gN),
            }

    def getGrid(self, gid):
        """Update the self.current* attributes according to the given grid

        Parameters:
            gid: the grid identifier
        """
        if gid not in self.__grids.keys():
            mainlogger.debug("the grid does not exist: %s" % gid)
            return
        self.currentmin = self.__grids[gid]["min"]
        self.currentmax = self.__grids[gid]["max"]
        self.currentval = (
            self.__grids[gid]["val"]
            if self.__grids[gid]["val"] is not None
            else self.defaultval
        )
        self.currenttype = self.__grids[gid]["type"]
        self.currentN = self.__grids[gid]["N"] if self.currenttype == "grid" else 1

    def getGridPoints(self, gid):
        """Get the number and the list of points in the grid

        Parameters:
            gid: the grid identifier

        Output:
            a tuple containing the number and the list of points
        """
        if gid not in self.__grids.keys():
            mainlogger.debug("the grid does not exist: %s" % gid)
            return (None, None)
        self.getGrid(gid)
        return (self.__grids[gid]["N"], self.__grids[gid]["pts"])

    def deleteGrid(self, gid):
        """delete a grid

        Parameters:
            gid: the grid identifier

        Output:
            True in case of success, False otherwise
        """
        try:
            del self.__grids[gid]
        except KeyError:
            mainlogger.exception("Cannot delete grid: %s" % gid)
            return False
        self.resetCurrent()
        return True

    def paramLabel(self, mathTex):
        """Use matplotlib to convert mathTex in a QImage

        Parameters:
            mathTex: the latex text to convert

        Output:
            a QImage
        """
        return imageFromTex(mathTex)


class Parameters(QObject):
    """Class including a description and
    the configuration for all the parameters
    """

    paramOrder = __paramOrder__
    paramOrderParth = __paramOrderParth__

    def __init__(self):
        """Define the list of all the parameters
        with their default values
        """
        self.all = {
            "eta10": Parameter("eta10", r"$\eta_{10}$", 2, 9, 6.13832),
            "DeltaNnu": Parameter("DeltaNnu", r"$\Delta N_\nu$", -3, 3, 0),
            "taun": Parameter("taun", r"$\tau_n$", 876.4, 882.4, 879.4),
            "csinue": Parameter("csinue", r"$\xi_{\nu_e}$", -1, 1, 0),
            "csinux": Parameter("csinux", r"$\xi_{\nu_X}$", -1, 1, 0),
            "rhoLambda": Parameter("rhoLambda", r"$\rho_\Lambda$", 0, 1, 0),
        }
        self.gridsList = {}

    def resetCurrent(self):
        """Reset the current value for each parameter in self.all"""
        for p in self.all.values():
            p.resetCurrent()

    def getGrid(self, gid):
        """Update the content of a given grid to the "current*" settings
        for each parameter in self.all

        Parameters:
            gid: the grid id
        """
        for p in self.all.values():
            p.getGrid(gid)

    def getGridPoints(self, gid):
        """Update the content of a given grid to the "current*" settings
        for each parameter in self.all, and return a list of grid points

        Parameters:
            gid: the grid id

        Output:
            a numpy array of shape (6,N) with the N points in the grid
        """
        self.getGrid(gid)
        pts = (
            np.mgrid[
                self.all["eta10"]
                .currentmin : self.all["eta10"]
                .currentmax : self.all["eta10"]
                .currentN
                * 1j,
                self.all["DeltaNnu"]
                .currentmin : self.all["DeltaNnu"]
                .currentmax : self.all["DeltaNnu"]
                .currentN
                * 1j,
                self.all["taun"]
                .currentmin : self.all["taun"]
                .currentmax : self.all["taun"]
                .currentN
                * 1j,
                self.all["csinue"]
                .currentmin : self.all["csinue"]
                .currentmax : self.all["csinue"]
                .currentN
                * 1j,
                self.all["csinux"]
                .currentmin : self.all["csinux"]
                .currentmax : self.all["csinux"]
                .currentN
                * 1j,
                self.all["rhoLambda"]
                .currentmin : self.all["rhoLambda"]
                .currentmax : self.all["rhoLambda"]
                .currentN
                * 1j,
            ]
            .reshape(
                6,
                self.all["eta10"].currentN
                * self.all["DeltaNnu"].currentN
                * self.all["taun"].currentN
                * self.all["csinue"].currentN
                * self.all["csinux"].currentN
                * self.all["rhoLambda"].currentN,
            )
            .T
        )
        return pts

    def deleteGrid(self, gid):
        """Delete a grid for all the parameters in self.all

        Parameters:
            gid: the grid id
        """
        for p in sorted(self.all.keys()):
            if not self.all[p].deleteGrid(gid):
                mainlogger.debug("Cannot delete grid for %s" % p)
                return False
        try:
            del self.gridsList[gid]
        except Exception:
            mainlogger.exception("Cannot delete grid index")
            return False
        return True
