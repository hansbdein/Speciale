"""Main window for the PArthENoPE3.0 GUI interface and related functions"""

import signal
import sys
import traceback

# PySide2
from PySide2.QtCore import Qt, QUrl, Signal
from PySide2.QtGui import (
    QDesktopServices,
    QFont,
    QGuiApplication,
    QIcon,
    QImage,
    QPixmap,
    QTextDocument,
)
from PySide2.QtWidgets import (
    QAction,
    QFrame,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
)

# PArthENoPE
try:
    import parthenopegui
    import parthenopegui.resourcesPySide2
    from parthenopegui.configuration import (
        Configuration,
        Nuclides,
        Parameters,
        PGFont,
        Reactions,
        askYesNo,
    )
    from parthenopegui.errorManager import pGUIErrorManager
    from parthenopegui.plotter import MWPlotPanel
    from parthenopegui.setrun import MWRunSettingsPanel
    from parthenopegui.texts import PGText
except ImportError:
    print("[mainWindow] Parthenopegui submodules for opening the GUI not found!")
    raise


class MWDescriptionPanel(QFrame):
    """`QFrame` extension to create a panel where to write the
    PArthENoPE description, with logos and webpage link
    """

    def __init__(self, parent=None):
        """Extension of `QFrame.__init__`, adds a layout
        and a QTextBrowser to the frame, with some images and text

        Parameter:
            parent: the parent widget
        """
        super(MWDescriptionPanel, self).__init__(parent)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.text = QTextBrowser(self)
        self.text.setOpenLinks(False)
        self.text.anchorClicked.connect(QDesktopServices.openUrl)

        self.document = QTextDocument()
        self.text.setDocument(self.document)
        layout.addWidget(self.text)
        font = QFont()
        font.setPointSize(Configuration.fontsizeinfo)
        self.document.setDefaultFont(font)

        for short, fname in (
            ("infn", "logo_INFN.png"),
            ("unina", "logofed_trasp.png"),
        ):
            self.document.addResource(
                QTextDocument.ImageResource,
                QUrl("mydata://%s.png" % short),
                QImage(":/images/%s" % fname).scaledToHeight(
                    Configuration.imageHeaderHeight, Qt.SmoothTransformation
                ),
            )
        self.document.addResource(
            QTextDocument.ImageResource,
            QUrl("mydata://logo.png"),
            QImage(":/images/sirenab.png").scaledToHeight(
                Configuration.logoHeight, Qt.SmoothTransformation
            ),
        )
        self.text.setHtml(
            "<center>"
            + '<img src="mydata://unina.png"/>'
            + '<img src="mydata://infn.png"/>'
            + "<br><br><br>"
            + '<span style="font-size: 30px"><i><b>%s</b></i></span>'
            % (PGText.parthenopename)
            + "<br>"
            + '<img src="mydata://logo.png" />'
            + "<br><br>"
            + "<i>%s</i>" % PGText.parthenopenamelong
            + "<br><br>"
            + PGText.parthenopedescription
            + "<br><br>"
            + "<a href='%s'>%s</a>"
            % (parthenopegui.__website__, parthenopegui.__website__)
            + "</center>"
        )
        self.text.setStyleSheet("background-color: white; color: black")


class MainWindow(QMainWindow):
    """Main window for the PArthENoPE GUI"""

    errormessage = Signal(type, type, Exception, traceback)

    def __init__(self):
        """Define some properties and build layout"""
        QMainWindow.__init__(self)
        self.errormessage.connect(self.excepthook)
        self.mainStatusBar = QStatusBar()
        self.nuclides = Nuclides()
        self.parameters = Parameters()
        self.reactions = Reactions()
        self.running = False

        self.createMainLayout()
        self.createMenusAndToolBar()
        self.setIcon()
        self.setWindowTitle(PGText.appname)

        # Catch Ctrl+C in shell
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        availableWidth = QGuiApplication.primaryScreen().availableGeometry().width()
        availableHeight = QGuiApplication.primaryScreen().availableGeometry().height()
        self.setGeometry(0, 0, availableWidth, availableHeight)

    def setIcon(self):
        """Set the icon of the main window"""
        appIcon = QIcon(":/images/icon.png")
        self.setWindowIcon(appIcon)

    def excepthook(self, cls, exception, trcbk):
        """Function that will replace `sys.excepthook` to log
        any error that occurs

        Parameters:
            cls, exception, trcbk as in `sys.excepthook`
        """
        pGUIErrorManager.error(PGText.errorUnhandled, exc_info=(cls, exception, trcbk))

    def closeEvent(self, event):
        """Intercept close events. Ask before closing if process is running

        Parameter:
            event: a QEvent
        """
        if self.running:
            if not askYesNo(PGText.warningRunningProcesses):
                event.ignore()
            else:
                event.accept()
        else:
            event.accept()

    def createMenusAndToolBar(self):
        """Create the actions and set the content of the menus"""
        self.exitAct = QAction(
            QIcon(":/images/application-exit.png"),
            PGText.menuActionExitTitle,
            self,
            shortcut="Ctrl+Q",
            statusTip=PGText.menuActionExitToolTip,
            triggered=self.close,
        )

        self.aboutAct = QAction(
            QIcon(":/images/help-about.png"),
            PGText.menuActionAboutTitle,
            self,
            statusTip=PGText.menuActionAboutToolTip,
            triggered=self.showAbout,
        )

        self.menuBar().clear()
        self.menuBar().setFont(PGFont())
        self.fileMenu = self.menuBar().addMenu(PGText.menuFileTitle)
        self.fileMenu.setFont(PGFont())
        self.fileMenu.addAction(self.exitAct)

        self.helpMenu = self.menuBar().addMenu(PGText.menuHelpTitle)
        self.helpMenu.setFont(PGFont())
        self.helpMenu.addAction(self.aboutAct)

    def showAbout(self):
        """Function to show the About dialog"""
        mbox = QMessageBox(
            QMessageBox.Information,
            PGText.aboutTitle,
            "Graphical interface for dealing with the PArthENoPE BBN code."
            + "<br><br>"
            + PGText.copyright
            + "<br><br>"
            + "For more information, contact:<br>"
            + "<b>%s</b><br><i>Phone</i>: %s<br><i>Email</i>: %s"
            % (PGText.contactname, PGText.contactphone, PGText.contactemail)
            + "<br><br>"
            + "<b>Author:</b> %s " % parthenopegui.__author__
            + "<i>&lt;%s&gt;</i><br>" % parthenopegui.__email__
            + "<b>Version:</b> %s (%s)<br>"
            % (parthenopegui.__version__, parthenopegui.__version_date__)
            + "<b>Python version</b>: %s" % sys.version,
            parent=self,
        )
        mbox.setTextFormat(Qt.RichText)
        mbox.setIconPixmap(QPixmap(":/images/icon.png"))
        mbox.exec_()

    def createMainLayout(self):
        """Set the layout of the main window,
        i.e. create the tabs with the various panels panels
        """
        self.tabWidget = QTabWidget(self)
        self.tabWidget.setFont(PGFont())
        self.tabWidget.setTabsClosable(False)
        self.setCentralWidget(self.tabWidget)

        for ix, [attrname, obj, desc, tooltip] in enumerate(
            [
                [
                    "descriptionPanel",
                    MWDescriptionPanel,
                    "panelTitleDescription",
                    "panelToolTipDescription",
                ],
                [
                    "runSettingsTab",
                    MWRunSettingsPanel,
                    "panelTitleRun",
                    "panelToolTipRun",
                ],
                ["plotPanel", MWPlotPanel, "panelTitlePlot", "panelToolTipPlot"],
            ]
        ):
            setattr(self, attrname, obj(self))
            self.tabWidget.addTab(getattr(self, attrname), getattr(PGText, desc))
            self.tabWidget.setTabToolTip(ix, getattr(PGText, tooltip))
