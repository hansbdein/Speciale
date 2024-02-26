"""Utilities for in the tests of the physbiblio modules.

This file is part of the physbiblio package.
"""
import datetime
import glob
import logging
import os
import pickle
import shutil
import sys
import time
import traceback
from multiprocessing import Pool, cpu_count

import matplotlib
import numpy as np
import six

matplotlib.use("Qt5Agg")
os.environ["QT_API"] = "pyside2"
import matplotlib.pyplot as plt
from appdirs import AppDirs
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

if sys.version_info[0] < 3:
    import unittest2 as unittest
    from mock import MagicMock, call, patch
    from StringIO import StringIO

    USE_AUTOSPEC_CLASS = False
else:
    import unittest
    from io import StringIO
    from unittest.mock import MagicMock, call, patch

    USE_AUTOSPEC_CLASS = True

from PySide2.QtCore import (
    QByteArray,
    QItemSelectionModel,
    QMimeData,
    QModelIndex,
    QObject,
    QPointF,
    QSignalBlocker,
    QSize,
    Qt,
    QThread,
    QUrl,
    Signal,
)
from PySide2.QtGui import (
    QDesktopServices,
    QDropEvent,
    QFont,
    QGuiApplication,
    QIcon,
    QImage,
    QPalette,
    QPixmap,
    QTextDocument,
)
from PySide2.QtTest import QTest
from PySide2.QtWidgets import (
    QAbstractItemView,
    QAction,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    import parthenopegui
    import parthenopegui.basic as pgb
    from parthenopegui.configuration import (
        Configuration,
        DDTableWidget,
        Nuclides,
        Parameter,
        Parameters,
        PGFont,
        PGLabel,
        PGLabelButton,
        PGPushButton,
        Reactions,
        UnicodeSymbols,
        askDirName,
        askFileName,
        askFileNames,
        askGenericText,
        askSaveFileName,
        askYesNo,
        cacheImageFromTex,
        imageFromTex,
        paramsHidePath,
        paramsRealPath,
    )
    from parthenopegui.errorManager import (
        ErrorStream,
        PGErrorManagerClass,
        PGErrorManagerClassGui,
        mainlogger,
        pGUIErrorManager,
    )
    from parthenopegui.mainWindow import MainWindow, MWDescriptionPanel
    from parthenopegui.plotter import (
        AbundancesEvolutionPanel,
        AbundancesGenericPanel,
        AbundancesParams1DPanel,
        AbundancesParams2DPanel,
        Add1DLineFromPoint,
        Add2DContourFromPoint,
        AddEvoLineFromPoint,
        AddFromPoint,
        AddLineFromPoint,
        Chi2ParamConfig,
        CMapItems,
        CurrentPlotContent,
        ExportPlotCode,
        GridLoader,
        MWPlotPanel,
        PGEmptyTableModel,
        PGListContoursModel,
        PGListLinesModel,
        PGListPointsModel,
        PlotPlaceholderPanel,
        PlotSettings,
        ShowPlotWidget,
    )
    from parthenopegui.plotUtils import (
        PGContour,
        PGLine,
        PGPlotObject,
        chi2func,
        chi2labels,
        cmaps,
        defaultCmap,
        extendOptions,
        markerOptions,
        styleOptions,
    )
    from parthenopegui.runner import RunPArthENoPE, Thread_poolRunner
    from parthenopegui.runUtils import NoGUIRun, runSingle, shellCommand
    from parthenopegui.setrun import (
        ConfigNuclidesOutput,
        EditParameters,
        EditReaction,
        MWNetworkPanel,
        MWOutputPanel,
        MWPhysParamsPanel,
        MWRunPanel,
        MWRunSettingsPanel,
        ReactionsTableWidget,
        ResumeGrid,
    )
    from parthenopegui.texts import PGText
except ImportError:
    print("[tests] Necessary parthenopegui submodules not found!")
    raise
except Exception:
    print(traceback.format_exc())


class PGTestsConfig(object):
    """Collects some configuration variables for the tests"""

    test_con = True
    test_err = True
    test_mWi = True
    test_plt = True
    test_run = True
    test_set = True
    globalQApp = QApplication()

    def __init__(self):
        """Init some strings that define the temporary folders,
        and a MainWindow instance
        """
        self.today = datetime.datetime.today()
        self.today_ymd = datetime.datetime.today().strftime("%y%m%d")
        self.testRunDirEmpty = "test_run_%s_empty" % self.today_ymd
        self.testRunDirDefaultSample = "test_run_%s_sample" % self.today_ymd
        self.testRunDirFullExample = "test_run_example_multi%s" % sys.version_info[0]
        self.mainW = MainWindow()
        rst = self.mainW.runSettingsTab
        rst.outputParams.outputFolderName.setText(self.testRunDirFullExample)
        if os.path.exists(
            os.path.join(self.testRunDirFullExample, Configuration.runSettingsObj)
        ):
            with open(
                os.path.join(self.testRunDirFullExample, Configuration.runSettingsObj),
                "rb",
            ) as _f:
                cp, gp = pickle.load(_f)
            cp = paramsRealPath(cp, self.testRunDirFullExample)
        else:
            # prepare a real run, if not existing, for test purposes (this will not be deleted)
            self.mainW.parameters.gridsList[0] = 13
            for p in Parameters.paramOrderParth[0:2]:
                self.mainW.parameters.all[p].addGrid(
                    0,
                    2,
                    self.mainW.parameters.all[p].defaultmin,
                    self.mainW.parameters.all[p].defaultmax,
                )
            p = Parameters.paramOrderParth[2]
            self.mainW.parameters.all[p].addGrid(
                0,
                3,
                self.mainW.parameters.all[p].defaultmin,
                self.mainW.parameters.all[p].defaultmax,
            )
            for p in Parameters.paramOrderParth[3:]:
                self.mainW.parameters.all[p].addGrid(
                    0,
                    1,
                    self.mainW.parameters.all[p].defaultval,
                    self.mainW.parameters.all[p].defaultval,
                )
            self.mainW.parameters.gridsList[1] = 1
            for p in Parameters.paramOrderParth:
                self.mainW.parameters.all[p].addGrid(
                    1,
                    1,
                    self.mainW.parameters.all[p].defaultval,
                    self.mainW.parameters.all[p].defaultval,
                )
            rst.networkParams.smallNet.setChecked(True)
            rst.networkParams.updateReactions()
            rst.outputParams.nuclidesInOutput.setChecked(True)
            mainlogger.info("Preparing test runs...")
            with patch("os.remove"):
                with patch("parthenopegui.setrun.datetime") as _dt:
                    _dt.today.return_value = self.today
                    cp, gp = rst.runPanel.startRunCustom()
                while not np.all(
                    [
                        os.path.exists(cp["output_file_log"] % ix)
                        and os.path.exists(cp["output_file_nuclides"] % ix)
                        and os.path.exists(cp["output_file_parthenope"] % ix)
                        for ix, pt in enumerate(rst.mainW.runner.getGridPointsList())
                    ]
                ):
                    time.sleep(0.2)
            time.sleep(2)
            mainlogger.info("Test runs have finished, reading...")
            for ix, pt in enumerate(rst.mainW.runner.getGridPointsList()):
                rst.mainW.runner.readNuclides(ix)
                rst.mainW.runner.readParthenope(ix)
            with patch("os.remove"):
                rst.mainW.runner.allHaveFinished()
            mainlogger.info("Test runs done")
        self.commonParamsGrid = cp
        self.gridPointsGrid = gp
        rst.outputParams.outputFolderName.setText(self.testRunDirDefaultSample)
        self.sampleRunner = RunPArthENoPE(cp, gp, self.mainW)
        self.sampleRunner.readAllResults()


testConfig = PGTestsConfig()


def setUpModule():
    """Create some dirs and files that will be used in the tests"""
    os.makedirs(testConfig.testRunDirEmpty)
    fileobj = os.path.join(testConfig.testRunDirEmpty, Configuration.runSettingsObj)
    with open(fileobj, "w") as _f:
        _f.write("")
    fileobj = os.path.join(
        testConfig.testRunDirDefaultSample, Configuration.runSettingsObj
    )
    with patch("parthenopegui.setrun.datetime") as _dt:
        _dt.today.return_value = testConfig.today
        cp, gp = testConfig.mainW.runSettingsTab.runPanel.startRunDefault()
    testConfig.commonParamsDefault = cp
    testConfig.gridPointsDefault = gp


def tearDownModule():
    """Remove dirs created for the tests"""
    shutil.rmtree(testConfig.testRunDirEmpty)
    shutil.rmtree(testConfig.testRunDirDefaultSample)


class PGTestCase(unittest.TestCase):
    """define the class that will be used for database tests"""

    @classmethod
    def setUpClass(self):
        """Assign a temporary QApplication to the class instance"""
        self.maxDiff = None
        self.qapp = PGTestsConfig.globalQApp

    @classmethod
    def tearDownClass(self):
        """Remove a temporary QApplication from the class instance"""
        del self.qapp

    def assertEqualArray(self, a, b):
        """Assert that two np.ndarrays (a, b) are equal
        using np.all(a == b)

        Parameters:
            a, b: the two np.ndarrays to compare
        """
        self.assertTrue(np.allclose(a, b, equal_nan=True))

    def assertGeometry(self, obj, x, y, w, h):
        """Test the geometry of an object

        Parameters:
            obj: the object to be tested
            x: the expected x position
            y: the expected y position
            w: the expected width
            h: the expected height
        """
        geom = obj.geometry()
        self.assertEqual(geom.x(), x)
        self.assertEqual(geom.y(), y)
        self.assertEqual(geom.width(), w)
        self.assertEqual(geom.height(), h)


class PGTestCasewMainW(PGTestCase):
    """Class that manages GUI tests which need a testing instance
    of the MainWindow class
    """

    @classmethod
    def setUpClass(self):
        """Call the parent method and instantiate a testing MainWindow"""
        super(PGTestCasewMainW, self).setUpClass()
        self.mainW = MainWindow()


########## package
class TestParthenopegui(PGTestCase):
    """Tests for package variables"""

    def test_attributes(self):
        """test package variables"""
        self.assertTrue(hasattr(parthenopegui, "__all__"))
        self.assertTrue(hasattr(parthenopegui, "__author__"))
        self.assertTrue(hasattr(parthenopegui, "__email__"))
        self.assertTrue(hasattr(parthenopegui, "__website__"))
        self.assertTrue(hasattr(parthenopegui, "__version__"))
        self.assertTrue(hasattr(parthenopegui, "__version_date__"))
        self.assertTrue(hasattr(parthenopegui, "FileNotFoundErrorClass"))
        self.assertTrue(hasattr(pgb, "PGConfiguration"))
        self.assertIsInstance(getattr(pgb, "__paramOrder__"), list)
        self.assertIsInstance(getattr(pgb, "__paramOrderParth__"), list)
        self.assertIsInstance(getattr(pgb, "__nuclideOrder__"), dict)
        self.assertTrue(hasattr(pgb, "paramsHidePath"))
        self.assertTrue(hasattr(pgb, "paramsRealPath"))
        self.assertTrue(hasattr(pgb, "runInGUI"))
        self.assertTrue(hasattr(pgb, "runNoGUI"))

    def test_PGConfiguration(self):
        """test attributes of the PGConfiguration class"""
        pgc = pgb.PGConfiguration
        self.assertTrue(hasattr(pgc, "fontsize"))
        self.assertTrue(hasattr(pgc, "fontsizeinfo"))
        self.assertTrue(hasattr(pgc, "logoHeight"))
        self.assertTrue(hasattr(pgc, "imageHeaderHeight"))
        self.assertTrue(hasattr(pgc, "rowImageHeight"))
        self.assertEqual(pgc.infoFilename, "info")
        self.assertEqual(pgc.nuclidesFilename, "nuclides")
        self.assertEqual(pgc.parthenopeFilename, "parthenope")
        self.assertEqual(pgc.failedRunsInstructionsFilename, "failed_instructions.txt")
        self.assertEqual(pgc.fortranExecutableName, "parthenope3.0")
        self.assertEqual(pgc.fortranOutputFilename, "fortran_output")
        self.assertEqual(pgc.runSettingsObj, "settings.obj")
        self.assertEqual(
            pgc.limitNuclides,
            {"smallNet": 10, "interNet": 19, "complNet": 27},
        )
        self.assertEqual(
            pgc.limitReactions,
            {"smallNet": 41, "interNet": 74, "complNet": 101},
        )
        self.assertEqual(pgc.formatParam, "%.6g")
        a = pgc()
        self.assertEqual(pgc.dataPath, AppDirs("PArthENoPE").user_data_dir)
        self.assertEqual(pgc.cachePath, AppDirs("PArthENoPE").user_cache_dir)

    def test_paramsHidePath(self):
        """test paramsHidePath"""
        a = {
            "output_folder": "abc/def/",
            "a": 1,
            "b": "2",
            "c": "abc/def/ghi",
            "d": "aaabc/def/ghi",
        }
        self.assertEqual(
            pgb.paramsHidePath(a, "abc/def"),
            {
                "output_folder": "abc/def/",
                "a": 1,
                "b": "2",
                "c": "FOLDER#/ghi",
                "d": "aaabc/def/ghi",
            },
        )

    def test_paramsRealPath(self):
        """test paramsRealPath"""
        a = {
            "output_folder": "abc/def/",
            "a": 1,
            "b": "2",
            "c": "FOLDER#/ghi",
            "d": "aaFOLDER#",
        }
        self.assertEqual(
            pgb.paramsRealPath(a, "abc/def"),
            {
                "output_folder": "abc/def/",
                "a": 1,
                "b": "2",
                "c": "abc/def/ghi",
                "d": "aaabc/def",
            },
        )

    def test_runInGUI(self):
        """test runInGUI"""
        self.qapp.exec_ = MagicMock(return_value=10)
        mw = MainWindow()
        mw.show = MagicMock()
        mw.raise_ = MagicMock()
        with patch("logging.Logger.info") as _i, patch("sys.exit") as _e, patch(
            "PySide2.QtWidgets.QApplication", return_value=self.qapp
        ) as _qa, patch(
            "parthenopegui.mainWindow.MainWindow", return_value=mw
        ) as _mw, patch(
            "parthenopegui.setrun.MWRunPanel.startRunPickle"
        ) as _srp:
            pgb.runInGUI(["a"])
            _i.assert_not_called()
            _srp.assert_not_called()
            _e.assert_called_once_with(10)
            _qa.assert_called_once_with(["a"])
            _mw.assert_called_once_with()
            mw.show.assert_called_once_with()
            mw.raise_.assert_called_once_with()
            self.qapp.exec_.assert_called_once_with()
            _e.reset_mock()
            _qa.reset_mock()
            _mw.reset_mock()
            mw.show.reset_mock()
            mw.raise_.reset_mock()
            self.qapp.exec_.reset_mock()

            pgb.runInGUI(["a", "b"])
            _i.assert_called_once_with(
                "Preparing to run with imported grid settings.\n"
                + "Notice that the GUI will not show the run settings properly!\n"
                + "You cannot edit the previous settings nor check them."
            )
            _e.assert_called_once_with(10)
            _qa.assert_called_once_with(["a", "b"])
            _mw.assert_called_once_with()
            mw.show.assert_called_once_with()
            mw.raise_.assert_called_once_with()
            self.qapp.exec_.assert_called_once_with()
            _srp.assert_called_once_with("b")
            self.assertEqual(mw.tabWidget.currentWidget(), mw.runSettingsTab)

    def test_runNoGUI(self):
        """test runInGUI"""
        prevlev = mainlogger.level
        ngr = NoGUIRun()
        ngr.prepareRunFromPickle = MagicMock()
        ngr.run = MagicMock()
        with patch("logging.Logger.info") as _i, patch("sys.exit") as _e, patch(
            "parthenopegui.runUtils.NoGUIRun", autospec=True, return_value=ngr
        ) as _ngr:
            pgb.runNoGUI(["a"])
            _i.assert_called_once_with(
                "Nothing to do: cannot import PySide2 and no folder to run was indicated!"
            )
            _e.assert_called_once_with(1)
            _ngr.assert_not_called()
            pgb.runNoGUI(["a", "b"])
            _i.assert_called_once_with(
                "Nothing to do: cannot import PySide2 and no folder to run was indicated!"
            )
            _e.assert_called_once_with(1)
            _ngr.assert_called_once_with()
            ngr.prepareRunFromPickle.assert_called_once_with("b")
            ngr.run.assert_called_once_with()
        mainlogger.setLevel(prevlev)


########## texts
class TestPGText(unittest.TestCase):
    """Tests for text class"""

    def test_hasattributes(self):
        """Test that the given classes have all the necessary attributes
        and that they are of the correct type
        """
        strings = [
            "appname",
            "askReplace",
            "copyright",
            "contactname",
            "contactphone",
            "contactemail",
            "parthenopename",
            "parthenopenamelong",
            "parthenopedescription",
            "aboutTitle",
            "smallAsciiArrow",
            "comma",
            "buttonAccept",
            "buttonCancel",
            "buttonDelete",
            "buttonEdit",
            "cannotWrite",
            "category",
            "current",
            "editReactionFirstLabel",
            "editReactionTitle",
            "editReactionToolTip",
            "editReactionValueToolTip",
            "editParametersAdd",
            "editParametersTitle",
            "emptyFile",
            "errorCannotFindGrid",
            "errorCannotFindIndex",
            "errorCannotLoadGrid",
            "errorCannotWriteFile",
            "errorInvalidField",
            "errorInvalidFieldSet",
            "errorInvalidParamLimits",
            "errorLoadingGrid",
            "errorNoPhysicalParams",
            "errorNoOutputFolder",
            "errorNoReactionNetwork",
            "errorReadTable",
            "errorUnhandled",
            "failedRunsInstructions",
            "failedRunsMessage",
            "fileNotFound",
            "interpolationPerformedContour",
            "interpolationPerformedLine",
            "lineNotFound",
            "loadGridAsk",
            "mean",
            "menuFileTitle",
            "menuHelpTitle",
            "menuActionAboutTitle",
            "menuActionAboutToolTip",
            "menuActionExitTitle",
            "menuActionExitToolTip",
            "networkCustomizeRate",
            "networkSelect",
            "noGrid",
            "nuclidesOthersTitle",
            "nuclidesOthersToolTip",
            "nuclidesSelectAllText",
            "nuclidesSelectAllToolTip",
            "nuclidesSelectedTitle",
            "nuclidesSelectedToolTip",
            "nuclidesUnselectAllText",
            "nuclidesUnselectAllToolTip",
            "outputPanelDirectoryAsk",
            "outputPanelDirectoryDialogTitle",
            "outputPanelDirectoryInitialTitle",
            "outputPanelDirectoryToolTip",
            "outputPanelNuclidesInOutput",
            "outputPanelSelectDescription",
            "panelTitleDescription",
            "panelTitleRun",
            "panelTitlePlot",
            "panelToolTipDescription",
            "panelToolTipRun",
            "panelToolTipPlot",
            "parentAttributeMissing",
            "physicalParamsAdd",
            "physicalParamsAddToolTip",
            "physicalParamsDescription",
            "physicalParamsEmpty",
            "physicalParamsNumberSummary",
            "plotAbundancesTypeError",
            "plotAskDeleteLine",
            "plotAvailablePoints",
            "plotChi2AskLabel",
            "plotChi2AskCheckToolTip",
            "plotChi2AskMeanToolTip",
            "plotChi2AskStdToolTip",
            "plotContourCMap",
            "plotContourExtend",
            "plotContourExtendToolTip",
            "plotContourFilled",
            "plotContourFilledToolTip",
            "plotContourHasCbar",
            "plotContourHasCbarToolTip",
            "plotContourLevelsInvalidLen",
            "plotContourLevelsInvalidLevel",
            "plotContourLevelsInvalidSyntax",
            "plotContourLevelsInvalidType",
            "plotContourLevelsLabel",
            "plotContourLevelsToolTip",
            "plotContoursInUse",
            "plotContourZLabel",
            "plotContourZLabelToolTip",
            "plotInvalidColor",
            "plotInvalidMean",
            "plotInvalidStddev",
            "plotInvalidWidth",
            "plotLineColor",
            "plotLineColorToolTip",
            "plotLineLabel",
            "plotLineLabelToolTip",
            "plotLineMarker",
            "plotLineMarkerToolTip",
            "plotLineStyle",
            "plotLineStyleToolTip",
            "plotLineWidth",
            "plotLineWidthToolTip",
            "plotLinesInUse",
            "plotPlaceholderText",
            "plotSaved",
            "plotScriptSaved",
            "plotSelectChi2AddLine",
            "plotSelectChi2AddLineToolTip",
            "plotSelectChi2Type",
            "plotSelectGridText",
            "plotSelectNuclide",
            "plotSelectNuclideLine",
            "plotSelectNuclideContour",
            "plotSelectNuclideToolTip",
            "plotSelectPlotType",
            "plotSettAskResetImage",
            "plotSettAxesTextSize",
            "plotSettAxesTextSizeToolTip",
            "plotSettExportImage",
            "plotSettExportImageToolTip",
            "plotSettFigSize",
            "plotSettFigSizeToolTipH",
            "plotSettFigSizeToolTipV",
            "plotSettFigSizeWarning",
            "plotSettLegend",
            "plotSettLegendToolTip",
            "plotSettLegendLoc",
            "plotSettLegendLocToolTip",
            "plotSettLegendNumCols",
            "plotSettLegendNumColsToolTip",
            "plotSettLegendTextSize",
            "plotSettLegendTextSizeToolTip",
            "plotSettRefreshImage",
            "plotSettRefreshImageToolTip",
            "plotSettRevertImage",
            "plotSettRevertImageToolTip",
            "plotSettResetImage",
            "plotSettResetImageToolTip",
            "plotSettSaveImage",
            "plotSettSaveImageToolTip",
            "plotSettTabTitleAxes",
            "plotSettTabTitleFigure",
            "plotSettTabTitleLegend",
            "plotSettTight",
            "plotSettTightToolTip",
            "plotSettTitleLab",
            "plotSettTitleLabToolTip",
            "plotSettXLab",
            "plotSettXLabToolTip",
            "plotSettXLims",
            "plotSettXLimsToolTipLow",
            "plotSettXLimsToolTipUpp",
            "plotSettXScale",
            "plotSettXScaleToolTip",
            "plotSettYLabToolTip",
            "plotSettYLab",
            "plotSettYLims",
            "plotSettYLimsToolTipLow",
            "plotSettYLimsToolTipUpp",
            "plotSettYScale",
            "plotSettYScaleToolTip",
            "plotSettZLab",
            "plotSettZLabToolTip",
            "plotWhereToSave",
            "plotWhereToSaveScript",
            "reactionEditToolTip",
            "resumeGridAskDelete",
            "resumeGridDeleteToolTip",
            "resumeGridEditToolTip",
            "runCompletedErrorFile",
            "runCompletedErrorMessage",
            "runCompletedFailing",
            "runCompletedFailingCheck",
            "runCompletedFailingManual",
            "runCompletedSuccessfully",
            "runFailed",
            "runnerAskStop",
            "runnerDone",
            "runnerFinished",
            "runnerParthenopeNotFound",
            "runnerRunning",
            "runnerStopped",
            "runPanelCustomTitle",
            "runPanelCustomToolTip",
            "runPanelDefaultTitle",
            "runPanelDefaultToolTip",
            "runPanelDescription",
            "runPanelProgressBarToolTip",
            "runPanelSaveCustomTitle",
            "runPanelSaveCustomToolTip",
            "runPanelSaveDefaultTitle",
            "runPanelSaveDefaultToolTip",
            "runPanelSpaceInFolderName",
            "runPanelStopTitle",
            "runPanelStopToolTip",
            "runSingleArgsType",
            "selectGridToolTip",
            "standardDeviation",
            "startRun",
            "startRunI",
            "tryToOpenFolder",
            "warningLargeValue",
            "warningMaxEmptyList",
            "warningMissingLine",
            "warningRunningProcesses",
            "warningWrongType",
        ]
        lists = [
            "editReactionCombo",
            "editParametersColumnLabels",
            "editParametersCombo",
            "reactionColumnHeaders",
            "reactionColumnHeaderToolTips",
            "resumeGridHeaders",
            "resumeGridHeadersToolTip",
        ]
        dicts = [
            "editParametersToolTips",
            "networkDescription",
            "networkToolTips",
            "parameterDescriptions",
            "PArthENoPEWatermark",
            "plotSelectChi2TypeContents",
            "plotTypeDescription",
        ]
        for c in (PGText,):
            for s in strings:
                self.assertIsInstance(getattr(c, s), six.string_types)
            for l in lists:
                self.assertIsInstance(getattr(c, l), list)
            for d in dicts:
                self.assertIsInstance(getattr(c, d), dict)
            for k, v in c.plotTypeDescription.items():
                self.assertIsInstance(v, list)
                self.assertTrue(len(v) > 1)


########## errorManager
if PGTestsConfig.test_err:

    class TestErrors(unittest.TestCase):
        """Test PGErrorManagerClass stuff"""

        @classmethod
        def setUpClass(self):
            """settings"""
            self.logFileName = "test.log"
            self.pGErrorManager = PGErrorManagerClass(
                level=logging.INFO,
                logfilename=self.logFileName,
                loggerString="testlogger",
            )

        @classmethod
        def tearDownClass(self):
            """remove temporary log"""
            if os.path.exists(self.logFileName):
                os.remove(self.logFileName)

        def reset(self, m):
            """Clear a StringIO object"""
            m.seek(0)
            m.truncate(0)

        def resetLogFile(self):
            """Reset the logfile content"""
            open(self.logFileName, "w").close()

        def readLogFile(self):
            """Read the logfile content"""
            with open(self.logFileName) as logFile:
                log_new = logFile.read()
            return log_new

        def test_critical(self):
            """Test pBLogger.critical with and without exception"""
            self.resetLogFile()
            try:
                raise Exception("Fake error")
            except Exception as e:
                self.pGErrorManager.logger.critical(str(e))
            log_new = self.readLogFile()
            self.assertIn(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), log_new)
            self.assertIn("CRITICAL : [tests.test_critical] Fake error", log_new)
            self.assertNotIn("Traceback (most recent call last):", log_new)
            try:
                raise Exception("Fake error")
            except Exception as e:
                self.pGErrorManager.logger.critical(str(e), exc_info=True)
            log_new = self.readLogFile()
            self.assertIn("Traceback (most recent call last):", log_new)

        def test_exception(self):
            """Test pBLogger.exception"""
            self.resetLogFile()
            try:
                raise Exception("Fake error")
            except Exception as e:
                self.pGErrorManager.logger.exception(str(e))
            try:
                raise Exception("New fake error")
            except Exception as e:
                self.pGErrorManager.logger.error(str(e))
            log_new = self.readLogFile()
            self.assertIn(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), log_new)
            self.assertIn("ERROR : [tests.test_exception] Fake error", log_new)
            self.assertIn("ERROR : [tests.test_exception] New fake error", log_new)
            self.assertIn("Traceback (most recent call last):", log_new)

        def test_warning(self):
            """Test pBLogger.warning"""
            self.resetLogFile()
            try:
                raise Exception("Fake warning")
            except Exception as e:
                self.pGErrorManager.logger.warning(str(e))
            log_new = self.readLogFile()
            self.assertIn(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), log_new)
            self.assertIn("WARNING : [tests.test_warning] Fake warning", log_new)

        def test_info(self):
            """Test pBLogger.info"""
            self.resetLogFile()
            self.pGErrorManager.logger.info("Some info")
            log_new = self.readLogFile()
            self.assertIn(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), log_new)
            self.assertIn("INFO : [tests.test_info] Some info", log_new)

        def test_default(self):
            """Test default properties for the PGErrorManagerClass"""
            with patch(
                "logging.handlers.RotatingFileHandler", autospec=USE_AUTOSPEC_CLASS
            ) as _r, patch("logging.getLogger", autospec=True) as _g:
                self.tmp = PGErrorManagerClass()
                _r.assert_called_once_with(
                    "gui.log", backupCount=5, maxBytes=5.0 * 2**20
                )
                _g.assert_called_once_with("parthenopelog")
            self.assertEqual(self.tmp.loglevel, logging.WARNING)

    class TestGUIErrorManager(PGTestCase):
        """Test the functions in ErrorManager"""

        def test_ErrorStream(self):
            """Test functions in ErrorStream"""
            es = ErrorStream()
            self.assertIsInstance(es, StringIO)
            self.assertEqual(es.lastMBox, None)
            self.assertEqual(es.priority, 1)
            es.setPriority(2)
            self.assertEqual(es.priority, 2)
            mb = MagicMock()
            with patch(
                "parthenopegui.errorManager.QMessageBox",
                return_value=mb,  # , autospec=True
            ) as _mb:
                es.write("sometext")
                _mb.assert_called_once_with(_mb.Information, "Information", "")
            mb.setText.assert_called_once_with("sometext")
            mb.setWindowTitle.assert_called_once_with("Error")
            mb.setIcon.assert_called_once_with(_mb.Critical)
            mb.exec_.assert_called_once_with()
            self.assertEqual(es.priority, 1)
            es.setPriority(0)
            mb = MagicMock()
            with patch(
                "parthenopegui.errorManager.QMessageBox",
                return_value=mb,  # , autospec=True
            ) as _mb:
                es.write("some\ntext")
                _mb.assert_called_once_with(_mb.Information, "Information", "")
            mb.setText.assert_called_once_with("some<br>text")
            mb.setWindowTitle.assert_called_once_with("Information")
            mb.setIcon.assert_called_once_with(_mb.Information)
            mb.exec_.assert_called_once_with()
            mb = MagicMock()
            with patch(
                "parthenopegui.errorManager.QMessageBox",
                return_value=mb,  # , autospec=True
            ) as _mb:
                es.write("some new text")
                _mb.assert_called_once_with(_mb.Information, "Information", "")
            mb.setText.assert_called_once_with("some new text")
            mb.setWindowTitle.assert_called_once_with("Warning")
            mb.setIcon.assert_called_once_with(_mb.Warning)
            mb.exec_.assert_called_once_with()
            with patch(
                "parthenopegui.errorManager.QMessageBox",
                return_value=mb,  # , autospec=True
            ) as _mb:
                self.assertEqual(es.write(" "), None)
                self.assertEqual(_mb.call_count, 0)

        def test_PGErrorManagerClassGui(self):
            """Test functions in ErrorStream"""
            el = PGErrorManagerClassGui()
            self.assertIsInstance(el, PGErrorManagerClass)
            self.assertIsInstance(el.guiStream, ErrorStream)
            l = el.loggerPriority(25)
            self.assertEqual(el.guiStream.priority, 25)
            for fn, p in (
                ("debug", 0),
                ("info", 0),
                ("warning", 1),
                ("error", 2),
                ("critical", 2),
                ("exception", 2),
            ):
                with patch("logging.Logger.%s" % fn) as _f, patch(
                    "parthenopegui.errorManager.PGErrorManagerClassGui.loggerPriority",
                    autospec=True,
                ) as _p:
                    getattr(el, fn)("test message %s" % fn)
                    _p.assert_called_once_with(el, p)
                    _f.assert_called_once_with("test message %s" % fn)

        def test_objects(self):
            """test that the module contains the appropriate objects"""
            import parthenopegui.errorManager as pgem

            self.assertIsInstance(pgem.pGUIErrorManager, pgem.PGErrorManagerClassGui)
            self.assertIsInstance(pgem.pGUIErrorManager, PGErrorManagerClass)


########## configuration
if PGTestsConfig.test_con:

    class TestUnicodeSymbols(unittest.TestCase):
        """Test UnicodeSymbols"""

        def test_unicode(self):
            """test l2u dictionary has the necessary keys and fromLatex"""
            ustrings = [
                "leftrightarrow",
                "rightarrow",
                "mu",
                "sigma",
                "tau",
                "Omega",
                "Lambda",
                "pi",
                "rho",
                "csi",
                "nu",
                "alpha",
                "gamma",
                "Delta",
                "^1",
                "^2",
                "^3",
                "^4",
                "^6",
                "^7",
                "^8",
                "^9",
                "^{1}",
                "^{2}",
                "^{3}",
                "^{4}",
                "^{6}",
                "^{7}",
                "^{8}",
                "^{9}",
                "^{10}",
                "^{11}",
                "^{12}",
                "^{13}",
                "^{14}",
                "^{15}",
                "^{16}",
                "^-",
                "^{-}",
                "^+",
                "^{+}",
                "bar",
                "barnue",
                "nue",
            ]
            self.assertIsInstance(UnicodeSymbols.l2u, dict)
            for k in ustrings:
                self.assertIn(k, UnicodeSymbols.l2u.keys())
            a = UnicodeSymbols()
            self.assertIsInstance(a, QObject)

    class TestConfiguration(unittest.TestCase):
        """test the Configuration class"""

        def test_init(self):
            """test __init__"""
            self.assertIsInstance(Configuration(), pgb.PGConfiguration)

    class TestAskFunctions(PGTestCase):
        """Test the ask* functions"""

        def test_askDirName(self):
            """Test askDirName"""
            with patch(
                "parthenopegui.configuration.QFileDialog.getExistingDirectory",
                side_effect=["/abc/def/ghi/", ""],
            ) as _fd:
                self.assertEqual(
                    askDirName(title="mytitle", dir="/tmp"),
                    "/abc/def/ghi/",
                )
                _fd.assert_called_once_with(None, caption="mytitle", dir="/tmp")
                _fd.reset_mock()
                self.assertEqual(askDirName(), "")

        def test_askFileName(self):
            """Test askFileName"""
            with patch(
                "parthenopegui.configuration.QFileDialog.getOpenFileName",
                side_effect=[("/abc/def/ghi/", "abc"), ("", "")],
            ) as _fd:
                self.assertEqual(
                    askFileName(title="mytitle", dir="/tmp", filter="random"),
                    "/abc/def/ghi/",
                )
                _fd.assert_called_once_with(
                    None,
                    caption="mytitle",
                    dir="/tmp",
                    filter="random",
                    options=QFileDialog.Option.DontConfirmOverwrite,
                )
                _fd.reset_mock()
                self.assertEqual(askFileName(), "")

        def test_askFileNames(self):
            """Test askFileNames"""
            with patch(
                "parthenopegui.configuration.QFileDialog.getOpenFileNames",
                side_effect=[(["/abc", "/def/ghi/"], "abc"), ("", "")],
            ) as _fd:
                self.assertEqual(
                    askFileNames(title="mytitle", dir="/tmp", filter="random"),
                    ["/abc", "/def/ghi/"],
                )
                _fd.assert_called_once_with(
                    None,
                    caption="mytitle",
                    dir="/tmp",
                    filter="random",
                    options=QFileDialog.Option.DontConfirmOverwrite,
                )
                _fd.reset_mock()
                self.assertEqual(askFileNames(), "")

        def test_askGenericText(self):
            """Test askGenericText"""
            p = MagicMock()
            qid = MagicMock()
            qid.exec_.side_effect = [True, False]
            qid.textValue.side_effect = ["abc", "def"]
            with patch(
                "parthenopegui.configuration.QInputDialog",
                return_value=qid,
                autospec=True,
            ) as _qid:
                self.assertEqual(askGenericText("mymessage", "mytitle"), "abc")
                qid.setInputMode.assert_called_once_with(_qid.TextInput)
                qid.setLabelText.assert_called_once_with("mymessage")
                qid.setWindowTitle.assert_called_once_with("mytitle")
                qid.exec_.assert_called_once_with()
                qid.textValue.assert_called_once_with()
                self.assertEqual(askGenericText("mymessage", "mytitle", parent=p), "")

        def test_askSaveFileName(self):
            """Test askSaveFileName"""
            with patch(
                "parthenopegui.configuration.QFileDialog.getSaveFileName",
                side_effect=[("/abc/def/ghi/", "abc"), ("", "")],
            ) as _fd:
                self.assertEqual(
                    askSaveFileName(title="mytitle", dir="/tmp", filter="random"),
                    "/abc/def/ghi/",
                )
                _fd.assert_called_once_with(
                    None,
                    caption="mytitle",
                    dir="/tmp",
                    filter="random",
                    options=QFileDialog.Option.DontConfirmOverwrite,
                )
                _fd.reset_mock()
                self.assertEqual(askSaveFileName(), "")

        def test_askYesNo(self):
            """Test askYesNo"""
            mb = MagicMock()
            mb.addButton.side_effect = ["yes", "no", "yes", "no"]
            mb.clickedButton.side_effect = ["yes", "no"]
            with patch(
                "parthenopegui.configuration.QMessageBox",
                return_value=mb,
                # autospec=True,
            ) as _mb:
                self.assertTrue(askYesNo("mymessage"))
                _mb.assert_called_once_with(_mb.Question, "Question", "mymessage")
                mb.addButton.assert_has_calls([call(_mb.Yes), call(_mb.No)])
                mb.setDefaultButton.assert_called_once_with("no")
                mb.exec_.assert_called_once_with()
                mb.clickedButton.assert_called_once_with()
                _mb.reset_mock()
                self.assertFalse(askYesNo("mymessage", title="mytitle"))
                _mb.assert_called_once_with(_mb.Question, "mytitle", "mymessage")

    class TestPGClasses(unittest.TestCase):
        """test some of the simple extensions of the Qt classes:
        PGFont, PGLabel, PGPushButton
        """

        def testPGFont(self):
            """Test fontsize and inheritance of PGFont"""
            f = PGFont()
            self.assertIsInstance(f, QFont)
            self.assertEqual(f.pointSize(), Configuration.fontsize)

        def testPGLabel(self):
            """Test fontsize and inheritance of PGLabel, and setCenteredText"""
            l = PGLabel("a")
            self.assertIsInstance(l, QLabel)
            self.assertEqual(l.font().pointSize(), Configuration.fontsize)
            self.assertEqual(l.text(), "a")
            self.assertTrue(l.wordWrap())
            with patch("PySide2.QtWidgets.QLabel.setText", autospec=True) as _s:
                l.setCenteredText("abc")
                _s.assert_called_once_with("<center>abc</center>")

        def testPGPushButton(self):
            """Test fontsize and inheritance of PGPushButton"""
            p = PGPushButton("a")
            self.assertIsInstance(p, QPushButton)
            self.assertEqual(p.font().pointSize(), Configuration.fontsize)
            self.assertEqual(p.text(), "a")

    class TestPGLabelButton(PGTestCase):
        """Test the PGLabelButton class"""

        def test_init(self):
            """test __init__"""
            p = QWidget()
            b = PGLabelButton()
            self.assertIsInstance(b, PGPushButton)
            self.assertTrue(b._PGLabelButton__m)
            self.assertTrue(b._PGLabelButton__s)
            self.assertEqual(b.parent(), None)
            b = PGLabelButton(parent=p)
            self.assertIsInstance(b, PGPushButton)
            self.assertEqual(b.parent(), p)
            self.assertIsInstance(b._PGLabelButton__lyt, QHBoxLayout)
            self.assertEqual(b.layout(), b._PGLabelButton__lyt)
            self.assertIsInstance(b._PGLabelButton__lbl, PGLabel)
            self.assertEqual(b.layout().itemAt(0).widget(), b._PGLabelButton__lbl)
            self.assertTrue(
                b._PGLabelButton__lbl.testAttribute(Qt.WA_TranslucentBackground)
            )
            self.assertTrue(
                b._PGLabelButton__lbl.testAttribute(Qt.WA_TransparentForMouseEvents)
            )
            self.assertEqual(b._PGLabelButton__lbl.text(), "")
            b = PGLabelButton(text="abc")
            self.assertEqual(b.parent(), None)
            self.assertEqual(b._PGLabelButton__lbl.text(), "abc")
            b = PGLabelButton(p, "abc")
            self.assertEqual(b.parent(), p)
            self.assertEqual(b._PGLabelButton__lbl.text(), "abc")

        def test_setMaximumWidth(self):
            """test setMaximumWidth"""
            b = PGLabelButton()
            b.setMaximumWidth(1234)
            self.assertEqual(QPushButton.maximumWidth(b), 1234)
            self.assertEqual(b._PGLabelButton__lbl.maximumWidth(), 1234)

        def test_setText(self):
            """test setText"""
            b = PGLabelButton()
            b.setText("abc")
            self.assertEqual(QPushButton.text(b), "")
            self.assertEqual(b._PGLabelButton__lbl.text(), "abc")

        def test_sizeHint(self):
            """test sizeHint"""
            b = PGLabelButton()
            s = QSize(112, 133)
            b._PGLabelButton__lbl.sizeHint = MagicMock(return_value=s)
            r = b.sizeHint()
            self.assertIsInstance(r, QSize)
            self.assertEqual(r.width(), 112 + 2 * b._PGLabelButton__m)
            self.assertEqual(r.height(), 133 + 2 * b._PGLabelButton__m)

        def test_text(self):
            """test text"""
            p = QWidget()
            b = PGLabelButton("a")
            self.assertEqual(b.text(), "")
            b = PGLabelButton(p, "a")
            self.assertEqual(b.text(), "a")
            b._PGLabelButton__lbl.setText("b")
            self.assertEqual(b.text(), "b")

    class TestDDTableWidget(PGTestCase):
        """Test the DDTableWidget class"""

        def test_init(self):
            """test init"""
            p = QWidget()
            mddtw = DDTableWidget(p, "head")
            self.assertIsInstance(mddtw, QTableWidget)
            self.assertEqual(mddtw.parent(), p)
            self.assertEqual(mddtw.columnCount(), 1)
            self.assertTrue(mddtw.dragEnabled())
            self.assertTrue(mddtw.acceptDrops())
            self.assertFalse(mddtw.dragDropOverwriteMode())
            self.assertEqual(mddtw.selectionBehavior(), QAbstractItemView.SelectRows)
            with patch(
                "PySide2.QtWidgets.QTableWidget.setHorizontalHeaderLabels",
                autospec=True,
            ) as _shl:
                mddtw = DDTableWidget(p, "head")
                _shl.assert_called_once_with(["head"])

        def test_dropMimeData(self):
            """test dropMimeData"""
            p = QWidget()
            mddtw = DDTableWidget(p, "head")
            self.assertTrue(mddtw.dropMimeData(12, 0, None, None))
            self.assertEqual(mddtw.lastDropRow, 12)

        def test_dropEvent(self):
            """test dropEvent"""
            p = QWidget()
            sender = DDTableWidget(p, "head")
            item = QTableWidgetItem("")
            item.setData(Qt.DecorationRole, QPixmap(Nuclides.all["9Be"]))
            item.setData(Qt.UserRole, "9Be")
            sender.insertRow(0)
            sender.setItem(0, 0, item)
            item = QTableWidgetItem("n")
            item.setData(Qt.UserRole, "n")
            sender.insertRow(1)
            sender.setItem(1, 0, item)
            sender.selectionModel().select(
                sender.model().index(0, 0), QItemSelectionModel.Select
            )

            mddtw = DDTableWidget(p, "head")
            item = QTableWidgetItem("p")
            item.setData(Qt.UserRole, "p")
            mddtw.insertRow(0)
            mddtw.setItem(0, 0, item)
            item = QTableWidgetItem("")
            item.setData(Qt.DecorationRole, QPixmap(Nuclides.all["6Li"]))
            item.setData(Qt.UserRole, "6Li")
            mddtw.insertRow(1)
            mddtw.setItem(1, 0, item)

            mimedata = QMimeData()
            if sys.version_info[0] < 3:
                mimedata.setData(
                    "application/x-qabstractitemmodeldatalist", QByteArray("n")
                )
            else:
                mimedata.setData(
                    "application/x-qabstractitemmodeldatalist", QByteArray(b"n")
                )
            ev = QDropEvent(
                QPointF(1, 1),
                Qt.DropActions(Qt.MoveAction),
                mimedata,
                Qt.MouseButtons(Qt.LeftButton),
                Qt.KeyboardModifiers(Qt.NoModifier),
            )
            with patch(
                "PySide2.QtGui.QDropEvent.source", return_value=sender, autospec=True
            ) as _s:
                mddtw.dropEvent(ev)
                _s.assert_called_once_with()
            self.assertEqual(sender.rowCount(), 1)
            self.assertEqual(mddtw.rowCount(), 3)
            self.assertEqual(sender.item(0, 0).data(Qt.UserRole), "9Be")
            self.assertEqual(mddtw.item(0, 0).data(Qt.UserRole), "n")
            self.assertEqual(mddtw.item(1, 0).data(Qt.UserRole), "p")
            self.assertEqual(mddtw.item(2, 0).data(Qt.UserRole), "6Li")

            mddtw.selectionModel().select(
                mddtw.model().index(1, 0), QItemSelectionModel.Select
            )
            if sys.version_info[0] < 3:
                mimedata.setData(
                    "application/x-qabstractitemmodeldatalist", QByteArray("p")
                )
            else:
                mimedata.setData(
                    "application/x-qabstractitemmodeldatalist", QByteArray(b"p")
                )
            ev = QDropEvent(
                QPointF(1, 1),
                Qt.DropActions(Qt.MoveAction),
                mimedata,
                Qt.MouseButtons(Qt.LeftButton),
                Qt.KeyboardModifiers(Qt.NoModifier),
            )
            with patch(
                "PySide2.QtGui.QDropEvent.source", return_value=mddtw, autospec=True
            ) as _s:
                mddtw.dropEvent(ev)
                _s.assert_called_once_with()
            self.assertEqual(sender.rowCount(), 1)
            self.assertEqual(mddtw.rowCount(), 3)
            self.assertEqual(sender.item(0, 0).data(Qt.UserRole), "9Be")
            self.assertEqual(mddtw.item(0, 0).data(Qt.UserRole), "n")
            self.assertEqual(mddtw.item(1, 0).data(Qt.UserRole), "p")
            self.assertEqual(mddtw.item(2, 0).data(Qt.UserRole), "6Li")

        def test_getselectedRowsFast(self):
            """test getselectedRowsFast"""
            p = QWidget()
            mddtw = DDTableWidget(p, "head")
            a = mddtw.getselectedRowsFast()
            self.assertEqual(a, [])
            with patch(
                "PySide2.QtWidgets.QTableWidget.selectedItems",
                return_value=[QTableWidgetItem() for i in range(5)],
                autospec=True,
            ) as _si, patch(
                "PySide2.QtWidgets.QTableWidgetItem.row",
                side_effect=[1, 8, 3, 1, 2],
                autospec=True,
            ) as _r, patch(
                "PySide2.QtWidgets.QTableWidgetItem.text",
                side_effect=["de", "fg", "ab", "hi", "bibkey"],
                autospec=True,
            ) as _t:
                a = mddtw.getselectedRowsFast()
                self.assertEqual(a, [1, 2, 3, 8])

        def test_reorder(self):
            """test reorder"""
            p = QWidget()
            sender = DDTableWidget(p, "head")
            qsb = QSignalBlocker(sender)
            item = QTableWidgetItem("")
            item.setData(Qt.DecorationRole, QPixmap(Nuclides.all["9Be"]))
            item.setData(Qt.UserRole, "9Be")
            sender.insertRow(0)
            sender.setItem(0, 0, item)
            item = QTableWidgetItem("n")
            item.setData(Qt.UserRole, "n")
            sender.insertRow(1)
            sender.setItem(1, 0, item)
            item = QTableWidgetItem("")
            item.setData(Qt.DecorationRole, QPixmap(Nuclides.all["16O"]))
            item.setData(Qt.UserRole, "16O")
            sender.insertRow(2)
            sender.setItem(2, 0, item)
            item = QTableWidgetItem("")
            item.setData(Qt.DecorationRole, QPixmap(Nuclides.all["3He"]))
            item.setData(Qt.UserRole, "3He")
            sender.insertRow(3)
            sender.setItem(3, 0, item)
            self.assertEqual(sender.item(0, 0).data(Qt.UserRole), "9Be")
            self.assertEqual(sender.item(1, 0).data(Qt.UserRole), "n")
            self.assertEqual(sender.item(2, 0).data(Qt.UserRole), "16O")
            self.assertEqual(sender.item(3, 0).data(Qt.UserRole), "3He")
            del qsb
            with patch(
                "parthenopegui.configuration.DDTableWidget.reorder", autospec=True
            ) as _f:
                sender.cellChanged.emit(0, 0)
                _f.assert_called_once_with(sender, 0, 0)
            sender.cellChanged.emit(0, 0)
            self.assertEqual(sender.item(0, 0).data(Qt.UserRole), "n")
            self.assertEqual(sender.item(1, 0).data(Qt.UserRole), "3He")
            self.assertEqual(sender.item(2, 0).data(Qt.UserRole), "9Be")
            self.assertEqual(sender.item(3, 0).data(Qt.UserRole), "16O")

    class TestNuclides(unittest.TestCase):
        """test the Nuclides class"""

        def test_attributes(self):
            """test the attributes of the Nuclides class"""
            self.assertIsInstance(Nuclides.updated, Signal)
            self.assertIsInstance(Nuclides.all, dict)
            self.assertIsInstance(Nuclides.current, dict)
            self.assertIsInstance(Nuclides.nuclideOrder, dict)
            nucl = [
                "n",
                "p",
                "2H",
                "3H",
                "3He",
                "4He",
                "6Li",
                "7Li",
                "7Be",
                "8Li",
                "8B",
                "9Be",
                "10B",
                "11B",
                "11C",
                "12B",
                "12C",
                "12N",
                "13C",
                "13N",
                "14C",
                "14N",
                "14O",
                "15N",
                "15O",
                "16O",
            ]
            for i, q in enumerate(nucl):
                self.assertIn(q, Nuclides.all.keys())
                self.assertIsInstance(Nuclides.all[q], QImage)
                self.assertIn(q, Nuclides.nuclideOrder.keys())
                self.assertEqual(Nuclides.nuclideOrder[q], i + 1)
            a = Nuclides()
            with MagicMock() as _f:
                a.updated.connect(_f)
                _f.assert_not_called()
                self.assertEqual(a.current, {})
                a.updateCurrent(["n", "p"])
                self.assertEqual(
                    a.current, {"n": Nuclides.all["n"], "p": Nuclides.all["p"]}
                )
                _f.assert_called_once_with()
            self.assertEqual(Nuclides.current, {})

    class TestReactions(unittest.TestCase):
        """Test reaction-related functions"""

        def test_imageFromTex(self):
            """test the imageFromTex function"""
            qi = imageFromTex(r"$\nu_\mu$")
            self.assertIsInstance(qi, QImage)
            with patch(
                "matplotlib.figure.Figure.__init__", return_value=None, autospec=True
            ) as _fi, self.assertRaises(AttributeError):
                qi = imageFromTex(r"$\nu_\mu$")
                _fi.assert_called_once_with()
            with patch(
                "matplotlib.axes.Axes.text", return_value=None, autospec=True
            ) as _t, self.assertRaises(AttributeError):
                qi = imageFromTex(r"$\nu_\mu$")
                _t.assert_called_once_with(
                    0,
                    0,
                    r"$\nu_\mu$",
                    ha="left",
                    va="bottom",
                    fontsize=pbConfig.params["bibListFontSize"],
                )
            fc = FigureCanvasAgg(matplotlib.figure.Figure())
            qii = QImage()
            with patch(
                "matplotlib.backends.backend_agg." + "FigureCanvasAgg.print_to_buffer",
                return_value=("buf", ["size0", "size1"]),
                autospec=True,
            ) as _ptb, patch(
                "parthenopegui.configuration." + "FigureCanvasAgg",
                return_value=fc,
                autospec=USE_AUTOSPEC_CLASS,
            ), patch(
                "PySide2.QtGui.QImage", return_value=qii  # , autospec=True
            ) as _qii, self.assertRaises(
                AttributeError
            ):
                qi = imageFromTex(r"$\nu_\mu$")
                _ptb.assert_called_once_with(fc)
                _qii.assert_calle_once_with(
                    "buf", "size0", "size1", QImage.Format_ARGB32
                )

        def test_cacheImageFromTex(self):
            """test the doReaction function"""
            fname = os.path.join(Configuration.cachePath, "file.png")
            img = QImage()
            img.load = MagicMock(side_effect=[False, True])
            img1 = QImage()
            img1.save = MagicMock()
            with patch(
                "os.path.exists",
                side_effect=[False, True, False, False, True, True, True],
            ) as _ope, patch("os.makedirs") as _om, patch(
                "parthenopegui.configuration.QImage", return_value=img
            ) as _qi, patch(
                "parthenopegui.configuration.imageFromTex",
                return_value=img1,
                autospec=True,
            ) as _ift:
                # no file, no cache
                r = cacheImageFromTex("somestring")
                self.assertEqual(_ope.call_count, 0)
                _ift.assert_called_once_with("somestring")
                self.assertEqual(r, img1)
                _ift.reset_mock()
                # img not existing, folder exists
                r = cacheImageFromTex("somestring", "file")
                _ope.assert_has_calls([call(fname), call(Configuration.cachePath)])
                self.assertEqual(_qi.call_count, 0)
                img1.save.assert_called_once_with(fname)
                self.assertEqual(_om.call_count, 0)
                self.assertEqual(r, img1)
                img1.save.reset_mock()
                # img not existing, need to create folder
                r = cacheImageFromTex("somestring", "file")
                self.assertEqual(_qi.call_count, 0)
                img1.save.assert_called_once_with(fname)
                _om.assert_called_once_with(Configuration.cachePath)
                self.assertEqual(r, img1)
                _ift.reset_mock()
                # img exists, fail to load
                r = cacheImageFromTex("somestring", "file")
                self.assertEqual(_qi.call_count, 1)
                img.load.assert_called_once_with(fname)
                _ift.assert_called_once_with("somestring")
                self.assertEqual(r, img1)
                # img exists, loaded successfully
                r = cacheImageFromTex("somestring", "file")
                self.assertEqual(_ift.call_count, 1)
                self.assertEqual(r, img)

        def test_attributes(self):
            """test the attributes of the Reactions class"""
            self.assertIsInstance(Reactions.updated, Signal)
            self.assertIsInstance(Reactions.all, dict)
            self.assertIsInstance(Reactions.current, dict)
            for i in range(1, 101):
                self.assertIn(i, Reactions.all.keys())
                self.assertIsInstance(Reactions.all[i], tuple)
                self.assertEqual(len(Reactions.all[i]), 2)
            a = Reactions()
            with MagicMock() as _f:
                a.updated.connect(_f)
                _f.assert_not_called()
                self.assertEqual(a.current, {})
                a.updateCurrent([1, 100])
                self.assertEqual(
                    a.current, {1: Reactions.all[1], 100: Reactions.all[100]}
                )
                _f.assert_called_once_with()
            self.assertEqual(Reactions.current, {})

    class TestParameter(unittest.TestCase):
        """Test the Parameters class"""

        def test_init(self):
            """test the init function and the class attributes"""
            with patch(
                "parthenopegui.configuration.Parameter.paramLabel",
                return_value=QImage(),
            ) as _pl:
                p = Parameter("alpha", "\alpha", 0, 1, 0.5)
                _pl.assert_called_once_with("\alpha")
            self.assertEqual(p.name, "alpha")
            self.assertEqual(p.label, "\alpha")
            self.assertEqual(p.defaultmin, 0.0)
            self.assertEqual(p.defaultmax, 1.0)
            self.assertEqual(p.defaultval, 0.5)
            self.assertEqual(p.currentmin, 0.0)
            self.assertEqual(p.currentmax, 1.0)
            self.assertEqual(p.currentval, 0.5)
            self.assertEqual(p.currentN, 2)
            self.assertEqual(p.currenttype, "single")
            self.assertIsInstance(p.fig, QPixmap)
            self.assertIsInstance(p._Parameter__grids, dict)

        def test_resetCurrent(self):
            """test the resetCurrent function"""
            with patch(
                "parthenopegui.configuration.Parameter.paramLabel",
                return_value=QImage(),
            ) as _pl:
                p = Parameter("alpha", "\alpha", 0, 1, 0.5)
            p.currentmin = 10.0
            p.currentmax = 11.0
            p.currentval = 10.5
            p.currentN = 200
            p.currenttype = "grid"
            p.resetCurrent()
            self.assertEqual(p.currentmin, 0.0)
            self.assertEqual(p.currentmax, 1.0)
            self.assertEqual(p.currentval, 0.5)
            self.assertEqual(p.currentN, 2)
            self.assertEqual(p.currenttype, "single")

        def test_addGrid(self):
            """test the addGrid function"""
            with patch(
                "parthenopegui.configuration.Parameter.paramLabel",
                return_value=QImage(),
            ) as _pl:
                p = Parameter("alpha", "\alpha", 0, 1, 0.5)
            self.assertEqual(len(p._Parameter__grids), 0)
            with patch("logging.Logger.warning") as _w:
                self.assertEqual(p.addGrid(123, 0, 1, 2), None)
                _w.assert_called_once_with(
                    "Invalid number of points for a grid:" + " N=0 for parameter alpha"
                )
            self.assertEqual(len(p._Parameter__grids), 0)
            self.assertEqual(p.addGrid(123, 1, 1, 2), None)
            self.assertEqual(len(p._Parameter__grids), 1)
            a = p._Parameter__grids[123]
            self.assertEqual(a["type"], "single")
            self.assertEqual(a["N"], 1)
            self.assertEqual(a["val"], 1.5)
            self.assertEqual(a["min"], 1.5)
            self.assertEqual(a["max"], 1.5)
            self.assertEqual(a["pts"], np.asarray([1.5]))
            self.assertEqual(p.addGrid(321, 3, 1, 4), None)
            self.assertEqual(len(p._Parameter__grids), 2)
            a = p._Parameter__grids[321]
            self.assertEqual(a["type"], "grid")
            self.assertEqual(a["N"], 3)
            self.assertEqual(a["val"], None)
            self.assertEqual(a["min"], 1)
            self.assertEqual(a["max"], 4)
            self.assertEqual(list(a["pts"]), list(np.linspace(1, 4, 3)))
            self.assertEqual(p.addGrid(123, 3, 1, 4), None)
            self.assertEqual(len(p._Parameter__grids), 2)
            a = p._Parameter__grids[123]
            self.assertEqual(a["type"], "grid")
            self.assertEqual(a["N"], 3)
            self.assertEqual(a["val"], None)
            self.assertEqual(a["min"], 1)
            self.assertEqual(a["max"], 4)
            self.assertEqual(list(a["pts"]), list(np.linspace(1, 4, 3)))

        def test_getGrid(self):
            """test the getGrid function"""
            with patch(
                "parthenopegui.configuration.Parameter.paramLabel",
                return_value=QImage(),
            ) as _pl:
                p = Parameter("alpha", "\alpha", 0, 1, 0.5)
            p.addGrid(12, 3, 1, 4)
            p.addGrid(23, 1, 5, 7)
            self.assertEqual(p.currentmin, 0.0)
            self.assertEqual(p.currentmax, 1.0)
            self.assertEqual(p.currentval, 0.5)
            self.assertEqual(p.currentN, 2)
            self.assertEqual(p.currenttype, "single")
            p.getGrid(12)
            self.assertEqual(p.currentmin, 1.0)
            self.assertEqual(p.currentmax, 4.0)
            self.assertEqual(p.currentval, 0.5)
            self.assertEqual(p.currentN, 3)
            self.assertEqual(p.currenttype, "grid")
            p.getGrid(23)
            self.assertEqual(p.currentmin, 6.0)
            self.assertEqual(p.currentmax, 6.0)
            self.assertEqual(p.currentval, 6)
            self.assertEqual(p.currentN, 1)
            self.assertEqual(p.currenttype, "single")
            with patch("logging.Logger.debug") as _d:
                p.getGrid(123)
                _d.assert_called_once_with("the grid does not exist: 123")
            self.assertEqual(p.currentmin, 6.0)
            self.assertEqual(p.currentmax, 6.0)
            self.assertEqual(p.currentval, 6)
            self.assertEqual(p.currentN, 1)
            self.assertEqual(p.currenttype, "single")

        def test_getGridPoints(self):
            """test the getGridPoints function"""
            with patch(
                "parthenopegui.configuration.Parameter.paramLabel",
                return_value=QImage(),
            ) as _pl:
                p = Parameter("alpha", "\alpha", 0, 1, 0.5)
            p.addGrid(12, 3, 1, 4)
            p.addGrid(23, 1, 5, 7)
            with patch(
                "parthenopegui.configuration.Parameter.getGrid", autospec=True
            ) as _g:
                r = p.getGridPoints(12)
                self.assertEqual(len(r), 2)
                self.assertEqual(r[0], 3)
                self.assertEqual(list(r[1]), [1, 2.5, 4])
                _g.assert_called_once_with(p, 12)
            r = p.getGridPoints(23)
            self.assertEqual(len(r), 2)
            self.assertEqual(r[0], 1)
            self.assertEqual(list(r[1]), [6.0])
            with patch(
                "parthenopegui.configuration.Parameter.getGrid", autospec=True
            ) as _g, patch("logging.Logger.debug") as _d:
                self.assertEqual(p.getGridPoints(123), (None, None))
                _d.assert_called_once_with("the grid does not exist: 123")
                self.assertEqual(_g.call_count, 0)

        def test_deleteGrid(self):
            """test the deleteGrid function"""
            with patch(
                "parthenopegui.configuration.Parameter.paramLabel",
                return_value=QImage(),
            ) as _pl:
                p = Parameter("alpha", "\alpha", 0, 1, 0.5)
            p.addGrid(12, 3, 1, 4)
            p.addGrid(23, 1, 5, 7)
            with patch(
                "parthenopegui.configuration.Parameter.resetCurrent", autospec=True
            ) as _rc:
                self.assertTrue(p.deleteGrid(23))
                _rc.assert_called_once_with(p)
            self.assertEqual(list(p._Parameter__grids.keys()), [12])
            with patch(
                "parthenopegui.configuration.Parameter.resetCurrent", autospec=True
            ) as _rc, patch("logging.Logger.exception") as _e:
                self.assertFalse(p.deleteGrid(23))
                _e.assert_called_once_with("Cannot delete grid: 23")
                self.assertEqual(_rc.call_count, 0)
            self.assertEqual(list(p._Parameter__grids.keys()), [12])

        def test_paramLabel(self):
            """test the paramLabel function"""
            with patch(
                "parthenopegui.configuration.Parameter.paramLabel",
                return_value=QImage(),
            ) as _pl:
                p = Parameter("alpha", "\alpha", 0, 1, 0.5)
            with patch(
                "parthenopegui.configuration.imageFromTex", return_value=QImage()
            ) as _ift:
                qi = p.paramLabel(r"$\nu_\mu$")
                _ift.assert_called_once_with(r"$\nu_\mu$")

    class TestParameters(unittest.TestCase):
        """Test the Parameters class"""

        @classmethod
        def setUpClass(self):
            """Assign a temporary QApplication to the class instance"""
            self.p = Parameters()

        def test_init(self):
            """test the init function and the class attributes"""
            self.assertIsInstance(Parameters.paramOrder, list)
            self.assertEqual(
                Parameters.paramOrder,
                ["eta10", "DeltaNnu", "taun", "csinue", "csinux", "rhoLambda"],
            )
            self.assertIsInstance(Parameters.paramOrderParth, list)
            self.assertEqual(
                Parameters.paramOrderParth,
                ["DeltaNnu", "csinue", "csinux", "taun", "rhoLambda", "eta10"],
            )
            self.assertIsInstance(self.p.all, dict)
            self.assertIsInstance(self.p.gridsList, dict)
            for a in Parameters.paramOrder:
                self.assertIn(a, self.p.all.keys())
                self.assertIsInstance(self.p.all[a], Parameter)
            with patch(
                "parthenopegui.configuration.Parameter", autospec=USE_AUTOSPEC_CLASS
            ) as _pi:
                p = Parameters()
                _pi.assert_has_calls(
                    [
                        call("eta10", r"$\eta_{10}$", 2, 9, 6.13832),
                        call("DeltaNnu", r"$\Delta N_\nu$", -3, 3, 0),
                        call("taun", r"$\tau_n$", 876.4, 882.4, 879.4),
                        call("csinue", r"$\xi_{\nu_e}$", -1, 1, 0),
                        call("csinux", r"$\xi_{\nu_X}$", -1, 1, 0),
                        call("rhoLambda", r"$\rho_\Lambda$", 0, 1, 0),
                    ]
                )

        def test_resetCurrent(self):
            """test the resetCurrent function"""
            with patch(
                "parthenopegui.configuration.Parameter.resetCurrent", autospec=True
            ) as _pi:
                self.p.resetCurrent()
                _pi.assert_has_calls([call(v) for v in self.p.all.values()])

        def test_getGrid(self):
            """test the getGrid function"""
            with patch(
                "parthenopegui.configuration.Parameter.getGrid", autospec=True
            ) as _pi:
                self.p.getGrid(123)
                _pi.assert_has_calls([call(v, 123) for v in self.p.all.values()])

        def test_getGridPoints(self):
            """test the getGridPoints function"""
            self.p.all["eta10"].currentmin = 1
            self.p.all["eta10"].currentmax = 1
            self.p.all["eta10"].currentN = 1
            self.p.all["DeltaNnu"].currentmin = 0
            self.p.all["DeltaNnu"].currentmax = 0
            self.p.all["DeltaNnu"].currentN = 1
            self.p.all["taun"].currentmin = 880
            self.p.all["taun"].currentmax = 880
            self.p.all["taun"].currentN = 1
            self.p.all["csinue"].currentmin = -1
            self.p.all["csinue"].currentmax = 1
            self.p.all["csinue"].currentN = 2
            self.p.all["csinux"].currentmin = -1
            self.p.all["csinux"].currentmax = 1
            self.p.all["csinux"].currentN = 2
            self.p.all["rhoLambda"].currentmin = 0.7
            self.p.all["rhoLambda"].currentmax = 0.7
            self.p.all["rhoLambda"].currentN = 1
            with patch(
                "parthenopegui.configuration.Parameters.getGrid", autospec=True
            ) as _pi:
                res = self.p.getGridPoints(123)
                self.assertEqual(
                    res.tolist(),
                    [
                        [1, 0, 880, -1, -1, 0.7],
                        [1, 0, 880, -1, 1, 0.7],
                        [1, 0, 880, 1, -1, 0.7],
                        [1, 0, 880, 1, 1, 0.7],
                    ],
                )
                _pi.assert_called_once_with(self.p, 123)

        def test_deleteGrid(self):
            """test the deleteGrid function"""
            self.p.gridsList[123] = 987
            self.p.gridsList[1] = 5
            with patch(
                "parthenopegui.configuration.Parameter.deleteGrid",
                autospec=True,
                return_value=True,
            ) as _pi:
                self.assertTrue(self.p.deleteGrid(123))
                _pi.assert_has_calls(
                    [call(self.p.all[p], 123) for p in sorted(self.p.all.keys())]
                )
            self.assertEqual(list(self.p.gridsList.keys()), [1])
            with patch(
                "parthenopegui.configuration.Parameter.deleteGrid",
                autospec=True,
                side_effect=[True, False],
            ) as _pi, patch("logging.Logger.debug") as _d:
                self.assertFalse(self.p.deleteGrid(123))
                _pi.assert_has_calls(
                    [call(self.p.all[p], 123) for p in sorted(self.p.all.keys())[0:2]]
                )
                _d.assert_called_once_with("Cannot delete grid for csinue")
            self.assertEqual(list(self.p.gridsList.keys()), [1])
            with patch(
                "parthenopegui.configuration.Parameter.deleteGrid",
                autospec=True,
                return_value=True,
            ) as _pi, patch("logging.Logger.exception") as _e:
                self.assertFalse(self.p.deleteGrid(123))
                _e.assert_called_once_with("Cannot delete grid index")
            self.assertEqual(list(self.p.gridsList.keys()), [1])


########## runner and runUtils
if PGTestsConfig.test_run:

    class TestRunSingle(PGTestCase):
        """Test the runSingle function"""

        def test_shellCommand(self):
            """Test the value of the string shellCommand"""
            self.assertEqual(shellCommand, "./%s < %s | tee %s > /dev/null")

        def test_func(self):
            """Test the runSingle function"""
            ic = os.path.join(testConfig.testRunDirDefaultSample, "test_123.in")
            lg = os.path.join(testConfig.testRunDirDefaultSample, "test_123.log")
            with patch("logging.Logger.exception") as _e, patch(
                "logging.Logger.info"
            ) as _i:
                self.assertEqual(runSingle("0", "a", "b"), -1)
                _e.assert_called_once_with(PGText.runSingleArgsType, exc_info=True)
                _e.reset_mock()
                self.assertEqual(runSingle(1, "a", 3), -1)
                _e.assert_called_once_with(PGText.runSingleArgsType, exc_info=True)
                _e.reset_mock()
                self.assertEqual(runSingle(1, 2, "b"), -1)
                _e.assert_called_once_with(PGText.runSingleArgsType, exc_info=True)
                self.assertEqual(_i.call_count, 0)
            with patch("os.system") as _s, patch("os.remove") as _r, patch(
                "logging.Logger.info"
            ) as _i:
                self.assertEqual(runSingle(123, ic, lg), 123)
                with open(ic.replace("card", "in")) as _f:
                    f = _f.read()
                self.assertEqual(f, "c\n%s\n" % ic)
                _s.assert_called_once_with(
                    shellCommand
                    % (
                        Configuration.fortranExecutableName,
                        ic.replace("card", "in"),
                        lg,
                    )
                )
                self.assertEqual(_r.call_count, 0)
                _i.assert_any_call(PGText.startRunI % 123)

    class TestNoGuiRun(PGTestCase):
        """Test the NoGuiRun class"""

        def test_init(self):
            """test __init__"""
            with patch("parthenopegui.runUtils.NoGUIRun.makeFortranExecutable") as _f:
                ngr = NoGUIRun()
                _f.assert_called_once_with()
            self.assertIsInstance(ngr.finishedRuns, list)
            self.assertIsInstance(ngr.failedRuns, list)
            self.assertIsInstance(ngr.nuclidesEvolution, dict)
            self.assertIsInstance(ngr.nuclidesHeader, list)
            self.assertIsInstance(ngr.parthenopeHeader, list)
            self.assertIsInstance(ngr.parthenopeOutPoints, dict)
            self.assertFalse(hasattr(self, "guilogger"))

        def test_allHaveFinished(self):
            """test allHaveFinished"""
            rp = NoGUIRun()
            rp.defineParams(
                testConfig.commonParamsGrid,
                testConfig.gridPointsGrid,
            )
            for ix, pt in enumerate(rp.getGridPointsList()):
                with patch("os.remove") as _r:
                    rp.oneHasFinished(ix)
            with patch("os.remove") as _rm, patch("numpy.savetxt") as _s, patch(
                "logging.Logger.debug"
            ) as _d, patch(
                "parthenopegui.runUtils.NoGUIRun.fixParthenopeOutPoints",
                autospec=True,
            ) as _fp:
                rp.allHaveFinished()
                _fp.assert_called_once_with(rp)
                for ix, pt in enumerate(rp.getGridPointsList()):
                    _rm.assert_has_calls(
                        [
                            call(
                                testConfig.commonParamsGrid["output_file_parthenope"]
                                % ix
                            )
                        ]
                    )
                self.assertEqual(_d.call_count, 0)
                _s.assert_called_once()
                self.assertEqual(
                    _s.call_args[0][0], testConfig.commonParamsGrid["output_file_grid"]
                )
                self.assertEqualArray(
                    np.asarray(
                        [
                            rp.parthenopeOutPoints[ix]
                            for ix in sorted(rp.parthenopeOutPoints.keys())
                        ]
                    ),
                    _s.call_args[0][1],
                )
                self.assertEqual(_s.call_args[1]["fmt"], "%14.7e")
                self.assertEqual(
                    _s.call_args[1]["header"], "  ".join(rp.parthenopeHeader)
                )

        def test_createInputCard(self):
            """test createInputCard"""
            now = datetime.datetime.today().strftime("%y%m%d_%H%M%S")
            inputdict = {
                "taun": 880.0,
                "DeltaNnu": 1.0,
                "csinue": 0.1,
                "csinux": -0.1,
                "rhoLambda": 0.7,
                "eta10": 6.1,
                "N_changed_rates": 3,
                "N_stored_nuclides": 4,
                "changed_rates_list": [[2, 1, 1.0], [3, 2, 1.0], [4, 3, "1.12"]],
                "inputcard_filename": os.path.join(
                    testConfig.testRunDirDefaultSample, "input_" + now + "_%d.card" % 1
                ),
                "num_nuclides_net": 18,
                "onScreenOutput": False,
                "output_file_nuclides": os.path.join(
                    testConfig.testRunDirDefaultSample,
                    "nuclides_" + now + "_%d.out" % 1,
                ),
                "output_file_info": os.path.join(
                    testConfig.testRunDirDefaultSample,
                    "info_" + now + "_%d.out" % 1,
                ),
                "output_file_parthenope": os.path.join(
                    testConfig.testRunDirDefaultSample,
                    "parthenope_" + now + "_%d.out" % 1,
                ),
                "output_overwrite": True,
                "output_save_nuclides": True,
                "stored_nuclides": ["n", "p", "3He", "4He"],
            }
            rp = NoGUIRun()
            rp.defineParams(
                testConfig.commonParamsGrid,
                testConfig.gridPointsGrid,
            )
            rp.createInputCard(**inputdict)
            with open(inputdict["inputcard_filename"]) as _f:
                c = _f.read()
            self.assertRegex(c, "TAU[ ]*880.0000000 ")
            self.assertRegex(c, "DNNU[ ]*1.0000000 ")
            self.assertRegex(c, "XIE[ ]*0.1000000 ")
            self.assertRegex(c, "XIX[ ]*-0.1000000 ")
            self.assertRegex(c, "RHOLMBD[ ]*0.7000000 ")
            self.assertRegex(c, "OVERWRITE[ ]*T ")
            self.assertRegex(c, "FOLLOW[ ]*F ")
            self.assertRegex(c, "ETA10[ ]*6.1000000 ")
            self.assertRegex(c, "OUTPUT[ ]*T 4 1 2 5 6 ")
            self.assertRegex(c, "NETWORK[ ]*18 ")
            self.assertRegex(
                c,
                (
                    "FILES[ ]*{td:}/{pt:}_{now:}_1.out"
                    + " {td:}/{nc:}_{now:}_1.out"
                    + " {td:}/{inf:}_{now:}_1.out "
                ).format(
                    td=testConfig.testRunDirDefaultSample,
                    now=now,
                    pt=Configuration.parthenopeFilename,
                    nc=Configuration.nuclidesFilename,
                    inf=Configuration.infoFilename,
                ),
            )
            self.assertRegex(c, "RATES([ ]*)3 \(2 1 1.0\) \(3 2 1.0\) \(4 3 1.12\)")
            self.assertRegex(c, "EXIT[ ]*terminates input")

        def test_defineParams(self):
            """test defineParams"""
            ngr = NoGUIRun()
            self.assertFalse(hasattr(ngr, "gridPoints"))
            self.assertFalse(hasattr(ngr, "commonParams"))
            self.assertFalse(hasattr(ngr, "totalRuns"))
            with patch(
                "parthenopegui.runUtils.NoGUIRun.getGridPointsList",
                return_value=["a", "b", "c"],
            ) as _g:
                ngr.defineParams("cp", "gp")
            self.assertEqual(ngr.gridPoints, "gp")
            self.assertEqual(ngr.commonParams, "cp")
            self.assertEqual(ngr.totalRuns, 3)
            _g.assert_called_once_with()

        def test_fixParthenopeOutPoints(self):
            """test fixParthenopeOutPoints"""
            rp = NoGUIRun()
            rp.defineParams(
                testConfig.commonParamsGrid,
                testConfig.gridPointsGrid,
            )
            rp.parthenopeHeader = ["a", "b", "c", "d", "e", "f", "g"]
            rp.parthenopeOutPoints = {
                0: [1, 2, 3, 4, 5, 6, 7],
                1: [11, 12, 13, 14, 15],
                2: [21, 22, 23, 24, 25, 26, 27, 28],
            }
            rp.fixParthenopeOutPoints()
            self.assertEqualArray(rp.parthenopeOutPoints[0], [1, 2, 3, 4, 5, 6, 7])
            self.assertEqualArray(
                rp.parthenopeOutPoints[1], [11, 12, 13, 14, 15, np.nan, np.nan]
            )
            self.assertEqualArray(
                rp.parthenopeOutPoints[2], [21, 22, 23, 24, 25, 26, 27]
            )

            with patch("os.remove") as _rm, patch("numpy.savetxt") as _s, patch(
                "logging.Logger.debug"
            ) as _d:
                rp.allHaveFinished()
                for ix, pt in enumerate(rp.getGridPointsList()):
                    _rm.assert_has_calls(
                        [
                            call(
                                testConfig.commonParamsGrid["output_file_parthenope"]
                                % ix
                            )
                        ]
                    )

            with patch("os.remove", side_effect=IOError) as _rm, patch(
                "numpy.savetxt"
            ) as _s, patch("logging.Logger.debug") as _d:
                rp.allHaveFinished()
                for ix, pt in enumerate(rp.getGridPointsList()):
                    _rm.assert_has_calls(
                        [
                            call(
                                testConfig.commonParamsGrid["output_file_parthenope"]
                                % ix
                            )
                        ]
                    )
                    _d.assert_has_calls(
                        [
                            call(
                                PGText.fileNotFound
                                % testConfig.commonParamsGrid["output_file_parthenope"]
                                % ix,
                            )
                        ]
                    )

            rp.failedRuns = [1, 12]
            failedInstructionsFilename = os.path.join(
                rp.commonParams["output_folder"],
                Configuration.failedRunsInstructionsFilename,
            )
            if os.path.exists(failedInstructionsFilename):
                os.remove(failedInstructionsFilename)
            with patch("os.remove") as _rm, patch("numpy.savetxt") as _s, patch(
                "logging.Logger.debug"
            ) as _d, patch("logging.Logger.warning") as _w:
                rp.allHaveFinished()

                numfailedruns = len(rp.failedRuns)
                numtotalruns = len(rp.getGridPointsList())
                _w.assert_called_once_with(
                    PGText.failedRunsMessage
                    % (
                        numfailedruns,
                        numtotalruns,
                        failedInstructionsFilename,
                    )
                )
                self.assertEqual(_rm.call_count, 0)
            self.assertTrue(os.path.exists(failedInstructionsFilename))
            with open(failedInstructionsFilename) as _f:
                content = _f.read()
            self.assertEqual(
                content,
                PGText.failedRunsInstructions.format(
                    numfailed=numfailedruns,
                    numtotal=numtotalruns,
                    cardslist="\n".join(
                        [
                            rp.commonParams["inputcard_filename"] % ix
                            for ix in rp.failedRuns
                        ]
                    ),
                    executablename=Configuration.fortranExecutableName,
                    runcommandslist="\n".join(
                        [
                            shellCommand
                            % (
                                Configuration.fortranExecutableName,
                                (rp.commonParams["inputcard_filename"] % ix).replace(
                                    "card", "in"
                                ),
                                rp.commonParams["output_file_log"] % ix,
                            )
                            for ix in rp.failedRuns
                        ]
                    ),
                    parthenopefile=rp.commonParams["output_file_grid"],
                ),
            )

        def test_getGridPointsList(self):
            """test getGridPointsList"""
            cp = {"output_folder": testConfig.testRunDirDefaultSample}
            gp = np.asarray(
                [
                    np.mgrid[-2:2:5j, 1:1:1j, 1:1:1j, 1:1:1j, 1:1:1j, 1:1:1j]
                    .reshape(6, 5)
                    .T
                ]
            )
            rp = NoGUIRun()
            rp.defineParams(cp, gp)
            self.assertEqualArray(rp.getGridPointsList(), gp[0])
            self.assertEqual(len(rp.getGridPointsList()), 5)
            g1 = (
                np.mgrid[-2:2:5j, 1:1:1j, 1:1:1j, 1:1:1j, 1:1:1j, 1:1:1j]
                .reshape(6, 5)
                .T
            )
            g2 = (
                np.mgrid[1:1:1j, 1:1:1j, 1:1:1j, 1:1:1j, 1:1:1j, -2:2:5j]
                .reshape(6, 5)
                .T
            )
            rp = NoGUIRun()
            rp.defineParams(cp, np.asarray([g1, g2]))
            self.assertEqualArray(
                rp.getGridPointsList(), np.concatenate(np.asarray([g1, g2]))
            )
            self.assertEqual(len(rp.getGridPointsList()), 10)

        def test_getNuclidesHeader(self):
            """test getNuclidesHeader"""
            rp = NoGUIRun()
            rp.defineParams(
                testConfig.commonParamsGrid,
                testConfig.gridPointsGrid,
            )
            fn = rp.commonParams["output_file_nuclides"] % 20
            if os.path.exists(fn):
                os.remove(fn)
            self.assertEqual(rp.nuclidesHeader, [])
            with patch("logging.Logger.debug") as _d:
                rp.getNuclidesHeader(fn)
                _d.assert_called_once_with(PGText.fileNotFound % fn)
            self.assertEqual(rp.nuclidesHeader, [])
            with open(fn, "w") as _f:
                _f.write("")
            with patch("logging.Logger.debug") as _d:
                rp.getNuclidesHeader(fn)
                _d.assert_called_once_with(PGText.lineNotFound % (0, fn), exc_info=True)
            self.assertEqual(rp.nuclidesHeader, [])
            with open(fn, "w") as _f:
                _f.write("#")
            self.assertEqual(rp.nuclidesHeader, [])
            with patch("logging.Logger.debug") as _d:
                rp.getNuclidesHeader(rp.commonParams["output_file_nuclides"] % 0)
                self.assertEqual(_d.call_count, 0)
            self.assertEqual(
                rp.nuclidesHeader,
                [
                    "T(MeV)",
                    "phi_e",
                    "thetah",
                    "tnuh",
                    "nbh",
                    "N",
                    "P",
                    "H2",
                    "H3",
                    "He3",
                    "He4",
                    "Li6",
                    "Li7",
                    "Be7",
                ],
            )

        def test_getParthenopeHeader(self):
            """test getParthenopeHeader"""
            rp = NoGUIRun()
            rp.defineParams(
                testConfig.commonParamsGrid,
                testConfig.gridPointsGrid,
            )
            fn = rp.commonParams["output_file_parthenope"] % 20
            if os.path.exists(fn):
                os.remove(fn)
            self.assertEqual(rp.parthenopeHeader, [])
            with patch("logging.Logger.debug") as _d:
                rp.getParthenopeHeader(fn)
                _d.assert_called_once_with(PGText.fileNotFound % fn)

            self.assertEqual(rp.parthenopeHeader, ["undefined"] * 6)
            with open(fn, "w") as _f:
                _f.write("")
            rp.failedRuns = []
            with patch("logging.Logger.debug") as _d:
                rp.getParthenopeHeader(fn)
                _d.assert_called_once_with(PGText.lineNotFound % (0, fn), exc_info=True)
            self.assertEqual(rp.parthenopeHeader, ["undefined"] * 6)
            rp.parthenopeHeader = ["a", "b", "c", "d"]
            with open(fn, "w") as _f:
                _f.write("#")
            with patch("logging.Logger.debug") as _d:
                rp.getParthenopeHeader(fn)
                self.assertEqual(_d.call_count, 0)
            self.assertEqual(rp.parthenopeHeader, ["undefined"] * 6)
            with open(fn, "w") as _f:
                _f.write("# a b c d ")
            with patch("logging.Logger.debug") as _d:
                rp.getParthenopeHeader(fn)
                self.assertEqual(_d.call_count, 0)
            self.assertEqual(
                rp.parthenopeHeader, ["a", "b", "c", "d", "undefined", "undefined"]
            )
            rp.parthenopeHeader = []
            with patch("logging.Logger.debug") as _d:
                rp.getParthenopeHeader(rp.commonParams["output_file_parthenope"] % 0)
                self.assertEqual(_d.call_count, 0)
            self.assertEqual(
                rp.parthenopeHeader,
                [
                    "N_eff",
                    "xie",
                    "xix",
                    "tau",
                    "rholmbd",
                    "eta10",
                    "OmegaBh^2",
                    "phie",
                    "thetah",
                    "tnuh",
                    "nbh",
                    "N/H",
                    "Y_H",
                    "H2/H",
                    "H3/H",
                    "He3/H",
                    "Y_p",
                    "Li6/H",
                    "Li7/H",
                    "Be7/H",
                ],
            )

        def test_makeFortranExecutable(self):
            """test makeFortranExecutable"""
            ngr = NoGUIRun()
            fn = pgb.PGConfiguration.fortranExecutableName
            with patch(
                "os.path.exists", side_effect=[False, True, True, False, False]
            ) as _e, patch("os.system") as _s, patch("logging.Logger.critical") as _c:
                ngr.makeFortranExecutable()
                _e.assert_has_calls([call(fn), call(fn)])
                _s.assert_called_once_with("make")
                _c.assert_not_called()
                ngr.makeFortranExecutable()
                _s.assert_called_once_with("make")
                _c.assert_not_called()
                with self.assertRaises(SystemExit):
                    ngr.makeFortranExecutable()
                self.assertEqual(_s.call_count, 2)
                _c.assert_called_once_with(PGText.runnerParthenopeNotFound)

        def test_oneHasFinished(self):
            """test oneHasFinished"""
            ic = os.path.join(testConfig.commonParamsGrid["inputcard_filename"] % 0)
            rp = NoGUIRun()
            rp.defineParams(
                testConfig.commonParamsGrid,
                testConfig.gridPointsGrid,
            )
            self.assertEqual(rp.finishedRuns, [])
            self.assertEqual(rp.failedRuns, [])
            rp.oneHasFinished(-1)
            self.assertEqual(rp.finishedRuns, [])
            self.assertEqual(rp.failedRuns, [])
            with patch("logging.Logger.warning") as _w, patch(
                "logging.Logger.debug"
            ) as _d, patch(
                "parthenopegui.runUtils.NoGUIRun.readNuclides", autospec=True
            ) as _rn, patch(
                "parthenopegui.runUtils.NoGUIRun.readParthenope", autospec=True
            ) as _rp, patch(
                "os.remove"
            ) as _r:
                rp.oneHasFinished(0)
                _rn.assert_called_once_with(rp, 0)
                _rp.assert_called_once_with(rp, 0)
                self.assertEqual(_d.call_count, 0)
                self.assertEqual(_w.call_count, 0)
                _r.assert_has_calls([call(ic.replace("card", "in")), call(ic)])
            self.assertEqual(rp.finishedRuns, [0])
            self.assertEqual(rp.failedRuns, [])

            fn = rp.commonParams["output_file_log"] % 21
            with open(fn, "w") as _f:
                _f.write("a\nb\nc\nd")
            with patch("logging.Logger.warning") as _w, patch(
                "logging.Logger.debug"
            ) as _d, patch(
                "parthenopegui.runUtils.NoGUIRun.readNuclides", autospec=True
            ) as _rn, patch(
                "parthenopegui.runUtils.NoGUIRun.readParthenope", autospec=True
            ) as _rp, patch(
                "os.remove"
            ) as _r:
                rp.oneHasFinished(21)
                _w.assert_called_once_with(PGText.runFailed % (21, fn))
                self.assertEqual(_d.call_count, 0)
                _rn.assert_called_once_with(rp, 21)
                _rp.assert_called_once_with(rp, 21)
                self.assertEqual(_r.call_count, 0)
            self.assertEqual(rp.finishedRuns, [0])
            self.assertEqual(rp.failedRuns, [21])
            fn = rp.commonParams["output_file_log"] % 22
            if os.path.exists(fn):
                os.remove(fn)
            with patch("logging.Logger.warning") as _w, patch(
                "logging.Logger.debug"
            ) as _d, patch(
                "parthenopegui.runUtils.NoGUIRun.readNuclides", autospec=True
            ) as _rn, patch(
                "parthenopegui.runUtils.NoGUIRun.readParthenope", autospec=True
            ) as _rp:
                rp.oneHasFinished(22)
                _d.assert_called_once_with(PGText.fileNotFound % fn)
                _w.assert_called_once_with(PGText.runFailed % (22, fn))
                _rn.assert_called_once_with(rp, 22)
                _rp.assert_called_once_with(rp, 22)
            self.assertEqual(rp.finishedRuns, [0])
            self.assertEqual(rp.failedRuns, [21, 22])

            fn = rp.commonParams["output_file_log"] % 23
            with open(fn, "w") as _f:
                _f.write(
                    "%s\nsome message\n%s\nsome error\n%s"
                    % (
                        PGText.runCompletedFailing,
                        PGText.runCompletedFailing,
                        PGText.runCompletedFailingCheck,
                    )
                )
            with patch("logging.Logger.warning") as _w, patch(
                "logging.Logger.debug"
            ) as _d, patch(
                "parthenopegui.runUtils.NoGUIRun.readNuclides", autospec=True
            ) as _rn, patch(
                "parthenopegui.runUtils.NoGUIRun.readParthenope", autospec=True
            ) as _rp, patch(
                "os.remove"
            ) as _r:
                rp.oneHasFinished(23)
                _w.assert_called_once_with(
                    PGText.runFailed % (23, fn)
                    + PGText.runCompletedErrorMessage
                    + "some error"
                    + "\n"
                    + PGText.runCompletedErrorFile
                    % fn.replace(
                        Configuration.nuclidesFilename, Configuration.infoFilename
                    )
                )
                self.assertEqual(_d.call_count, 0)
                _rn.assert_called_once_with(rp, 23)
                _rp.assert_called_once_with(rp, 23)
                self.assertEqual(_r.call_count, 0)
            self.assertEqual(rp.finishedRuns, [0])
            self.assertEqual(rp.failedRuns, [21, 22, 23])

            fn = rp.commonParams["output_file_log"] % 24
            with open(fn, "w") as _f:
                _f.write(
                    "%s\nsome message\n%s\nsome error"
                    % (PGText.runCompletedFailing, PGText.runCompletedFailing)
                )
            with patch("logging.Logger.warning") as _w, patch(
                "logging.Logger.debug"
            ) as _d, patch(
                "parthenopegui.runUtils.NoGUIRun.readNuclides", autospec=True
            ) as _rn, patch(
                "parthenopegui.runUtils.NoGUIRun.readParthenope", autospec=True
            ) as _rp, patch(
                "os.remove"
            ) as _r:
                rp.oneHasFinished(24)
                _w.assert_called_once_with(PGText.runFailed % (24, fn))
                self.assertEqual(_d.call_count, 0)
                _rn.assert_called_once_with(rp, 24)
                _rp.assert_called_once_with(rp, 24)
                self.assertEqual(_r.call_count, 0)
            self.assertEqual(rp.finishedRuns, [0])
            self.assertEqual(rp.failedRuns, [21, 22, 23, 24])

            fn = rp.commonParams["output_file_log"] % 23
            with open(fn, "w") as _f:
                _f.write(
                    "%s\nsome message\n%s\nsome error\n%s"
                    % (
                        PGText.runCompletedFailing,
                        PGText.runCompletedFailing,
                        PGText.runCompletedFailingManual,
                    )
                )
            with patch("logging.Logger.warning") as _w, patch(
                "logging.Logger.debug"
            ) as _d, patch(
                "parthenopegui.runUtils.NoGUIRun.readNuclides", autospec=True
            ) as _rn, patch(
                "parthenopegui.runUtils.NoGUIRun.readParthenope", autospec=True
            ) as _rp, patch(
                "os.remove"
            ) as _r:
                rp.oneHasFinished(23)
                _w.assert_called_once_with(
                    PGText.runFailed % (23, fn)
                    + PGText.runCompletedErrorMessage
                    + "some error"
                    + "\n"
                    + PGText.runCompletedErrorFile
                    % fn.replace(
                        Configuration.nuclidesFilename, Configuration.infoFilename
                    )
                )
                self.assertEqual(_d.call_count, 0)
                _rn.assert_called_once_with(rp, 23)
                _rp.assert_called_once_with(rp, 23)
                self.assertEqual(_r.call_count, 0)
            self.assertEqual(rp.finishedRuns, [0])
            self.assertEqual(rp.failedRuns, [21, 22, 23, 24])

        def test_prepareArgs(self):
            """test prepareArgs"""
            rp = NoGUIRun()
            rp.defineParams(
                testConfig.commonParamsDefault,
                testConfig.gridPointsDefault,
            )
            params = rp.commonParams.copy()
            ix = 0
            pt = rp.gridPoints[0][0]
            for ip, p in enumerate(Parameters.paramOrder):
                params[p] = pt[ip]
            params["runid"] = ix
            for f in (
                "output_file_parthenope",
                "output_file_nuclides",
                "output_file_info",
                "output_file_run",
                "output_file_log",
                "inputcard_filename",
            ):
                params[f] = params[f] % ix
            with patch(
                "parthenopegui.runUtils.NoGUIRun.createInputCard", autospec=True
            ) as _ci, patch("glob.iglob", return_value=["a", "b"]) as _g, patch(
                "os.remove"
            ) as _r:
                args = rp.prepareArgs()
                path = os.path.join(
                    rp.commonParams["output_folder"],
                    "%s_%s_*"
                    % (Configuration.parthenopeFilename, rp.commonParams["now"]),
                )
                _g.assert_called_once()
                self.assertEqual(_g.call_args_list[0][0][0], path)
                _r.assert_has_calls([call("a"), call("b")])
                _ci.assert_called_once_with(rp, **params)
                self.assertEqual(
                    args, [[0, params["inputcard_filename"], params["output_file_log"]]]
                )

            rp.defineParams(
                testConfig.commonParamsGrid,
                testConfig.gridPointsGrid,
            )
            with patch(
                "parthenopegui.runUtils.NoGUIRun.createInputCard", autospec=True
            ) as _ci, patch("glob.iglob", return_value=["a", "b"]) as _g, patch(
                "os.remove"
            ) as _r:
                args = rp.prepareArgs()
                path = os.path.join(
                    rp.commonParams["output_folder"],
                    "%s_%s_*"
                    % (Configuration.parthenopeFilename, rp.commonParams["now"]),
                )
                _g.assert_called_once()
                self.assertEqual(_g.call_args_list[0][0][0], path)
                _r.assert_has_calls([call("a"), call("b")])
                self.assertEqual(_ci.call_count, len(rp.getGridPointsList()))
                for ix, pt in enumerate(rp.getGridPointsList()):
                    params = rp.commonParams.copy()
                    for ip, p in enumerate(Parameters.paramOrder):
                        params[p] = pt[ip]
                    params["runid"] = ix
                    for f in (
                        "output_file_parthenope",
                        "output_file_nuclides",
                        "output_file_info",
                        "output_file_run",
                        "output_file_log",
                        "inputcard_filename",
                    ):
                        params[f] = params[f] % ix
                    _ci.assert_any_call(rp, **params)
                    self.assertEqual(
                        args[ix],
                        [ix, params["inputcard_filename"], params["output_file_log"]],
                    )

        def test_prepareRunFromPickle(self):
            """test prepareRunFromPickle"""
            rp = NoGUIRun()
            with patch(
                "os.path.isfile", side_effect=[False, True, True, True, True]
            ) as _f, patch("logging.Logger.info") as _i, patch(
                "logging.Logger.error"
            ) as _e, patch(
                "parthenopegui.runUtils.paramsRealPath", return_value="abcdef"
            ) as _p, patch(
                "parthenopegui.runUtils.NoGUIRun.defineParams"
            ) as _d, patch(
                "pickle.load", side_effect=[ValueError, ValueError, 1234, ("A", "B")]
            ) as _l:
                rp.prepareRunFromPickle("abc/def")
                _e.assert_called_once_with(PGText.errorCannotLoadGrid)
                _l.assert_not_called()
                _e.reset_mock()
                rp.prepareRunFromPickle(testConfig.testRunDirDefaultSample)
                _i.assert_called_once_with(
                    PGText.tryToOpenFolder % testConfig.testRunDirDefaultSample
                )
                _l.assert_called_once()
                _e.assert_called_once_with(PGText.errorCannotLoadGrid)
                _d.assert_not_called()
                rp.prepareRunFromPickle(testConfig.testRunDirDefaultSample)
                _d.assert_not_called()
                rp.prepareRunFromPickle(testConfig.testRunDirDefaultSample)
                _d.assert_not_called()
                _e.reset_mock()
                rp.prepareRunFromPickle(testConfig.testRunDirDefaultSample)
                _p.assert_called_once_with("A", testConfig.testRunDirDefaultSample)
                _d.assert_called_once_with("abcdef", "B")
                _e.assert_not_called()

        def test_readAllResults(self):
            """test readAllResults"""
            rp = NoGUIRun()
            rp.defineParams(
                testConfig.commonParamsGrid,
                testConfig.gridPointsGrid,
            )
            self.assertEqual(rp.nuclidesEvolution, {})
            self.assertEqual(rp.parthenopeOutPoints, {})
            self.assertEqual(rp.nuclidesHeader, [])
            self.assertEqual(rp.parthenopeHeader, [])
            rp.readAllResults()
            self.assertEqual(
                rp.nuclidesHeader,
                [
                    "T(MeV)",
                    "phi_e",
                    "thetah",
                    "tnuh",
                    "nbh",
                    "N",
                    "P",
                    "H2",
                    "H3",
                    "He3",
                    "He4",
                    "Li6",
                    "Li7",
                    "Be7",
                ],
            )
            self.assertEqual(
                rp.parthenopeHeader,
                [
                    "N_eff",
                    "xie",
                    "xix",
                    "tau",
                    "rholmbd",
                    "eta10",
                    "OmegaBh^2",
                    "phie",
                    "thetah",
                    "tnuh",
                    "nbh",
                    "N/H",
                    "Y_H",
                    "H2/H",
                    "H3/H",
                    "He3/H",
                    "Y_p",
                    "Li6/H",
                    "Li7/H",
                    "Be7/H",
                ],
            )
            self.assertEqual(
                sorted(list(rp.nuclidesEvolution.keys())), [i for i in range(13)]
            )
            for v in rp.nuclidesEvolution.values():
                self.assertIsInstance(v, np.ndarray)
                self.assertEqual(v.shape[1], len(rp.nuclidesHeader))
            self.assertEqual(
                sorted(list(rp.parthenopeOutPoints.keys())), [i for i in range(13)]
            )
            for v in rp.parthenopeOutPoints.values():
                self.assertIsInstance(v, np.ndarray)
                self.assertEqual(v.shape, (len(rp.parthenopeHeader),))

        def test_readNuclides(self):
            """test readNuclides"""
            rp = NoGUIRun()
            rp.defineParams(
                testConfig.commonParamsGrid,
                testConfig.gridPointsGrid,
            )
            fn = rp.commonParams["output_file_nuclides"] % 20
            if os.path.exists(fn):
                os.remove(fn)
            self.assertEqual(rp.parthenopeOutPoints, {})
            self.assertEqual(rp.parthenopeHeader, [])
            with patch("logging.Logger.debug") as _d:
                rp.readNuclides(20)
                _d.assert_has_calls(
                    [
                        call(PGText.fileNotFound % fn),
                        call(PGText.fileNotFound % fn),
                    ]
                )
            self.assertEqual(rp.parthenopeOutPoints, {})
            self.assertEqual(rp.parthenopeHeader, [])
            with open(fn, "w") as _f:
                _f.write("")
            with patch("logging.Logger.debug") as _d:
                rp.readNuclides(20)
                _d.assert_has_calls(
                    [
                        call(PGText.lineNotFound % (0, fn), exc_info=True),
                        call(PGText.emptyFile % fn),
                    ]
                )
            self.assertEqual(rp.parthenopeOutPoints, {})
            self.assertEqual(rp.parthenopeHeader, [])
            with open(fn, "w") as _f:
                _f.write("#")
            with patch("logging.Logger.debug") as _d:
                rp.readNuclides(20)
                _d.assert_called_once_with(PGText.emptyFile % fn)
            self.assertEqual(rp.parthenopeOutPoints, {})
            self.assertEqual(rp.parthenopeHeader, [])
            with patch("logging.Logger.debug") as _d:
                rp.readNuclides(0)
                self.assertEqual(_d.call_count, 0)
            self.assertEqual(
                rp.nuclidesHeader,
                [
                    "T(MeV)",
                    "phi_e",
                    "thetah",
                    "tnuh",
                    "nbh",
                    "N",
                    "P",
                    "H2",
                    "H3",
                    "He3",
                    "He4",
                    "Li6",
                    "Li7",
                    "Be7",
                ],
            )
            self.assertEqual(sorted(list(rp.nuclidesEvolution.keys())), [0])
            for v in rp.nuclidesEvolution.values():
                self.assertIsInstance(v, np.ndarray)
                self.assertEqual(v.shape[1], len(rp.nuclidesHeader))

        def test_readParthenope(self):
            """test readParthenope"""
            rp = NoGUIRun()
            rp.defineParams(
                testConfig.commonParamsGrid,
                testConfig.gridPointsGrid,
            )
            fn = rp.commonParams["output_file_parthenope"] % 20
            if os.path.exists(fn):
                os.remove(fn)
            self.assertEqual(rp.parthenopeOutPoints, {})
            self.assertEqual(rp.parthenopeHeader, [])
            with patch("logging.Logger.debug") as _d:
                rp.readParthenope(20)
                _d.assert_any_call(PGText.fileNotFound % fn)

            self.assertEqual(list(rp.parthenopeOutPoints.keys()), [20])
            self.assertEqualArray(
                rp.parthenopeOutPoints[20],
                np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
            )
            self.assertEqual(rp.parthenopeHeader, ["undefined"] * 6)
            with open(fn, "w") as _f:
                _f.write("")
            rp.failedRuns = []
            with patch("logging.Logger.debug") as _d:
                rp.readParthenope(20)
                _d.assert_has_calls(
                    [
                        call(PGText.lineNotFound % (0, fn), exc_info=True),
                        call(PGText.emptyFile % fn),
                    ]
                )
            self.assertEqual(rp.failedRuns, [20])
            self.assertEqual(list(rp.parthenopeOutPoints.keys()), [20])
            self.assertEqualArray(
                rp.parthenopeOutPoints[20],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            )
            self.assertEqual(rp.parthenopeHeader, ["undefined"] * 6)
            rp.parthenopeHeader = ["a", "b", "c", "d"]
            with open(fn, "w") as _f:
                _f.write("#")
            with patch("logging.Logger.debug") as _d:
                rp.readParthenope(20)
                _d.assert_any_call(PGText.emptyFile % fn)
            self.assertEqual(rp.failedRuns, [20])
            self.assertEqual(list(rp.parthenopeOutPoints.keys()), [20])
            self.assertEqualArray(
                rp.parthenopeOutPoints[20],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            )
            self.assertEqual(
                rp.parthenopeHeader, ["a", "b", "c", "d", "undefined", "undefined"]
            )
            rp.parthenopeHeader = []
            with patch("logging.Logger.debug") as _d:
                rp.readParthenope(0)
                self.assertEqual(_d.call_count, 0)
            self.assertEqual(rp.failedRuns, [20])
            self.assertEqual(
                rp.parthenopeHeader,
                [
                    "N_eff",
                    "xie",
                    "xix",
                    "tau",
                    "rholmbd",
                    "eta10",
                    "OmegaBh^2",
                    "phie",
                    "thetah",
                    "tnuh",
                    "nbh",
                    "N/H",
                    "Y_H",
                    "H2/H",
                    "H3/H",
                    "He3/H",
                    "Y_p",
                    "Li6/H",
                    "Li7/H",
                    "Be7/H",
                ],
            )
            self.assertEqual(sorted(list(rp.parthenopeOutPoints.keys())), [0, 20])
            for v in rp.parthenopeOutPoints.values():
                self.assertIsInstance(v, np.ndarray)
            self.assertEqual(rp.parthenopeOutPoints[0].shape, (20,))
            self.assertEqual(rp.parthenopeOutPoints[20].shape, (6,))
            rp.fixParthenopeOutPoints()
            self.assertEqual(rp.parthenopeOutPoints[20].shape, (20,))

        def test_run(self):
            """test run"""
            rp = NoGUIRun()
            po = Pool(processes=cpu_count())
            a = MagicMock()
            a.wait = MagicMock()
            b = MagicMock()
            b.wait = MagicMock()
            po.apply_async = MagicMock(side_effect=[a, b])
            po.close = MagicMock()
            with patch(
                "parthenopegui.runUtils.NoGUIRun.prepareArgs", return_value=["a", "b"]
            ) as _pa, patch("logging.Logger.info") as _i, patch(
                "parthenopegui.runUtils.Pool", return_value=po
            ) as _po, patch(
                "parthenopegui.runUtils.NoGUIRun.allHaveFinished"
            ) as _a:
                rp.run()
                _pa.assert_called_once_with()
                _i.assert_has_calls(
                    [call(PGText.startRun), call(PGText.runnerFinished)]
                )
                _po.assert_called_once_with(processes=cpu_count())
                po.apply_async.assert_has_calls(
                    [
                        call(runSingle, "a", callback=rp.oneHasFinished),
                        call(runSingle, "b", callback=rp.oneHasFinished),
                    ]
                )
                po.close.assert_called_once_with()
                a.wait.assert_called_once_with()
                b.wait.assert_called_once_with()
                _a.assert_called_once_with()
            po.terminate()

        def test_useGuiLogger(self):
            """test useGuiLogger"""
            ngr = NoGUIRun()
            self.assertEqual(ngr.useGuiLogger(), mainlogger)
            ngr.guilogger = pGUIErrorManager
            self.assertEqual(ngr.useGuiLogger(), pGUIErrorManager)

    class TestThread_poolRunner(PGTestCasewMainW):
        """Test the Thread_poolRunner class"""

        def setUp(self):
            """Create a Pool to be used in the tests"""
            self.pool = Pool(processes=2)

        def tearDown(self):
            """Close the Pool (required in order to avoid stalling"""
            self.pool.close()

        def test_init(self):
            """test init"""
            pool = MagicMock()
            args = MagicMock()
            cbf = MagicMock()
            pa = QWidget()
            tr = Thread_poolRunner(pool, args, cbf, pa)
            self.assertIsInstance(tr, QThread)
            self.assertEqual(tr.pool, pool)
            self.assertEqual(tr.args, args)
            self.assertEqual(tr.callbackFunc, cbf)
            self.assertEqual(tr.parentWidget, pa)

        def test_run(self):
            """test run"""
            rp = RunPArthENoPE(
                testConfig.commonParamsDefault, testConfig.gridPointsDefault, self.mainW
            )
            phf = MagicMock()
            rp.poolHasFinished.connect(phf)
            us = MagicMock()
            rp.updateStatus.connect(us)
            pool = self.pool
            args = [(2, "a", "b"), (3, "c", "d")]
            cbf = MagicMock()
            tr = Thread_poolRunner(pool, args, cbf, rp)
            tr.finished.connect(tr.deleteLater)
            _r = MagicMock()
            _r.wait = MagicMock()
            with patch(
                "multiprocessing.pool.Pool.apply_async", autospec=True, return_value=_r
            ) as _aa, patch("multiprocessing.pool.Pool.close", autospec=True) as _c:
                tr.run()
                _aa.assert_has_calls(
                    [
                        call(pool, runSingle, (2, "a", "b"), callback=cbf),
                        call(pool, runSingle, (3, "c", "d"), callback=cbf),
                    ]
                )
                _c.assert_called_once_with(pool)
                _r.wait.assert_has_calls([call(), call()])
                phf.assert_called_once_with()
                us.assert_called_once_with(PGText.runnerFinished)

    class TestRunPArthENoPE(PGTestCasewMainW):
        """Test the RunPArthENoPE class"""

        def setUp(self):
            """Create a Pool to be used in the tests"""
            self.pool = Pool(processes=2)

        def tearDown(self):
            """Close the Pool(s) (required in order to avoid stalling"""
            self.pool.close()
            try:
                self.poolTerm.terminate()
            except Exception:
                pass

        def test_init(self):
            """test init"""
            cp = {"output_folder": testConfig.testRunDirDefaultSample}
            gp = np.asarray(
                [
                    np.mgrid[0:1:2j, 1:1:1j, 1:1:1j, 1:1:1j, 1:1:1j, 1:1:1j]
                    .reshape(6, 2)
                    .T
                ]
            )
            rp = RunPArthENoPE(cp, gp, parent=self.mainW)
            self.assertIsInstance(rp, NoGUIRun)
            self.assertIsInstance(rp, QObject)
            for s in (
                "poolHasFinished",
                "runHasFinished",
                "updateStatus",
                "updateProgressBar",
            ):
                self.assertIsInstance(getattr(rp, s), Signal)
            self.assertEqual(rp.mainW, self.mainW)
            self.assertEqual(rp.commonParams, cp)
            self.assertEqualArray(rp.gridPoints, gp)
            self.assertEqual(rp.totalRuns, 2)
            self.assertEqual(rp.finishedRuns, [])
            self.assertEqual(rp.failedRuns, [])
            self.assertEqual(rp.nuclidesEvolution, {})
            self.assertEqual(rp.parthenopeOutPoints, {})
            self.assertEqual(rp.nuclidesHeader, [])
            self.assertEqual(rp.parthenopeHeader, [])

            with patch("os.path.exists", return_value=False) as _e, patch(
                "logging.Logger.critical"
            ) as _c, patch("os.system") as _s:
                with self.assertRaises(SystemExit):
                    rp = RunPArthENoPE(cp, gp, parent=self.mainW)
                _e.assert_has_calls(
                    [
                        call(Configuration.fortranExecutableName),
                        call(Configuration.fortranExecutableName),
                    ]
                )
                _s.assert_called_once_with("make")
                _c.assert_called_once_with(PGText.runnerParthenopeNotFound)
            with patch("os.path.exists", side_effect=[False, True, True]) as _e, patch(
                "logging.Logger.critical"
            ) as _c, patch("os.system") as _s, patch("os.makedirs") as _m:
                rp = RunPArthENoPE(cp, gp, parent=self.mainW)
                _e.assert_has_calls(
                    [
                        call(Configuration.fortranExecutableName),
                        call(Configuration.fortranExecutableName),
                    ]
                )
                _s.assert_called_once_with("make")
                self.assertEqual(_c.call_count, 0)
                self.assertEqual(_m.call_count, 0)
            with patch("os.path.exists", side_effect=[True, False]) as _e, patch(
                "os.makedirs"
            ) as _m:
                rp = RunPArthENoPE(
                    {"output_folder": "/non/existent/folder"}, gp, parent=self.mainW
                )
                _e.assert_has_calls(
                    [
                        call(Configuration.fortranExecutableName),
                        call("/non/existent/folder"),
                    ]
                )
                _m.assert_called_once_with("/non/existent/folder")

        def test_oneHasFinished(self):
            """test oneHasFinished"""
            probar = self.mainW.runSettingsTab.runPanel.progressBar
            ic = os.path.join(testConfig.commonParamsGrid["inputcard_filename"] % 0)
            rp = RunPArthENoPE(
                testConfig.commonParamsGrid,
                testConfig.gridPointsGrid,
                parent=self.mainW,
            )
            rp.updateProgressBar.connect(probar.setValue)
            self.assertEqual(rp.finishedRuns, [])
            self.assertEqual(rp.failedRuns, [])
            probar.setValue(12)
            rp.oneHasFinished(-1)
            self.assertEqual(rp.finishedRuns, [])
            self.assertEqual(rp.failedRuns, [])
            self.assertEqual(probar.value(), 12)
            with patch("logging.Logger.warning") as _w, patch(
                "logging.Logger.debug"
            ) as _d, patch(
                "parthenopegui.runUtils.NoGUIRun.readNuclides", autospec=True
            ) as _rn, patch(
                "parthenopegui.runUtils.NoGUIRun.readParthenope", autospec=True
            ) as _rp, patch(
                "os.remove"
            ) as _r:
                rp.oneHasFinished(0)
                _rn.assert_called_once_with(rp, 0)
                _rp.assert_called_once_with(rp, 0)
                self.assertEqual(_d.call_count, 0)
                self.assertEqual(_w.call_count, 0)
                _r.assert_has_calls([call(ic.replace("card", "in")), call(ic)])
            self.assertEqual(rp.finishedRuns, [0])
            self.assertEqual(probar.value(), len(rp.finishedRuns))
            self.assertEqual(rp.failedRuns, [])

            fn = rp.commonParams["output_file_log"] % 21
            with open(fn, "w") as _f:
                _f.write("a\nb\nc\nd")
            with patch("logging.Logger.warning") as _w, patch(
                "logging.Logger.debug"
            ) as _d, patch(
                "parthenopegui.runUtils.NoGUIRun.readNuclides", autospec=True
            ) as _rn, patch(
                "parthenopegui.runUtils.NoGUIRun.readParthenope", autospec=True
            ) as _rp, patch(
                "os.remove"
            ) as _r:
                rp.oneHasFinished(21)
                _w.assert_called_once_with(PGText.runFailed % (21, fn))
                self.assertEqual(_d.call_count, 0)
                _rn.assert_called_once_with(rp, 21)
                _rp.assert_called_once_with(rp, 21)
                self.assertEqual(_r.call_count, 0)
            self.assertEqual(rp.finishedRuns, [0])
            self.assertEqual(probar.value(), len(rp.finishedRuns))
            self.assertEqual(rp.failedRuns, [21])
            fn = rp.commonParams["output_file_log"] % 22
            if os.path.exists(fn):
                os.remove(fn)
            with patch("logging.Logger.warning") as _w, patch(
                "logging.Logger.debug"
            ) as _d, patch(
                "parthenopegui.runUtils.NoGUIRun.readNuclides", autospec=True
            ) as _rn, patch(
                "parthenopegui.runUtils.NoGUIRun.readParthenope", autospec=True
            ) as _rp:
                rp.oneHasFinished(22)
                _d.assert_called_once_with(PGText.fileNotFound % fn)
                _w.assert_called_once_with(PGText.runFailed % (22, fn))
                _rn.assert_called_once_with(rp, 22)
                _rp.assert_called_once_with(rp, 22)
            self.assertEqual(rp.finishedRuns, [0])
            self.assertEqual(probar.value(), len(rp.finishedRuns))
            self.assertEqual(rp.failedRuns, [21, 22])

            fn = rp.commonParams["output_file_log"] % 23
            with open(fn, "w") as _f:
                _f.write(
                    "%s\nsome message\n%s\nsome error\n%s"
                    % (
                        PGText.runCompletedFailing,
                        PGText.runCompletedFailing,
                        PGText.runCompletedFailingCheck,
                    )
                )
            with patch("logging.Logger.warning") as _w, patch(
                "logging.Logger.debug"
            ) as _d, patch(
                "parthenopegui.runUtils.NoGUIRun.readNuclides", autospec=True
            ) as _rn, patch(
                "parthenopegui.runUtils.NoGUIRun.readParthenope", autospec=True
            ) as _rp, patch(
                "os.remove"
            ) as _r:
                rp.oneHasFinished(23)
                _w.assert_called_once_with(
                    PGText.runFailed % (23, fn)
                    + PGText.runCompletedErrorMessage
                    + "some error"
                    + "\n"
                    + PGText.runCompletedErrorFile
                    % fn.replace(
                        Configuration.nuclidesFilename, Configuration.infoFilename
                    )
                )
                self.assertEqual(_d.call_count, 0)
                _rn.assert_called_once_with(rp, 23)
                _rp.assert_called_once_with(rp, 23)
                self.assertEqual(_r.call_count, 0)
            self.assertEqual(rp.finishedRuns, [0])
            self.assertEqual(probar.value(), len(rp.finishedRuns))
            self.assertEqual(rp.failedRuns, [21, 22, 23])

            fn = rp.commonParams["output_file_log"] % 24
            with open(fn, "w") as _f:
                _f.write(
                    "%s\nsome message\n%s\nsome error"
                    % (PGText.runCompletedFailing, PGText.runCompletedFailing)
                )
            with patch("logging.Logger.warning") as _w, patch(
                "logging.Logger.debug"
            ) as _d, patch(
                "parthenopegui.runUtils.NoGUIRun.readNuclides", autospec=True
            ) as _rn, patch(
                "parthenopegui.runUtils.NoGUIRun.readParthenope", autospec=True
            ) as _rp, patch(
                "os.remove"
            ) as _r:
                rp.oneHasFinished(24)
                _w.assert_called_once_with(PGText.runFailed % (24, fn))
                self.assertEqual(_d.call_count, 0)
                _rn.assert_called_once_with(rp, 24)
                _rp.assert_called_once_with(rp, 24)
                self.assertEqual(_r.call_count, 0)
            self.assertEqual(rp.finishedRuns, [0])
            self.assertEqual(probar.value(), len(rp.finishedRuns))
            self.assertEqual(rp.failedRuns, [21, 22, 23, 24])

            fn = rp.commonParams["output_file_log"] % 23
            with open(fn, "w") as _f:
                _f.write(
                    "%s\nsome message\n%s\nsome error\n%s"
                    % (
                        PGText.runCompletedFailing,
                        PGText.runCompletedFailing,
                        PGText.runCompletedFailingManual,
                    )
                )
            with patch("logging.Logger.warning") as _w, patch(
                "logging.Logger.debug"
            ) as _d, patch(
                "parthenopegui.runUtils.NoGUIRun.readNuclides", autospec=True
            ) as _rn, patch(
                "parthenopegui.runUtils.NoGUIRun.readParthenope", autospec=True
            ) as _rp, patch(
                "os.remove"
            ) as _r:
                rp.oneHasFinished(23)
                _w.assert_called_once_with(
                    PGText.runFailed % (23, fn)
                    + PGText.runCompletedErrorMessage
                    + "some error"
                    + "\n"
                    + PGText.runCompletedErrorFile
                    % fn.replace(
                        Configuration.nuclidesFilename, Configuration.infoFilename
                    )
                )
                self.assertEqual(_d.call_count, 0)
                _rn.assert_called_once_with(rp, 23)
                _rp.assert_called_once_with(rp, 23)
                self.assertEqual(_r.call_count, 0)
            self.assertEqual(rp.finishedRuns, [0])
            self.assertEqual(probar.value(), len(rp.finishedRuns))
            self.assertEqual(rp.failedRuns, [21, 22, 23, 24])

        def test_run(self):
            """test run"""
            rp = RunPArthENoPE(
                testConfig.commonParamsDefault,
                testConfig.gridPointsDefault,
                parent=self.mainW,
            )
            params = rp.commonParams.copy()
            ix = 0
            pt = rp.gridPoints[0][0]
            for ip, p in enumerate(Parameters.paramOrder):
                params[p] = pt[ip]
            params["runid"] = ix
            for f in (
                "output_file_parthenope",
                "output_file_nuclides",
                "output_file_info",
                "output_file_run",
                "output_file_log",
                "inputcard_filename",
            ):
                params[f] = params[f] % ix
            us = MagicMock()
            rp.updateStatus.connect(us)
            pool = self.pool
            rf = MagicMock()
            thr = Thread_poolRunner(pool, [], rf, self.mainW)
            thr.start = MagicMock()
            thr.deleteLater = MagicMock()
            with patch(
                "parthenopegui.runner.RunPArthENoPE.createInputCard", autospec=True
            ) as _ci, patch(
                "parthenopegui.runner.Pool", return_value=pool
            ) as _p, patch(
                "glob.iglob", return_value=["a", "b"]
            ) as _g, patch(
                "os.remove"
            ) as _r, patch(
                "parthenopegui.runner.Thread_poolRunner",
                return_value=thr,
                autospec=USE_AUTOSPEC_CLASS,
            ) as _t:
                rp.run()
                path = os.path.join(
                    rp.commonParams["output_folder"],
                    "%s_%s_*"
                    % (Configuration.parthenopeFilename, rp.commonParams["now"]),
                )
                _g.assert_called_once()
                self.assertEqual(_g.call_args_list[0][0][0], path)
                _r.assert_has_calls([call("a"), call("b")])
                _ci.assert_called_once_with(rp, **params)
                _p.assert_called_once_with(processes=cpu_count())
                _t.assert_called_once_with(
                    pool,
                    [
                        [
                            ix,
                            rp.commonParams["inputcard_filename"] % ix,
                            rp.commonParams["output_file_log"] % ix,
                        ]
                        for ix, pt in enumerate(rp.getGridPointsList())
                    ],
                    rp.runHasFinished.emit,
                    rp,
                )
            us.assert_called_once_with(PGText.runnerRunning)
            thr.start.assert_called_once_with()
            thr.finished.emit()
            thr.deleteLater.assert_called_once()

            rp = RunPArthENoPE(
                testConfig.commonParamsGrid,
                testConfig.gridPointsGrid,
                parent=self.mainW,
            )
            rp.updateStatus.connect(us)
            with patch(
                "parthenopegui.runner.RunPArthENoPE.createInputCard", autospec=True
            ) as _ci, patch(
                "parthenopegui.runner.Pool", return_value=pool
            ) as _p, patch(
                "glob.iglob", return_value=["a", "b"]
            ) as _g, patch(
                "os.remove"
            ) as _r, patch(
                "parthenopegui.runner.Thread_poolRunner",
                return_value=thr,
                autospec=USE_AUTOSPEC_CLASS,
            ) as _t:
                rp.run()
                path = os.path.join(
                    rp.commonParams["output_folder"],
                    "%s_%s_*"
                    % (Configuration.parthenopeFilename, rp.commonParams["now"]),
                )
                _g.assert_called_once()
                self.assertEqual(_g.call_args_list[0][0][0], path)
                _r.assert_has_calls([call("a"), call("b")])
                self.assertEqual(_ci.call_count, len(rp.getGridPointsList()))
                for ix, pt in enumerate(rp.getGridPointsList()):
                    params = rp.commonParams.copy()
                    for ip, p in enumerate(Parameters.paramOrder):
                        params[p] = pt[ip]
                    params["runid"] = ix
                    for f in (
                        "output_file_parthenope",
                        "output_file_nuclides",
                        "output_file_info",
                        "output_file_run",
                        "output_file_log",
                        "inputcard_filename",
                    ):
                        params[f] = params[f] % ix
                    _ci.assert_any_call(rp, **params)
                _p.assert_called_once_with(processes=cpu_count())
                _t.assert_called_once_with(
                    pool,
                    [
                        [
                            ix,
                            rp.commonParams["inputcard_filename"] % ix,
                            rp.commonParams["output_file_log"] % ix,
                        ]
                        for ix, pt in enumerate(rp.getGridPointsList())
                    ],
                    rp.runHasFinished.emit,
                    rp,
                )

        def test_stop(self):
            """test stop"""
            rp = RunPArthENoPE(
                testConfig.commonParamsDefault,
                testConfig.gridPointsDefault,
                parent=self.mainW,
            )
            rp.updateStatus.connect(
                self.mainW.runSettingsTab.runPanel.statusLabel.setCenteredText
            )
            thr = MagicMock()
            thr.quit = MagicMock()
            with patch(
                "parthenopegui.runner.Thread_poolRunner", return_value=thr
            ) as _t:
                rp.run()
            self.poolTerm = rp.pool
            self.mainW.runSettingsTab.runPanel.statusLabel.setText("abc")
            with patch("parthenopegui.runner.askYesNo", return_value=True) as _a, patch(
                "multiprocessing.pool.Pool.terminate"
            ) as _t, patch("logging.Logger.debug") as _d:
                rp.stop()
                _a.assert_called_once_with(PGText.runnerAskStop)
                _t.assert_called_once_with()
                thr.quit.assert_called_once_with()
                self.assertEqual(
                    self.mainW.runSettingsTab.runPanel.statusLabel.text(),
                    "<center>%s</center>" % PGText.runnerStopped,
                )
                self.assertEqual(_d.call_count, 0)
            self.mainW.runSettingsTab.runPanel.statusLabel.setText("abc")
            thr.quit = MagicMock(side_effect=RuntimeError)
            with patch("parthenopegui.runner.askYesNo", return_value=True) as _a, patch(
                "multiprocessing.pool.Pool.terminate"
            ) as _t, patch("logging.Logger.debug") as _d:
                rp.stop()
                _a.assert_called_once_with(PGText.runnerAskStop)
                _t.assert_called_once_with()
                thr.quit.assert_called_once_with()
                _d.assert_called_once_with("", exc_info=True)
                self.assertEqual(
                    self.mainW.runSettingsTab.runPanel.statusLabel.text(), "abc"
                )

        def test_failingruns(self):
            """Test runs when something is going wrong"""
            shutil.rmtree(testConfig.testRunDirDefaultSample)
            mainW = MainWindow()
            rst = mainW.runSettingsTab
            rst.outputParams.outputFolderName.setText(
                testConfig.testRunDirDefaultSample
            )
            # generate grid where 1 point fails,
            # test that parthenope.out is recreated when missing
            mainW.parameters.gridsList[0] = 1
            for p in Parameters.paramOrderParth:
                if p == "taun":
                    mainW.parameters.all[p].addGrid(
                        0,
                        1,
                        -0.12,
                        -0.12,
                    )
                else:
                    mainW.parameters.all[p].addGrid(
                        0,
                        1,
                        mainW.parameters.all[p].defaultval,
                        mainW.parameters.all[p].defaultval,
                    )
            mainW.parameters.gridsList[1] = 1
            for p in Parameters.paramOrderParth:
                mainW.parameters.all[p].addGrid(
                    1,
                    1,
                    mainW.parameters.all[p].defaultval,
                    mainW.parameters.all[p].defaultval,
                )
            rst.networkParams.smallNet.setChecked(True)
            rst.networkParams.updateReactions()
            rst.outputParams.nuclidesInOutput.setChecked(True)
            today = datetime.datetime.today()
            with patch("parthenopegui.setrun.datetime") as _dt:
                _dt.today.return_value = today
                cp, gp = rst.runPanel.startRunCustom()
            time.sleep(3)
            with patch("os.remove"), patch("logging.Logger.warning"):
                mainW.runner.oneHasFinished(0)
                mainW.runner.oneHasFinished(1)
                mainW.runner.allHaveFinished()
            self.assertEqual(mainW.runner.failedRuns, [0])
            self.assertEqual(mainW.runner.finishedRuns, [1])
            self.assertTrue(
                os.path.exists(
                    os.path.join(
                        cp["output_folder"],
                        Configuration.failedRunsInstructionsFilename,
                    )
                )
            )
            with open(cp["output_file_grid"]) as _f:
                lines = _f.readlines()
            self.assertIn("tau", lines[0])
            self.assertIn("nan", lines[1])
            self.assertIn("6.13832", lines[2])
            os.remove(cp["output_file_grid"])
            with patch("logging.Logger.warning"):
                mainW.runner.readAllResults()
            self.assertTrue(os.path.exists(cp["output_file_grid"]))
            with open(cp["output_file_grid"]) as _f:
                lines = _f.readlines()
            self.assertIn("tau", lines[0])
            self.assertIn("nan", lines[1])
            self.assertIn("6.13832", lines[2])

            # test that interpolation fixes work when one nan is present
            mainW.parameters.deleteGrid(0)
            mainW.parameters.deleteGrid(1)
            mainW.runner.failedRuns = []
            mainW.runner.finishedRuns = []
            mainW.parameters.gridsList[0] = 4
            for p in Parameters.paramOrderParth:
                if p == "taun":
                    mainW.parameters.all[p].addGrid(
                        0,
                        4,
                        874.0,
                        884.0,
                    )
                else:
                    mainW.parameters.all[p].addGrid(
                        0,
                        1,
                        mainW.parameters.all[p].defaultval,
                        mainW.parameters.all[p].defaultval,
                    )
            with patch("parthenopegui.setrun.datetime") as _dt:
                _dt.today.return_value = today
                cp, gp = rst.runPanel.startRunCustom()
            time.sleep(3)
            with patch("os.remove"), patch("logging.Logger.warning"):
                for i in range(4):
                    mainW.runner.oneHasFinished(i)
                mainW.runner.allHaveFinished()
            self.assertEqual(mainW.runner.failedRuns, [0])
            self.assertEqual(mainW.runner.finishedRuns, [1, 2, 3])
            mainW.plotPanel.specificPlotPanelStack.setCurrentIndex(
                mainW.plotPanel.stackAttrs.index("1Ddependence")
            )
            with patch(
                "parthenopegui.plotter.askDirName",
                return_value=testConfig.testRunDirDefaultSample,
            ) as _adn, patch("logging.Logger.warning"):
                mainW.plotPanel.gridLoader.loadGrid()
            afp = Add1DLineFromPoint(
                mainW,
                0,
                "abc",
                mainW.plotPanel.stackWidgets["1Ddependence"].pointsModel.dataList[
                    0, -2
                ],
                mainW.plotPanel.stackWidgets["1Ddependence"].runner.parthenopeHeader,
                mainW.plotPanel.stackWidgets["1Ddependence"].pointsModel.dataList[
                    0, -1
                ],
            )
            afp.nuclideCombo.setCurrentIndex(5)
            afp.exec_ = MagicMock()
            afp.result = MagicMock(return_value=True)
            idx = QModelIndex()
            idx.isValid = MagicMock(return_value=True)
            idx.row = MagicMock(return_value=0)
            with patch(
                "parthenopegui.plotter.Add1DLineFromPoint", return_value=afp
            ) as _afp, patch("logging.Logger.warning"), patch(
                "matplotlib.pyplot.plot"
            ) as _plot:
                mainW.plotPanel.stackWidgets["1Ddependence"].cellDoubleClickPoints(idx)
                self.assertEqualArray(
                    _plot.call_args_list[1][0][1],
                    [9.3293e-10, 9.3808e-10, 9.4322e-10],
                )

        def test_useGuiLogger(self):
            """test useGuiLogger"""
            cp = {"output_folder": testConfig.testRunDirDefaultSample}
            gp = np.asarray(
                [
                    np.mgrid[0:1:2j, 1:1:1j, 1:1:1j, 1:1:1j, 1:1:1j, 1:1:1j]
                    .reshape(6, 2)
                    .T
                ]
            )
            rp = RunPArthENoPE(cp, gp, parent=self.mainW)
            self.assertIsInstance(rp.useGuiLogger(), PGErrorManagerClassGui)


########## setrun
if PGTestsConfig.test_set:

    class TestEditReaction(PGTestCase):
        """Test the EditReaction class"""

        def test_init(self):
            """test init"""
            p = QWidget()
            er = EditReaction(
                {
                    "rea": Reactions.all[12][0],
                    "type": Reactions.all[12][1],
                    "corr": "high",
                    "val": "1.0",
                },
                p,
            )
            self.assertEqual(er.windowTitle(), PGText.editReactionTitle)
            self.assertIsInstance(er.layout(), QGridLayout)

            self.assertIsInstance(er.layout().itemAtPosition(0, 0).widget(), PGLabel)
            self.assertEqual(
                er.layout().itemAtPosition(0, 0).widget().text(),
                PGText.editReactionFirstLabel,
            )
            self.assertIsInstance(er.layout().itemAtPosition(0, 2).widget(), PGLabel)
            self.assertEqual(er.layout().itemAtPosition(0, 2).widget().text(), "")
            self.assertIsInstance(
                er.layout().itemAtPosition(0, 2).widget().pixmap(), QPixmap
            )
            self.assertIsInstance(er.layout().itemAtPosition(0, 3).widget(), PGLabel)
            self.assertRegex(er.layout().itemAtPosition(0, 3).widget().text(), "[ ]*")
            self.assertIsInstance(er.layout().itemAtPosition(0, 4).widget(), PGLabel)
            self.assertEqual(er.layout().itemAtPosition(0, 4).widget().text(), "")
            self.assertIsInstance(
                er.layout().itemAtPosition(0, 4).widget().pixmap(), QPixmap
            )

            self.assertIsInstance(er.corrType, QComboBox)
            self.assertEqual(er.layout().itemAtPosition(1, 0).widget(), er.corrType)
            for i, f in enumerate(PGText.editReactionCombo):
                self.assertEqual(er.corrType.itemText(i), "%s" % f)
            self.assertEqual(er.corrType.currentText(), "high")
            with patch(
                "parthenopegui.setrun.EditReaction.updateFactor", autospec=True
            ) as _f:
                er.corrType.setCurrentIndex(0)
                _f.assert_called_once_with(er, 0)

            self.assertIsInstance(er.corrValue, QLineEdit)
            self.assertEqual(er.corrValue.text(), "1.0")
            self.assertEqual(er.layout().itemAtPosition(1, 2).widget(), er.corrValue)
            self.assertFalse(er.corrValue.isEnabled())
            er.corrType.setCurrentIndex(3)
            self.assertTrue(er.corrValue.isEnabled())

            self.assertIsInstance(er.acceptButton, PGPushButton)
            self.assertEqual(er.acceptButton.text(), PGText.buttonAccept)
            self.assertEqual(er.acceptButton, er.layout().itemAtPosition(2, 1).widget())
            with patch(
                "parthenopegui.setrun.EditReaction.testAccept", autospec=True
            ) as _f:
                QTest.mouseClick(er.acceptButton, Qt.LeftButton)
                _f.assert_called_once_with(er)

            self.assertIsInstance(er.cancelButton, PGPushButton)
            self.assertEqual(er.cancelButton.text(), PGText.buttonCancel)
            self.assertEqual(er.cancelButton, er.layout().itemAtPosition(2, 2).widget())
            er.reject = MagicMock()
            QTest.mouseClick(er.cancelButton, Qt.LeftButton)
            er.reject.assert_called_once_with()

            er = EditReaction(
                {"rea": "np", "type": "weak", "corr": "unknown", "val": "1.0"}, p
            )
            self.assertEqual(er.layout().itemAtPosition(0, 2).widget().text(), "np")
            self.assertEqual(er.layout().itemAtPosition(0, 2).widget().pixmap(), None)
            self.assertEqual(er.layout().itemAtPosition(0, 4).widget().text(), "weak")
            self.assertEqual(er.layout().itemAtPosition(0, 4).widget().pixmap(), None)
            self.assertEqual(er.corrType.currentText(), "default")

        def test_updateFactor(self):
            """test init"""
            p = QWidget()
            er = EditReaction(
                {
                    "rea": Reactions.all[1][0],
                    "type": Reactions.all[1][1],
                    "corr": "default",
                    "val": "1.0",
                },
                p,
            )
            er.corrType.setCurrentIndex(3)
            self.assertTrue(er.corrValue.isEnabled())
            self.assertEqual(er.corrValue.text(), "1.0")
            er.corrType.setCurrentIndex(0)
            self.assertFalse(er.corrValue.isEnabled())
            self.assertEqual(er.corrValue.text(), "1.0")
            er.corrType.setCurrentIndex(1)
            self.assertFalse(er.corrValue.isEnabled())
            self.assertEqual(er.corrValue.text(), "-" + UnicodeSymbols.l2u["sigma"])
            er.corrType.setCurrentIndex(2)
            self.assertFalse(er.corrValue.isEnabled())
            self.assertEqual(er.corrValue.text(), "+" + UnicodeSymbols.l2u["sigma"])

        def test_testAccept(self):
            """test testAccept"""
            p = QWidget()
            er = EditReaction(
                {
                    "rea": Reactions.all[1][0],
                    "type": Reactions.all[1][1],
                    "corr": "default",
                    "val": "1.0",
                },
                p,
            )
            er.accept = MagicMock()
            with patch("logging.Logger.warning") as _w:
                er.corrValue.setText("abc")
                er.testAccept()
                self.assertEqual(_w.call_count, 0)
                self.assertEqual(er.accept.call_count, 1)
                er.corrType.setCurrentIndex(3)
                er.corrValue.setText("2.0")
                er.testAccept()
                self.assertEqual(_w.call_count, 0)
                self.assertEqual(er.accept.call_count, 2)
                er.corrValue.setText("abc")
                er.testAccept()
                _w.assert_called_once_with(
                    PGText.errorInvalidField % ("value", "reaction rate", "abc")
                )
                self.assertEqual(er.accept.call_count, 2)
                _w.reset_mock()
                er.corrValue.setText("-12")
                er.testAccept()
                _w.assert_called_once_with(
                    PGText.errorInvalidField
                    % ("negative value", "reaction rate", -12.0)
                )
                self.assertEqual(er.accept.call_count, 2)
                _w.reset_mock()
                er.corrValue.setText("123456789")
                er.testAccept()
                _w.assert_called_once_with(
                    PGText.warningLargeValue % ("the reaction rate", 123456789.0)
                )
                self.assertEqual(er.accept.call_count, 2)

    class TestReactionsTableWidget(PGTestCasewMainW):
        """Test the ReactionsTableWidget class"""

        def test_init(self):
            """test init"""
            with patch(
                "parthenopegui.setrun.ReactionsTableWidget.fillReactions", autospec=True
            ) as _f:
                rw = ReactionsTableWidget(self.mainW)
                _f.assert_called_once_with(rw)
            self.assertIsInstance(rw, QTableWidget)
            self.assertEqual(rw.mainW, self.mainW)
            self.assertEqual(rw.columnCount(), 5)
            for i, h in enumerate(PGText.reactionColumnHeaders):
                self.assertEqual(rw.horizontalHeaderItem(i).text(), h)
            self.assertEqual(rw.colKeys, ["rea", "type", "corr", "val"])
            self.assertEqual(rw.selectionBehavior(), QAbstractItemView.SelectRows)
            self.assertEqual(rw.reactionsData, {})
            with patch(
                "parthenopegui.setrun.ReactionsTableWidget.onCellDoubleClick",
                autospec=True,
            ) as _f:
                rw.cellDoubleClicked.emit(0, 0)
                _f.assert_called_once_with(rw, 0, 0)

        def test_fillReactions(self):
            """test fillReactions"""
            with patch(
                "parthenopegui.setrun.ReactionsTableWidget.fillReactions", autospec=True
            ) as _f:
                rw = ReactionsTableWidget(self.mainW)
            self.assertEqual(rw.reactionsData, {})
            rw.fillReactions()
            self.assertEqual(
                rw.reactionsData,
                {
                    re: {"rea": uc[0], "type": uc[1], "corr": "default", "val": "1.0"}
                    for re, uc in self.mainW.reactions.current.items()
                },
            )
            with patch(
                "PySide2.QtWidgets.QTableWidget.resizeColumnsToContents"
            ) as _cr, patch(
                "PySide2.QtWidgets.QTableWidget.resizeRowsToContents"
            ) as _rr:
                rw.fillReactions()
                _cr.assert_called_once_with()
                _rr.assert_called_once_with()
            self.assertEqual(
                rw.reactionsData,
                {
                    re: {"rea": uc[0], "type": uc[1], "corr": "default", "val": "1.0"}
                    for re, uc in self.mainW.reactions.current.items()
                },
            )
            self.assertEqual(rw.rowCount(), len(self.mainW.reactions.current))
            for i, r in enumerate(rw.reactionsData.values()):
                for c, k in enumerate(rw.colKeys):
                    if isinstance(r[k], QImage):
                        self.assertEqual(rw.item(i, c).text(), "")
                        self.assertIsInstance(
                            rw.item(i, c).data(Qt.DecorationRole), QPixmap
                        )
                    else:
                        self.assertEqual(rw.item(i, c).text(), r[k])
                        self.assertEqual(rw.item(i, c).data(Qt.DecorationRole), None)
                    self.assertEqual(
                        rw.item(i, c).flags(), Qt.ItemIsSelectable | Qt.ItemIsEnabled
                    )
                self.assertEqual(
                    rw.item(i, 4).flags(), Qt.ItemIsSelectable | Qt.ItemIsEnabled
                )
                self.assertIsInstance(rw.item(i, 4).data(Qt.DecorationRole), QIcon)
            self.mainW.reactions.updateCurrent(
                [
                    n
                    for n in Reactions.all.keys()
                    if n < Configuration.limitReactions["complNet"]
                ]
            )
            rw.fillReactions()
            self.assertEqual(
                rw.reactionsData,
                {
                    re: {"rea": uc[0], "type": uc[1], "corr": "default", "val": "1.0"}
                    for re, uc in self.mainW.reactions.current.items()
                },
            )
            self.mainW.reactions.updateCurrent(
                [
                    n
                    for n in Reactions.all.keys()
                    if n < Configuration.limitReactions["smallNet"]
                ]
            )

        def test_onCellDoubleClick(self):
            """test onCellDoubleClick"""
            rw = ReactionsTableWidget(self.mainW)
            with patch("logging.Logger.debug") as _d:
                rw.onCellDoubleClick(1000, 1)
                _d.assert_called_once_with(PGText.errorReadTable)
            er = EditReaction(rw.reactionsData[1], self.mainW)
            er.exec_ = MagicMock(side_effect=[False, True])
            er.corrType.setCurrentText("custom factor")
            er.corrValue.setText("1.23")
            with patch(
                "PySide2.QtWidgets.QTableWidget.resizeColumnsToContents"
            ) as _cr, patch(
                "PySide2.QtWidgets.QTableWidget.resizeRowsToContents"
            ) as _rr, patch(
                "parthenopegui.setrun.EditReaction", return_value=er
            ) as _er, patch(
                "parthenopegui.setrun.ReactionsTableWidget.updateRow", autospec=True
            ) as _ur:
                rw.onCellDoubleClick(0, 0)
                self.assertEqual(_er.call_count, 0)
                rw.onCellDoubleClick(2, 0)
                self.assertEqual(_er.call_count, 0)
                rw.onCellDoubleClick(0, 4)
                _er.assert_called_once_with(rw.reactionsData[1], rw)
                er.exec_.assert_called_once_with()
                self.assertEqual(_cr.call_count, 0)
                self.assertEqual(_rr.call_count, 0)
                self.assertEqual(_ur.call_count, 0)
                rw.onCellDoubleClick(0, 4)
                self.assertEqual(_er.call_count, 2)
                self.assertEqual(er.exec_.call_count, 2)
                _cr.assert_called_once_with()
                _rr.assert_called_once_with()
                _ur.assert_called_once_with(rw, 0, "custom factor", "1.23")

        def test_updateRow(self):
            """test updateRow"""
            rw = ReactionsTableWidget(self.mainW)
            self.assertEqual(rw.reactionsData[1]["corr"], "default")
            self.assertEqual(rw.reactionsData[1]["val"], "1.0")
            self.assertEqual(rw.item(0, 2).text(), "default")
            self.assertEqual(rw.item(0, 3).text(), "1.0")
            rw.updateRow(0, "custom factor", "1.23")
            self.assertEqual(rw.reactionsData[1]["corr"], "custom factor")
            self.assertEqual(rw.reactionsData[1]["val"], "1.23")
            self.assertEqual(rw.item(0, 2).text(), "custom factor")
            self.assertEqual(rw.item(0, 3).text(), "1.23")
            rw.updateRow(0, "test", 3.21)
            self.assertEqual(rw.reactionsData[1]["corr"], "test")
            self.assertEqual(rw.reactionsData[1]["val"], "3.21")
            self.assertEqual(rw.item(0, 2).text(), "test")
            self.assertEqual(rw.item(0, 3).text(), "3.21")

        def test_readline(self):
            """test readline"""
            rw = ReactionsTableWidget(self.mainW)
            rw.updateRow(0, "custom factor", "1.23")
            rw.updateRow(1, "high", 3.21)
            rw.updateRow(2, "low", 1.23)
            self.assertEqual(rw.readline(0), [1, 3, "1.23"])
            self.assertEqual(rw.readline(1), [2, 2, "1.0"])
            self.assertEqual(rw.readline(2), [3, 1, "1.0"])
            self.assertEqual(rw.readline(3), [4, 0, "1.0"])

    class TestMWNetworkPanel(PGTestCasewMainW):
        """Test the MWNetworkPanel class"""

        def test_init(self):
            """test init"""
            np = MWNetworkPanel(self.mainW)
            self.assertIsInstance(np, QFrame)
            self.assertEqual(np.mainW, self.mainW)
            self.assertIsInstance(np.layout(), QGridLayout)
            self.assertIsInstance(np.groupBox, QGroupBox)
            self.assertTrue(np.groupBox.isFlat())
            self.assertEqual(np.groupBox, np.layout().itemAt(0).widget())
            self.assertIsInstance(np.groupBox.layout(), QVBoxLayout)
            for i, (a, d) in enumerate(PGText.networkDescription.items()):
                self.assertIsInstance(getattr(np, a), QRadioButton)
                self.assertEqual(getattr(np, a).text(), d)
                if a == "smallNet":
                    self.assertTrue(getattr(np, a).isChecked())
                else:
                    self.assertFalse(getattr(np, a).isChecked())
                self.assertEqual(
                    np.groupBox.layout().itemAt(i).widget(), getattr(np, a)
                )
                with patch(
                    "parthenopegui.setrun.MWNetworkPanel.updateReactions", autospec=True
                ) as _f:
                    getattr(np, a).toggled.emit(True)
                    _f.assert_called_once_with(np)

            self.assertIsInstance(np.layout().itemAt(1).widget(), PGLabel)
            self.assertEqual(
                np.layout().itemAt(1).widget().text(), PGText.networkCustomizeRate
            )
            self.assertIsInstance(np.reactionsTable, ReactionsTableWidget)
            self.assertEqual(np.layout().itemAt(2).widget(), np.reactionsTable)
            with patch(
                "parthenopegui.setrun.ReactionsTableWidget.fillReactions", autospec=True
            ) as _f:
                self.mainW.reactions.updated.emit()
                self.assertGreater(_f.call_count, 0)

        def test_updateReactions(self):
            """test updateReactions"""
            np = MWNetworkPanel(self.mainW)
            self.assertEqual(len(self.mainW.nuclides.current), 9)
            self.assertEqual(len(self.mainW.reactions.current), 40)
            with patch(
                "parthenopegui.configuration.Nuclides.updateCurrent", autospec=True
            ) as _nu, patch(
                "parthenopegui.configuration.Reactions.updateCurrent", autospec=True
            ) as _ru:
                np.interNet.setChecked(True)
                _nu.assert_has_calls(
                    [
                        call(
                            self.mainW.nuclides,
                            [
                                n
                                for n in Nuclides.all.keys()
                                if Nuclides.nuclideOrder[n]
                                < Configuration.limitNuclides["interNet"]
                            ],
                        )
                    ]
                )
                _ru.assert_has_calls(
                    [
                        call(
                            self.mainW.reactions,
                            [
                                n
                                for n in Reactions.all.keys()
                                if n < Configuration.limitReactions["interNet"]
                            ],
                        )
                    ]
                )
            np.updateReactions()
            self.assertEqual(len(self.mainW.nuclides.current), 18)
            self.assertEqual(len(self.mainW.reactions.current), 73)
            with patch(
                "parthenopegui.configuration.Nuclides.updateCurrent", autospec=True
            ) as _nu, patch(
                "parthenopegui.configuration.Reactions.updateCurrent", autospec=True
            ) as _ru:
                np.complNet.setChecked(True)
                _nu.assert_has_calls(
                    [
                        call(
                            self.mainW.nuclides,
                            [
                                n
                                for n in Nuclides.all.keys()
                                if Nuclides.nuclideOrder[n]
                                < Configuration.limitNuclides["complNet"]
                            ],
                        )
                    ]
                )
                _ru.assert_has_calls(
                    [
                        call(
                            self.mainW.reactions,
                            [
                                n
                                for n in Reactions.all.keys()
                                if n < Configuration.limitReactions["complNet"]
                            ],
                        )
                    ]
                )
            np.updateReactions()
            self.assertEqual(len(self.mainW.nuclides.current), 26)
            self.assertEqual(len(self.mainW.reactions.current), 100)
            with patch(
                "parthenopegui.configuration.Nuclides.updateCurrent", autospec=True
            ) as _nu, patch(
                "parthenopegui.configuration.Reactions.updateCurrent", autospec=True
            ) as _ru:
                np.smallNet.setChecked(True)
                _nu.assert_has_calls(
                    [
                        call(
                            self.mainW.nuclides,
                            [
                                n
                                for n in Nuclides.all.keys()
                                if Nuclides.nuclideOrder[n]
                                < Configuration.limitNuclides["smallNet"]
                            ],
                        )
                    ]
                )
                _ru.assert_has_calls(
                    [
                        call(
                            self.mainW.reactions,
                            [
                                n
                                for n in Reactions.all.keys()
                                if n < Configuration.limitReactions["smallNet"]
                            ],
                        )
                    ]
                )
            np.updateReactions()
            self.assertEqual(len(self.mainW.nuclides.current), 9)
            self.assertEqual(len(self.mainW.reactions.current), 40)

    class TestEditParameters(PGTestCasewMainW):
        """Test the EditParameters class"""

        @classmethod
        def setUpClass(self):
            """Store a link to a MWPhysParamsPanel instance"""
            super(TestEditParameters, self).setUpClass()
            self.pp = self.mainW.runSettingsTab.physicsParams

        def test_init(self):
            """test init"""
            self.mainW.parameters.all["DeltaNnu"].currenttype = "grid"
            self.mainW.parameters.all["DeltaNnu"].currentN = 2
            ep = EditParameters(self.mainW, self.pp)
            ep.reject = MagicMock()
            self.assertIsInstance(ep, QDialog)
            self.assertEqual(ep.mainW, self.mainW)
            self.assertEqual(ep.physPanel, self.pp)
            self.assertEqual(ep.windowTitle(), PGText.editParametersTitle)
            self.assertIsInstance(ep.layout(), QGridLayout)
            self.assertIsInstance(ep.inputs, dict)

            self.assertIsInstance(ep.layout().itemAtPosition(0, 0).widget(), PGLabel)
            self.assertEqual(
                ep.layout().itemAtPosition(0, 0).widget().text(),
                PGText.editParametersAdd,
            )
            for i, desc in enumerate(PGText.editParametersColumnLabels):
                self.assertIsInstance(
                    ep.layout().itemAtPosition(1, i + 1).widget(), PGLabel
                )
                self.assertEqual(
                    ep.layout().itemAtPosition(1, i + 1).widget().text(), desc
                )

            for i, p in enumerate(Parameters.paramOrder):
                pa = self.mainW.parameters.all[p]
                self.assertIsInstance(
                    ep.layout().itemAtPosition(i + 2, 0).widget(), PGLabel
                )
                self.assertIsInstance(
                    ep.layout().itemAtPosition(i + 2, 0).widget().pixmap(), QPixmap
                )
                self.assertEqual(
                    ep.layout().itemAtPosition(i + 2, 0).widget().toolTip(),
                    PGText.parameterDescriptions[p],
                )

                self.assertIsInstance(ep.inputs[p]["combo"], QComboBox)
                self.assertEqual(
                    ep.layout().itemAtPosition(i + 2, 1).widget(), ep.inputs[p]["combo"]
                )
                for j, c in enumerate(PGText.editParametersCombo):
                    self.assertEqual(ep.inputs[p]["combo"].itemText(j), c)

                self.assertIsInstance(ep.inputs[p]["def"], QLineEdit)
                self.assertEqual(ep.inputs[p]["def"].text(), "%s" % pa.currentval)
                self.assertEqual(
                    ep.layout().itemAtPosition(i + 2, 2).widget(), ep.inputs[p]["def"]
                )
                self.assertIsInstance(ep.inputs[p]["min"], QLineEdit)
                self.assertEqual(ep.inputs[p]["min"].text(), "%s" % pa.currentmin)
                self.assertEqual(
                    ep.layout().itemAtPosition(i + 2, 3).widget(), ep.inputs[p]["min"]
                )
                self.assertIsInstance(ep.inputs[p]["max"], QLineEdit)
                self.assertEqual(ep.inputs[p]["max"].text(), "%s" % pa.currentmax)
                self.assertEqual(
                    ep.layout().itemAtPosition(i + 2, 4).widget(), ep.inputs[p]["max"]
                )
                self.assertIsInstance(ep.inputs[p]["num"], QLineEdit)
                self.assertEqual(ep.inputs[p]["num"].text(), "%s" % pa.currentN)
                self.assertEqual(
                    ep.layout().itemAtPosition(i + 2, 5).widget(), ep.inputs[p]["num"]
                )

                if p == "DeltaNnu":
                    self.assertEqual(ep.inputs[p]["combo"].currentText(), "grid")
                    self.assertFalse(ep.inputs[p]["def"].isEnabled())
                    self.assertTrue(ep.inputs[p]["min"].isEnabled())
                    self.assertTrue(ep.inputs[p]["max"].isEnabled())
                    self.assertTrue(ep.inputs[p]["num"].isEnabled())
                else:
                    self.assertEqual(
                        ep.inputs[p]["combo"].currentText(), "single point"
                    )
                    self.assertTrue(ep.inputs[p]["def"].isEnabled())
                    self.assertFalse(ep.inputs[p]["min"].isEnabled())
                    self.assertFalse(ep.inputs[p]["max"].isEnabled())
                    self.assertFalse(ep.inputs[p]["num"].isEnabled())
                with patch(
                    "parthenopegui.setrun.EditParameters.updateFactor", autospec=True
                ) as _f:
                    ep.inputs[p]["combo"].currentTextChanged.emit("abc")
                    _f.assert_called_once_with(ep, "abc", p)

            self.assertIsInstance(ep.acceptButton, PGPushButton)
            self.assertEqual(ep.acceptButton.text(), PGText.buttonAccept)
            self.assertEqual(
                ep.acceptButton,
                ep.layout().itemAtPosition(2 + len(ep.inputs), 2).widget(),
            )
            with patch(
                "parthenopegui.setrun.EditParameters.testAccept", autospec=True
            ) as _f:
                QTest.mouseClick(ep.acceptButton, Qt.LeftButton)
                _f.assert_called_once_with(ep)
            self.assertIsInstance(ep.cancelButton, PGPushButton)
            self.assertEqual(ep.cancelButton.text(), PGText.buttonCancel)
            self.assertEqual(
                ep.cancelButton,
                ep.layout().itemAtPosition(2 + len(ep.inputs), 3).widget(),
            )
            QTest.mouseClick(ep.cancelButton, Qt.LeftButton)
            ep.reject.assert_called_once_with()

        @patch("logging.Logger.warning")
        def test_testAccept(self, _w):
            """test testAccept"""
            self.mainW.parameters.all["DeltaNnu"].currenttype = "grid"
            self.mainW.parameters.all["DeltaNnu"].currentN = 2
            ep = EditParameters(self.mainW, self.pp)
            ep.accept = MagicMock()
            ep.inputs["taun"]["def"].setText("abc")
            ep.testAccept()
            _w.assert_any_call(PGText.errorInvalidField % ("value", "taun", "abc"))
            self.assertEqual(ep.accept.call_count, 0)
            ep.inputs["taun"]["def"].setText("1.0")
            ep.inputs["DeltaNnu"]["def"].setText("abc")
            ep.inputs["DeltaNnu"]["num"].setText("abc")
            ep.inputs["DeltaNnu"]["min"].setText("a")
            ep.inputs["DeltaNnu"]["max"].setText("b")
            ep.testAccept()
            _w.assert_any_call(
                PGText.errorInvalidField % ("number of points", "DeltaNnu", "abc")
            )
            self.assertEqual(ep.accept.call_count, 0)
            ep.inputs["DeltaNnu"]["num"].setText("1.2")
            ep.testAccept()
            _w.assert_any_call(
                PGText.errorInvalidField % ("number of points", "DeltaNnu", "1.2")
            )
            self.assertEqual(ep.accept.call_count, 0)
            ep.inputs["DeltaNnu"]["num"].setText("1")
            ep.testAccept()
            _w.assert_any_call(
                PGText.errorInvalidField
                % ("number of points (smaller than 2)", "DeltaNnu", 1)
            )
            self.assertEqual(ep.accept.call_count, 0)
            ep.inputs["DeltaNnu"]["num"].setText("3")
            ep.testAccept()
            _w.assert_any_call(PGText.errorInvalidField % ("minimum", "DeltaNnu", "a"))
            self.assertEqual(ep.accept.call_count, 0)
            ep.inputs["DeltaNnu"]["min"].setText("-3")
            ep.testAccept()
            _w.assert_any_call(PGText.errorInvalidField % ("maximum", "DeltaNnu", "b"))
            self.assertEqual(ep.accept.call_count, 0)
            ep.inputs["DeltaNnu"]["max"].setText("-3")
            ep.testAccept()
            _w.assert_any_call(
                PGText.errorInvalidParamLimits % ("DeltaNnu", -3.0, -3.0)
            )
            self.assertEqual(ep.accept.call_count, 0)
            ep.inputs["DeltaNnu"]["max"].setText("3")
            ep.testAccept()
            ep.accept.assert_called_once_with()

        def test_updateFactor(self):
            """test updateFactor"""
            self.mainW.parameters.all["DeltaNnu"].currenttype = "single point"
            self.mainW.parameters.all["DeltaNnu"].currentN = 1
            ep = EditParameters(self.mainW, self.pp)
            for param in Parameters.paramOrder:
                self.assertTrue(ep.inputs[param]["def"].isEnabled())
                self.assertFalse(ep.inputs[param]["min"].isEnabled())
                self.assertFalse(ep.inputs[param]["max"].isEnabled())
                self.assertFalse(ep.inputs[param]["num"].isEnabled())
                ep.updateFactor("grid", param)
                self.assertFalse(ep.inputs[param]["def"].isEnabled())
                self.assertTrue(ep.inputs[param]["min"].isEnabled())
                self.assertTrue(ep.inputs[param]["max"].isEnabled())
                self.assertTrue(ep.inputs[param]["num"].isEnabled())
                ep.updateFactor("single point", param)
                self.assertTrue(ep.inputs[param]["def"].isEnabled())
                self.assertFalse(ep.inputs[param]["min"].isEnabled())
                self.assertFalse(ep.inputs[param]["max"].isEnabled())
                self.assertFalse(ep.inputs[param]["num"].isEnabled())

        def test_emitRefresh(self):
            """test emitRefresh"""
            ep = EditParameters(self.mainW, self.pp)
            f = MagicMock()
            self.pp.needsRefresh.connect(f)
            with patch("logging.Logger.debug") as _d:
                ep.emitRefresh()
                self.assertEqual(_d.call_count, 0)
            f.assert_called_once_with()
            p = QWidget()
            ep.physPanel = p
            with patch("logging.Logger.debug") as _d:
                ep.emitRefresh()
                _d.assert_called_once_with("", exc_info=True)
            f.assert_called_once_with()

        @patch("parthenopegui.configuration.Parameter.addGrid", autospec=True)
        @patch("logging.Logger.warning")
        def test_processOutput(self, _w, _ag):
            """test processOutput"""
            self.mainW.parameters.gridsList = {2: 1}
            self.mainW.parameters.all["DeltaNnu"].defaultnum = 4
            self.mainW.parameters.all["DeltaNnu"].defaultmin = -2.4
            self.mainW.parameters.all["DeltaNnu"].defaultmax = 2.4
            ep = EditParameters(self.mainW, self.pp)
            self.mainW.parameters.all["DeltaNnu"].currenttype = "single point"
            ep.emitRefresh = MagicMock()
            ep.inputs["DeltaNnu"]["combo"].setCurrentText("grid")
            ep.inputs["DeltaNnu"]["def"].setText("abc")
            ep.inputs["DeltaNnu"]["num"].setText("abc")
            ep.inputs["DeltaNnu"]["min"].setText("a")
            ep.inputs["DeltaNnu"]["max"].setText("b")
            ep.inputs["taun"]["combo"].setCurrentText("grid")
            ep.inputs["taun"]["num"].setText("5")
            ep.inputs["taun"]["min"].setText("879")
            ep.inputs["taun"]["max"].setText("881.1")
            ep.inputs["csinue"]["def"].setText("0.1")
            ep.inputs["csinux"]["def"].setText("0.2")
            ep.inputs["rhoLambda"]["def"].setText("abc")

            ep.processOutput()
            self.assertEqual(self.mainW.parameters.gridsList, {2: 1, 3: 20})
            self.assertEqual(ep.emitRefresh.call_count, 1)
            self.assertEqual(_ag.call_count, 6)
            _ag.assert_any_call(
                self.mainW.parameters.all["eta10"], 3, 1, 6.13832, 6.13832
            )
            _ag.assert_any_call(self.mainW.parameters.all["DeltaNnu"], 3, 4, -2.4, 2.4)
            _ag.assert_any_call(self.mainW.parameters.all["taun"], 3, 5, 879.0, 881.1)
            _ag.assert_any_call(self.mainW.parameters.all["csinue"], 3, 1, 0.1, 0.1)
            _ag.assert_any_call(self.mainW.parameters.all["csinux"], 3, 1, 0.2, 0.2)
            _ag.assert_any_call(self.mainW.parameters.all["rhoLambda"], 3, 1, 0.0, 0.0)
            self.assertEqual(_w.call_count, 4)
            _w.assert_any_call(PGText.errorInvalidField % ("value", "rhoLambda", "abc"))
            _w.assert_any_call(PGText.errorInvalidField % ("minimum", "DeltaNnu", "a"))
            _w.assert_any_call(PGText.errorInvalidField % ("maximum", "DeltaNnu", "b"))
            _w.assert_any_call(
                PGText.errorInvalidField % ("number of points", "DeltaNnu", "abc")
            )

            ep.inputs["DeltaNnu"]["num"].setText("3")
            ep.inputs["DeltaNnu"]["min"].setText("-3")
            ep.inputs["DeltaNnu"]["max"].setText("3")
            ep.inputs["rhoLambda"]["def"].setText("0.3")
            ep.processOutput()
            self.assertEqual(self.mainW.parameters.gridsList, {2: 1, 3: 20, 4: 15})
            self.assertEqual(ep.emitRefresh.call_count, 2)
            self.assertEqual(_ag.call_count, 12)
            _ag.assert_any_call(self.mainW.parameters.all["DeltaNnu"], 4, 3, -3, 3)
            _ag.assert_any_call(self.mainW.parameters.all["rhoLambda"], 4, 1, 0.3, 0.3)
            _ag.assert_any_call(
                self.mainW.parameters.all["eta10"], 4, 1, 6.13832, 6.13832
            )
            _ag.assert_any_call(self.mainW.parameters.all["taun"], 4, 5, 879.0, 881.1)
            _ag.assert_any_call(self.mainW.parameters.all["csinue"], 4, 1, 0.1, 0.1)
            _ag.assert_any_call(self.mainW.parameters.all["csinux"], 4, 1, 0.2, 0.2)
            self.assertEqual(_w.call_count, 4)

            self.mainW.parameters.gridsList = {}
            with patch("logging.Logger.debug") as _d:
                ep.processOutput()
                _d.assert_called_once_with(PGText.warningMaxEmptyList)
            self.assertEqual(self.mainW.parameters.gridsList, {0: 15})

            self.mainW.parameters.gridsList = []
            with patch("logging.Logger.debug") as _d:
                ep.processOutput("a")
                _d.assert_any_call(
                    PGText.warningWrongType % ("Parameters.gridsList", "dict")
                )
                _d.assert_any_call(
                    PGText.errorInvalidFieldSet % ("value", "newid", "a", 0)
                )
            self.assertEqual(self.mainW.parameters.gridsList, {0: 15})

            self.mainW.parameters.gridsList = "abcd"
            with patch("logging.Logger.debug") as _d:
                ep.processOutput(9)
                _d.assert_called_once_with(
                    PGText.warningWrongType % ("Parameters.gridsList", "dict")
                )
            self.assertEqual(self.mainW.parameters.gridsList, {9: 15})

    class TestResumeGrid(PGTestCasewMainW):
        """Test the ResumeGrid class"""

        @classmethod
        def setUpClass(self):
            """fill the parameter grids with some examples to test"""
            super(TestResumeGrid, self).setUpClass()
            self.mainW.parameters.gridsList[0] = 9
            for p in Parameters.paramOrderParth[0:3]:
                self.mainW.parameters.all[p].addGrid(
                    0,
                    2,
                    self.mainW.parameters.all[p].defaultmin,
                    self.mainW.parameters.all[p].defaultmax,
                )
            for p in Parameters.paramOrderParth[3:]:
                self.mainW.parameters.all[p].addGrid(
                    0,
                    1,
                    self.mainW.parameters.all[p].defaultval,
                    self.mainW.parameters.all[p].defaultval,
                )
            self.mainW.parameters.gridsList[1] = 1
            for p in Parameters.paramOrderParth:
                self.mainW.parameters.all[p].addGrid(
                    1,
                    1,
                    self.mainW.parameters.all[p].defaultval,
                    self.mainW.parameters.all[p].defaultval,
                )
            self.pp = self.mainW.runSettingsTab.physicsParams

        def test_init(self):
            """test init"""
            with patch("PySide2.QtWidgets.QTableWidget.resizeColumnsToContents") as _rs:
                rg = ResumeGrid(0, self.mainW, self.pp)
                _rs.assert_called_once_with()
            self.assertIsInstance(rg, QFrame)
            self.assertEqual(rg.idGrid, 0)
            self.assertEqual(rg.mainW, self.mainW)
            self.assertEqual(rg.physPanel, self.pp)

            self.assertIsInstance(rg.layout(), QGridLayout)

            self.assertIsInstance(rg.editGrid, PGPushButton)
            self.assertEqual(rg.editGrid.text(), PGText.buttonEdit)
            self.assertEqual(rg.editGrid, rg.layout().itemAtPosition(1, 1).widget())
            with patch("parthenopegui.setrun.ResumeGrid.onEdit", autospec=True) as _f:
                QTest.mouseClick(rg.editGrid, Qt.LeftButton)
                _f.assert_called_once_with(rg)

            self.assertIsInstance(rg.deleteGrid, PGPushButton)
            self.assertEqual(rg.deleteGrid.text(), PGText.buttonDelete)
            self.assertEqual(rg.deleteGrid, rg.layout().itemAtPosition(2, 1).widget())
            with patch("parthenopegui.setrun.ResumeGrid.onDelete", autospec=True) as _f:
                QTest.mouseClick(rg.deleteGrid, Qt.LeftButton)
                _f.assert_called_once_with(rg)

            self.assertIsInstance(rg.table, QTableWidget)
            self.assertEqual(rg.table, rg.layout().itemAtPosition(0, 0).widget())
            self.assertEqual(rg.table.columnCount(), 3)
            for i, h in enumerate(PGText.resumeGridHeaders):
                self.assertEqual(rg.table.horizontalHeaderItem(i).text(), h)
            self.assertEqual(rg.minimumHeight(), int(rg.table.rowHeight(0) * 7.5))
            self.assertEqual(rg.table.selectionBehavior(), QAbstractItemView.SelectRows)
            self.assertEqual(rg.table.rowCount(), len(Parameters.paramOrder))
            for i, p in enumerate(Parameters.paramOrder):
                pa = self.mainW.parameters.all[p]
                npts, grid = pa.getGridPoints(0)
                self.assertEqual(rg.table.item(i, 0).text(), "")
                self.assertEqual(
                    rg.table.item(i, 0).flags(), Qt.ItemIsSelectable | Qt.ItemIsEnabled
                )
                self.assertEqual(rg.table.item(i, 1).text(), "%s" % npts)
                self.assertEqual(
                    rg.table.item(i, 1).flags(), Qt.ItemIsSelectable | Qt.ItemIsEnabled
                )
                self.assertEqual(rg.table.item(i, 2).text(), "%s" % grid)
                self.assertEqual(
                    rg.table.item(i, 2).flags(), Qt.ItemIsSelectable | Qt.ItemIsEnabled
                )

            qi = QIcon()
            it = QTableWidgetItem("")
            with patch(
                "PySide2.QtWidgets.QTableWidget.resizeColumnsToContents"
            ) as _rs, patch(
                "parthenopegui.setrun.QTableWidgetItem", return_value=it
            ) as _wi, patch(
                "parthenopegui.setrun.QIcon", return_value=qi
            ) as _qi:
                rg = ResumeGrid(0, self.mainW, self.pp)
                _rs.assert_called_once_with()
                for p in Parameters.paramOrder:
                    _qi.assert_any_call(self.mainW.parameters.all[p].fig)
                _wi.assert_any_call(qi, "")

        def test_onEdit(self):
            """test onEdit"""
            rg = ResumeGrid(1, self.mainW, self.pp)
            ep = EditParameters(self.mainW, self.pp)
            ep.exec_ = MagicMock(side_effect=[False, True])
            ep.processOutput = MagicMock()
            with patch(
                "parthenopegui.setrun.EditParameters", return_value=ep
            ) as _ep, patch(
                "parthenopegui.configuration.Parameters.getGrid", autospec=True
            ) as _gg:
                rg.onEdit()
                _gg.assert_called_once_with(self.mainW.parameters, 1)
                _ep.assert_called_once_with(self.mainW, self.pp)
                ep.exec_.assert_called_once_with()
                self.assertEqual(ep.processOutput.call_count, 0)
                rg.onEdit()
                self.assertEqual(ep.exec_.call_count, 2)
                ep.processOutput.assert_called_once_with(newid=1)

        def test_onDelete(self):
            """test onDelete"""
            rg = ResumeGrid(0, self.mainW, self.pp)
            nc = MagicMock()
            self.pp.needsRefresh.connect(nc)
            with patch(
                "parthenopegui.setrun.askYesNo",
                autospec=True,
                side_effect=[False, True, True],
            ) as _a, patch(
                "parthenopegui.configuration.Parameters.deleteGrid",
                autospec=True,
                side_effect=[False, True],
            ) as _dg, patch(
                "logging.Logger.warning"
            ) as _w:
                rg.onDelete()
                _a.assert_called_once_with(PGText.resumeGridAskDelete)
                self.assertEqual(_dg.call_count, 0)
                rg.onDelete()
                self.assertEqual(_a.call_count, 2)
                _dg.assert_called_once_with(self.mainW.parameters, 0)
                self.assertEqual(nc.call_count, 0)
                _w.assert_called_once_with(PGText.errorLoadingGrid)
                rg.onDelete()
                self.assertEqual(_a.call_count, 3)
                self.assertEqual(_dg.call_count, 2)
                nc.assert_called_once_with()
                _w.assert_called_once_with(PGText.errorLoadingGrid)

    class TestMWPhysParamsPanel(PGTestCasewMainW):
        """Test the MWPhysParamsPanel class"""

        def test_init(self):
            """test init"""
            with patch("logging.Logger.error") as _e, patch(
                "parthenopegui.setrun.MWPhysParamsPanel.refreshContent", autospec=True
            ) as _r:
                ppp = MWPhysParamsPanel(self.mainW)
                self.assertEqual(_e.call_count, 0)
                _r.assert_called_once_with(ppp)
            self.assertIsInstance(ppp, QScrollArea)
            self.assertIsInstance(ppp.needsRefresh, Signal)
            self.assertIsInstance(ppp.inwidget, QWidget)
            self.assertIsInstance(ppp.inwidget.layout(), QVBoxLayout)
            with patch(
                "parthenopegui.setrun.MWPhysParamsPanel.refreshContent", autospec=True
            ) as _r:
                ppp.needsRefresh.emit()
                _r.assert_called_once_with(ppp)

            p = QWidget(self.mainW)
            with patch("logging.Logger.error") as _e, patch(
                "parthenopegui.setrun.MWPhysParamsPanel.refreshContent", autospec=True
            ) as _r, self.assertRaises(AttributeError):
                ppp = MWPhysParamsPanel(p)
                _e.assert_called_once_with(PGText.parentAttributeMissing)
                self.assertEqual(_r.call_count, 0)

        def test_cleanLayout(self):
            """test cleanLayout"""
            ppp = MWPhysParamsPanel(self.mainW)
            p1 = QWidget(ppp)
            p2 = QWidget(ppp)
            ppp.inwidget.layout().addWidget(p1)
            ppp.inwidget.layout().addWidget(p2)
            self.assertGreater(ppp.inwidget.layout().count(), 2)
            ppp.cleanLayout()
            self.assertEqual(ppp.inwidget.layout().count(), 0)

        def test_refreshContent(self):
            """test refreshContent"""
            self.mainW.parameters = Parameters()
            nBaseWidgets = 3
            with patch(
                "parthenopegui.setrun.MWPhysParamsPanel.refreshContent", autospec=True
            ) as _r:
                ppp = MWPhysParamsPanel(self.mainW)
            self.assertFalse(hasattr(ppp, "addNew"))
            self.assertFalse(hasattr(ppp, "gridNumberInfo"))
            self.assertFalse(hasattr(ppp, "tableList"))
            self.assertEqual(ppp.inwidget.layout().count(), 0)

            with patch(
                "parthenopegui.setrun.MWPhysParamsPanel.cleanLayout", autospec=True
            ) as _c:
                ppp.refreshContent()
                _c.assert_called_once_with(ppp)
            self.assertEqual(ppp.inwidget.layout().count(), nBaseWidgets)
            self.assertIsInstance(ppp.inwidget.layout().itemAt(0).widget(), PGLabel)
            self.assertEqual(
                ppp.inwidget.layout().itemAt(0).widget().text(),
                PGText.physicalParamsDescription,
            )
            self.assertIsInstance(ppp.addNew, PGPushButton)
            self.assertEqual(ppp.addNew.text(), PGText.physicalParamsAdd)
            self.assertEqual(ppp.inwidget.layout().itemAt(1).widget(), ppp.addNew)
            self.assertIsInstance(ppp.inwidget.layout().itemAt(2).widget(), PGLabel)
            self.assertEqual(
                ppp.inwidget.layout().itemAt(2).widget().text(),
                PGText.physicalParamsEmpty,
            )
            self.assertEqual(ppp.tableList, {})
            self.assertEqual(ppp.backgroundRole(), QPalette.Base)
            self.assertEqual(ppp.widget(), ppp.inwidget)
            self.assertTrue(ppp.widgetResizable())

            self.mainW.parameters.gridsList[0] = 9
            for p in Parameters.paramOrderParth[0:3]:
                self.mainW.parameters.all[p].addGrid(
                    0,
                    2,
                    self.mainW.parameters.all[p].defaultmin,
                    self.mainW.parameters.all[p].defaultmax,
                )
            for p in Parameters.paramOrderParth[3:]:
                self.mainW.parameters.all[p].addGrid(
                    0,
                    1,
                    self.mainW.parameters.all[p].defaultval,
                    self.mainW.parameters.all[p].defaultval,
                )
            self.mainW.parameters.gridsList[1] = 1
            for p in Parameters.paramOrderParth:
                self.mainW.parameters.all[p].addGrid(
                    1,
                    1,
                    self.mainW.parameters.all[p].defaultval,
                    self.mainW.parameters.all[p].defaultval,
                )
            ppp.cleanLayout()
            with patch(
                "parthenopegui.setrun.MWPhysParamsPanel.cleanLayout", autospec=True
            ) as _c:
                ppp.refreshContent()
                _c.assert_called_once_with(ppp)
            self.assertEqual(ppp.inwidget.layout().count(), nBaseWidgets + 2)
            self.assertIsInstance(ppp.inwidget.layout().itemAt(0).widget(), PGLabel)
            self.assertEqual(
                ppp.inwidget.layout().itemAt(0).widget().text(),
                PGText.physicalParamsDescription,
            )
            self.assertIsInstance(ppp.addNew, PGPushButton)
            self.assertEqual(ppp.addNew.text(), PGText.physicalParamsAdd)
            self.assertEqual(ppp.inwidget.layout().itemAt(1).widget(), ppp.addNew)
            self.assertIsInstance(ppp.inwidget.layout().itemAt(2).widget(), PGLabel)
            self.assertEqual(
                ppp.inwidget.layout().itemAt(2).widget().text(),
                PGText.physicalParamsNumberSummary
                % (
                    len(self.mainW.parameters.gridsList.keys()),
                    np.sum(list(self.mainW.parameters.gridsList.values())),
                ),
            )
            self.assertEqual(ppp.backgroundRole(), QPalette.Base)
            self.assertEqual(ppp.widget(), ppp.inwidget)
            self.assertTrue(ppp.widgetResizable())
            self.assertIsInstance(ppp.tableList, dict)
            self.assertEqual(len(ppp.tableList), 2)
            for i in sorted(self.mainW.parameters.gridsList.keys()):
                self.assertIsInstance(ppp.tableList[i], ResumeGrid)
                self.assertEqual(ppp.tableList[i].mainW, ppp.mainW)
                self.assertEqual(ppp.tableList[i].idGrid, i)
                self.assertEqual(ppp.tableList[i].physPanel, ppp)
                self.assertEqual(
                    ppp.inwidget.layout().itemAt(nBaseWidgets + i).widget(),
                    ppp.tableList[i],
                )

        def test_onAddPoints(self):
            """test onAddPoints"""
            ppp = MWPhysParamsPanel(self.mainW)
            ep = EditParameters(ppp.mainW, ppp)
            ep.exec_ = MagicMock(side_effect=[False, True])
            ep.processOutput = MagicMock()
            with patch(
                "parthenopegui.configuration.Parameters.resetCurrent", autospec=True
            ) as _rc, patch(
                "parthenopegui.setrun.EditParameters", return_value=ep
            ) as _ep:
                ppp.onAddPoints()
                _rc.assert_called_once_with(self.mainW.parameters)
                _ep.assert_called_once_with(ppp.mainW, ppp)
                ep.exec_.assert_called_once_with()
                self.assertEqual(ep.processOutput.call_count, 0)
                ppp.onAddPoints()
                self.assertEqual(ep.exec_.call_count, 2)
                ep.processOutput.assert_called_once_with()

    class TestConfigNuclidesOutput(PGTestCasewMainW):
        """Test the ConfigNuclidesOutput class"""

        def test_init(self):
            """test init"""
            with patch(
                "parthenopegui.setrun.ConfigNuclidesOutput.fillNuclides", autospec=True
            ) as _f:
                cno = ConfigNuclidesOutput(self.mainW)
                _f.assert_called_once_with(cno)
            self.assertIsInstance(cno, QWidget)
            self.assertEqual(cno.items, [])
            self.assertEqual(cno.selected, [])

            self.assertIsInstance(cno.layout(), QGridLayout)

            self.assertIsInstance(cno.listAll, DDTableWidget)
            self.assertEqual(
                cno.listAll.horizontalHeaderItem(0).text(), PGText.nuclidesOthersTitle
            )
            self.assertEqual(cno.listAll, cno.layout().itemAtPosition(0, 2).widget())
            self.assertIsInstance(cno.listSel, DDTableWidget)
            self.assertEqual(
                cno.listSel.horizontalHeaderItem(0).text(), PGText.nuclidesSelectedTitle
            )
            self.assertEqual(cno.listSel, cno.layout().itemAtPosition(0, 0).widget())

            self.assertIsInstance(cno.selAllButton, PGPushButton)
            self.assertEqual(cno.selAllButton.text(), PGText.nuclidesSelectAllText)
            self.assertEqual(
                cno.selAllButton.toolTip(), PGText.nuclidesSelectAllToolTip
            )
            self.assertEqual(
                cno.selAllButton, cno.layout().itemAtPosition(1, 1).widget()
            )
            with patch(
                "parthenopegui.setrun.ConfigNuclidesOutput.fillNuclides", autospec=True
            ) as _f:
                QTest.mouseClick(cno.selAllButton, Qt.LeftButton)
                _f.assert_called_once_with(cno, addall=True)
            self.assertIsInstance(cno.unselAllButton, PGPushButton)
            self.assertEqual(cno.unselAllButton.text(), PGText.nuclidesUnselectAllText)
            self.assertEqual(
                cno.unselAllButton.toolTip(), PGText.nuclidesUnselectAllToolTip
            )
            self.assertEqual(
                cno.unselAllButton, cno.layout().itemAtPosition(2, 1).widget()
            )
            with patch(
                "parthenopegui.setrun.ConfigNuclidesOutput.unselectAll", autospec=True
            ) as _f:
                QTest.mouseClick(cno.unselAllButton, Qt.LeftButton)
                _f.assert_called_once_with(cno)

        def test_readCurrent(self):
            """test readCurrent"""
            with patch(
                "parthenopegui.setrun.ConfigNuclidesOutput.fillNuclides", autospec=True
            ) as _f:
                cno = ConfigNuclidesOutput(self.mainW)
            for isel, n in enumerate(["n", "p", "3H", "6Li"]):
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
                cno.listSel.insertRow(isel)
                cno.listSel.setItem(isel, 0, item)
            cno.selected = "abc"
            cno.readCurrent()
            self.assertEqual(cno.selected, ["n", "p", "3H", "6Li"])

        def test_unselectAll(self):
            """test unselectAll"""
            with patch(
                "parthenopegui.setrun.ConfigNuclidesOutput.fillNuclides", autospec=True
            ) as _f:
                cno = ConfigNuclidesOutput(self.mainW)
            cno.selected = ["n", "p", "3H", "6Li"]
            with patch(
                "parthenopegui.setrun.ConfigNuclidesOutput.fillNuclides", autospec=True
            ) as _f:
                cno.unselectAll()
                _f.assert_called_once_with(cno, clear=False)
            self.assertEqual(cno.selected, [])

        def test_fillNuclides(self):
            """test fillNuclides"""
            with patch(
                "parthenopegui.setrun.ConfigNuclidesOutput.fillNuclides", autospec=True
            ) as _f:
                cno = ConfigNuclidesOutput(self.mainW)
            self.assertEqual(cno.listAll.rowCount(), 0)
            self.assertEqual(cno.listSel.rowCount(), 0)
            cno.selected = ["n", "p", "3H", "6Li"]
            qp = QPixmap()
            with patch("parthenopegui.setrun.QPixmap", return_value=qp) as _p:
                cno.fillNuclides(True)
                _p.assert_has_calls(
                    [
                        call(Nuclides.all[n])
                        for n in sorted(
                            self.mainW.nuclides.current.keys(),
                            key=lambda n: Nuclides.nuclideOrder[n],
                        )[2:]
                    ]
                )
            self.assertEqual(cno.selected, list(self.mainW.nuclides.current.keys()))
            self.assertEqual(cno.listAll.rowCount(), 0)
            self.assertEqual(
                cno.listSel.rowCount(), len(self.mainW.nuclides.current.keys())
            )
            for i, n in enumerate(
                sorted(
                    self.mainW.nuclides.current.keys(),
                    key=lambda n: Nuclides.nuclideOrder[n],
                )
            ):
                self.assertEqual(cno.listSel.item(i, 0).data(Qt.UserRole), n)
                if isinstance(Nuclides.all[n], QImage):
                    self.assertIsInstance(
                        cno.listSel.item(i, 0).data(Qt.DecorationRole), QImage
                    )
                    self.assertEqual(cno.listSel.item(i, 0).data(Qt.DisplayRole), "")
                else:
                    self.assertEqual(cno.listSel.item(i, 0).data(Qt.DisplayRole), n)
                self.assertEqual(
                    cno.listSel.item(i, 0).flags(),
                    Qt.ItemIsSelectable
                    | Qt.ItemIsEnabled
                    | Qt.ItemIsDragEnabled
                    | Qt.ItemIsDropEnabled,
                )

            cno.selected = ["n", "p", "3H", "6Li"]
            cno.fillNuclides(False)
            self.assertEqual(cno.selected, ["n", "p", "3H", "6Li"])
            isel = 0
            iall = 0
            for n in sorted(
                self.mainW.nuclides.current.keys(),
                key=lambda n: Nuclides.nuclideOrder[n],
            ):
                if n in cno.selected:
                    self.assertEqual(cno.listSel.item(isel, 0).data(Qt.UserRole), n)
                    if isinstance(Nuclides.all[n], QImage):
                        self.assertIsInstance(
                            cno.listSel.item(isel, 0).data(Qt.DecorationRole), QImage
                        )
                        self.assertEqual(
                            cno.listSel.item(isel, 0).data(Qt.DisplayRole), ""
                        )
                    else:
                        self.assertEqual(
                            cno.listSel.item(isel, 0).data(Qt.DisplayRole), n
                        )
                    self.assertEqual(
                        cno.listSel.item(isel, 0).flags(),
                        Qt.ItemIsSelectable
                        | Qt.ItemIsEnabled
                        | Qt.ItemIsDragEnabled
                        | Qt.ItemIsDropEnabled,
                    )
                    isel += 1
                else:
                    self.assertEqual(cno.listAll.item(iall, 0).data(Qt.UserRole), n)
                    if isinstance(Nuclides.all[n], QImage):
                        self.assertIsInstance(
                            cno.listAll.item(iall, 0).data(Qt.DecorationRole), QImage
                        )
                        self.assertEqual(
                            cno.listAll.item(iall, 0).data(Qt.DisplayRole), ""
                        )
                    else:
                        self.assertEqual(
                            cno.listAll.item(iall, 0).data(Qt.DisplayRole), n
                        )
                    self.assertEqual(
                        cno.listAll.item(iall, 0).flags(),
                        Qt.ItemIsSelectable
                        | Qt.ItemIsEnabled
                        | Qt.ItemIsDragEnabled
                        | Qt.ItemIsDropEnabled,
                    )
                    iall += 1

            # check what happens when resetting networks
            new = [
                n
                for n in Nuclides.all.keys()
                if Nuclides.nuclideOrder[n] < Configuration.limitNuclides["smallNet"]
            ]
            self.mainW.nuclides.updateCurrent(new)
            with patch("parthenopegui.setrun.QPixmap", return_value=qp) as _p:
                cno.fillNuclides(True, True)
                _p.assert_has_calls(
                    [
                        call(Nuclides.all[n])
                        for n in sorted(
                            self.mainW.nuclides.current.keys(),
                            key=lambda n: Nuclides.nuclideOrder[n],
                        )[2:]
                    ]
                )
            smallselect = list(cno.selected)
            self.assertEqual(cno.selected, list(self.mainW.nuclides.current.keys()))
            self.assertEqual(cno.listAll.rowCount(), 0)
            self.assertEqual(
                cno.listSel.rowCount(), len(self.mainW.nuclides.current.keys())
            )
            for i, n in enumerate(
                sorted(
                    self.mainW.nuclides.current.keys(),
                    key=lambda n: Nuclides.nuclideOrder[n],
                )
            ):
                self.assertEqual(cno.listSel.item(i, 0).data(Qt.UserRole), n)

            new = [
                n
                for n in Nuclides.all.keys()
                if Nuclides.nuclideOrder[n] < Configuration.limitNuclides["complNet"]
            ]
            self.mainW.nuclides.updateCurrent(new)
            with patch("parthenopegui.setrun.QPixmap", return_value=qp) as _p:
                cno.fillNuclides(True, True)
                _p.assert_has_calls(
                    [
                        call(Nuclides.all[n])
                        for n in sorted(
                            self.mainW.nuclides.current.keys(),
                            key=lambda n: Nuclides.nuclideOrder[n],
                        )[2:]
                    ]
                )
            self.assertEqual(cno.selected, list(self.mainW.nuclides.current.keys()))
            self.assertEqual(cno.listAll.rowCount(), 0)
            self.assertEqual(
                cno.listSel.rowCount(), len(self.mainW.nuclides.current.keys())
            )
            for i, n in enumerate(
                sorted(
                    self.mainW.nuclides.current.keys(),
                    key=lambda n: Nuclides.nuclideOrder[n],
                )
            ):
                self.assertEqual(cno.listSel.item(i, 0).data(Qt.UserRole), n)

            with patch("parthenopegui.setrun.QPixmap", return_value=qp) as _p:
                cno.fillNuclides(True, False)
                _p.assert_has_calls(
                    [
                        call(Nuclides.all[n])
                        for n in sorted(
                            self.mainW.nuclides.current.keys(),
                            key=lambda n: Nuclides.nuclideOrder[n],
                        )[2:]
                    ]
                )
            self.assertEqual(sorted(cno.selected), sorted(smallselect))
            self.assertEqual(
                cno.listAll.rowCount(),
                Configuration.limitNuclides["complNet"]
                - Configuration.limitNuclides["smallNet"],
            )
            self.assertEqual(
                cno.listSel.rowCount(), Configuration.limitNuclides["smallNet"] - 1
            )
            for i, n in enumerate(
                sorted(
                    smallselect,
                    key=lambda n: Nuclides.nuclideOrder[n],
                )
            ):
                self.assertEqual(cno.listSel.item(i, 0).data(Qt.UserRole), n)
            j = 0
            for i, n in enumerate(
                sorted(
                    self.mainW.nuclides.current.keys(),
                    key=lambda n: Nuclides.nuclideOrder[n],
                )
            ):
                if n not in smallselect:
                    self.assertEqual(cno.listAll.item(j, 0).data(Qt.UserRole), n)
                    j += 1

        def test_clearNuclides(self):
            """test clearNuclides"""
            cno = ConfigNuclidesOutput(self.mainW)
            cno.items = ["a", "b"]
            with patch(
                "parthenopegui.configuration.DDTableWidget.clearContents", autospec=True
            ) as _c:
                cno.clearNuclides()
                self.assertEqual(cno.items, [])
                self.assertEqual(_c.call_count, 2)
                self.assertEqual(cno.listAll.rowCount(), 0)
                self.assertEqual(cno.listSel.rowCount(), 0)

    class TestMWOutputPanel(PGTestCasewMainW):
        """Test the MWOutputPanel class"""

        def test_init(self):
            """test init"""
            op = MWOutputPanel(self.mainW)
            self.assertIsInstance(op, QFrame)
            self.assertEqual(op.mainW, self.mainW)
            self.assertIsInstance(op.layout(), QGridLayout)

            self.assertIsInstance(op.layout().itemAtPosition(0, 0).widget(), PGLabel)
            self.assertEqual(
                op.layout().itemAtPosition(0, 0).widget().text(),
                "<center>%s</center>" % PGText.outputPanelSelectDescription,
            )
            self.assertIsInstance(op.nuclidesOutput, ConfigNuclidesOutput)
            self.assertEqual(
                op.layout().itemAtPosition(1, 0).widget(), op.nuclidesOutput
            )
            with patch(
                "parthenopegui.setrun.ConfigNuclidesOutput.fillNuclides", autospec=True
            ) as _f:
                op.mainW.nuclides.updated.emit()
                _f.assert_any_call(op.nuclidesOutput)

            self.assertIsInstance(op.layout().itemAtPosition(2, 1).widget(), PGLabel)
            self.assertEqual(
                op.layout().itemAtPosition(2, 1).widget().text(),
                PGText.outputPanelDirectoryAsk,
            )
            self.assertIsInstance(op.outputFolderName, PGLabelButton)
            self.assertEqual(
                op.layout().itemAtPosition(3, 1).widget(), op.outputFolderName
            )
            self.assertEqual(
                op.outputFolderName.text(), PGText.outputPanelDirectoryInitialTitle
            )
            self.assertEqual(
                op.outputFolderName.toolTip(), PGText.outputPanelDirectoryToolTip
            )
            with patch(
                "parthenopegui.setrun.MWOutputPanel.askDirectoryName", autospec=True
            ) as _f:
                QTest.mouseClick(op.outputFolderName, Qt.LeftButton)
                _f.assert_called_once_with(op)

            self.assertIsInstance(op.layout().itemAtPosition(4, 1).widget(), PGLabel)
            self.assertEqual(op.layout().itemAtPosition(4, 1).widget().text(), "")
            self.assertIsInstance(op.layout().itemAtPosition(5, 1).widget(), PGLabel)
            self.assertEqual(
                op.layout().itemAtPosition(5, 1).widget().text(),
                PGText.outputPanelNuclidesInOutput,
            )
            self.assertIsInstance(op.nuclidesInOutput, QCheckBox)
            self.assertEqual(
                op.layout().itemAtPosition(6, 1).widget(), op.nuclidesInOutput
            )
            self.assertEqual(op.nuclidesInOutput.text(), "")
            self.assertTrue(op.nuclidesInOutput.isChecked())

        def test_askDirectoryName(self):
            """test askDirectoryName"""
            op = MWOutputPanel(self.mainW)
            with patch(
                "parthenopegui.setrun.askDirName",
                autospec=True,
                side_effect=["", " abc ", "abc"],
            ) as _a, patch("logging.Logger.error") as _w:
                op.askDirectoryName()
                _a.assert_called_once_with(
                    op,
                    PGText.outputPanelDirectoryDialogTitle,
                    op.outputFolderName.text(),
                )
                self.assertEqual(
                    op.outputFolderName.text(), PGText.outputPanelDirectoryInitialTitle
                )
                self.assertEqual(
                    self.mainW.runSettingsTab.runPanel.saveDefault.toolTip(),
                    PGText.runPanelSaveDefaultToolTip % "folder",
                )
                self.assertEqual(
                    self.mainW.runSettingsTab.runPanel.saveCustom.toolTip(),
                    PGText.runPanelSaveCustomToolTip % "folder",
                )
                self.assertEqual(_w.call_count, 0)
                op.askDirectoryName()
                _w.assert_called_once_with(PGText.runPanelSpaceInFolderName)
                self.assertEqual(
                    op.outputFolderName.text(), PGText.outputPanelDirectoryInitialTitle
                )
                self.assertEqual(
                    self.mainW.runSettingsTab.runPanel.saveDefault.toolTip(),
                    PGText.runPanelSaveDefaultToolTip % "folder",
                )
                self.assertEqual(
                    self.mainW.runSettingsTab.runPanel.saveCustom.toolTip(),
                    PGText.runPanelSaveCustomToolTip % "folder",
                )
                op.askDirectoryName()
                _w.assert_called_once_with(PGText.runPanelSpaceInFolderName)
                self.assertEqual(op.outputFolderName.text(), "abc")
                self.assertEqual(
                    self.mainW.runSettingsTab.runPanel.saveDefault.toolTip(),
                    PGText.runPanelSaveDefaultToolTip % "abc",
                )
                self.assertEqual(
                    self.mainW.runSettingsTab.runPanel.saveCustom.toolTip(),
                    PGText.runPanelSaveCustomToolTip % "abc",
                )

    class TestMWRunPanel(PGTestCasewMainW):
        """Test the MWRunPanel class"""

        def test_init(self):
            """test init"""
            rp = MWRunPanel(self.mainW)
            self.assertIsInstance(rp, QFrame)
            self.assertEqual(rp.mainW, self.mainW)
            self.assertIsInstance(rp.layout(), QGridLayout)

            self.assertIsInstance(rp.text, PGLabel)
            self.assertEqual(
                rp.text.text(), "<center>%s</center>" % PGText.runPanelDescription
            )
            self.assertEqual(rp.layout().itemAtPosition(0, 0).widget(), rp.text)

            self.assertIsInstance(rp.runDefault, PGPushButton)
            self.assertEqual(rp.runDefault.text(), PGText.runPanelDefaultTitle)
            self.assertEqual(rp.runDefault.toolTip(), PGText.runPanelDefaultToolTip)
            self.assertEqual(rp.layout().itemAtPosition(1, 1).widget(), rp.runDefault)
            with patch(
                "parthenopegui.setrun.MWRunPanel.startRunDefault", autospec=True
            ) as _f:
                QTest.mouseClick(rp.runDefault, Qt.LeftButton)
                _f.assert_called_once_with(rp)

            self.assertIsInstance(rp.saveDefault, PGPushButton)
            self.assertEqual(rp.saveDefault.text(), PGText.runPanelSaveDefaultTitle)
            self.assertEqual(
                rp.saveDefault.toolTip(), PGText.runPanelSaveDefaultToolTip % "folder"
            )
            self.assertEqual(rp.layout().itemAtPosition(1, 2).widget(), rp.saveDefault)
            with patch(
                "parthenopegui.setrun.MWRunPanel.prepareRun", autospec=True
            ) as _f:
                QTest.mouseClick(rp.saveDefault, Qt.LeftButton)
                _f.assert_called_once_with(rp, False)

            self.assertIsInstance(rp.runCustom, PGPushButton)
            self.assertEqual(rp.runCustom.text(), PGText.runPanelCustomTitle)
            self.assertEqual(rp.runCustom.toolTip(), PGText.runPanelCustomToolTip)
            self.assertEqual(rp.layout().itemAtPosition(2, 1).widget(), rp.runCustom)
            with patch(
                "parthenopegui.setrun.MWRunPanel.startRunCustom", autospec=True
            ) as _f:
                QTest.mouseClick(rp.runCustom, Qt.LeftButton)
                _f.assert_called_once_with(rp)

            self.assertIsInstance(rp.saveCustom, PGPushButton)
            self.assertEqual(rp.saveCustom.text(), PGText.runPanelSaveCustomTitle)
            self.assertEqual(
                rp.saveCustom.toolTip(), PGText.runPanelSaveCustomToolTip % "folder"
            )
            self.assertEqual(rp.layout().itemAtPosition(2, 2).widget(), rp.saveCustom)
            with patch(
                "parthenopegui.setrun.MWRunPanel.prepareRun", autospec=True
            ) as _f:
                QTest.mouseClick(rp.saveCustom, Qt.LeftButton)
                _f.assert_called_once_with(rp, default=False)

            self.assertIsInstance(rp.statusLabel, PGLabel)
            self.assertEqual(rp.statusLabel.text(), "")
            self.assertEqual(rp.layout().itemAtPosition(3, 0).widget(), rp.statusLabel)

            self.assertIsInstance(rp.stopButton, PGPushButton)
            self.assertEqual(rp.stopButton.text(), PGText.runPanelStopTitle)
            self.assertEqual(rp.stopButton.toolTip(), PGText.runPanelStopToolTip)
            self.assertEqual(rp.layout().itemAtPosition(4, 1).widget(), rp.stopButton)
            self.assertFalse(rp.stopButton.isEnabled())
            rp.stopButton.setEnabled(True)
            with patch("parthenopegui.setrun.MWRunPanel.stopRun", autospec=True) as _f:
                QTest.mouseClick(rp.stopButton, Qt.LeftButton)
                _f.assert_called_once_with(rp)

            self.assertIsInstance(rp.progressBar, QProgressBar)
            self.assertEqual(rp.progressBar.minimum(), 0)
            self.assertEqual(rp.layout().itemAtPosition(5, 0).widget(), rp.progressBar)

        def test_startRunDefault(self):
            """test startRunDefault"""
            rp = MWRunPanel(self.mainW)
            with patch(
                "parthenopegui.setrun.MWRunPanel.prepareRun",
                autospec=True,
                return_value=["a", "b"],
            ) as _p, patch(
                "parthenopegui.setrun.MWRunPanel.startRun", autospec=True
            ) as _r:
                self.assertEqual(rp.startRunDefault(), ("a", "b"))
                _p.assert_called_once_with(rp)
                _r.assert_called_once_with(rp)
            with patch(
                "parthenopegui.setrun.MWRunPanel.prepareRun",
                autospec=True,
                return_value=None,
            ) as _p, patch(
                "parthenopegui.setrun.MWRunPanel.startRun", autospec=True
            ) as _r:
                self.assertEqual(rp.startRunDefault(), (None, None))
                _p.assert_called_once_with(rp)
                _r.assert_not_called()

        def test_startRunCustom(self):
            """test startRunCustom"""
            rp = MWRunPanel(self.mainW)
            with patch(
                "parthenopegui.setrun.MWRunPanel.prepareRun",
                autospec=True,
                return_value=["a", "b"],
            ) as _p, patch(
                "parthenopegui.setrun.MWRunPanel.startRun", autospec=True
            ) as _r:
                self.assertEqual(rp.startRunCustom(), ("a", "b"))
                _p.assert_called_once_with(rp, default=False)
                _r.assert_called_once_with(rp)
            with patch(
                "parthenopegui.setrun.MWRunPanel.prepareRun",
                autospec=True,
                return_value=None,
            ) as _p, patch(
                "parthenopegui.setrun.MWRunPanel.startRun", autospec=True
            ) as _r:
                self.assertEqual(rp.startRunCustom(), (None, None))
                _p.assert_called_once_with(rp, default=False)
                _r.assert_not_called()

        def test_startRunPickle(self):
            """test startRunPickle"""
            rp = MWRunPanel(self.mainW)
            run = RunPArthENoPE(
                testConfig.commonParamsGrid, testConfig.gridPointsGrid, self.mainW
            )
            with patch(
                "parthenopegui.setrun.RunPArthENoPE",
                autospec=USE_AUTOSPEC_CLASS,
                return_value=run,
            ) as _p, patch(
                "parthenopegui.setrun.MWRunPanel.startRun", autospec=True
            ) as _r, patch(
                "logging.Logger.info"
            ) as _i:
                cp, gp = rp.startRunPickle(testConfig.testRunDirFullExample)
                self.assertEqual(cp, testConfig.commonParamsGrid)
                self.assertTrue(
                    np.all(
                        [np.all(a == b) for a, b in zip(gp, testConfig.gridPointsGrid)]
                    )
                )
                self.assertEqual(_p.call_count, 1)
                self.assertEqual(_p.call_args[0][0], testConfig.commonParamsGrid)
                self.assertTrue(
                    np.all(
                        [
                            np.all(a == b)
                            for a, b in zip(
                                _p.call_args[0][1], testConfig.gridPointsGrid
                            )
                        ]
                    )
                )
                self.assertEqual(_p.call_args[1]["parent"], self.mainW)
                _r.assert_called_once_with(rp)
                _i.assert_any_call(
                    PGText.tryToOpenFolder % testConfig.testRunDirFullExample
                )
                _i.assert_any_call(PGText.startRun)

            with patch(
                "parthenopegui.setrun.RunPArthENoPE",
                autospec=USE_AUTOSPEC_CLASS,
                return_value=run,
            ) as _p, patch("logging.Logger.error") as _e, patch(
                "logging.Logger.info"
            ) as _i:
                self.assertEqual(rp.startRunPickle("/nonexistent/run/folder"), None)
                self.assertEqual(_p.call_count, 0)
                _e.assert_called_once_with(PGText.errorCannotLoadGrid)
                self.assertEqual(_i.call_count, 0)

            with open(Configuration.runSettingsObj, "w") as _f:
                _f.write("")
            with patch(
                "parthenopegui.setrun.RunPArthENoPE",
                autospec=USE_AUTOSPEC_CLASS,
                return_value=run,
            ) as _p, patch("logging.Logger.error") as _e:
                self.assertEqual(rp.startRunPickle("."), None)
                self.assertEqual(_p.call_count, 0)
                _e.assert_called_once_with(PGText.errorCannotLoadGrid)
            os.remove(Configuration.runSettingsObj)

        def test_prepareRun(self):
            """test prepareRun"""
            rp = MWRunPanel(self.mainW)
            # empty folder
            self.mainW.runSettingsTab.outputParams.outputFolderName.setText(
                PGText.outputPanelDirectoryInitialTitle
            )
            with patch("logging.Logger.warning") as _e:
                self.assertEqual(rp.prepareRun(), None)
                _e.assert_called_once_with(PGText.errorNoOutputFolder)
                _e.reset_mock()
                self.assertEqual(rp.prepareRun(False), None)
                _e.assert_called_once_with(PGText.errorNoOutputFolder)
                _e.reset_mock()
                self.mainW.runSettingsTab.outputParams.outputFolderName.setText("")
                self.assertEqual(rp.prepareRun(), None)
                _e.assert_called_once_with(PGText.errorNoOutputFolder)
                _e.reset_mock()
                self.assertEqual(rp.prepareRun(False), None)
                _e.assert_called_once_with(PGText.errorNoOutputFolder)
            self.mainW.runSettingsTab.outputParams.outputFolderName.setText(
                testConfig.testRunDirDefaultSample
            )
            run = RunPArthENoPE(
                testConfig.commonParamsDefault, testConfig.gridPointsDefault, self.mainW
            )
            today = datetime.datetime.today()
            now = today.strftime("%y%m%d_%H%M%S")
            # default
            with patch("parthenopegui.setrun.datetime") as _dt, patch(
                "parthenopegui.setrun.RunPArthENoPE",
                return_value=run,
                autospec=USE_AUTOSPEC_CLASS,
            ) as _rp, patch("pickle.dump") as _dm:
                _dt.today.return_value = today
                cp, gp = rp.prepareRun()
                self.assertEqual(_dm.call_count, 1)
                self.assertEqual(
                    _dm.call_args[0][0][0], paramsHidePath(cp, cp["output_folder"])
                )
                self.assertEqualArray(_dm.call_args[0][0][1], gp)
                self.assertEqual(_rp.call_count, 1)
                self.assertEqual(_rp.call_args[0][0], cp)
                self.assertEqualArray(_rp.call_args[0][1], gp)
                self.assertEqual(_rp.call_args[1]["parent"], self.mainW)
                self.assertEqual(rp.mainW.runner, run)
                self.assertEqual(
                    cp,
                    {
                        "N_changed_rates": 0,
                        "N_stored_nuclides": 9,
                        "changed_rates_list": [],
                        "inputcard_filename": os.path.join(
                            testConfig.testRunDirDefaultSample,
                            "input_" + now + "_%d.card",
                        ),
                        "now": now,
                        "num_nuclides_net": 9,
                        "onScreenOutput": False,
                        "output_file_grid": os.path.join(
                            testConfig.testRunDirDefaultSample,
                            "parthenope_" + now + ".out",
                        ),
                        "output_file_log": os.path.join(
                            testConfig.testRunDirDefaultSample,
                            "%s_" % Configuration.fortranOutputFilename
                            + now
                            + "_%d.out",
                        ),
                        "output_file_nuclides": os.path.join(
                            testConfig.testRunDirDefaultSample,
                            "nuclides_" + now + "_%d.out",
                        ),
                        "output_file_info": os.path.join(
                            testConfig.testRunDirDefaultSample,
                            "info_" + now + "_%d.out",
                        ),
                        "output_file_parthenope": os.path.join(
                            testConfig.testRunDirDefaultSample,
                            "parthenope_" + now + "_%d.out",
                        ),
                        "output_file_run": os.path.join(
                            testConfig.testRunDirDefaultSample, "run_" + now + "_%d.out"
                        ),
                        "output_folder": testConfig.testRunDirDefaultSample,
                        "output_fortran_log": os.path.join(
                            testConfig.testRunDirDefaultSample,
                            "fortran_output_" + now + ".out",
                        ),
                        "output_overwrite": True,
                        "output_save_nuclides": True,
                        "stored_nuclides": [
                            "n",
                            "p",
                            "2H",
                            "3H",
                            "3He",
                            "4He",
                            "6Li",
                            "7Li",
                            "7Be",
                        ],
                    },
                )
                self.assertEqualArray(
                    gp,
                    np.asarray(
                        [
                            [
                                [
                                    self.mainW.parameters.all[p].defaultval
                                    for p in Parameters.paramOrder
                                ]
                            ]
                        ]
                    ),
                )
            # custom but no points
            with patch("logging.Logger.warning") as _w:
                self.assertEqual(rp.prepareRun(default=False), None)
                _w.assert_called_once_with(PGText.errorNoPhysicalParams)

            # prepare for custom
            self.mainW.parameters.gridsList[0] = 5
            for p in Parameters.paramOrderParth[1:]:
                self.mainW.parameters.all[p].addGrid(
                    0,
                    1,
                    self.mainW.parameters.all[p].defaultval,
                    self.mainW.parameters.all[p].defaultval,
                )
            self.mainW.parameters.all["DeltaNnu"].addGrid(0, 5, -2, 2)
            self.mainW.runSettingsTab.networkParams.interNet.setChecked(True)
            self.mainW.runSettingsTab.networkParams.updateReactions()
            self.mainW.runSettingsTab.outputParams.nuclidesInOutput.setChecked(False)
            self.mainW.runSettingsTab.networkParams.reactionsTable.updateRow(
                0, "low", 0
            )
            self.mainW.runSettingsTab.networkParams.reactionsTable.updateRow(
                1, "high", 0
            )
            self.mainW.runSettingsTab.networkParams.reactionsTable.updateRow(
                2, "custom factor", 1.12
            )
            # default (ignore changed values)
            with patch("parthenopegui.setrun.datetime") as _dt, patch(
                "parthenopegui.setrun.RunPArthENoPE",
                return_value=run,
                autospec=USE_AUTOSPEC_CLASS,
            ) as _rp, patch("pickle.dump") as _dm:
                _dt.today.return_value = today
                cp, gp = rp.prepareRun()
                self.assertEqual(_dm.call_count, 1)
                self.assertEqual(
                    _dm.call_args[0][0][0], paramsHidePath(cp, cp["output_folder"])
                )
                self.assertEqualArray(_dm.call_args[0][0][1], gp)
                self.assertEqual(_rp.call_count, 1)
                self.assertEqual(_rp.call_args[0][0], cp)
                self.assertEqualArray(_rp.call_args[0][1], gp)
                self.assertEqual(_rp.call_args[1]["parent"], self.mainW)
                self.assertEqual(rp.mainW.runner, run)
                self.assertEqual(
                    cp,
                    {
                        "N_changed_rates": 0,
                        "N_stored_nuclides": 9,
                        "changed_rates_list": [],
                        "inputcard_filename": os.path.join(
                            testConfig.testRunDirDefaultSample,
                            "input_" + now + "_%d.card",
                        ),
                        "now": now,
                        "num_nuclides_net": 9,
                        "onScreenOutput": False,
                        "output_file_grid": os.path.join(
                            testConfig.testRunDirDefaultSample,
                            "parthenope_" + now + ".out",
                        ),
                        "output_file_log": os.path.join(
                            testConfig.testRunDirDefaultSample,
                            "%s_" % Configuration.fortranOutputFilename
                            + now
                            + "_%d.out",
                        ),
                        "output_file_nuclides": os.path.join(
                            testConfig.testRunDirDefaultSample,
                            "nuclides_" + now + "_%d.out",
                        ),
                        "output_file_info": os.path.join(
                            testConfig.testRunDirDefaultSample,
                            "info_" + now + "_%d.out",
                        ),
                        "output_file_parthenope": os.path.join(
                            testConfig.testRunDirDefaultSample,
                            "parthenope_" + now + "_%d.out",
                        ),
                        "output_file_run": os.path.join(
                            testConfig.testRunDirDefaultSample, "run_" + now + "_%d.out"
                        ),
                        "output_folder": testConfig.testRunDirDefaultSample,
                        "output_fortran_log": os.path.join(
                            testConfig.testRunDirDefaultSample,
                            "fortran_output_" + now + ".out",
                        ),
                        "output_overwrite": True,
                        "output_save_nuclides": True,
                        "stored_nuclides": [
                            "n",
                            "p",
                            "2H",
                            "3H",
                            "3He",
                            "4He",
                            "6Li",
                            "7Li",
                            "7Be",
                        ],
                    },
                )
                self.assertEqualArray(
                    gp,
                    np.asarray(
                        [
                            [
                                [
                                    self.mainW.parameters.all[p].defaultval
                                    for p in Parameters.paramOrder
                                ]
                            ]
                        ]
                    ),
                )
            # custom
            self.mainW.runSettingsTab.networkParams.interNet.setChecked(True)
            self.mainW.runSettingsTab.networkParams.updateReactions()
            self.mainW.runSettingsTab.outputParams.nuclidesInOutput.setChecked(False)
            self.mainW.runSettingsTab.networkParams.reactionsTable.updateRow(
                0, "low", 0
            )
            self.mainW.runSettingsTab.networkParams.reactionsTable.updateRow(
                1, "high", 0
            )
            self.mainW.runSettingsTab.networkParams.reactionsTable.updateRow(
                2, "custom factor", 1.12
            )
            with patch("parthenopegui.setrun.datetime") as _dt, patch(
                "parthenopegui.setrun.RunPArthENoPE",
                return_value=run,
                autospec=USE_AUTOSPEC_CLASS,
            ) as _rp, patch("pickle.dump") as _dm:
                _dt.today.return_value = today
                cp, gp = rp.prepareRun(default=False)
                self.assertEqual(_dm.call_count, 1)
                self.assertEqual(
                    _dm.call_args[0][0][0], paramsHidePath(cp, cp["output_folder"])
                )
                self.assertEqualArray(_dm.call_args[0][0][1], gp)
                self.assertEqual(_rp.call_count, 1)
                self.assertEqual(_rp.call_args[0][0], cp)
                self.assertEqualArray(_rp.call_args[0][1], gp)
                self.assertEqual(_rp.call_args[1]["parent"], self.mainW)
                self.assertEqual(rp.mainW.runner, run)
                self.assertEqual(
                    cp,
                    {
                        "N_changed_rates": 3,
                        "N_stored_nuclides": 9,
                        "changed_rates_list": [
                            [1, 1, "1.0"],
                            [2, 2, "1.0"],
                            [3, 3, "1.12"],
                        ],
                        "inputcard_filename": os.path.join(
                            testConfig.testRunDirDefaultSample,
                            "input_" + now + "_%d.card",
                        ),
                        "now": now,
                        "num_nuclides_net": 18,
                        "onScreenOutput": False,
                        "output_file_grid": os.path.join(
                            testConfig.testRunDirDefaultSample,
                            "parthenope_" + now + ".out",
                        ),
                        "output_file_log": os.path.join(
                            testConfig.testRunDirDefaultSample,
                            "%s_" % Configuration.fortranOutputFilename
                            + now
                            + "_%d.out",
                        ),
                        "output_file_nuclides": os.path.join(
                            testConfig.testRunDirDefaultSample,
                            "nuclides_" + now + "_%d.out",
                        ),
                        "output_file_info": os.path.join(
                            testConfig.testRunDirDefaultSample,
                            "info_" + now + "_%d.out",
                        ),
                        "output_file_parthenope": os.path.join(
                            testConfig.testRunDirDefaultSample,
                            "parthenope_" + now + "_%d.out",
                        ),
                        "output_file_run": os.path.join(
                            testConfig.testRunDirDefaultSample, "run_" + now + "_%d.out"
                        ),
                        "output_folder": testConfig.testRunDirDefaultSample,
                        "output_fortran_log": os.path.join(
                            testConfig.testRunDirDefaultSample,
                            "fortran_output_" + now + ".out",
                        ),
                        "output_overwrite": True,
                        "output_save_nuclides": False,
                        "stored_nuclides": [
                            "n",
                            "p",
                            "2H",
                            "3H",
                            "3He",
                            "4He",
                            "6Li",
                            "7Li",
                            "7Be",
                        ],
                    },
                )
                self.assertEqualArray(
                    gp,
                    np.asarray(
                        [
                            [
                                [
                                    self.mainW.parameters.all[p].defaultval
                                    if p != "DeltaNnu"
                                    else a
                                    for p in Parameters.paramOrder
                                ]
                                for a in np.linspace(-2, 2, 5)
                            ]
                        ]
                    ),
                )

        def test_startRun(self):
            """test startRun"""
            rp = MWRunPanel(self.mainW)
            self.mainW.runSettingsTab.outputParams.outputFolderName.setText(
                testConfig.testRunDirDefaultSample
            )
            rp.prepareRun()
            self.mainW.runner.totalRuns = 123
            rp.mainW.running = False
            rp.mainW.runSettingsTab.outputParams.setEnabled(True)
            rp.mainW.runSettingsTab.networkParams.setEnabled(True)
            rp.mainW.runSettingsTab.physicsParams.setEnabled(True)
            rp.runDefault.setEnabled(True)
            rp.saveDefault.setEnabled(True)
            rp.runCustom.setEnabled(True)
            rp.saveCustom.setEnabled(True)
            rp.stopButton.setEnabled(False)
            with patch("parthenopegui.runner.RunPArthENoPE.run", autospec=True) as _r:
                rp.startRun()
                _r.assert_called_once_with(self.mainW.runner)
            with patch(
                "parthenopegui.runner.RunPArthENoPE.oneHasFinished", autospec=True
            ) as _f:
                rp.mainW.runner.runHasFinished.emit(41)
                _f.assert_called_once_with(rp.mainW.runner, 41)
            with patch(
                "parthenopegui.setrun.MWRunPanel.enableAll", autospec=True
            ) as _f1, patch(
                "parthenopegui.runner.RunPArthENoPE.allHaveFinished", autospec=True
            ) as _f2:
                rp.mainW.runner.poolHasFinished.emit()
                _f1.assert_called_once_with(rp)
                _f2.assert_called_once_with(rp.mainW.runner)
            self.assertEqual(rp.progressBar.value(), 0)
            self.assertEqual(rp.progressBar.maximum(), 123)
            self.assertTrue(rp.mainW.running)
            self.assertFalse(rp.mainW.runSettingsTab.outputParams.isEnabled())
            self.assertFalse(rp.mainW.runSettingsTab.networkParams.isEnabled())
            self.assertFalse(rp.mainW.runSettingsTab.physicsParams.isEnabled())
            self.assertFalse(rp.runDefault.isEnabled())
            self.assertFalse(rp.saveDefault.isEnabled())
            self.assertFalse(rp.runCustom.isEnabled())
            self.assertFalse(rp.saveCustom.isEnabled())
            self.assertTrue(rp.stopButton.isEnabled())
            self.mainW.runner.updateStatus.emit("newtext")
            self.assertEqual(rp.statusLabel.text(), "<center>newtext</center>")
            self.mainW.runner.updateProgressBar.emit(111)
            self.assertEqual(rp.progressBar.value(), 111)

        def test_stopRun(self):
            """test stopRun"""
            rp = MWRunPanel(self.mainW)
            self.mainW.runSettingsTab.outputParams.outputFolderName.setText(
                testConfig.testRunDirDefaultSample
            )
            rp.prepareRun()
            with patch(
                "parthenopegui.setrun.MWRunPanel.enableAll", autospec=True
            ) as _e, patch(
                "parthenopegui.runner.RunPArthENoPE.stop", autospec=True
            ) as _s:
                rp.stopRun()
                _s.assert_called_once_with(self.mainW.runner)
                _e.assert_called_once_with(rp)

        def test_enableAll(self):
            """test enableAll"""
            rp = MWRunPanel(self.mainW)
            rp.mainW.running = True
            rp.mainW.runSettingsTab.outputParams.setEnabled(False)
            rp.mainW.runSettingsTab.networkParams.setEnabled(False)
            rp.mainW.runSettingsTab.physicsParams.setEnabled(False)
            rp.runDefault.setEnabled(False)
            rp.saveDefault.setEnabled(False)
            rp.runCustom.setEnabled(False)
            rp.saveCustom.setEnabled(False)
            rp.stopButton.setEnabled(True)
            rp.enableAll()
            self.assertFalse(rp.mainW.running)
            self.assertTrue(rp.mainW.runSettingsTab.outputParams.isEnabled())
            self.assertTrue(rp.mainW.runSettingsTab.networkParams.isEnabled())
            self.assertTrue(rp.mainW.runSettingsTab.physicsParams.isEnabled())
            self.assertTrue(rp.runDefault.isEnabled())
            self.assertTrue(rp.saveDefault.isEnabled())
            self.assertTrue(rp.runCustom.isEnabled())
            self.assertTrue(rp.saveCustom.isEnabled())
            self.assertFalse(rp.stopButton.isEnabled())

    class TestMWRunSettingsPanel(PGTestCasewMainW):
        """Test the MWRunSettingsPanel class"""

        def test_init(self):
            """test init"""
            sp = MWRunSettingsPanel(self.mainW)
            self.assertIsInstance(sp, QFrame)
            self.assertEqual(sp.mainW, self.mainW)
            self.assertIsInstance(sp.layout(), QGridLayout)

            self.assertIsInstance(sp.runPanel, MWRunPanel)
            self.assertIsInstance(sp.outputParams, MWOutputPanel)
            self.assertIsInstance(sp.networkParams, MWNetworkPanel)
            self.assertIsInstance(sp.physicsParams, MWPhysParamsPanel)

            self.assertEqual(
                sp.layout().itemAtPosition(0, 0).widget(), sp.networkParams
            )
            self.assertEqual(
                sp.layout().itemAtPosition(0, 1).widget(), sp.physicsParams
            )
            self.assertEqual(sp.layout().itemAtPosition(1, 0).widget(), sp.outputParams)
            self.assertEqual(sp.layout().itemAtPosition(1, 1).widget(), sp.runPanel)


########## plotter
if PGTestsConfig.test_plt:

    class TestPlotterGlobals(PGTestCase):
        """Test global variables in the plotter module"""

        def test_globals(self):
            """Test the global variables"""
            import parthenopegui.plotter as p

            self.assertIsInstance(p.cmaps, dict)
            for v in p.cmaps.values():
                self.assertIsInstance(v, list)
                for m in v:
                    try:
                        self.assertIsInstance(
                            matplotlib.cm.get_cmap(m), matplotlib.colors.Colormap
                        )
                    except ValueError:
                        pass
            self.assertIsInstance(p.defaultCmap, six.string_types)
            self.assertIsInstance(p.extendOptions, list)
            self.assertIsInstance(p.markerOptions, list)
            self.assertIsInstance(p.styleOptions, list)
            self.assertIsInstance(p.chi2labels, dict)
            for k, v in p.chi2labels.items():
                self.assertIsInstance(v, six.string_types)

        def test_chi2func(self):
            """test the function that computes the chi2"""
            self.assertEqual(chi2func(1.12, 1.12, 0.4), 0.0)
            self.assertAlmostEqual(chi2func(1.52, 1.12, 0.4), 1.0)
            rv = chi2func(np.asarray([0.72, 1.92, 0.0]), 1.12, 0.4)
            for r, e in zip(rv, [1.0, 4.0, 7.84]):
                self.assertAlmostEqual(r, e)
            rv = chi2func([0.72, 1.92, 0.0], 1.12, 0.4)
            for r, e in zip(rv, [1.0, 4.0, 7.84]):
                self.assertAlmostEqual(r, e)

    class TestPGPlotObject(PGTestCase):
        """Test the PGPlotObject"""

        def test_getYlabel(self):
            """test the function that returns the ylabel of the object"""
            p = PGPlotObject()
            p.chi2 = False
            p.ylabel = "abc"
            self.assertEqual(p.getYlabel(), "abc")
            for k in ("chi2", "lh", "nllh", "pllh"):
                p.chi2 = k
                self.assertEqual(p.getYlabel(), chi2labels[k])
            p.chi2 = True
            with self.assertRaises(KeyError):
                p.getYlabel()
            p.z = True
            self.assertEqual(p.getYlabel(), "abc")
            p.chi2 = False
            self.assertEqual(p.getYlabel(), "abc")

        def test_getZlabel(self):
            """test the function that returns the zlabel of the object"""
            p = PGPlotObject()
            p.chi2 = False
            p.zlabel = "abc"
            self.assertEqual(p.getZlabel(), "")
            p.z = True
            self.assertEqual(p.getZlabel(), "abc")
            del p.zlabel
            self.assertEqual(p.getZlabel(), "")
            p.zlabel = "abc"
            for k in ("chi2", "lh", "nllh", "pllh"):
                p.chi2 = k
                self.assertEqual(p.getZlabel(), chi2labels[k])
            p.chi2 = True
            with self.assertRaises(KeyError):
                p.getZlabel()

    class TestPGLine(PGTestCase):
        """Test the PGLine class"""

        def test_init(self):
            """test init"""
            l = PGLine("x", "y", 0)
            self.assertIsInstance(l, PGPlotObject)
            self.assertEqual(l.x, "x")
            self.assertEqual(l.y, "y")
            self.assertEqual(l.pointRow, 0)
            self.assertFalse(l.chi2)
            self.assertEqual(l.color, "k")
            self.assertEqual(l.description, "")
            self.assertEqual(l.label, "")
            self.assertEqual(l.marker, "")
            self.assertEqual(l.style, "-")
            self.assertEqual(l.width, 1)
            self.assertEqual(l.xlabel, "")
            self.assertEqual(l.ylabel, "")
            l = PGLine(
                [0, 1, 2],
                [3, 4, 5],
                1,
                c="r",
                c2="chi2",
                d="desc",
                l="label",
                m="o",
                s=":",
                w=1.12,
                xl="xl",
                yl="yl",
            )
            self.assertEqual(l.x, [0, 1, 2])
            self.assertEqual(l.y, [3, 4, 5])
            self.assertEqual(l.pointRow, 1)
            self.assertTrue(l.chi2)
            self.assertEqual(l.color, "r")
            self.assertEqual(l.description, "desc")
            self.assertEqual(l.label, "label")
            self.assertEqual(l.marker, "o")
            self.assertEqual(l.style, ":")
            self.assertEqual(l.width, 1.12)
            self.assertEqual(l.xlabel, "xl")
            self.assertEqual(l.ylabel, "yl")

    class TestPGContour(PGTestCase):
        """Test the PGContour class"""

        def test_init(self):
            """test init"""
            c = PGContour("x", "y", "z", 0)
            self.assertEqual(c.x, "x")
            self.assertEqual(c.y, "y")
            self.assertEqual(c.z, "z")
            self.assertEqual(c.pointRow, 0)
            self.assertFalse(c.chi2)
            self.assertEqual(c.cmap, defaultCmap)
            self.assertEqual(c.description, "")
            self.assertEqual(c.extend, None)
            self.assertTrue(c.filled)
            self.assertTrue(c.hascbar)
            self.assertEqual(c.label, "")
            self.assertEqual(c.levels, None)
            self.assertEqual(c.xlabel, "")
            self.assertEqual(c.ylabel, "")
            self.assertEqual(c.zlabel, "")
            c = PGContour(
                [0, 1],
                [2, 3],
                [4, 5, 6, 7],
                1,
                c2="chi2",
                cm="winter",
                d="desc",
                ex="both",
                f=False,
                hcb=False,
                l="lab",
                lvs=[-1, 2],
                xl="xl",
                yl="yl",
                zl="zl",
            )
            self.assertEqual(c.x, [0, 1])
            self.assertEqual(c.y, [2, 3])
            self.assertEqual(c.z, [4, 5, 6, 7])
            self.assertEqual(c.pointRow, 1)
            self.assertTrue(c.chi2)
            self.assertEqual(c.cmap, "winter")
            self.assertEqual(c.description, "desc")
            self.assertEqual(c.extend, "both")
            self.assertFalse(c.filled)
            self.assertFalse(c.hascbar)
            self.assertEqual(c.label, "lab")
            self.assertEqual(c.levels, [-1, 2])
            self.assertEqual(c.xlabel, "xl")
            self.assertEqual(c.ylabel, "yl")
            self.assertEqual(c.zlabel, "zl")

    class TestCurrentPlotContent(PGTestCase):
        """tests for the CurrentPlotContent class"""

        @classmethod
        def setUpClass(self):
            """Call the parent method and instantiate a testing MainWindow"""
            super(TestCurrentPlotContent, self).setUpClass()
            self.ps = PlotSettings()

        def test_init(self):
            """test the init method"""
            cpc = CurrentPlotContent(self.ps)
            self.assertEqual(cpc.pSett, self.ps)
            with patch(
                "parthenopegui.plotter.CurrentPlotContent.clearCurrent", autospec=True
            ) as _cc:
                cpc = CurrentPlotContent(self.ps)
                _cc.assert_called_once_with(cpc)

        def test_clearCurrent(self):
            """test the clearCurrent method"""
            cpc = CurrentPlotContent(self.ps)
            cpc.lines = [1, 2, 3]
            cpc.contours = "empty"
            cpc.axesTextSize = "xx-large"
            cpc.title = "title"
            cpc.tight = False
            cpc.legend = False
            cpc.legendLoc = "upper left"
            cpc.legendNcols = 4
            cpc.legendTextSize = "xx-large"
            cpc.figsize = [1, 1]
            with patch(
                "parthenopegui.plotter.PlotSettings.clearCurrent", autospec=True
            ) as _cc:
                cpc.clearCurrent()
                _cc.assert_called_once_with(self.ps)
            self.assertEqual(cpc.lines, [])
            self.assertEqual(cpc.contours, [])
            self.assertEqual(cpc.axesTextSize, self.ps._possibleTextSizesDefault)
            self.assertEqual(cpc.title, "")
            self.assertTrue(cpc.tight)
            self.assertTrue(cpc.legend)
            self.assertEqual(cpc.legendLoc, self.ps._possibleLegendLoc[0])
            self.assertEqual(cpc.legendNcols, 1)
            self.assertEqual(cpc.legendTextSize, self.ps._possibleTextSizesDefault)
            self.assertEqual(cpc.figsize, None)

        def test_readSettings(self):
            """test the readSettings method"""
            cpc = CurrentPlotContent(self.ps)
            self.assertEqual(cpc.axesTextSize, self.ps._possibleTextSizesDefault)
            self.assertEqual(cpc.title, "")
            self.assertTrue(cpc.tight)
            self.assertTrue(cpc.legend)
            self.assertEqual(cpc.legendLoc, self.ps._possibleLegendLoc[0])
            self.assertEqual(cpc.legendNcols, 1)
            self.assertEqual(cpc.legendTextSize, self.ps._possibleTextSizesDefault)
            self.assertEqual(cpc.figsize, None)
            self.ps.axesTextSize = "large"
            self.ps.figsize = (4, 4)
            self.ps.legend = False
            self.ps.legendLoc = "upper left"
            self.ps.legendNcols = 4
            self.ps.legendTextSize = "xx-large"
            self.ps.tight = False
            self.ps.title = "title"
            cpc.readSettings()
            self.assertEqual(cpc.axesTextSize, "large")
            self.assertEqual(cpc.title, "title")
            self.assertFalse(cpc.tight)
            self.assertFalse(cpc.legend)
            self.assertEqual(cpc.legendLoc, "upper left")
            self.assertEqual(cpc.legendNcols, 4)
            self.assertEqual(cpc.legendTextSize, "xx-large")
            self.assertEqual(cpc.figsize, (4, 4))

        def test_doPlot(self):
            """test the doPlot method"""
            cpc = CurrentPlotContent(self.ps)
            cpc.lines.append(
                PGLine(
                    1,
                    1,
                    0,
                    c="r",
                    d="desc",
                    l="first line",
                    m=".",
                    s=":",
                    w=2.0,
                    xl="xlab1",
                    yl="ylab1",
                )
            )
            cpc.lines.append(
                PGLine(
                    2,
                    2,
                    0,
                    c="b",
                    c2="chi2",
                    d="ription",
                    l="second line",
                    m="o",
                    s="--",
                    w=0.5,
                    xl="xlab2",
                    yl="ylab2",
                )
            )
            cpc.pSett.axesTextSize = "small"
            cpc.pSett.figsize = [6, 6]
            cpc.pSett.legend = True
            cpc.pSett.legendLoc = "center"
            cpc.pSett.legendNcols = 2
            cpc.pSett.legendTextSize = "large"
            cpc.pSett.tight = True
            cpc.pSett.title = "abc"
            cpc.pSett.xlabel = "xl"
            cpc.pSett.ylabel = "yl"
            cpc.pSett.xscale = "log"
            cpc.pSett.yscale = "log"
            cpc.pSett.xlims = (1, 10)
            cpc.pSett.ylims = (1, 10)
            fig = plt.figure()
            with patch(
                "matplotlib.pyplot.figure", autospec=True, return_value=fig
            ) as _fig, patch("matplotlib.pyplot.plot", autospec=True) as _plot, patch(
                "matplotlib.pyplot.contour", autospec=True
            ) as _ct, patch(
                "matplotlib.pyplot.contourf", autospec=True
            ) as _cf, patch(
                "matplotlib.pyplot.colorbar", autospec=True
            ) as _cb, patch(
                "matplotlib.pyplot.xlabel", autospec=True
            ) as _xl, patch(
                "matplotlib.pyplot.ylabel", autospec=True
            ) as _yl, patch(
                "matplotlib.pyplot.xscale", autospec=True
            ) as _xs, patch(
                "matplotlib.pyplot.yscale", autospec=True
            ) as _ys, patch(
                "matplotlib.pyplot.xlim", autospec=True
            ) as _xm, patch(
                "matplotlib.pyplot.ylim", autospec=True
            ) as _ym, patch(
                "matplotlib.pyplot.title", autospec=True
            ) as _pt, patch(
                "matplotlib.pyplot.tight_layout", autospec=True
            ) as _tl, patch(
                "matplotlib.pyplot.legend", autospec=True
            ) as _le, patch(
                "matplotlib.pyplot.savefig", autospec=True
            ) as _sa, patch(
                "matplotlib.pyplot.close", autospec=True
            ) as _cl:
                self.assertEqual(cpc.doPlot(), fig)
                _fig.assert_called_once_with(figsize=(6.0, 6.0))
                _plot.assert_has_calls(
                    [
                        call(np.nan, np.nan),
                        call(
                            1,
                            1,
                            color="r",
                            label="first line",
                            ls=":",
                            lw=2.0,
                            marker=".",
                        ),
                        call(
                            2,
                            2,
                            color="b",
                            label="second line",
                            ls="--",
                            lw=0.5,
                            marker="o",
                        ),
                    ]
                )
                self.assertEqual(_ct.call_count, 0)
                self.assertEqual(_cf.call_count, 0)
                self.assertEqual(_cb.call_count, 0)
                _xl.assert_called_once_with("xl", fontsize="small")
                _yl.assert_called_once_with("yl", fontsize="small")
                _xs.assert_called_once_with("log")
                _ys.assert_called_once_with("log")
                _xm.assert_called_once_with((1.0, 10.0))
                _ym.assert_called_once_with((1.0, 10.0))
                _pt.assert_called_once_with("abc")
                _tl.assert_called_once_with()
                _le.assert_called_once_with(loc="center", ncol=2, fontsize="large")
                self.assertEqual(_sa.call_count, 0)
                _cl.assert_called_once_with()

            cpc.clearCurrent()
            cpc.contours.append(
                PGContour(
                    1,
                    1,
                    1,
                    0,
                    cm="winter",
                    d="desc",
                    f=False,
                    hcb=False,
                    l="leg",
                    xl="xl1",
                    yl="yl1",
                    zl="zl1",
                )
            )
            cm1 = matplotlib.cm.get_cmap("winter")
            cpc.contours.append(
                PGContour(
                    2,
                    2,
                    2,
                    0,
                    c2="chi2",
                    cm="rainbow",
                    d="ription",
                    ex="both",
                    l="end",
                    lvs=[0, 1, 2],
                    xl="xl2",
                    yl="yl2",
                    zl="zl2",
                )
            )
            cpc.pSett.axesTextSize = "medium"
            cm2 = matplotlib.cm.get_cmap("rainbow")
            cpc.pSett.legend = False
            cpc.pSett.tight = False
            cpc.pSett.xlabel = "xl"
            cpc.pSett.ylabel = "yl"
            cpc.pSett.xscale = "linear"
            cpc.pSett.yscale = "linear"
            cf = plt.contourf([0, 1], [0, 1], [[0, 0.25], [0.25, 0.5]])
            cb = plt.colorbar(cf)
            with patch(
                "matplotlib.pyplot.figure", autospec=True, return_value=fig
            ) as _fig, patch("matplotlib.pyplot.plot", autospec=True) as _plot, patch(
                "matplotlib.pyplot.contour", autospec=True
            ) as _ct, patch(
                "matplotlib.pyplot.contourf", autospec=True
            ) as _cf, patch(
                "matplotlib.pyplot.colorbar", autospec=True, return_value=cb
            ) as _cb, patch(
                "matplotlib.cm.get_cmap", autospec=True, side_effect=[cm1, cm2]
            ) as _cm, patch(
                "matplotlib.axes.Axes.set_ylabel", autospec=True
            ) as _zl, patch(
                "matplotlib.pyplot.xlabel", autospec=True
            ) as _xl, patch(
                "matplotlib.pyplot.ylabel", autospec=True
            ) as _yl, patch(
                "matplotlib.pyplot.xscale", autospec=True
            ) as _xs, patch(
                "matplotlib.pyplot.yscale", autospec=True
            ) as _ys, patch(
                "matplotlib.pyplot.xlim", autospec=True
            ) as _xm, patch(
                "matplotlib.pyplot.ylim", autospec=True
            ) as _ym, patch(
                "matplotlib.pyplot.title", autospec=True
            ) as _pt, patch(
                "matplotlib.pyplot.tight_layout", autospec=True
            ) as _tl, patch(
                "matplotlib.pyplot.legend", autospec=True
            ) as _le, patch(
                "matplotlib.pyplot.savefig", autospec=True
            ) as _sa, patch(
                "matplotlib.pyplot.close", autospec=True
            ) as _cl:
                self.assertEqual(cpc.doPlot(123), fig)
                _fig.assert_called_once_with(figsize=(5.0, 5.0))
                _plot.assert_called_once_with(np.nan, np.nan)
                _ct.assert_called_once_with(1, 1, 1, cmap=cm1)
                _cf.assert_called_once_with(
                    2, 2, 2, cmap=cm2, extend="both", levels=[0, 1, 2]
                )
                _cb.assert_called_once_with(_cf())
                _zl.assert_called_once_with(cb.ax, "zl2", fontsize="medium")
                _xl.assert_called_once_with("xl", fontsize="medium")
                _yl.assert_called_once_with("yl", fontsize="medium")
                _xs.assert_called_once_with("linear")
                _ys.assert_called_once_with("linear")
                _xm.assert_called_once_with((0.0, 1.0))
                _ym.assert_called_once_with((0.0, 1.0))
                self.assertEqual(_pt.call_count, 0)
                self.assertEqual(_tl.call_count, 0)
                self.assertEqual(_le.call_count, 0)
                self.assertEqual(_sa.call_count, 0)
            cpc.pSett.zlabel = "myzl"
            with patch(
                "matplotlib.pyplot.figure", autospec=True, return_value=fig
            ) as _fig, patch("matplotlib.pyplot.plot", autospec=True) as _plot, patch(
                "matplotlib.pyplot.contour", autospec=True
            ) as _ct, patch(
                "matplotlib.pyplot.contourf", autospec=True
            ) as _cf, patch(
                "matplotlib.pyplot.colorbar", autospec=True, return_value=cb
            ) as _cb, patch(
                "matplotlib.cm.get_cmap", autospec=True, side_effect=[cm1, cm2]
            ) as _cm, patch(
                "matplotlib.axes.Axes.set_ylabel", autospec=True
            ) as _zl, patch(
                "matplotlib.pyplot.xlabel", autospec=True
            ) as _xl, patch(
                "matplotlib.pyplot.ylabel", autospec=True
            ) as _yl, patch(
                "matplotlib.pyplot.xscale", autospec=True
            ) as _xs, patch(
                "matplotlib.pyplot.yscale", autospec=True
            ) as _ys, patch(
                "matplotlib.pyplot.xlim", autospec=True
            ) as _xm, patch(
                "matplotlib.pyplot.ylim", autospec=True
            ) as _ym, patch(
                "matplotlib.pyplot.title", autospec=True
            ) as _pt, patch(
                "matplotlib.pyplot.tight_layout", autospec=True
            ) as _tl, patch(
                "matplotlib.pyplot.legend", autospec=True
            ) as _le, patch(
                "matplotlib.pyplot.text", autospec=True
            ) as _tx, patch(
                "matplotlib.pyplot.savefig", autospec=True
            ) as _sa, patch(
                "matplotlib.pyplot.close", autospec=True
            ) as _cl:
                self.assertEqual(cpc.doPlot("savepath.png"), fig)
                _zl.assert_called_once_with(cb.ax, "myzl", fontsize="medium")
                _sa.assert_called_once_with("savepath.png")
                _tx.assert_called_once_with(
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
            plt.close()

        def test_doPlot_interp_line(self):
            """test the doPlot method when failed points are present in lines"""
            cpc = CurrentPlotContent(self.ps)
            cpc.lines.append(
                PGLine(
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    [np.nan, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                    0,
                )
            )
            cpc.lines.append(
                PGLine(
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    [10, 11, 12, np.nan, 13, 15, 16, 17, 18, 19, 20],
                    0,
                )
            )
            cpc.lines.append(
                PGLine(
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, np.nan],
                    0,
                )
            )
            cpc.lines.append(
                PGLine(
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    [np.nan, np.nan, 10, 11, 12, 15, 16, 17, np.nan, np.nan, np.nan],
                    0,
                )
            )
            cpc.lines.append(
                PGLine(
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                    [10, 11, 12, np.nan, np.nan, 15, 16, 17, 18, 19, 20, np.nan],
                    0,
                )
            )
            fig = plt.figure()
            with patch(
                "matplotlib.pyplot.figure", autospec=True, return_value=fig
            ) as _fig, patch("matplotlib.pyplot.plot", autospec=True) as _plot, patch(
                "matplotlib.pyplot.xlabel", autospec=True
            ) as _xl, patch(
                "matplotlib.pyplot.ylabel", autospec=True
            ) as _yl, patch(
                "matplotlib.pyplot.xscale", autospec=True
            ) as _xs, patch(
                "matplotlib.pyplot.yscale", autospec=True
            ) as _ys, patch(
                "matplotlib.pyplot.xlim", autospec=True
            ) as _xm, patch(
                "matplotlib.pyplot.ylim", autospec=True
            ) as _ym, patch(
                "matplotlib.pyplot.title", autospec=True
            ) as _pt, patch(
                "matplotlib.pyplot.tight_layout", autospec=True
            ) as _tl, patch(
                "matplotlib.pyplot.legend", autospec=True
            ) as _le, patch(
                "matplotlib.pyplot.savefig", autospec=True
            ) as _sa, patch(
                "matplotlib.pyplot.close", autospec=True
            ) as _cl, patch(
                "logging.Logger.warning"
            ) as _w:
                self.assertEqual(cpc.doPlot(), fig)
                self.assertEqualArray(
                    _plot.call_args_list[1][0][0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                )
                self.assertEqualArray(
                    _plot.call_args_list[1][0][1],
                    [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                )
                self.assertEqualArray(
                    _plot.call_args_list[2][0][0], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                )
                self.assertEqualArray(
                    _plot.call_args_list[2][0][1],
                    [10, 11, 12, 12.5, 13, 15, 16, 17, 18, 19, 20],
                )
                self.assertEqualArray(
                    _plot.call_args_list[3][0][0], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                )
                self.assertEqualArray(
                    _plot.call_args_list[3][0][1],
                    [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                )
                self.assertEqualArray(_plot.call_args_list[4][0][0], [2, 3, 4, 5, 6, 7])
                self.assertEqualArray(
                    _plot.call_args_list[4][0][1],
                    [10, 11, 12, 15, 16, 17],
                )
                self.assertEqualArray(
                    _plot.call_args_list[5][0][0], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                )
                self.assertEqualArray(
                    _plot.call_args_list[5][0][1],
                    [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                )
                _w.assert_has_calls(
                    [
                        call(
                            PGText.interpolationPerformedLine
                            % (
                                (
                                    np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                                    np.array(
                                        [
                                            np.nan,
                                            11.0,
                                            12.0,
                                            13.0,
                                            14.0,
                                            15.0,
                                            16.0,
                                            17.0,
                                            18.0,
                                            19.0,
                                            20.0,
                                        ]
                                    ),
                                ),
                                (
                                    np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                                    np.array(
                                        [
                                            11.0,
                                            12.0,
                                            13.0,
                                            14.0,
                                            15.0,
                                            16.0,
                                            17.0,
                                            18.0,
                                            19.0,
                                            20.0,
                                        ]
                                    ),
                                ),
                            )
                        ),
                        call(
                            PGText.interpolationPerformedLine
                            % (
                                (
                                    np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                                    np.array(
                                        [10, 11, 12, np.nan, 13, 15, 16, 17, 18, 19, 20]
                                    ),
                                ),
                                (
                                    np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                                    np.array(
                                        [
                                            10.0,
                                            11.0,
                                            12.0,
                                            12.5,
                                            13.0,
                                            15.0,
                                            16.0,
                                            17.0,
                                            18.0,
                                            19.0,
                                            20.0,
                                        ]
                                    ),
                                ),
                            )
                        ),
                        call(
                            PGText.interpolationPerformedLine
                            % (
                                (
                                    np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                                    np.array(
                                        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, np.nan]
                                    ),
                                ),
                                (
                                    np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                                    np.array(
                                        [
                                            10.0,
                                            11.0,
                                            12.0,
                                            13.0,
                                            14.0,
                                            15.0,
                                            16.0,
                                            17.0,
                                            18.0,
                                            19.0,
                                        ]
                                    ),
                                ),
                            )
                        ),
                        call(
                            PGText.interpolationPerformedLine
                            % (
                                (
                                    np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                                    np.array(
                                        [
                                            np.nan,
                                            np.nan,
                                            10,
                                            11,
                                            12,
                                            15,
                                            16,
                                            17,
                                            np.nan,
                                            np.nan,
                                            np.nan,
                                        ]
                                    ),
                                ),
                                (
                                    np.array([2, 3, 4, 5, 6, 7]),
                                    np.array([10.0, 11.0, 12.0, 15.0, 16.0, 17.0]),
                                ),
                            )
                        ),
                        call(
                            PGText.interpolationPerformedLine
                            % (
                                (
                                    np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
                                    np.array(
                                        [
                                            10.0,
                                            11.0,
                                            12.0,
                                            np.nan,
                                            np.nan,
                                            15.0,
                                            16.0,
                                            17.0,
                                            18.0,
                                            19.0,
                                            20.0,
                                            np.nan,
                                        ]
                                    ),
                                ),
                                (
                                    np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                                    np.array(
                                        [
                                            10.0,
                                            11.0,
                                            12.0,
                                            13.0,
                                            14.0,
                                            15.0,
                                            16.0,
                                            17.0,
                                            18.0,
                                            19.0,
                                            20.0,
                                        ]
                                    ),
                                ),
                            )
                        ),
                    ]
                )

        def test_doPlot_interp_contour(self):
            """test the doPlot method when failed points are present in contours"""
            cpc = CurrentPlotContent(self.ps)
            cpc.contours.append(
                PGContour(
                    [0, 1, 2],
                    [0, 1, 2, 3],
                    [[np.nan, 2, 3], [11, 12, np.nan], [16, 17, np.nan], [31, 32, 33]],
                    0,
                    hcb=False,
                )
            )
            cpc.contours.append(
                PGContour(
                    [0, 1, 2],
                    [0, 1, 2, 3],
                    [[1, np.nan, 3], [np.nan, 12, 13], [11, 22, 23], [31, np.nan, 33]],
                    0,
                    hcb=False,
                )
            )
            cpc.contours.append(
                PGContour(
                    [0, 1, 2],
                    [0, 1, 2, 3],
                    [[1, 2, 3], [11, np.nan, 15], [21, 22, 23], [np.nan, 30, 33]],
                    0,
                    hcb=False,
                )
            )
            cpc.contours.append(
                PGContour(
                    [0, 1, 2],
                    [0, 1, 2, 3],
                    [[1, 2, 3], [9, np.nan, 13], [21, np.nan, 25], [31, 32, 33]],
                    0,
                    hcb=False,
                )
            )
            cpc.contours.append(
                PGContour(
                    [0, 1, 2],
                    [0, 1, 2, 3],
                    [
                        [1, 5, np.nan],
                        [np.nan, 12, 13],
                        [np.nan, 22, 23],
                        [16, 32, np.nan],
                    ],
                    0,
                    hcb=False,
                )
            )
            fig = plt.figure()
            with patch(
                "matplotlib.pyplot.figure", autospec=True, return_value=fig
            ) as _fig, patch(
                "matplotlib.pyplot.contourf", autospec=True
            ) as _plot, patch(
                "matplotlib.pyplot.xlabel", autospec=True
            ) as _xl, patch(
                "matplotlib.pyplot.ylabel", autospec=True
            ) as _yl, patch(
                "matplotlib.pyplot.xscale", autospec=True
            ) as _xs, patch(
                "matplotlib.pyplot.yscale", autospec=True
            ) as _ys, patch(
                "matplotlib.pyplot.xlim", autospec=True
            ) as _xm, patch(
                "matplotlib.pyplot.ylim", autospec=True
            ) as _ym, patch(
                "matplotlib.pyplot.title", autospec=True
            ) as _pt, patch(
                "matplotlib.pyplot.tight_layout", autospec=True
            ) as _tl, patch(
                "matplotlib.pyplot.legend", autospec=True
            ) as _le, patch(
                "matplotlib.pyplot.savefig", autospec=True
            ) as _sa, patch(
                "matplotlib.pyplot.close", autospec=True
            ) as _cl, patch(
                "logging.Logger.warning"
            ) as _w:
                self.assertEqual(cpc.doPlot(), fig)
                self.assertEqualArray(_plot.call_args_list[0][0][0], [0, 1, 2])
                self.assertEqualArray(
                    _plot.call_args_list[0][0][1],
                    [0, 1, 2, 3],
                )
                self.assertEqualArray(
                    _plot.call_args_list[0][0][2],
                    [[3.5, 2, 3], [11, 12, 13], [16, 17, 23], [31, 32, 33]],
                )
                _w.assert_any_call(
                    PGText.interpolationPerformedContour
                    % (
                        np.array(
                            [
                                [np.nan, 2.0, 3.0],
                                [11.0, 12.0, np.nan],
                                [16.0, 17.0, np.nan],
                                [31.0, 32.0, 33.0],
                            ]
                        ),
                        np.array(
                            [[3.5, 2, 3], [11, 12, 13], [16, 17, 23], [31, 32, 33]]
                        ),
                    )
                )
                self.assertEqualArray(
                    _plot.call_args_list[1][0][2],
                    [[1, 2, 3], [6, 12, 13], [11, 22, 23], [31, 32, 33]],
                )
                _w.assert_any_call(
                    PGText.interpolationPerformedContour
                    % (
                        np.array(
                            [
                                [1, np.nan, 3],
                                [np.nan, 12, 13],
                                [11, 22, 23],
                                [31, np.nan, 33],
                            ]
                        ),
                        np.array(
                            [[1.0, 2, 3], [6, 12, 13], [11, 22, 23], [31, 32, 33]]
                        ),
                    )
                )
                self.assertEqualArray(
                    _plot.call_args_list[2][0][2],
                    [[1, 2, 3], [11, 12.5, 15], [21, 22, 23], [29, 30, 33]],
                )
                _w.assert_any_call(
                    PGText.interpolationPerformedContour
                    % (
                        np.array(
                            [
                                [1, 2, 3],
                                [11, np.nan, 15],
                                [21, 22, 23],
                                [np.nan, 30, 33],
                            ]
                        ),
                        np.array(
                            [[1, 2, 3], [11, 12.5, 15], [21, 22, 23], [29, 30, 33]]
                        ),
                    )
                )
                self.assertEqualArray(
                    _plot.call_args_list[3][0][2],
                    [[1, 2, 3], [9, 11.5, 13], [21, 22.375, 25], [31, 32, 33]],
                )
                _w.assert_any_call(
                    PGText.interpolationPerformedContour
                    % (
                        np.array(
                            [[1, 2, 3], [9, np.nan, 13], [21, np.nan, 25], [31, 32, 33]]
                        ),
                        np.array(
                            [[1, 2, 3], [9, 11.5, 13], [21, 22.375, 25], [31, 32, 33]]
                        ),
                    )
                )
                self.assertEqualArray(
                    _plot.call_args_list[4][0][2],
                    [[1, 5, 6], [6, 12, 13], [11, 22, 23], [16, 32, 40.5]],
                )
                _w.assert_any_call(
                    PGText.interpolationPerformedContour
                    % (
                        np.array(
                            [
                                [1, 5, np.nan],
                                [np.nan, 12, 13],
                                [np.nan, 22, 23],
                                [16, 32, np.nan],
                            ]
                        ),
                        np.array(
                            [[1, 5, 6], [6, 12, 13], [11, 22, 23], [16, 32, 40.5]]
                        ),
                    )
                )

    class TestGridLoader(PGTestCasewMainW):
        """Test the GridLoader class"""

        def test_init(self):
            """test the init method"""
            gl = GridLoader(parent=self.mainW.plotPanel, mainW=self.mainW)
            self.assertIsInstance(gl.reloadedGrid, Signal)
            self.assertEqual(gl.mainW, self.mainW)
            self.assertTrue(hasattr(gl, "runner"))
            self.assertIsInstance(gl.layout(), QVBoxLayout)
            self.assertIsInstance(gl.layout().itemAt(0).widget(), PGLabel)
            self.assertEqual(
                gl.layout().itemAt(0).widget().text(), PGText.plotSelectGridText
            )
            self.assertIsInstance(gl.openGrid, PGLabelButton)
            self.assertEqual(gl.layout().itemAt(1).widget(), gl.openGrid)
            with patch(
                "parthenopegui.plotter.GridLoader.loadGrid", autospec=True
            ) as _o:
                QTest.mouseClick(gl.openGrid, Qt.LeftButton)
                _o.assert_called_once_with(gl)
            self.assertIsInstance(gl.groupBox, QGroupBox)
            self.assertIsInstance(gl.layout().itemAt(2).widget(), PGLabel)
            self.assertEqual(gl.layout().itemAt(2).widget().text(), "")
            self.assertEqual(gl.layout().itemAt(3).widget(), gl.groupBox)
            self.assertIsInstance(gl.groupBox.layout(), QHBoxLayout)
            for i, (attr, [title, desc]) in enumerate(
                PGText.plotTypeDescription.items()
            ):
                self.assertIsInstance(getattr(gl, attr), QRadioButton)
                self.assertEqual(getattr(gl, attr).text(), title)
                self.assertEqual(getattr(gl, attr).toolTip(), desc)
                self.assertEqual(
                    gl.groupBox.layout().itemAt(i).widget(), getattr(gl, attr)
                )
                with patch(
                    "parthenopegui.plotter.MWPlotPanel.updatePlotTypePanel",
                    autospec=True,
                ) as _u:
                    QTest.mouseClick(getattr(gl, attr), Qt.LeftButton)
                    if i == 0:
                        _u.assert_called_once_with(self.mainW.plotPanel, True)
                    else:
                        _u.assert_has_calls(
                            [
                                call(self.mainW.plotPanel, False),
                                call(self.mainW.plotPanel, True),
                            ]
                        )

        def test_loadGrid(self):
            """test the loadGrid method"""
            gl = GridLoader(parent=self.mainW.plotPanel, mainW=self.mainW)
            with patch(
                "parthenopegui.plotter.askDirName",
                autospec=True,
                side_effect=["", "/nonexistent/parthenope/directory"],
            ) as _adn, patch("logging.Logger.error") as _e:
                gl.loadGrid()
                _adn.assert_called_once_with(gl, PGText.loadGridAsk)
                _e.assert_called_once_with(PGText.errorCannotFindGrid)
            with patch(
                "parthenopegui.plotter.askDirName",
                autospec=True,
                return_value="/nonexistent/parthenope/directory",
            ) as _adn, patch("logging.Logger.error") as _e, patch(
                "os.path.isfile", return_value=True
            ) as _i:
                gl.loadGrid()
                _e.assert_called_once_with(PGText.errorCannotLoadGrid)
            with patch(
                "parthenopegui.plotter.askDirName",
                autospec=True,
                return_value=testConfig.testRunDirEmpty,
            ) as _adn, patch("logging.Logger.error") as _e, patch(
                "os.path.isfile", return_value=True
            ) as _i:
                gl.loadGrid()
                _e.assert_called_once_with(PGText.errorCannotLoadGrid)
            with patch(
                "parthenopegui.plotter.askDirName",
                autospec=True,
                return_value=testConfig.testRunDirEmpty,
            ) as _adn, patch("logging.Logger.error") as _e, patch(
                "os.path.isfile", return_value=True
            ) as _i, patch(
                "pickle.load", side_effect=ValueError
            ) as _l:
                gl.loadGrid()
                _e.assert_called_once_with(PGText.errorCannotLoadGrid)
                self.assertEqual(_l.call_count, 1)
            rel = MagicMock()
            rp = RunPArthENoPE(
                testConfig.commonParamsDefault, testConfig.gridPointsDefault, self.mainW
            )
            rp.readAllResults = MagicMock()
            self.mainW.runSettingsTab.outputParams.outputFolderName.setText(
                testConfig.testRunDirDefaultSample
            )
            with patch(
                "parthenopegui.plotter.askDirName",
                autospec=True,
                return_value=testConfig.testRunDirDefaultSample,
            ) as _adn, patch("logging.Logger.error") as _e, patch(
                "parthenopegui.plotter.RunPArthENoPE",
                return_value=rp,
                autospec=USE_AUTOSPEC_CLASS,
            ) as _rp:
                gl.reloadedGrid.connect(rel)
                gl.loadGrid()
                self.assertEqual(_e.call_count, 0)
                self.assertEqual(gl.openGrid.text(), testConfig.testRunDirDefaultSample)
                self.assertEqual(_rp.call_count, 1)
                self.assertEqual(_rp.call_args[0][0], testConfig.commonParamsDefault)
                self.assertEqualArray(_rp.call_args[0][1], testConfig.gridPointsDefault)
                self.assertEqual(_rp.call_args[1]["parent"], self.mainW)
                self.assertEqual(gl.runner, rp)
                rp.readAllResults.assert_called_once_with()
                rel.assert_called_once_with()

    class TestShowPlotWidget(PGTestCasewMainW):
        """Test the ShowPlotWidget class"""

        def test_init(self):
            """test the init method"""
            with patch(
                "parthenopegui.plotter.ShowPlotWidget.updatePlots", autospec=True
            ) as _up:
                spw = ShowPlotWidget(self.mainW, self.mainW)
                _up.assert_called_once_with(spw)
            self.assertEqual(spw.mainW, self.mainW)
            self.assertEqual(spw.fig, None)
            self.assertEqual(spw.canvas, None)
            self.assertIsInstance(spw.layout(), QGridLayout)

        def test_updatePlots(self):
            """test the updatePlots method"""
            spw = ShowPlotWidget(self.mainW, self.mainW)
            fig = plt.figure()
            plt.plot(np.nan, np.nan)
            plt.close()
            canvas = FigureCanvas(fig)
            canvas.draw = MagicMock()
            spw.fig = None
            spw.canvas = None
            with patch("matplotlib.pyplot.figure", return_value=fig) as _fi, patch(
                "parthenopegui.plotter.FigureCanvas", return_value=canvas
            ) as _ca:
                spw.updatePlots()
                self.assertGreater(_fi.call_count, 0)
                self.assertEqual(spw.fig, fig)
                self.assertEqual(spw.canvas, canvas)
                _ca.assert_called_once_with(fig)
                canvas.draw.assert_called_once_with()
                self.assertEqual(spw.layout().itemAtPosition(0, 0).widget(), spw.canvas)
            spw.fig = None
            spw.canvas = None
            with patch("matplotlib.pyplot.figure", return_value=fig) as _fi:
                spw.updatePlots(fig)
                self.assertEqual(_fi.call_count, 0)
                self.assertEqual(spw.fig, fig)
                self.assertIsInstance(spw.canvas, FigureCanvas)
                self.assertNotEqual(spw.canvas, canvas)
                self.assertEqual(spw.layout().itemAtPosition(0, 0).widget(), spw.canvas)

        def test_saveAction(self):
            """test the saveAction method"""
            spw = ShowPlotWidget(self.mainW, self.mainW)
            fig = plt.figure()
            plt.plot(np.nan, np.nan)
            plt.close()
            with patch(
                "parthenopegui.plotter.askSaveFileName", autospec=True, return_value=""
            ) as _as, patch(
                "parthenopegui.plotter.ShowPlotWidget.updatePlots", autospec=True
            ) as _up:
                spw.saveAction()
                _as.assert_called_once_with(spw, PGText.plotWhereToSave)
                self.assertEqual(_up.call_count, 0)
            with patch(
                "parthenopegui.plotter.askSaveFileName",
                autospec=True,
                return_value="abc",
            ) as _as, patch(
                "parthenopegui.plotter.ShowPlotWidget.updatePlots",
                autospec=True,
                side_effect=AttributeError,
            ) as _up, patch(
                "parthenopegui.plotter.CurrentPlotContent.doPlot",
                autospec=True,
                return_value=fig,
            ) as _dp, patch(
                "logging.Logger.warning"
            ) as _w:
                spw.saveAction()
                _as.assert_called_once_with(spw, PGText.plotWhereToSave)
                _dp.assert_called_once_with(
                    self.mainW.plotPanel.currentPlotContent, savefig="abc"
                )
                _up.assert_called_once_with(spw, fig=fig)
                _w.assert_called_once_with("", exc_info=True)
            with patch(
                "parthenopegui.plotter.askSaveFileName",
                autospec=True,
                return_value="abc",
            ) as _as, patch(
                "parthenopegui.plotter.ShowPlotWidget.updatePlots", autospec=True
            ) as _up, patch(
                "parthenopegui.plotter.CurrentPlotContent.doPlot",
                autospec=True,
                return_value=fig,
            ) as _dp, patch(
                "logging.Logger.info"
            ) as _m:
                spw.saveAction()
                _as.assert_called_once_with(spw, PGText.plotWhereToSave)
                _dp.assert_called_once_with(
                    self.mainW.plotPanel.currentPlotContent, savefig="abc"
                )
                _up.assert_called_once_with(spw, fig=fig)
                _m.assert_called_once_with(PGText.plotSaved)

    class TestPlotSettings(PGTestCasewMainW):
        """Test the PlotSettings class"""

        def test_init(self):
            """test init"""
            with patch(
                "parthenopegui.plotter.PlotSettings._doAxesTab", autospec=True
            ) as _da, patch(
                "parthenopegui.plotter.PlotSettings._doLegendTab", autospec=True
            ) as _dl, patch(
                "parthenopegui.plotter.PlotSettings._doFigureTab", autospec=True
            ) as _df, patch(
                "parthenopegui.plotter.PlotSettings._doButtons", autospec=True
            ) as _db:
                ps = PlotSettings(parent=self.mainW)
                _da.assert_called_once_with(ps)
                _dl.assert_called_once_with(ps)
                _df.assert_called_once_with(ps)
                _db.assert_called_once_with(ps)
            self.assertIsInstance(ps._scales, list)
            self.assertIsInstance(ps._possibleLegendLoc, list)
            self.assertIsInstance(ps._possibleLegendMaxNcols, int)
            self.assertIsInstance(ps._possibleTextSizes, list)
            self.assertIn(ps._possibleTextSizesDefault, ps._possibleTextSizes)
            for p in (
                "_default_xlimsd",
                "_default_xlimsu",
                "_default_ylimsd",
                "_default_ylimsu",
                "_default_figsizex",
                "_default_figsizey",
            ):
                self.assertIsInstance(getattr(ps, p), float)
            self.assertIsInstance(ps.automaticSettings, dict)
            self.assertIsInstance(ps.layout(), QVBoxLayout)
            self.assertTrue(hasattr(ps, "tabWidget"))
            self.assertIsInstance(ps.tabWidget, QTabWidget)
            self.assertFalse(ps.tabWidget.tabsClosable())
            self.assertEqual(ps.layout().itemAt(0).widget(), ps.tabWidget)

        def test_doAxesTab(self):
            """test _doAxesTab"""
            with patch(
                "parthenopegui.plotter.PlotSettings._doAxesTab", autospec=True
            ) as _f:
                ps = PlotSettings(parent=self.mainW)
                _f.assert_called_once_with(ps)
            self.assertFalse(hasattr(ps, "axesTab"))
            ps._doAxesTab()
            self.assertTrue(hasattr(ps, "axesTab"))
            self.assertIsInstance(ps.axesTab, QWidget)
            self.assertIsInstance(ps.axesTab.layout(), QGridLayout)
            ps = PlotSettings(parent=self.mainW)
            self.assertEqual(ps.tabWidget.widget(0), ps.axesTab)
            lay = ps.axesTab.layout()

            # first row: xlabel, ylabel
            r = 0
            self.assertIsInstance(lay.itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                lay.itemAtPosition(r, 0).widget().text(), PGText.plotSettXLab
            )
            self.assertIsInstance(ps.xlabelEd, QLineEdit)
            self.assertEqual(ps.xlabelEd.text(), "")
            self.assertEqual(lay.itemAtPosition(r, 1).widget(), ps.xlabelEd)
            self.assertIsInstance(lay.itemAtPosition(r, 4).widget(), PGLabel)
            self.assertEqual(
                lay.itemAtPosition(r, 4).widget().text(), PGText.plotSettYLab
            )
            self.assertIsInstance(ps.ylabelEd, QLineEdit)
            self.assertEqual(ps.ylabelEd.text(), "")
            self.assertEqual(lay.itemAtPosition(r, 5).widget(), ps.ylabelEd)

            # second row: zlabel, title
            r += 1
            self.assertIsInstance(lay.itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                lay.itemAtPosition(r, 0).widget().text(), PGText.plotSettZLab
            )
            self.assertIsInstance(ps.zlabelEd, QLineEdit)
            self.assertEqual(ps.zlabelEd.text(), "")
            self.assertEqual(lay.itemAtPosition(r, 1).widget(), ps.zlabelEd)

            # third row: xscale, yscale
            r += 1
            self.assertIsInstance(lay.itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                lay.itemAtPosition(r, 0).widget().text(), PGText.plotSettXScale
            )
            self.assertIsInstance(lay.itemAtPosition(r, 4).widget(), PGLabel)
            self.assertEqual(
                lay.itemAtPosition(r, 4).widget().text(), PGText.plotSettYScale
            )
            for i, c in enumerate(["xscaleCombo", "yscaleCombo"]):
                self.assertEqual(
                    getattr(ps, c), lay.itemAtPosition(r, 1 + i * 4).widget()
                )
                self.assertIsInstance(getattr(ps, c), QComboBox)
                self.assertEqual(getattr(ps, c).count(), len(ps._scales))
                self.assertEqual(getattr(ps, c).currentText(), ps._scales[0])
                for j, s in enumerate(ps._scales):
                    self.assertEqual(getattr(ps, c).itemText(j), s)

            # fourth row: xlims, ylims
            r += 1
            for i, l in enumerate(
                [
                    PGText.plotSettXLims,
                    PGText.smallAsciiArrow,
                    PGText.plotSettYLims,
                    PGText.smallAsciiArrow,
                ]
            ):
                self.assertIsInstance(lay.itemAtPosition(r, i * 2).widget(), PGLabel)
                self.assertEqual(lay.itemAtPosition(r, i * 2).widget().text(), l)
            for i, c in enumerate(["xlimsLow", "xlimsUpp", "ylimsLow", "ylimsUpp"]):
                self.assertIsInstance(getattr(ps, c), QLineEdit)
                self.assertEqual(getattr(ps, c).text(), "")
                self.assertEqual(
                    lay.itemAtPosition(r, 1 + i * 2).widget(), getattr(ps, c)
                )

            # text size
            r += 1
            self.assertIsInstance(lay.itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                lay.itemAtPosition(r, 0).widget().text(),
                PGText.plotSettAxesTextSize,
            )
            self.assertEqual(ps.axesTextSizeCombo, lay.itemAtPosition(r, 1).widget())
            self.assertIsInstance(ps.axesTextSizeCombo, QComboBox)
            self.assertEqual(ps.axesTextSizeCombo.count(), len(ps._possibleTextSizes))
            self.assertEqual(
                ps.axesTextSizeCombo.currentText(), ps._possibleTextSizesDefault
            )
            for j, s in enumerate(ps._possibleTextSizes):
                self.assertEqual(ps.axesTextSizeCombo.itemText(j), s)

        def test_doButtons(self):
            """test _doButtons"""
            with patch(
                "parthenopegui.plotter.PlotSettings._doButtons", autospec=True
            ) as _f:
                ps = PlotSettings(parent=self.mainW)
                _f.assert_called_once_with(ps)
            self.assertFalse(hasattr(ps, "_buttonsA"))
            ps._doButtons()
            for j, [o, l, b] in enumerate(
                [
                    [
                        "_buttonsA",
                        "_buttonsALayout",
                        [
                            ["refreshButton", PGText.plotSettRefreshImage],
                            ["revertButton", PGText.plotSettRevertImage],
                            ["resetButton", PGText.plotSettResetImage],
                        ],
                    ],
                    [
                        "_buttonsB",
                        "_buttonsBLayout",
                        [
                            ["saveButton", PGText.plotSettSaveImage],
                            ["exportButton", PGText.plotSettExportImage],
                        ],
                    ],
                ]
            ):
                self.assertIsInstance(getattr(ps, o), QWidget)
                self.assertEqual(ps.layout().itemAt(1 + j).widget(), getattr(ps, o))
                self.assertIsInstance(getattr(ps, l), QHBoxLayout)
                self.assertEqual(getattr(ps, o).layout(), getattr(ps, l))
                for i, [c, t] in enumerate(b):
                    self.assertIsInstance(getattr(ps, c), PGPushButton)
                    self.assertEqual(getattr(ps, c).text(), t)
                    self.assertEqual(getattr(ps, l).itemAt(i).widget(), getattr(ps, c))

        def test_doFigureTab(self):
            """test _doFigureTab"""
            with patch(
                "parthenopegui.plotter.PlotSettings._doFigureTab", autospec=True
            ) as _f:
                ps = PlotSettings(parent=self.mainW)
                _f.assert_called_once_with(ps)
            self.assertFalse(hasattr(ps, "figureTab"))
            ps._doFigureTab()
            self.assertTrue(hasattr(ps, "figureTab"))
            self.assertIsInstance(ps.figureTab, QWidget)
            self.assertIsInstance(ps.figureTab.layout(), QGridLayout)
            ps = PlotSettings(parent=self.mainW)
            self.assertEqual(ps.tabWidget.widget(2), ps.figureTab)
            lay = ps.figureTab.layout()

            # title
            r = 0
            self.assertIsInstance(lay.itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                lay.itemAtPosition(r, 0).widget().text(),
                PGText.plotSettTitleLab,
            )
            self.assertIsInstance(ps.titleEd, QLineEdit)
            self.assertEqual(ps.titleEd.text(), "")
            self.assertEqual(lay.itemAtPosition(r + 1, 0).widget(), ps.titleEd)

            # figure size
            r += 2
            for i, [l, a, b] in enumerate(
                [
                    [PGText.plotSettFigSize, r, 0],
                    [PGText.comma, r + 1, 1],
                    [PGText.plotSettFigSizeWarning, r + 2, 0],
                ]
            ):
                self.assertIsInstance(lay.itemAtPosition(a, b).widget(), PGLabel)
                self.assertEqual(lay.itemAtPosition(a, b).widget().text(), l)
            r += 1
            for i, c in enumerate(["figsizex", "figsizey"]):
                self.assertIsInstance(getattr(ps, c), QLineEdit)
                self.assertEqual(getattr(ps, c).text(), "")
                self.assertEqual(lay.itemAtPosition(r, i * 2).widget(), getattr(ps, c))

            # tight
            r += 2
            self.assertIsInstance(ps.tightCheck, QCheckBox)
            self.assertEqual(ps.tightCheck, lay.itemAtPosition(r, 0).widget())
            self.assertEqual(ps.tightCheck.text(), PGText.plotSettTight)
            self.assertTrue(ps.tightCheck.isChecked())

        def test_doLegendTab(self):
            """test _doLegendTab"""
            with patch(
                "parthenopegui.plotter.PlotSettings._doLegendTab", autospec=True
            ) as _f:
                ps = PlotSettings(parent=self.mainW)
                _f.assert_called_once_with(ps)
            self.assertFalse(hasattr(ps, "legendTab"))
            ps._doLegendTab()
            self.assertTrue(hasattr(ps, "legendTab"))
            self.assertIsInstance(ps.legendTab, QWidget)
            self.assertIsInstance(ps.legendTab.layout(), QGridLayout)
            ps = PlotSettings(parent=self.mainW)
            self.assertEqual(ps.tabWidget.widget(1), ps.legendTab)
            lay = ps.legendTab.layout()

            # activate legend
            r = 0
            self.assertIsInstance(ps.legendCheck, QCheckBox)
            self.assertEqual(ps.legendCheck, lay.itemAtPosition(r, 0).widget())
            self.assertEqual(ps.legendCheck.text(), PGText.plotSettLegend)
            self.assertTrue(ps.legendCheck.isChecked())

            # legend position
            r += 1
            self.assertIsInstance(lay.itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                lay.itemAtPosition(r, 0).widget().text(),
                PGText.plotSettLegendLoc,
            )
            self.assertEqual(ps.legendLocCombo, lay.itemAtPosition(r, 1).widget())
            self.assertIsInstance(ps.legendLocCombo, QComboBox)
            self.assertEqual(ps.legendLocCombo.count(), len(ps._possibleLegendLoc))
            self.assertEqual(ps.legendLocCombo.currentText(), ps._possibleLegendLoc[0])
            for j, s in enumerate(ps._possibleLegendLoc):
                self.assertEqual(ps.legendLocCombo.itemText(j), s)

            # column number
            r += 1
            self.assertIsInstance(lay.itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                lay.itemAtPosition(r, 0).widget().text(),
                PGText.plotSettLegendNumCols,
            )
            self.assertEqual(ps.legendNumColsCombo, lay.itemAtPosition(r, 1).widget())
            self.assertIsInstance(ps.legendNumColsCombo, QComboBox)
            self.assertEqual(ps.legendNumColsCombo.count(), ps._possibleLegendMaxNcols)
            self.assertEqual(ps.legendNumColsCombo.currentText(), "1")
            for j in range(ps._possibleLegendMaxNcols):
                self.assertEqual(ps.legendNumColsCombo.itemText(j), "%d" % (j + 1))

            # text size
            r += 1
            self.assertIsInstance(lay.itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                lay.itemAtPosition(r, 0).widget().text(),
                PGText.plotSettLegendTextSize,
            )
            self.assertEqual(ps.legendTextSizeCombo, lay.itemAtPosition(r, 1).widget())
            self.assertIsInstance(ps.legendTextSizeCombo, QComboBox)
            self.assertEqual(ps.legendTextSizeCombo.count(), len(ps._possibleTextSizes))
            self.assertEqual(
                ps.legendTextSizeCombo.currentText(), ps._possibleTextSizesDefault
            )
            for j, s in enumerate(ps._possibleTextSizes):
                self.assertEqual(ps.legendTextSizeCombo.itemText(j), s)

        def test_clearCurrent(self):
            """test the clearCurrent method"""
            ps = PlotSettings()
            ps.axesTextSize = "large"
            ps.figsize = [10, 10]
            ps.legend = False
            ps.legendLoc = "upper right"
            ps.legendNcols = 2
            ps.legendTextSize = "large"
            ps.tight = False
            ps.title = "title"
            ps.xlabel = "x"
            ps.xlims = (-1, 10)
            ps.xscale = "log"
            ps.ylabel = "y"
            ps.ylims = (-1, 2)
            ps.yscale = "log"
            ps.zlabel = "z"
            ps.clearCurrent()
            self.assertEqual(ps.axesTextSize, ps._possibleTextSizesDefault)
            self.assertEqualArray(
                ps.figsize, [ps._default_figsizex, ps._default_figsizey]
            )
            self.assertTrue(ps.legend)
            self.assertEqual(ps.legendLoc, ps._possibleLegendLoc[0])
            self.assertEqual(ps.legendNcols, 1)
            self.assertEqual(ps.legendTextSize, ps._possibleTextSizesDefault)
            self.assertTrue(ps.tight)
            self.assertEqual(ps.title, "")
            self.assertEqual(ps.xlabel, "")
            self.assertEqualArray(ps.xlims, (ps._default_xlimsd, ps._default_xlimsu))
            self.assertEqual(ps.xscale, ps._scales[0])
            self.assertEqual(ps.ylabel, "")
            self.assertEqualArray(ps.ylims, (ps._default_ylimsd, ps._default_ylimsu))
            self.assertEqual(ps.yscale, ps._scales[0])
            self.assertEqual(ps.zlabel, "")
            with patch(
                "parthenopegui.plotter.PlotSettings.updateAutomaticSettings",
                autospec=True,
            ) as _f:
                ps.clearCurrent()
                _f.assert_called_once_with(
                    ps,
                    {
                        "xlabel": "",
                        "xlims": (ps._default_xlimsd, ps._default_xlimsu),
                        "xscale": ps._scales[0],
                        "ylabel": "",
                        "ylims": (ps._default_ylimsd, ps._default_ylimsu),
                        "yscale": ps._scales[0],
                        "zlabel": "",
                    },
                    force=True,
                )

        def test_restoreAutomaticSettings(self):
            """test restoreAutomaticSettings"""
            ps = PlotSettings()
            ps.xlabel = "x"
            ps.ylabel = "y"
            ps.xlims = (2, 10)
            ps.ylims = (3, 10)
            ps.automaticSettings = {
                "abc": "abc",
                "xlabel": "myvarA",
                "xlims": (1, 0),
            }
            self.assertFalse(hasattr(ps, "abc"))
            self.assertEqual(ps.xlabel, "x")
            self.assertEqual(ps.ylabel, "y")
            self.assertEqual(ps.xlims, (2, 10))
            self.assertEqual(ps.ylims, (3, 10))
            ps.restoreAutomaticSettings()
            self.assertTrue(hasattr(ps, "abc"))
            self.assertEqual(ps.abc, "abc")
            self.assertEqual(ps.xlabel, "myvarA")
            self.assertEqual(ps.ylabel, "y")
            self.assertEqual(ps.xlims, (1, 0))
            self.assertEqual(ps.ylims, (3, 10))

        def test_updateAutomaticSettings(self):
            """test updateAutomaticSettings"""
            ps = PlotSettings()
            ps.clearCurrent()
            self.assertEqual(
                ps.automaticSettings,
                {
                    "xlabel": "",
                    "xlims": (ps._default_xlimsd, ps._default_xlimsu),
                    "xscale": ps._scales[0],
                    "ylabel": "",
                    "ylims": (ps._default_ylimsd, ps._default_ylimsu),
                    "yscale": ps._scales[0],
                    "zlabel": "",
                },
            )
            self.assertEqual(ps.xlabel, "")
            self.assertEqual(ps.xlims, (ps._default_xlimsd, ps._default_xlimsu))
            self.assertEqual(ps.xscale, ps._scales[0])
            self.assertEqual(ps.ylabel, "")
            self.assertEqual(ps.ylims, (ps._default_ylimsd, ps._default_ylimsu))
            self.assertEqual(ps.yscale, ps._scales[0])
            self.assertEqual(ps.zlabel, "")

            ps.updateAutomaticSettings({"abc": "abc", "xlabel": "xv", "zlabel": "z"})
            self.assertEqual(
                ps.automaticSettings,
                {
                    "abc": "abc",
                    "xlabel": "xv",
                    "xlims": (ps._default_xlimsd, ps._default_xlimsu),
                    "xscale": ps._scales[0],
                    "ylabel": "",
                    "ylims": (ps._default_ylimsd, ps._default_ylimsu),
                    "yscale": ps._scales[0],
                    "zlabel": "z",
                },
            )
            self.assertEqual(ps.abc, "abc")
            self.assertEqual(ps.xlabel, "xv")
            self.assertEqual(ps.xlims, (ps._default_xlimsd, ps._default_xlimsu))
            self.assertEqual(ps.xscale, ps._scales[0])
            self.assertEqual(ps.ylabel, "")
            self.assertEqual(ps.ylims, (ps._default_ylimsd, ps._default_ylimsu))
            self.assertEqual(ps.yscale, ps._scales[0])
            self.assertEqual(ps.zlabel, "z")
            ps.xlabel = "x"
            ps.ylabel = "y"
            ps.xlims = (ps._default_xlimsd, 2 * ps._default_xlimsu)
            ps.ylims = (3, 10)

            ps.updateAutomaticSettings(
                {"xlabel": "nxv", "zlabel": "nz", "xlims": (0, 2)}
            )
            self.assertEqual(
                ps.automaticSettings,
                {
                    "abc": "abc",
                    "xlabel": "nxv",
                    "xlims": (0, 2),
                    "xscale": ps._scales[0],
                    "ylabel": "",
                    "ylims": (ps._default_ylimsd, ps._default_ylimsu),
                    "yscale": ps._scales[0],
                    "zlabel": "nz",
                },
            )
            self.assertEqual(ps.abc, "abc")
            self.assertEqual(ps.xlabel, "x")
            self.assertEqual(ps.xlims, (ps._default_xlimsd, 2 * ps._default_xlimsu))
            self.assertEqual(ps.xscale, ps._scales[0])
            self.assertEqual(ps.ylabel, "y")
            self.assertEqual(ps.ylims, (3, 10))
            self.assertEqual(ps.yscale, ps._scales[0])
            self.assertEqual(ps.zlabel, "nz")

            with patch("logging.Logger.debug") as _d, patch(
                "logging.Logger.error"
            ) as _e:
                ps.updateAutomaticSettings(
                    {
                        "xlabel": "mylab",
                        "zlabel": "",
                        "xlims": (10, 20),
                        "ylims": ("a", 19),
                    },
                    force=True,
                )
                _e.assert_called_once_with("Invalid lower ylims: a. Use 0")
                _d.assert_called_once_with(
                    "Value not accepted for ylims: %s" % (("a", 19),)
                )
            self.assertEqual(
                ps.automaticSettings,
                {
                    "abc": "abc",
                    "xlabel": "mylab",
                    "xlims": (10, 20),
                    "xscale": ps._scales[0],
                    "ylabel": "",
                    "ylims": (0, 19),
                    "yscale": ps._scales[0],
                    "zlabel": "",
                },
            )
            self.assertEqual(ps.abc, "abc")
            self.assertEqual(ps.xlabel, "mylab")
            self.assertEqual(ps.xlims, (10, 20))
            self.assertEqual(ps.xscale, ps._scales[0])
            self.assertEqual(ps.ylabel, "y")
            self.assertEqual(ps.ylims, (0, 19))
            self.assertEqual(ps.yscale, ps._scales[0])
            self.assertEqual(ps.zlabel, "")

            with patch("logging.Logger.debug") as _d, patch(
                "logging.Logger.error"
            ) as _e:
                ps.updateAutomaticSettings(
                    {
                        "xlabel": "x",
                        "zlabel": "z",
                        "xscale": ps._scales[1],
                        "xlims": ("a", 20),
                    }
                )
                _e.assert_called_once_with("Invalid lower xlims: a. Use 0")
                _d.assert_called_once_with(
                    "Value not accepted for xlims: %s" % (("a", 20),)
                )
            self.assertEqual(
                ps.automaticSettings,
                {
                    "abc": "abc",
                    "xlabel": "x",
                    "xlims": (0.0, 20.0),
                    "xscale": ps._scales[1],
                    "ylabel": "",
                    "ylims": (0, 19),
                    "yscale": ps._scales[0],
                    "zlabel": "z",
                },
            )
            self.assertEqual(ps.abc, "abc")
            self.assertEqual(ps.xlabel, "x")
            self.assertEqual(ps.xlims, (0, 20))
            self.assertEqual(ps.xscale, ps._scales[1])
            self.assertEqual(ps.ylabel, "y")
            self.assertEqual(ps.ylims, (0, 19))
            self.assertEqual(ps.yscale, ps._scales[0])
            self.assertEqual(ps.zlabel, "z")

        def test_axesTextSize(self):
            """test axesTextSize getter and setter"""
            ps = PlotSettings()
            self.assertEqual(ps.axesTextSize, "medium")
            ps.axesTextSize = 123
            self.assertEqual(ps.axesTextSize, "medium")
            ps.axesTextSize = "largex"
            self.assertEqual(ps.axesTextSize, "medium")
            for p in ps._possibleTextSizes:
                ps.axesTextSize = p
                self.assertEqual(ps.axesTextSize, p)

        def test_figsize(self):
            """test figsize getter and setter"""
            ps = PlotSettings()
            with patch("logging.Logger.error") as _e:
                self.assertEqual(
                    ps.figsize, (ps._default_figsizex, ps._default_figsizey)
                )
                self.assertEqual(_e.call_count, 2)
                ps.figsize = 2.3
                self.assertEqual(_e.call_count, 3)
                self.assertEqual(
                    ps.figsize, (ps._default_figsizex, ps._default_figsizey)
                )
                self.assertEqual(_e.call_count, 3)
                ps.figsize = "ab"
                self.assertEqual(_e.call_count, 4)
                self.assertEqual(
                    ps.figsize, (ps._default_figsizex, ps._default_figsizey)
                )
                self.assertEqual(_e.call_count, 4)
                ps.figsize = ["a", 2]
                self.assertEqual(_e.call_count, 5)
                self.assertEqual(ps.figsize, (ps._default_figsizex, 2.0))
                self.assertEqual(_e.call_count, 5)
                ps.figsize = [3, "a"]
                self.assertEqual(_e.call_count, 6)
                self.assertEqual(ps.figsize, (3.0, ps._default_figsizey))
                self.assertEqual(_e.call_count, 6)
                ps.figsize = [2.3, 4]
                self.assertEqual(ps.figsize, (2.3, 4))
                self.assertEqual(_e.call_count, 6)
                ps.figsize = (3.3, 4.2)
                self.assertEqual(ps.figsize, (3.3, 4.2))
                self.assertEqual(_e.call_count, 6)
                ps.figsize = (-3.3, 4.2)
                self.assertEqual(ps.figsize, (ps._default_figsizex, 4.2))
                self.assertEqual(_e.call_count, 7)
                ps.figsize = (3.3, -4.2)
                self.assertEqual(ps.figsize, (3.3, ps._default_figsizey))
                self.assertEqual(_e.call_count, 8)

        def test_legend(self):
            """test legend getter and setter"""
            ps = PlotSettings()
            self.assertTrue(ps.legend)
            ps.legend = None
            self.assertFalse(ps.legend)
            ps.legend = True
            self.assertTrue(ps.legend)
            ps.legend = False
            self.assertFalse(ps.legend)
            ps.legend = "a"
            self.assertTrue(ps.legend)

        def test_legendLoc(self):
            """test legendLoc getter and setter"""
            ps = PlotSettings()
            self.assertEqual(ps.legendLoc, "best")
            ps.legendLoc = 123
            self.assertEqual(ps.legendLoc, "best")
            ps.legendLoc = "test"
            self.assertEqual(ps.legendLoc, "best")
            for p in ps._possibleLegendLoc:
                ps.legendLoc = p
                self.assertEqual(ps.legendLoc, p)

        def test_legendNcols(self):
            """test legendNcols getter and setter"""
            ps = PlotSettings()
            self.assertEqual(ps.legendNcols, 1)
            ps.legendNcols = 123
            self.assertEqual(ps.legendNcols, 1)
            ps.legendNcols = "a"
            self.assertEqual(ps.legendNcols, 1)
            for p in [x for x in range(ps._possibleLegendMaxNcols)]:
                ps.legendNcols = p + 1
                self.assertEqual(ps.legendNcols, p + 1)

        def test_legendTextSize(self):
            """test legendTextSize getter and setter"""
            ps = PlotSettings()
            self.assertEqual(ps.legendTextSize, "medium")
            ps.legendTextSize = 123
            self.assertEqual(ps.legendTextSize, "medium")
            ps.legendTextSize = "largex"
            self.assertEqual(ps.legendTextSize, "medium")
            for p in ps._possibleTextSizes:
                ps.legendTextSize = p
                self.assertEqual(ps.legendTextSize, p)

        def test_tight(self):
            """test tight getter and setter"""
            ps = PlotSettings()
            self.assertTrue(ps.tight)
            ps.tight = None
            self.assertFalse(ps.tight)
            ps.tight = True
            self.assertTrue(ps.tight)
            ps.tight = False
            self.assertFalse(ps.tight)
            ps.tight = "a"
            self.assertTrue(ps.tight)

        def test_title(self):
            """test title getter and setter"""
            ps = PlotSettings()
            self.assertEqual(ps.title, "")
            self.assertEqual(ps.title, ps.titleEd.text())
            ps.titleEd.setText("abc")
            self.assertEqual(ps.title, "abc")
            self.assertEqual(ps.title, ps.titleEd.text())
            ps.title = "def"
            self.assertEqual(ps.title, "def")
            self.assertEqual(ps.title, ps.titleEd.text())
            ps.title = 123
            self.assertEqual(ps.title, "123")

        def test_xlabel(self):
            """test xlabel getter and setter"""
            ps = PlotSettings()
            self.assertEqual(ps.xlabel, "")
            self.assertEqual(ps.xlabel, ps.xlabelEd.text())
            ps.xlabelEd.setText("abc")
            self.assertEqual(ps.xlabel, "abc")
            self.assertEqual(ps.xlabel, ps.xlabelEd.text())
            ps.xlabel = "def"
            self.assertEqual(ps.xlabel, "def")
            self.assertEqual(ps.xlabel, ps.xlabelEd.text())
            ps.xlabel = 123
            self.assertEqual(ps.xlabel, "123")

        def test_xlims(self):
            """test xlims getter and setter"""
            ps = PlotSettings()
            with patch("logging.Logger.error") as _e:
                self.assertEqual(ps.xlims, (ps._default_xlimsd, ps._default_xlimsu))
                self.assertEqual(_e.call_count, 2)
                ps.xlims = 2.3
                self.assertEqual(_e.call_count, 3)
                self.assertEqual(ps.xlims, (ps._default_xlimsd, ps._default_xlimsu))
                self.assertEqual(_e.call_count, 3)
                ps.xlims = "ab"
                self.assertEqual(_e.call_count, 4)
                self.assertEqual(ps.xlims, (ps._default_xlimsd, ps._default_xlimsu))
                self.assertEqual(_e.call_count, 4)
                ps.xlims = ("a", 2)
                self.assertEqual(_e.call_count, 5)
                self.assertEqual(ps.xlims, (ps._default_xlimsd, 2.0))
                self.assertEqual(_e.call_count, 5)
                ps.xlims = (3, "a")
                self.assertEqual(_e.call_count, 6)
                self.assertEqual(ps.xlims, (3.0, ps._default_xlimsu))
                self.assertEqual(_e.call_count, 6)
                ps.xlims = (2.3, 4)
                self.assertEqual(ps.xlims, (2.3, 4))
                self.assertEqual(_e.call_count, 6)
                ps.xlims = (3.3, 4.2)
                self.assertEqual(ps.xlims, (3.3, 4.2))
                self.assertEqual(_e.call_count, 6)

        def test_xscale(self):
            """test xscale getter and setter"""
            ps = PlotSettings()
            with patch("logging.Logger.error") as _e:
                self.assertEqual(ps.xscale, "linear")
                self.assertEqual(ps.xscale, ps.xscaleCombo.currentText())
                ps.xscale = "abc"
                self.assertEqual(_e.call_count, 1)
                self.assertEqual(ps.xscale, "linear")
                ps.xscale = "log"
                self.assertEqual(ps.xscale, "log")
                ps.xscale = "lin"
                self.assertEqual(_e.call_count, 2)
                self.assertEqual(ps.xscale, "linear")

        def test_ylabel(self):
            """test ylabel getter and setter"""
            ps = PlotSettings()
            self.assertEqual(ps.ylabel, "")
            self.assertEqual(ps.ylabel, ps.ylabelEd.text())
            ps.ylabelEd.setText("abc")
            self.assertEqual(ps.ylabel, "abc")
            self.assertEqual(ps.ylabel, ps.ylabelEd.text())
            ps.ylabel = "def"
            self.assertEqual(ps.ylabel, "def")
            self.assertEqual(ps.ylabel, ps.ylabelEd.text())
            ps.ylabel = 123
            self.assertEqual(ps.ylabel, "123")

        def test_ylims(self):
            """test ylims getter and setter"""
            ps = PlotSettings()
            with patch("logging.Logger.error") as _e:
                self.assertEqual(ps.ylims, (ps._default_ylimsd, ps._default_ylimsu))
                self.assertEqual(_e.call_count, 2)
                ps.ylims = 2.3
                self.assertEqual(_e.call_count, 3)
                self.assertEqual(ps.ylims, (ps._default_ylimsd, ps._default_ylimsu))
                self.assertEqual(_e.call_count, 3)
                ps.ylims = "ab"
                self.assertEqual(_e.call_count, 4)
                self.assertEqual(ps.ylims, (ps._default_ylimsd, ps._default_ylimsu))
                self.assertEqual(_e.call_count, 4)
                ps.ylims = ("a", 2)
                self.assertEqual(_e.call_count, 5)
                self.assertEqual(ps.ylims, (ps._default_ylimsd, 2.0))
                self.assertEqual(_e.call_count, 5)
                ps.ylims = (3, "a")
                self.assertEqual(_e.call_count, 6)
                self.assertEqual(ps.ylims, (3.0, ps._default_ylimsu))
                self.assertEqual(_e.call_count, 6)
                ps.ylims = (2.3, 4)
                self.assertEqual(ps.ylims, (2.3, 4))
                self.assertEqual(_e.call_count, 6)
                ps.ylims = (3.3, 4.2)
                self.assertEqual(ps.ylims, (3.3, 4.2))
                self.assertEqual(_e.call_count, 6)

        def test_yscale(self):
            """test yscale getter and setter"""
            ps = PlotSettings()
            with patch("logging.Logger.error") as _e:
                self.assertEqual(ps.yscale, "linear")
                self.assertEqual(ps.yscale, ps.yscaleCombo.currentText())
                ps.yscale = "abc"
                self.assertEqual(_e.call_count, 1)
                self.assertEqual(ps.yscale, "linear")
                ps.yscale = "log"
                self.assertEqual(ps.yscale, "log")
                ps.yscale = "lin"
                self.assertEqual(_e.call_count, 2)
                self.assertEqual(ps.yscale, "linear")

        def test_zlabel(self):
            """test zlabel getter and setter"""
            ps = PlotSettings()
            self.assertEqual(ps.zlabel, "")
            self.assertEqual(ps.zlabel, ps.zlabelEd.text())
            ps.zlabelEd.setText("abc")
            self.assertEqual(ps.zlabel, "abc")
            self.assertEqual(ps.zlabel, ps.zlabelEd.text())
            ps.zlabel = "def"
            self.assertEqual(ps.zlabel, "def")
            self.assertEqual(ps.zlabel, ps.zlabelEd.text())
            ps.zlabel = 123
            self.assertEqual(ps.zlabel, "123")

    class TestPlotPlaceholderPanel(PGTestCasewMainW):
        """Test the PlotPlaceholderPanel class"""

        def test_init(self):
            """test init"""
            p = QWidget()
            pp = PlotPlaceholderPanel(p, self.mainW)
            self.assertIsInstance(pp, QFrame)
            self.assertEqual(pp.mainW, self.mainW)
            self.assertIsInstance(pp.layout(), QHBoxLayout)
            self.assertIsInstance(pp.text, QTextEdit)
            self.assertTrue(pp.text.isReadOnly())
            self.assertEqual(pp.text.toPlainText(), PGText.plotPlaceholderText)
            self.assertEqual(pp.text, pp.layout().itemAt(0).widget())

            self.assertEqual(pp.clearCurrent(), None)
            self.assertEqual(pp.reloadModel(), None)
            self.assertEqual(pp.updatePlotProperties(), None)

    class TestPGEmptyTableModel(PGTestCase):
        """test the PGEmptyTableModel class"""

        def test_columnCount(self):
            """test columnCount"""
            tm = PGEmptyTableModel()
            with self.assertRaises(AttributeError):
                tm.columnCount()
            tm.header = 1
            self.assertEqual(tm.columnCount(), 0)
            tm.header = ["a", "b", "c"]
            self.assertEqual(tm.columnCount(), 3)

        def test_flags(self):
            """test flags"""
            tm = PGEmptyTableModel()
            m = QModelIndex()
            with patch(
                "PySide2.QtCore.QModelIndex.isValid", side_effect=[False, True]
            ) as _v:
                self.assertEqual(tm.flags(m), Qt.NoItemFlags)
                _v.assert_called_once_with()
                self.assertEqual(tm.flags(m), Qt.ItemIsEnabled | Qt.ItemIsSelectable)

        def test_rowCount(self):
            """test rowCount"""
            tm = PGEmptyTableModel()
            with self.assertRaises(AttributeError):
                tm.rowCount()
            tm.dataList = 1
            self.assertEqual(tm.rowCount(), 0)
            tm.dataList = [0, 1, 2]
            self.assertEqual(tm.rowCount(), 3)

    class TestPGListLinesModel(PGTestCase):
        """Test the PGListLinesModel class"""

        def test_init(self):
            """test __init__"""
            p = QWidget()
            ll = PGListLinesModel([{"a": 1}, {"a": 2}, {"a": 3}], p)
            self.assertIsInstance(ll, PGEmptyTableModel)
            self.assertEqual(ll.dataList, [{"a": 1}, {"a": 2}, {"a": 3}])
            self.assertEqual(
                ll.header,
                [
                    "description",
                    "label",
                    "color",
                    "style",
                    "marker",
                    "width",
                    "Delete",
                    "Move up",
                    "Move down",
                ],
            )
            self.assertTrue(hasattr(ll, "headerToolTips"))

        def test_addLineToData(self):
            """test addLineToData"""
            p = QWidget()
            a = MagicMock()
            c = MagicMock()
            ll = PGListLinesModel([{"a": 1}, {"a": 2}, {"a": 3}], p)
            ll.layoutAboutToBeChanged.connect(a)
            ll.layoutChanged.connect(c)
            ll.addLineToData(4)
            a.assert_called_once()
            c.assert_called_once()
            self.assertEqual(ll.dataList, [{"a": 1}, {"a": 2}, {"a": 3}, 4])

        def test_data(self):
            """test data"""
            p = QWidget()
            qp = QPixmap()
            qp1 = QPixmap()
            qp.scaledToHeight = MagicMock(return_value=qp1)
            line = PGLine(
                [0, 1, 2],
                [3, 4, 5],
                1,
                c="r",
                c2="chi2",
                d="desc",
                l="label",
                m="o",
                s=":",
                w=1.12,
                xl="xl",
                yl="yl",
            )
            ll = PGListLinesModel([line, line, line], p)
            idx = QModelIndex()
            idx.isValid = MagicMock(return_value=False)
            idx.row = MagicMock(return_value=0)
            self.assertEqual(ll.data(idx, Qt.DisplayRole), None)
            idx.isValid.assert_called_once()
            self.assertEqual(idx.row.call_count, 0)
            idx.isValid = MagicMock(return_value=True)

            for c, f in (
                ("Delete", ":/images/delete.png"),
                ("Move down", ":/images/arrow-down.png"),
                ("Move up", ":/images/arrow-up.png"),
            ):
                idx.column = MagicMock(return_value=ll.header.index(c))
                idx.row = MagicMock(return_value=0)
                self.assertEqual(ll.data(idx, Qt.DisplayRole), "")
                self.assertEqual(ll.data(idx, Qt.UserRole), None)
                if c == "Delete":
                    with patch("parthenopegui.plotter.QPixmap", return_value=qp) as _qp:
                        self.assertEqual(ll.data(idx, Qt.DecorationRole), qp1)
                        _qp.assert_called_once_with(f)
                elif c == "Move down":
                    idx.row = MagicMock(side_effect=[0, 1, 2])
                    with patch("parthenopegui.plotter.QPixmap", return_value=qp) as _qp:
                        self.assertEqual(ll.data(idx, Qt.DecorationRole), qp1)
                        _qp.assert_called_once_with(f)
                    with patch("parthenopegui.plotter.QPixmap", return_value=qp) as _qp:
                        self.assertEqual(ll.data(idx, Qt.DecorationRole), qp1)
                        _qp.assert_called_once_with(f)
                    self.assertEqual(ll.data(idx, Qt.DecorationRole), None)
                elif c == "Move up":
                    idx.row = MagicMock(side_effect=[0, 1, 2])
                    self.assertEqual(ll.data(idx, Qt.DecorationRole), None)
                    with patch("parthenopegui.plotter.QPixmap", return_value=qp) as _qp:
                        self.assertEqual(ll.data(idx, Qt.DecorationRole), qp1)
                        _qp.assert_called_once_with(f)
                    with patch("parthenopegui.plotter.QPixmap", return_value=qp) as _qp:
                        self.assertEqual(ll.data(idx, Qt.DecorationRole), qp1)
                        _qp.assert_called_once_with(f)

            idx.row = MagicMock(return_value=0)
            for c in ("description", "label", "color", "style", "marker", "width"):
                idx.column = MagicMock(return_value=ll.header.index(c))
                self.assertEqual(ll.data(idx, Qt.UserRole), None)
                self.assertEqual(ll.data(idx, Qt.DisplayRole), "%s" % getattr(line, c))
            del line.width
            with patch("logging.Logger.exception") as _e:
                self.assertEqual(ll.data(idx, Qt.DisplayRole), None)
                _e.assert_called_once_with("")
            idx.column = MagicMock(return_value=12)
            with patch("logging.Logger.exception") as _e:
                self.assertEqual(ll.data(idx, Qt.DisplayRole), None)
                _e.assert_called_once_with(PGText.errorCannotFindIndex)
            idx.row = MagicMock(return_value=10)
            idx.column = MagicMock(return_value=2)
            with patch("logging.Logger.exception") as _e:
                self.assertEqual(ll.data(idx, Qt.DisplayRole), None)
                _e.assert_called_once_with("")

        def test_deleteLine(self):
            """test deleteLine"""
            p = QWidget()
            a = MagicMock()
            c = MagicMock()
            ll = PGListLinesModel([{"a": 1}, {"a": 2}, {"a": 3}], p)
            ll.layoutAboutToBeChanged.connect(a)
            ll.layoutChanged.connect(c)
            ll.deleteLine(2)
            a.assert_called_once()
            c.assert_called_once()
            self.assertEqual(ll.dataList, [{"a": 1}, {"a": 2}])
            with patch("logging.Logger.debug") as _w:
                ll.deleteLine(2)
                _w.assert_called_once_with(PGText.warningMissingLine)
            self.assertEqual(a.call_count, 2)
            self.assertEqual(c.call_count, 2)
            self.assertEqual(ll.dataList, [{"a": 1}, {"a": 2}])

        def test_headerData(self):
            """test headerData"""
            p = QWidget()
            ll = PGListLinesModel([{"a": 1}, {"a": 2}, {"a": 3}], p)
            self.assertEqual(ll.headerData(123, Qt.Horizontal, Qt.DecorationRole), None)
            self.assertEqual(ll.headerData(123, Qt.Vertical, Qt.DisplayRole), None)
            for c, t in enumerate(ll.header):
                self.assertEqual(ll.headerData(c, Qt.Horizontal, Qt.DisplayRole), t)

        def test_replaceLine(self):
            """test replaceLine"""
            p = QWidget()
            a = MagicMock()
            c = MagicMock()
            ll = PGListLinesModel([{"a": 1}, {"a": 2}, {"a": 3}], p)
            ll.layoutAboutToBeChanged.connect(a)
            ll.layoutChanged.connect(c)
            ll.replaceLine({"a": 8}, 2)
            a.assert_called_once()
            c.assert_called_once()
            self.assertEqual(ll.dataList, [{"a": 1}, {"a": 2}, {"a": 8}])
            with patch("logging.Logger.warning") as _w:
                ll.replaceLine({"a": 8}, 22)
                _w.assert_called_once_with(PGText.warningMissingLine)
            self.assertEqual(a.call_count, 2)
            self.assertEqual(c.call_count, 2)
            self.assertEqual(ll.dataList, [{"a": 1}, {"a": 2}, {"a": 8}])

    class TestChi2ParamConfig(PGTestCase):
        """Test the Chi2ParamConfig class"""

        def test_init(self):
            """test __init__"""
            a = QComboBox()
            b = QLineEdit("b")
            c = QLineEdit("c")
            with self.assertRaises(TypeError):
                Chi2ParamConfig("a", "b", "c")
            with self.assertRaises(TypeError):
                Chi2ParamConfig(a, "b", "c")
            with self.assertRaises(TypeError):
                Chi2ParamConfig(a, b, "c")
            o = Chi2ParamConfig(a, b, c)
            self.assertEqual(o.paramCombo, a)
            self.assertEqual(o.chi2Mean, b)
            self.assertEqual(o.chi2Std, c)
            self.assertIsInstance(o.paramLabel, PGLabel)
            self.assertIsInstance(o.meanLabel, PGLabel)
            self.assertIsInstance(o.stdLabel, PGLabel)
            self.assertEqual(o.paramLabel.text(), PGText.plotSelectNuclide)
            self.assertEqual(o.meanLabel.text(), PGText.mean)
            self.assertEqual(o.stdLabel.text(), PGText.standardDeviation)

    class TestAddFromPoint(PGTestCase):
        """Test the AddFromPoint class"""

        def test_init(self):
            """test __init__"""
            o = AddFromPoint()
            self.assertIsInstance(o, QDialog)

        def test_addChi2Line(self):
            """test _addChi2Line"""
            cw = QWidget()
            o = AddFromPoint()
            o.chi2Widget = cw
            cw.setLayout(QHBoxLayout())
            cw.layout().addWidget(QWidget())
            cw.layout().addWidget(QWidget())
            cw.layout().addWidget(QWidget())
            self.assertEqual(cw.layout().count(), 3)
            with patch(
                "parthenopegui.plotter.AddFromPoint._newChi2Widget", autospec=True
            ) as _n, patch(
                "parthenopegui.plotter.AddFromPoint._fillChi2Widget", autospec=True
            ) as _f:
                o._addChi2Line()
                _n.assert_called_once_with(o)
                _f.assert_called_once_with(o)
            self.assertEqual(cw.layout().count(), 0)

        def test_addHLine(self):
            """test _addHLine"""
            o = AddFromPoint()
            o.setLayout(QHBoxLayout())
            o.layout().addWidget = MagicMock()
            h = QFrame()
            h.setMinimumHeight = MagicMock()
            h.setFrameShape = MagicMock()
            with patch("parthenopegui.plotter.QFrame", return_value=h) as _f, patch(
                "parthenopegui.plotter.QFrame.HLine"
            ) as _l:
                o._addHLine(12)
                _f.assert_called_once_with()
                h.setFrameShape.assert_called_once_with(_l)
            h.setMinimumHeight.assert_called_once_with(2)
            o.layout().addWidget.assert_called_once_with(h, 12, 0, 1, 4)

        def test_doAddChi2Button(self):
            """test _doAddChi2Button"""
            w = AddFromPoint()
            self.assertFalse(hasattr(w, "addChi2Line"))
            w._doAddChi2Button()
            self.assertTrue(hasattr(w, "addChi2Line"))
            self.assertIsInstance(w.addChi2Line, PGPushButton)
            self.assertEqual(w.addChi2Line.text(), PGText.plotSelectChi2AddLine)
            self.assertEqual(
                w.addChi2Line.toolTip(), PGText.plotSelectChi2AddLineToolTip
            )
            with patch(
                "parthenopegui.plotter.AddFromPoint._addChi2Line", autospec=True
            ) as _f:
                QTest.mouseClick(w.addChi2Line, Qt.LeftButton)
                _f.assert_called_once_with(w)

        def test_doGroupBox(self):
            """test _doGroupBox"""
            w = AddFromPoint()
            self.assertFalse(hasattr(w, "groupBox"))
            w._doGroupBox()
            self.assertTrue(hasattr(w, "groupBox"))
            self.assertIsInstance(w.groupBox, QGroupBox)
            self.assertIsInstance(w.groupBox.layout(), QHBoxLayout)
            self.assertEqual(w.groupBox.title(), PGText.plotSelectChi2Type)
            for i, [attr, [title, desc]] in enumerate(
                PGText.plotSelectChi2TypeContents.items()
            ):
                self.assertIsInstance(getattr(w, attr), QRadioButton)
                self.assertEqual(getattr(w, attr).text(), title)
                self.assertEqual(getattr(w, attr).toolTip(), desc)
                self.assertEqual(
                    w.groupBox.layout().itemAt(i).widget(), getattr(w, attr)
                )
            self.assertTrue(w.chi2.isChecked())

        def test_fillChi2Widget(self):
            """test _fillChi2Widget"""
            w = AddFromPoint()
            w.chi2Widget = QWidget()
            l = QGridLayout()
            l.addWidget = MagicMock()
            w.chi2Widget.setLayout(l)
            w.chi2ParamConfigs = [
                Chi2ParamConfig(QComboBox(), QLineEdit("b"), QLineEdit("c")),
                Chi2ParamConfig(QComboBox(), QLineEdit("d"), QLineEdit("e")),
            ]
            w._fillChi2Widget()
            self.assertTrue(hasattr(w, "addChi2Line"))
            self.assertTrue(hasattr(w, "groupBox"))
            for r, o in enumerate(w.chi2ParamConfigs):
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
                    l.addWidget.assert_any_call(getattr(o, f), r, i)
            l.addWidget.assert_any_call(w.addChi2Line, r + 1, 0, 1, 2)
            l.addWidget.assert_any_call(w.groupBox, r + 2, 0, 1, 6)
            with patch(
                "parthenopegui.plotter.AddFromPoint._doAddChi2Button", autospec=True
            ) as _ac, patch(
                "parthenopegui.plotter.AddFromPoint._doGroupBox", autospec=True
            ) as _ag:
                w._fillChi2Widget()
                self.assertEqual(_ac.call_count, 0)
                self.assertEqual(_ag.call_count, 0)

        def test_getChi2Conv(self):
            """test _getChi2Conv"""
            w = AddFromPoint()
            w._doGroupBox()
            a = np.array([0, 1, 2, 3, 4])
            for k, v in (
                ("chi2", a),
                ("lh", np.exp(-0.5 * a)),
                ("nllh", 0.5 * a),
                ("pllh", -0.5 * a),
            ):
                getattr(w, k).setChecked(True)
                self.assertEqualArray(w._getChi2Conv(a), v)

        def test_getChi2Type(self):
            """test _getChi2Type"""
            w = AddFromPoint()
            w._doGroupBox()
            self.assertEqual(w._getChi2Type(False), False)
            self.assertEqual(w._getChi2Type(None), False)
            self.assertEqual(w._getChi2Type(0), False)
            for k in ("chi2", "lh", "nllh", "pllh"):
                getattr(w, k).setChecked(True)
                self.assertEqual(w._getChi2Type(True), k)
                chi2labels[k]

        def test_getDescLineParam(self):
            """test _getDescLineParam"""
            w = AddFromPoint()
            w._doGroupBox()
            for k, v in (
                ("chi2", "chi2(%s) "),
                ("lh", "lh(%s) "),
                ("nllh", "-llh(%s) "),
                ("pllh", "+llh(%s) "),
            ):
                getattr(w, k).setChecked(True)
                self.assertEqual(w._getDescLineParam(), v)

        def test_newChi2Widget(self):
            """test _newChi2Widget"""
            w = AddFromPoint()
            w.chi2ParamConfigs = []
            n = QComboBox()
            w._nuclideCombo = MagicMock(return_value=n)
            l1, l2 = QLineEdit(""), QLineEdit("")
            c = Chi2ParamConfig(n, l1, l2)
            with patch(
                "parthenopegui.plotter.Chi2ParamConfig", return_value=c
            ) as _c, patch(
                "parthenopegui.plotter.QLineEdit", side_effect=[l1, l2]
            ) as _l:
                w._newChi2Widget()
                _c.assert_called_once_with(n, l1, l2)
                w._nuclideCombo.assert_called_once_with()
                _l.assert_has_calls([call(""), call("")])
            self.assertEqual(l1.toolTip(), PGText.plotChi2AskMeanToolTip)
            self.assertEqual(l2.toolTip(), PGText.plotChi2AskStdToolTip)
            self.assertEqual(len(w.chi2ParamConfigs), 1)
            self.assertEqual(w.chi2ParamConfigs[0], c)

        def test_updateStack(self):
            """test _updateStack"""
            o = AddFromPoint()
            sw = QStackedWidget()
            o.stackWidget = sw
            w1 = QWidget()
            w2 = QWidget()
            sw.addWidget(w1)
            sw.addWidget(w2)
            self.assertEqual(sw.currentIndex(), 0)
            o._updateStack(Qt.Checked)
            self.assertEqual(sw.currentIndex(), 1)
            o._updateStack(Qt.Unchecked)
            self.assertEqual(sw.currentIndex(), 0)

    class TestAddLineFromPoint(PGTestCasewMainW):
        """Test the AddLineFromPoint class"""

        def test_init(self):
            """test __init__"""
            line = PGLine(
                [0, 1, 2],
                [3, 4, 5],
                1,
                c="r",
                c2="chi2",
                d="description",
                l="label",
                m="o",
                s=":",
                w=1.12,
                xl="xl",
                yl="yl",
            )
            # no line, no chi2
            afp = AddLineFromPoint(
                self.mainW,
                0,
                "desc",
                2,
                testConfig.sampleRunner.parthenopeHeader,
                testConfig.sampleRunner.nuclidesEvolution[0],
            )
            self.assertIsInstance(afp, AddFromPoint)
            self.assertEqual(afp.mainW, self.mainW)
            self.assertEqual(afp.pointRow, 0)
            self.assertEqual(afp.paramix, 2)
            self.assertEqual(afp.header, testConfig.sampleRunner.parthenopeHeader)
            self.assertEqualArray(
                afp.output, testConfig.sampleRunner.nuclidesEvolution[0]
            )
            self.assertIsInstance(afp.layout(), QGridLayout)
            self.assertEqual(afp.windowTitle(), PGText.plotSelectNuclideLine)
            self.assertIsInstance(afp.chi2ParamConfigs, list)
            self.assertIsInstance(afp.stackWidget, QStackedWidget)
            self.assertIsInstance(afp.paramWidget, QWidget)
            self.assertIsInstance(afp.paramWidget.layout(), QHBoxLayout)
            self.assertEqual(afp.stackWidget.widget(0), afp.paramWidget)
            self.assertIsInstance(afp.chi2Widget, QWidget)
            self.assertIsInstance(afp.chi2Widget.layout(), QGridLayout)
            self.assertEqual(afp.stackWidget.widget(1), afp.chi2Widget)

            self.assertEqual(afp.headerCut, 0)
            self.assertFalse(afp.askChi2)
            self.assertEqual(afp.line, None)
            self.assertEqual(afp.desc, "desc")

            r = 0
            self.assertIsInstance(afp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                afp.layout().itemAtPosition(r, 0).widget().text(), afp.desc
            )
            r += 1
            self.assertIsInstance(afp.layout().itemAtPosition(r, 0).widget(), QFrame)
            self.assertEqual(
                afp.layout().itemAtPosition(r, 0).widget().frameShape(), QFrame.HLine
            )
            r += 1
            self.assertIsInstance(
                afp.layout().itemAtPosition(r, 0).widget(), QStackedWidget
            )
            self.assertEqual(
                afp.layout().itemAtPosition(r, 0).widget(),
                afp.stackWidget,
            )
            self.assertIsInstance(afp.chi2Check, QCheckBox)
            self.assertEqual(afp.chi2Check.toolTip(), PGText.plotChi2AskCheckToolTip)
            self.assertFalse(afp.chi2Check.isChecked())
            self.assertTrue(afp.chi2Check.isHidden())
            self.assertIsInstance(afp.paramWidget.layout().itemAt(0).widget(), PGLabel)
            self.assertEqual(
                afp.paramWidget.layout().itemAt(0).widget().text(),
                PGText.plotSelectNuclide,
            )
            self.assertIsInstance(afp.nuclideCombo, QComboBox)
            for i, f in enumerate(afp.header[afp.headerCut :]):
                self.assertEqual(afp.nuclideCombo.itemText(i), "%s" % f)
            self.assertEqual(
                afp.paramWidget.layout().itemAt(1).widget(),
                afp.nuclideCombo,
            )
            r += 1
            self.assertIsInstance(afp.layout().itemAtPosition(r, 0).widget(), QFrame)
            self.assertEqual(
                afp.layout().itemAtPosition(r, 0).widget().frameShape(), QFrame.HLine
            )
            r += 1
            self.assertIsInstance(afp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                afp.layout().itemAtPosition(r, 0).widget().text(), PGText.plotLineLabel
            )
            self.assertIsInstance(afp.labelInput, QLineEdit)
            self.assertEqual(afp.labelInput.text(), "")
            self.assertEqual(afp.labelInput, afp.layout().itemAtPosition(r, 2).widget())
            r += 1
            self.assertIsInstance(afp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                afp.layout().itemAtPosition(r, 0).widget().text(), PGText.plotLineColor
            )
            self.assertIsInstance(afp.colorInput, QComboBox)
            self.assertTrue(afp.colorInput.isEditable())
            self.assertEqual(afp.colorInput.currentText(), "k")
            for i, q in enumerate(["k", "r", "b", "g", "m", "c", "y", "other (edit)"]):
                self.assertEqual(afp.colorInput.itemText(i), q)
            self.assertEqual(afp.colorInput, afp.layout().itemAtPosition(r, 1).widget())
            self.assertIsInstance(afp.layout().itemAtPosition(r, 2).widget(), PGLabel)
            self.assertEqual(
                afp.layout().itemAtPosition(r, 2).widget().text(), PGText.plotLineStyle
            )
            self.assertIsInstance(afp.styleCombo, QComboBox)
            for i, f in enumerate(styleOptions):
                self.assertEqual(afp.styleCombo.itemText(i), "%s" % f)
            self.assertEqual(afp.styleCombo.currentText(), styleOptions[0])
            self.assertEqual(afp.styleCombo, afp.layout().itemAtPosition(r, 3).widget())
            r += 1
            self.assertIsInstance(afp.layout().itemAtPosition(r, 2).widget(), PGLabel)
            self.assertEqual(
                afp.layout().itemAtPosition(r, 2).widget().text(), PGText.plotLineMarker
            )
            self.assertIsInstance(afp.markerCombo, QComboBox)
            for i, f in enumerate(markerOptions):
                self.assertEqual(afp.markerCombo.itemText(i), "%s" % f)
            self.assertEqual(afp.markerCombo.currentText(), markerOptions[0])
            self.assertEqual(
                afp.markerCombo, afp.layout().itemAtPosition(r, 3).widget()
            )
            self.assertIsInstance(afp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                afp.layout().itemAtPosition(r, 0).widget().text(), PGText.plotLineWidth
            )
            self.assertIsInstance(afp.widthInput, QLineEdit)
            self.assertEqual(afp.widthInput.text(), "1")
            self.assertEqual(afp.widthInput, afp.layout().itemAtPosition(r, 1).widget())
            r += 1
            self.assertIsInstance(afp.acceptButton, PGPushButton)
            self.assertEqual(afp.acceptButton.text(), PGText.buttonAccept)
            self.assertEqual(
                afp.acceptButton, afp.layout().itemAtPosition(r, 0).widget()
            )
            afp.testAccept = MagicMock()
            QTest.mouseClick(afp.acceptButton, Qt.LeftButton)
            afp.testAccept.assert_called_once_with()
            self.assertIsInstance(afp.cancelButton, PGPushButton)
            self.assertEqual(afp.cancelButton.text(), PGText.buttonCancel)
            self.assertEqual(
                afp.cancelButton, afp.layout().itemAtPosition(r, 2).widget()
            )
            afp.reject = MagicMock()
            QTest.mouseClick(afp.cancelButton, Qt.LeftButton)
            afp.reject.assert_called_once_with()

            # no line, w chi2
            cb = QCheckBox()
            with patch(
                "parthenopegui.plotter.AddFromPoint._newChi2Widget", autospec=True
            ) as _nc2, patch(
                "parthenopegui.plotter.AddLineFromPoint._nuclideCombo",
                autospec=True,
                return_value=cb,
            ) as _nc, patch(
                "parthenopegui.plotter.AddFromPoint._fillChi2Widget", autospec=True
            ) as _fc2:
                afp = Add1DLineFromPoint(
                    self.mainW,
                    0,
                    "desc",
                    0,
                    testConfig.sampleRunner.parthenopeHeader,
                    testConfig.sampleRunner.nuclidesEvolution[0],
                )
                self.assertEqual(afp.nuclideCombo, cb)
                _nc2.assert_called_once_with(afp)
                _nc.assert_called_once_with(afp)
                _fc2.assert_called_once_with(afp)
            afp = Add1DLineFromPoint(
                self.mainW,
                0,
                "desc",
                0,
                testConfig.sampleRunner.parthenopeHeader,
                testConfig.sampleRunner.nuclidesEvolution[0],
            )
            self.assertIsInstance(afp.chi2ParamConfigs, list)
            self.assertIsInstance(afp.stackWidget, QStackedWidget)
            self.assertIsInstance(afp.paramWidget, QWidget)
            self.assertIsInstance(afp.paramWidget.layout(), QHBoxLayout)
            self.assertEqual(afp.stackWidget.widget(0), afp.paramWidget)
            self.assertIsInstance(afp.chi2Widget, QWidget)
            self.assertIsInstance(afp.chi2Widget.layout(), QGridLayout)
            self.assertEqual(afp.stackWidget.widget(1), afp.chi2Widget)
            self.assertEqual(afp.headerCut, 6)
            self.assertTrue(afp.askChi2)
            self.assertEqual(afp.line, None)
            self.assertEqual(afp.desc, "desc")

            r = 0
            self.assertIsInstance(afp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                afp.layout().itemAtPosition(r, 0).widget().text(), afp.desc
            )
            r += 1
            self.assertIsInstance(afp.layout().itemAtPosition(r, 0).widget(), QFrame)
            self.assertEqual(
                afp.layout().itemAtPosition(r, 0).widget().frameShape(), QFrame.HLine
            )
            r += 1
            self.assertIsInstance(afp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                afp.layout().itemAtPosition(r, 0).widget().text(),
                PGText.plotChi2AskLabel,
            )
            self.assertIsInstance(afp.chi2Check, QCheckBox)
            self.assertEqual(afp.chi2Check.toolTip(), PGText.plotChi2AskCheckToolTip)
            self.assertFalse(afp.chi2Check.isChecked())
            with patch(
                "parthenopegui.plotter.AddFromPoint._updateStack", autospec=True
            ) as _f:
                afp.chi2Check.toggle()
                _f.assert_called_once_with(afp, 2)
                _f.reset_mock()
                afp.chi2Check.toggle()
                _f.assert_called_once_with(afp, 0)
            self.assertEqual(afp.layout().itemAtPosition(r, 3).widget(), afp.chi2Check)
            # paramWidget
            self.assertIsInstance(afp.paramWidget.layout().itemAt(0).widget(), PGLabel)
            self.assertEqual(
                afp.paramWidget.layout().itemAt(0).widget().text(),
                PGText.plotSelectNuclide,
            )
            self.assertIsInstance(afp.nuclideCombo, QComboBox)
            for i, f in enumerate(afp.header[afp.headerCut :]):
                self.assertEqual(afp.nuclideCombo.itemText(i), "%s" % f)
            self.assertEqual(
                afp.paramWidget.layout().itemAt(1).widget(),
                afp.nuclideCombo,
            )
            # chi2Widget
            for j, o in enumerate(afp.chi2ParamConfigs):
                for i, f in enumerate(
                    [
                        "paramLabel",
                        "paramCombo",
                        "meanLabel",
                        "chi2Mean",
                        "stdLabel",
                        "chi2Std",
                    ]
                ):
                    afp.chi2Widget.layout().addWidget(getattr(o, f), j, i)
            self.assertIsInstance(afp.addChi2Line, PGPushButton)
            self.assertEqual(afp.addChi2Line.text(), PGText.plotSelectChi2AddLine)
            self.assertEqual(
                afp.addChi2Line.toolTip(), PGText.plotSelectChi2AddLineToolTip
            )
            with patch(
                "parthenopegui.plotter.AddFromPoint._addChi2Line", autospec=True
            ) as _f:
                QTest.mouseClick(afp.addChi2Line, Qt.LeftButton)
                _f.assert_called_once_with(afp)
            self.assertEqual(
                afp.chi2Widget.layout().itemAtPosition(1, 0).widget(), afp.addChi2Line
            )

            r += 1
            self.assertIsInstance(
                afp.layout().itemAtPosition(r, 0).widget(), QStackedWidget
            )
            self.assertEqual(
                afp.layout().itemAtPosition(r, 0).widget(),
                afp.stackWidget,
            )
            r += 1
            self.assertIsInstance(afp.layout().itemAtPosition(r, 0).widget(), QFrame)
            self.assertEqual(
                afp.layout().itemAtPosition(r, 0).widget().frameShape(), QFrame.HLine
            )
            r += 1
            self.assertIsInstance(afp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                afp.layout().itemAtPosition(r, 0).widget().text(), PGText.plotLineLabel
            )
            self.assertIsInstance(afp.labelInput, QLineEdit)
            self.assertEqual(afp.labelInput.text(), "")
            self.assertEqual(afp.labelInput, afp.layout().itemAtPosition(r, 2).widget())
            r += 1
            self.assertIsInstance(afp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                afp.layout().itemAtPosition(r, 0).widget().text(), PGText.plotLineColor
            )
            self.assertIsInstance(afp.colorInput, QComboBox)
            self.assertTrue(afp.colorInput.isEditable())
            self.assertEqual(afp.colorInput.currentText(), "k")
            self.assertEqual(afp.colorInput, afp.layout().itemAtPosition(r, 1).widget())
            self.assertIsInstance(afp.layout().itemAtPosition(r, 2).widget(), PGLabel)
            self.assertEqual(
                afp.layout().itemAtPosition(r, 2).widget().text(), PGText.plotLineStyle
            )
            self.assertIsInstance(afp.styleCombo, QComboBox)
            for i, f in enumerate(styleOptions):
                self.assertEqual(afp.styleCombo.itemText(i), "%s" % f)
            self.assertEqual(afp.styleCombo.currentText(), styleOptions[0])
            self.assertEqual(afp.styleCombo, afp.layout().itemAtPosition(r, 3).widget())
            r += 1
            self.assertIsInstance(afp.layout().itemAtPosition(r, 2).widget(), PGLabel)
            self.assertEqual(
                afp.layout().itemAtPosition(r, 2).widget().text(), PGText.plotLineMarker
            )
            self.assertIsInstance(afp.markerCombo, QComboBox)
            for i, f in enumerate(markerOptions):
                self.assertEqual(afp.markerCombo.itemText(i), "%s" % f)
            self.assertEqual(afp.markerCombo.currentText(), markerOptions[0])
            self.assertEqual(
                afp.markerCombo, afp.layout().itemAtPosition(r, 3).widget()
            )
            self.assertIsInstance(afp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                afp.layout().itemAtPosition(r, 0).widget().text(), PGText.plotLineWidth
            )
            self.assertIsInstance(afp.widthInput, QLineEdit)
            self.assertEqual(afp.widthInput.text(), "1")
            self.assertEqual(afp.widthInput, afp.layout().itemAtPosition(r, 1).widget())

            # w line, no chi2
            afp = AddLineFromPoint(self.mainW, None, "", 0, [], [], line=line)
            self.assertEqual(afp.headerCut, 0)
            self.assertFalse(afp.askChi2)
            self.assertEqual(afp.line, line)
            self.assertEqual(afp.desc, "description")
            self.assertIsInstance(afp.chi2ParamConfigs, list)
            self.assertIsInstance(afp.stackWidget, QStackedWidget)
            self.assertIsInstance(afp.paramWidget, QWidget)
            self.assertIsInstance(afp.paramWidget.layout(), QHBoxLayout)
            self.assertEqual(afp.stackWidget.widget(0), afp.paramWidget)
            self.assertIsInstance(afp.chi2Widget, QWidget)
            self.assertIsInstance(afp.chi2Widget.layout(), QGridLayout)
            self.assertEqual(afp.stackWidget.widget(1), afp.chi2Widget)

            r = 0
            self.assertIsInstance(afp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                afp.layout().itemAtPosition(r, 0).widget().text(), afp.desc
            )
            self.assertFalse(hasattr(afp, "nuclideCombo"))
            self.assertFalse(hasattr(afp, "addChi2Line"))
            self.assertIsInstance(afp.chi2Check, QCheckBox)
            self.assertEqual(afp.chi2Check.toolTip(), PGText.plotChi2AskCheckToolTip)
            self.assertEqual(afp.chi2Check.isChecked(), bool(line.chi2))
            self.assertTrue(afp.chi2Check.isHidden())
            r += 1
            self.assertIsInstance(
                afp.layout().itemAtPosition(r, 0).widget(), QStackedWidget
            )
            self.assertEqual(
                afp.layout().itemAtPosition(r, 0).widget(),
                afp.stackWidget,
            )
            r += 1
            self.assertIsInstance(afp.layout().itemAtPosition(r, 0).widget(), QFrame)
            self.assertEqual(
                afp.layout().itemAtPosition(r, 0).widget().frameShape(), QFrame.HLine
            )
            r += 1
            self.assertIsInstance(afp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                afp.layout().itemAtPosition(r, 0).widget().text(), PGText.plotLineLabel
            )
            self.assertIsInstance(afp.labelInput, QLineEdit)
            self.assertEqual(afp.labelInput.text(), "label")
            self.assertEqual(afp.labelInput, afp.layout().itemAtPosition(r, 2).widget())
            r += 1
            self.assertIsInstance(afp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                afp.layout().itemAtPosition(r, 0).widget().text(), PGText.plotLineColor
            )
            self.assertIsInstance(afp.colorInput, QComboBox)
            self.assertTrue(afp.colorInput.isEditable())
            self.assertEqual(afp.colorInput.currentText(), "r")
            self.assertEqual(afp.colorInput, afp.layout().itemAtPosition(r, 1).widget())
            self.assertIsInstance(afp.layout().itemAtPosition(r, 2).widget(), PGLabel)
            self.assertEqual(
                afp.layout().itemAtPosition(r, 2).widget().text(), PGText.plotLineStyle
            )
            self.assertIsInstance(afp.styleCombo, QComboBox)
            for i, f in enumerate(styleOptions):
                self.assertEqual(afp.styleCombo.itemText(i), "%s" % f)
            self.assertEqual(afp.styleCombo.currentText(), ":")
            self.assertEqual(afp.styleCombo, afp.layout().itemAtPosition(r, 3).widget())
            r += 1
            self.assertIsInstance(afp.layout().itemAtPosition(r, 2).widget(), PGLabel)
            self.assertEqual(
                afp.layout().itemAtPosition(r, 2).widget().text(), PGText.plotLineMarker
            )
            self.assertIsInstance(afp.markerCombo, QComboBox)
            for i, f in enumerate(markerOptions):
                self.assertEqual(afp.markerCombo.itemText(i), "%s" % f)
            self.assertEqual(afp.markerCombo.currentText(), "o")
            self.assertEqual(
                afp.markerCombo, afp.layout().itemAtPosition(r, 3).widget()
            )
            self.assertIsInstance(afp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                afp.layout().itemAtPosition(r, 0).widget().text(), PGText.plotLineWidth
            )
            self.assertIsInstance(afp.widthInput, QLineEdit)
            self.assertEqual(afp.widthInput.text(), "1.12")
            self.assertEqual(afp.widthInput, afp.layout().itemAtPosition(r, 1).widget())

            # w line, w chi2
            afp = Add1DLineFromPoint(self.mainW, None, "", 0, [], [], line=line)
            self.assertIsInstance(afp.chi2ParamConfigs, list)
            self.assertIsInstance(afp.stackWidget, QStackedWidget)
            self.assertIsInstance(afp.paramWidget, QWidget)
            self.assertIsInstance(afp.paramWidget.layout(), QHBoxLayout)
            self.assertEqual(afp.stackWidget.widget(0), afp.paramWidget)
            self.assertIsInstance(afp.chi2Widget, QWidget)
            self.assertIsInstance(afp.chi2Widget.layout(), QGridLayout)
            self.assertEqual(afp.stackWidget.widget(1), afp.chi2Widget)

            self.assertEqual(afp.headerCut, 6)
            self.assertTrue(afp.askChi2)
            self.assertEqual(afp.line, line)
            self.assertEqual(afp.desc, "description")

            r = 0
            self.assertIsInstance(afp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                afp.layout().itemAtPosition(r, 0).widget().text(), afp.desc
            )
            self.assertFalse(hasattr(afp, "nuclideCombo"))
            self.assertFalse(hasattr(afp, "addChi2Line"))
            self.assertIsInstance(afp.chi2Check, QCheckBox)
            self.assertEqual(afp.chi2Check.toolTip(), PGText.plotChi2AskCheckToolTip)
            self.assertEqual(afp.chi2Check.isChecked(), bool(line.chi2))
            self.assertTrue(afp.chi2Check.isHidden())
            r += 1
            self.assertIsInstance(
                afp.layout().itemAtPosition(r, 0).widget(), QStackedWidget
            )
            self.assertEqual(
                afp.layout().itemAtPosition(r, 0).widget(),
                afp.stackWidget,
            )
            r += 1
            self.assertIsInstance(afp.layout().itemAtPosition(r, 0).widget(), QFrame)
            self.assertEqual(
                afp.layout().itemAtPosition(r, 0).widget().frameShape(), QFrame.HLine
            )
            r += 1
            self.assertIsInstance(afp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                afp.layout().itemAtPosition(r, 0).widget().text(), PGText.plotLineLabel
            )
            self.assertIsInstance(afp.labelInput, QLineEdit)
            self.assertEqual(afp.labelInput.text(), "label")
            self.assertEqual(afp.labelInput, afp.layout().itemAtPosition(r, 2).widget())
            r += 1
            self.assertIsInstance(afp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                afp.layout().itemAtPosition(r, 0).widget().text(), PGText.plotLineColor
            )
            self.assertIsInstance(afp.colorInput, QComboBox)
            self.assertTrue(afp.colorInput.isEditable())
            self.assertEqual(afp.colorInput.currentText(), "r")
            self.assertEqual(afp.colorInput, afp.layout().itemAtPosition(r, 1).widget())
            self.assertIsInstance(afp.layout().itemAtPosition(r, 2).widget(), PGLabel)
            self.assertEqual(
                afp.layout().itemAtPosition(r, 2).widget().text(), PGText.plotLineStyle
            )
            self.assertIsInstance(afp.styleCombo, QComboBox)
            for i, f in enumerate(styleOptions):
                self.assertEqual(afp.styleCombo.itemText(i), "%s" % f)
            self.assertEqual(afp.styleCombo.currentText(), ":")
            self.assertEqual(afp.styleCombo, afp.layout().itemAtPosition(r, 3).widget())
            r += 1
            self.assertIsInstance(afp.layout().itemAtPosition(r, 2).widget(), PGLabel)
            self.assertEqual(
                afp.layout().itemAtPosition(r, 2).widget().text(), PGText.plotLineMarker
            )
            self.assertIsInstance(afp.markerCombo, QComboBox)
            for i, f in enumerate(markerOptions):
                self.assertEqual(afp.markerCombo.itemText(i), "%s" % f)
            self.assertEqual(afp.markerCombo.currentText(), "o")
            self.assertEqual(
                afp.markerCombo, afp.layout().itemAtPosition(r, 3).widget()
            )
            self.assertIsInstance(afp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                afp.layout().itemAtPosition(r, 0).widget().text(), PGText.plotLineWidth
            )
            self.assertIsInstance(afp.widthInput, QLineEdit)
            self.assertEqual(afp.widthInput.text(), "1.12")
            self.assertEqual(afp.widthInput, afp.layout().itemAtPosition(r, 1).widget())

        def test_nuclideCombo(self):
            """test nuclideCombo"""
            afp = AddLineFromPoint(
                self.mainW,
                0,
                "desc",
                2,
                testConfig.sampleRunner.parthenopeHeader,
                testConfig.sampleRunner.nuclidesEvolution[0],
            )
            c = afp._nuclideCombo()
            self.assertIsInstance(c, QComboBox)
            self.assertEqual(c.toolTip(), PGText.plotSelectNuclideToolTip)
            for i, f in enumerate(afp.header[afp.headerCut :]):
                self.assertEqual(c.itemText(i), "%s" % f)

        def test_getLine(self):
            """test getLine"""
            l = PGLine([0], [1], 0)
            afp = Add1DLineFromPoint(
                self.mainW,
                0,
                "desc",
                0,
                testConfig.sampleRunner.parthenopeHeader,
                testConfig.sampleRunner.nuclidesEvolution[0],
            )
            afp.nuclideCombo.setCurrentText("thetah")
            afp.labelInput.setText("line1")
            afp.colorInput.setCurrentText("b")
            afp.styleCombo.setCurrentText("--")
            afp.markerCombo.setCurrentText("x")
            afp.widthInput.setText("3.2")
            afp.chi2Check.setChecked(True)
            afp.chi2ParamConfigs[0].paramCombo.setCurrentText("thetah")
            afp.chi2ParamConfigs[0].chi2Mean.setText("3.2")
            afp.chi2ParamConfigs[0].chi2Std.setText("1.1")
            with patch(
                "parthenopegui.plotter.PGLine",
                return_value=l,
                autospec=USE_AUTOSPEC_CLASS,
            ) as _l, patch(
                "parthenopegui.plotter.AddFromPoint._getChi2Conv",
                return_value=np.array([0.0, 1.0, 2.0]),
                autospec=True,
            ) as _gcc, patch(
                "parthenopegui.plotter.AddFromPoint._getChi2Type",
                return_value=True,
                autospec=True,
            ) as _gct, patch(
                "parthenopegui.plotter.AddFromPoint._getDescLineParam",
                return_value="chi2(%s) ",
                autospec=True,
            ) as _gdp:
                self.assertEqual(afp.getLine(), l)
                _gcc.assert_called_once()
                self.assertEqual(_gcc.call_args[0][0], afp)
                self.assertEqualArray(
                    _gcc.call_args[0][1],
                    chi2func(
                        testConfig.sampleRunner.nuclidesEvolution[0][
                            :, afp.headerCut + afp.nuclideCombo.currentIndex()
                        ],
                        3.2,
                        1.1,
                    ),
                )
                _gct.assert_called_once_with(afp, True)
                _gdp.assert_called_once_with(afp)
                self.assertEqualArray(
                    _l.call_args[0][0],
                    testConfig.sampleRunner.nuclidesEvolution[0][:, 0],
                )
                self.assertEqualArray(
                    _l.call_args[0][1],
                    np.array([0.0, 1.0, 2.0]),
                )
                self.assertEqual(
                    _l.call_args[1],
                    {
                        "c": "b",
                        "c2": True,
                        "d": "desc: chi2(thetah) ",
                        "l": "line1",
                        "m": "x",
                        "s": "--",
                        "w": 3.2,
                        "xl": "N_eff",
                        "yl": "thetah",
                    },
                )
            afp._newChi2Widget()
            afp.chi2ParamConfigs[0].chi2Mean.setText("1e-9")
            afp.chi2ParamConfigs[0].chi2Std.setText("5e-11")
            afp.chi2ParamConfigs[1].paramCombo.setCurrentText("Y_H")
            afp.chi2ParamConfigs[1].chi2Mean.setText("5e-6")
            afp.chi2ParamConfigs[1].chi2Std.setText("1e-6")
            with patch(
                "parthenopegui.plotter.PGLine",
                return_value=l,
                autospec=USE_AUTOSPEC_CLASS,
            ) as _l, patch(
                "parthenopegui.plotter.AddFromPoint._getChi2Conv",
                return_value=np.array([0.0, 1.0, 2.0]),
                autospec=True,
            ) as _gcc, patch(
                "parthenopegui.plotter.AddFromPoint._getChi2Type",
                return_value=True,
                autospec=True,
            ) as _gct, patch(
                "parthenopegui.plotter.AddFromPoint._getDescLineParam",
                return_value="chi2(%s) ",
                autospec=True,
            ) as _gdp:
                self.assertEqual(afp.getLine(), l)
                _gcc.assert_called_once()
                self.assertEqual(_gcc.call_args[0][0], afp)
                self.assertEqualArray(
                    _gcc.call_args[0][1],
                    chi2func(
                        testConfig.sampleRunner.nuclidesEvolution[0][
                            :,
                            afp.headerCut
                            + afp.chi2ParamConfigs[0].paramCombo.currentIndex(),
                        ],
                        1e-9,
                        5e-11,
                    )
                    + chi2func(
                        testConfig.sampleRunner.nuclidesEvolution[0][
                            :,
                            afp.headerCut
                            + afp.chi2ParamConfigs[1].paramCombo.currentIndex(),
                        ],
                        5e-6,
                        1e-6,
                    ),
                )
                _gct.assert_called_once_with(afp, True)
                self.assertEqual(_gdp.call_count, 2)
                self.assertEqualArray(
                    _l.call_args[0][0],
                    testConfig.sampleRunner.nuclidesEvolution[0][:, 0],
                )
                self.assertEqualArray(
                    _l.call_args[0][1],
                    np.array([0.0, 1.0, 2.0]),
                )
                self.assertEqual(
                    _l.call_args[1],
                    {
                        "c": "b",
                        "c2": True,
                        "d": "desc: chi2(thetah) chi2(Y_H) ",
                        "l": "line1",
                        "m": "x",
                        "s": "--",
                        "w": 3.2,
                        "xl": "N_eff",
                        "yl": "thetah",
                    },
                )
            afp.askChi2 = False
            with patch("parthenopegui.plotter.PGLine", return_value=l) as _l, patch(
                "parthenopegui.plotter.AddFromPoint._getChi2Conv", autospec=True
            ) as _gcc, patch(
                "parthenopegui.plotter.AddFromPoint._getChi2Type",
                return_value=False,
                autospec=True,
            ) as _gct, patch(
                "parthenopegui.plotter.AddFromPoint._getDescLineParam", autospec=True
            ) as _gdp:
                self.assertEqual(afp.getLine(), l)
                self.assertEqual(_gcc.call_count, 0)
                _gct.assert_called_once_with(afp, False)
                self.assertEqual(_gdp.call_count, 0)
                self.assertEqualArray(
                    _l.call_args[0][0],
                    testConfig.sampleRunner.nuclidesEvolution[0][:, 0],
                )
                self.assertEqualArray(
                    _l.call_args[0][1],
                    testConfig.sampleRunner.nuclidesEvolution[0][
                        :, afp.headerCut + afp.nuclideCombo.currentIndex()
                    ],
                )
                self.assertEqual(
                    _l.call_args[1],
                    {
                        "c": "b",
                        "c2": False,
                        "d": "desc: thetah",
                        "l": "line1",
                        "m": "x",
                        "s": "--",
                        "w": 3.2,
                        "xl": "N_eff",
                        "yl": "thetah",
                    },
                )
            afp.askChi2 = True
            afp.chi2Check.setChecked(False)
            with patch("parthenopegui.plotter.PGLine", return_value=l) as _l, patch(
                "parthenopegui.plotter.AddFromPoint._getChi2Conv", autospec=True
            ) as _gcc, patch(
                "parthenopegui.plotter.AddFromPoint._getChi2Type",
                return_value=False,
                autospec=True,
            ) as _gct, patch(
                "parthenopegui.plotter.AddFromPoint._getDescLineParam", autospec=True
            ) as _gdp:
                self.assertEqual(afp.getLine(), l)
                self.assertEqual(_gcc.call_count, 0)
                _gct.assert_called_once_with(afp, False)
                self.assertEqual(_gdp.call_count, 0)
                self.assertEqualArray(
                    _l.call_args[0][0],
                    testConfig.sampleRunner.nuclidesEvolution[0][:, 0],
                )
                self.assertEqualArray(
                    _l.call_args[0][1],
                    testConfig.sampleRunner.nuclidesEvolution[0][
                        :, afp.headerCut + afp.nuclideCombo.currentIndex()
                    ],
                )
                self.assertEqual(
                    _l.call_args[1],
                    {
                        "c": "b",
                        "c2": False,
                        "d": "desc: thetah",
                        "l": "line1",
                        "m": "x",
                        "s": "--",
                        "w": 3.2,
                        "xl": "N_eff",
                        "yl": "thetah",
                    },
                )

            line = PGLine(
                [0, 1, 2],
                [3, 4, 5],
                1,
                c="r",
                c2="chi2",
                d="desc",
                l="label",
                m="o",
                s=":",
                w=1.12,
                xl="xl",
                yl="yl",
            )
            afp = Add1DLineFromPoint(self.mainW, None, "", 0, [], [], line=line)
            afp.labelInput.setText("line1")
            afp.colorInput.setCurrentText("b")
            afp.styleCombo.setCurrentText("--")
            afp.markerCombo.setCurrentText("x")
            afp.widthInput.setText("3.2")
            afp._newChi2Widget()
            afp.chi2ParamConfigs[0].paramCombo.setCurrentText("thetah")
            afp.chi2ParamConfigs[0].chi2Mean.setText("3.2")
            afp.chi2ParamConfigs[0].chi2Std.setText("1.1")
            with patch("parthenopegui.plotter.PGLine", return_value=l) as _l:
                self.assertEqual(afp.getLine(), l)
                _l.assert_called_once_with(
                    line.x,
                    line.y,
                    line.pointRow,
                    d=line.description,
                    l="line1",
                    c="b",
                    s="--",
                    m="x",
                    w=3.2,
                    xl=line.xlabel,
                    yl=line.ylabel,
                    c2=line.chi2,
                )
            afp.askChi2 = False
            with patch("parthenopegui.plotter.PGLine", return_value=l) as _l:
                self.assertEqual(afp.getLine(), l)
                _l.assert_called_once_with(
                    line.x,
                    line.y,
                    line.pointRow,
                    d=line.description,
                    l="line1",
                    c="b",
                    s="--",
                    m="x",
                    w=3.2,
                    xl=line.xlabel,
                    yl=line.ylabel,
                    c2=line.chi2,
                )
            afp.askChi2 = True
            afp.chi2Check.setChecked(False)
            with patch("parthenopegui.plotter.PGLine", return_value=l) as _l:
                self.assertEqual(afp.getLine(), l)
                _l.assert_called_once_with(
                    line.x,
                    line.y,
                    line.pointRow,
                    d=line.description,
                    l="line1",
                    c="b",
                    s="--",
                    m="x",
                    w=3.2,
                    xl=line.xlabel,
                    yl=line.ylabel,
                    c2=line.chi2,
                )

        def test_testAccept(self):
            """test testAccept"""
            line = PGLine(
                [0, 1, 2],
                [3, 4, 5],
                1,
                c="r",
                c2="chi2",
                d="desc",
                l="label",
                m="o",
                s=":",
                w=1.12,
                xl="xl",
                yl="yl",
            )
            afp = Add1DLineFromPoint(self.mainW, None, "", 0, [], [], line=line)
            afp.accept = MagicMock()
            afp.colorInput.setCurrentText("abc")
            for t in ("-2.4", "abc"):
                afp.widthInput.setText(t)
                with patch("logging.Logger.warning") as _w:
                    afp.testAccept()
                    _w.assert_called_once_with(PGText.plotInvalidWidth % t)
            afp.widthInput.setText("1.12")
            afp._newChi2Widget()
            afp.chi2ParamConfigs[0].chi2Mean.setText("abc")
            with patch("logging.Logger.warning") as _w:
                afp.testAccept()
                _w.assert_called_once_with(PGText.plotInvalidMean % "abc")
            afp.chi2ParamConfigs[0].chi2Mean.setText("1.")
            for t in ("-2.4", "abc"):
                afp.chi2ParamConfigs[0].chi2Std.setText(t)
                with patch("logging.Logger.warning") as _w:
                    afp.testAccept()
                    _w.assert_called_once_with(PGText.plotInvalidStddev % t)
            afp.chi2ParamConfigs[0].chi2Std.setText("1.1")
            afp._newChi2Widget()
            with patch("logging.Logger.warning") as _w:
                afp.testAccept()
                _w.assert_called_once_with(PGText.plotInvalidMean % "")
            afp.chi2ParamConfigs[1].chi2Mean.setText("1.")
            for t in ("-2.4", "abc"):
                afp.chi2ParamConfigs[1].chi2Std.setText(t)
                with patch("logging.Logger.warning") as _w:
                    afp.testAccept()
                    _w.assert_called_once_with(PGText.plotInvalidStddev % t)
            afp.chi2ParamConfigs[1].chi2Std.setText("1.")
            afp.askChi2 = False
            afp.chi2ParamConfigs[0].chi2Mean.setText("abc")
            with patch("logging.Logger.warning") as _w, patch(
                "matplotlib.colors.is_color_like", return_value=False
            ) as _cl:
                afp.testAccept()
                _cl.assert_called_once_with("abc")
                _w.assert_called_once_with(PGText.plotInvalidColor % "abc")
                afp.askChi2 = True
                afp.chi2Check.setChecked(False)
                _w.reset_mock()
                afp.testAccept()
                _w.assert_called_once_with(PGText.plotInvalidColor % "abc")
            with patch("logging.Logger.warning") as _w, patch(
                "matplotlib.colors.is_color_like", side_effect=[False, True]
            ) as _cl:
                afp.testAccept()
                _cl.assert_called_once_with("abc")
                _w.assert_called_once_with(PGText.plotInvalidColor % "abc")
                self.assertEqual(afp.accept.call_count, 0)
                afp.testAccept()
                _w.assert_called_once()
            afp.accept.assert_called_once_with()
            afp.askChi2 = True
            afp.chi2Check.setChecked(True)
            afp.accept.reset_mock()
            with patch("logging.Logger.warning") as _w:
                afp.testAccept()
            afp.chi2ParamConfigs[0].chi2Mean.setText("1.")
            afp.chi2ParamConfigs[1].chi2Mean.setText("abc")
            with patch("logging.Logger.warning") as _w:
                afp.testAccept()
            afp.chi2ParamConfigs[1].chi2Mean.setText("1.")
            with patch("logging.Logger.warning") as _w, patch(
                "matplotlib.colors.is_color_like", return_value=True
            ) as _cl:
                afp.testAccept()
                _w.assert_not_called()
            afp.accept.assert_called_once_with()

    class TestPGListPointsModel(PGTestCasewMainW):
        """Test the PGListPointsModel class"""

        def test_init(self):
            """test __init__"""
            p = QWidget()
            dataList = ["a"]
            lp = PGListPointsModel(dataList, p, self.mainW)
            self.assertIsInstance(lp, PGEmptyTableModel)
            self.assertEqual(lp.mainW, self.mainW)
            self.assertEqual(lp.dataList, dataList)
            self.assertEqual(lp.inUse, [[]])
            self.assertEqual(
                lp.header, ["Used"] + list([p for p in Parameters.paramOrder])
            )
            self.assertTrue(hasattr(lp, "headerToolTips"))

        def test_data(self):
            """test data"""
            pw = QWidget()
            dataList = [
                [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                [21.0, 22.0, 23.0, 24.0, 25.0, 26.0],
            ]
            lp = PGListPointsModel(np.asarray(dataList), pw, self.mainW)
            qp = QPixmap()
            qp1 = QPixmap()
            qp.scaledToHeight = MagicMock(return_value=qp1)
            idx = QModelIndex()
            idx.isValid = MagicMock(return_value=False)
            idx.row = MagicMock(return_value=0)
            idx.column = MagicMock(return_value=10)

            self.assertEqual(lp.data(idx, Qt.DisplayRole), None)
            idx.isValid.assert_called_once()
            self.assertEqual(idx.row.call_count, 0)
            idx.isValid = MagicMock(return_value=True)

            with patch("logging.Logger.exception") as _e:
                self.assertEqual(lp.data(idx, Qt.DisplayRole), None)
                _e.assert_called_once_with(PGText.errorCannotFindIndex)
            idx.row = MagicMock(return_value=10)
            idx.column = MagicMock(return_value=0)
            with patch("logging.Logger.exception") as _e:
                self.assertEqual(lp.data(idx, Qt.DisplayRole), None)
                _e.assert_called_once_with(PGText.errorCannotFindIndex)
            idx.row = MagicMock(return_value=0)
            idx.column = MagicMock(return_value=0)

            self.assertEqual(lp.data(idx, Qt.DisplayRole), "")
            self.assertEqual(lp.data(idx, Qt.DecorationRole), None)
            lp.inUse[0].append(True)
            with patch("parthenopegui.plotter.QPixmap", return_value=qp) as _qp:
                self.assertEqual(lp.data(idx, Qt.DecorationRole), qp1)
                _qp.assert_called_once_with(":/images/dialog-ok-apply.png")

            for i, p in enumerate(Parameters.paramOrder):
                idx.column = MagicMock(return_value=i + 1)
                self.assertEqual(lp.data(idx, Qt.DecorationRole), None)
                self.assertEqual(
                    lp.data(idx, Qt.DisplayRole),
                    Configuration.formatParam % dataList[0][i],
                )

            dataList = [
                [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                ["1", "b", 3, "d", 5.0, "f"],
            ]
            idx.row = MagicMock(return_value=1)
            lp = PGListPointsModel(np.asarray(dataList), pw, self.mainW)
            for i, p in enumerate(Parameters.paramOrder):
                idx.column = MagicMock(return_value=i + 1)
                self.assertEqual(lp.data(idx, Qt.DecorationRole), None)
                self.assertEqual(lp.data(idx, Qt.DisplayRole), "%s" % dataList[1][i])

            idx.column = MagicMock(return_value=1)
            lp1 = PGListPointsModel(dataList, pw, self.mainW)
            with patch("logging.Logger.exception") as _e:
                self.assertEqual(lp1.data(idx, Qt.DisplayRole), None)
                _e.assert_called_once_with(
                    PGText.warningWrongType % ("self.dataList", "np.ndarray")
                )

        def test_headerData(self):
            """test headerData"""
            p = QWidget()
            lp = PGListPointsModel(["a"], p, self.mainW)
            self.assertEqual(lp.headerData(0, Qt.Horizontal, Qt.DisplayRole), "Used")
            self.assertEqual(lp.headerData(0, Qt.Vertical, Qt.DisplayRole), None)
            self.assertEqual(lp.headerData(0, Qt.Horizontal, Qt.DecorationRole), None)
            for i, p in enumerate(Parameters.paramOrder):
                self.assertEqual(
                    lp.headerData(1 + i, Qt.Horizontal, Qt.DisplayRole), None
                )
                self.assertEqual(
                    lp.headerData(1 + i, Qt.Vertical, Qt.DisplayRole), None
                )
                self.assertEqual(
                    lp.headerData(1 + i, Qt.Horizontal, Qt.DecorationRole),
                    self.mainW.parameters.all[lp.header[1 + i]].fig,
                )

        def test_replaceDataList(self):
            """test replaceDataList"""
            p = QWidget()
            a = MagicMock()
            c = MagicMock()
            dataList = ["a"]
            lp = PGListPointsModel(dataList, p, self.mainW)
            lp.layoutAboutToBeChanged.connect(a)
            lp.layoutChanged.connect(c)
            dataList1 = ["b", "c"]
            lp.replaceDataList(dataList1)
            a.assert_called_once()
            c.assert_called_once()
            self.assertEqual(lp.dataList, dataList1)
            self.assertEqual(lp.inUse, [[], []])

    class ExtAGPWLines(AbundancesGenericPanel):
        """Class to define self.hasLines and test AbundancesGenericPanel"""

        def __init__(self, *args, **kwargs):
            self.hasLines = True
            AbundancesGenericPanel.__init__(self, *args, **kwargs)

    class ExtAGPWoLines(AbundancesGenericPanel):
        """Class to define self.hasLines and test AbundancesGenericPanel"""

        def __init__(self, *args, **kwargs):
            self.hasLines = False
            AbundancesGenericPanel.__init__(self, *args, **kwargs)

    class TestAbundancesGenericPanel(PGTestCasewMainW):
        """Test the AbundancesGenericPanel class"""

        def test_init(self):
            """test __init__"""
            p = QWidget()

            agp = ExtAGPWLines(p, self.mainW)
            self.assertIsInstance(agp, AbundancesGenericPanel)
            self.assertIsInstance(agp, QFrame)
            self.assertEqual(agp.mainW, self.mainW)
            self.assertEqual(agp.runner, None)
            self.assertIsInstance(agp.layout(), QVBoxLayout)
            self.assertEqual(agp.dataList, [])

            self.assertIsInstance(agp.layout().itemAt(0).widget(), PGLabel)
            self.assertEqual(
                agp.layout().itemAt(0).widget().text(), PGText.plotAvailablePoints
            )

            self.assertIsInstance(agp.tableViewPts, QTableView)
            self.assertEqual(agp.tableViewPts, agp.layout().itemAt(1).widget())
            with patch(
                "parthenopegui.plotter.AbundancesGenericPanel.cellDoubleClickPoints",
                autospec=True,
            ) as _dc:
                agp.tableViewPts.doubleClicked.emit(QModelIndex())
                _dc.assert_called_once()
            self.assertIsInstance(agp.pointsModel, PGListPointsModel)
            self.assertEqual(agp.tableViewPts.model(), agp.pointsModel)

            self.assertIsInstance(agp.layout().itemAt(2).widget(), PGLabel)
            self.assertEqual(
                agp.layout().itemAt(2).widget().text(), PGText.plotLinesInUse
            )

            self.assertIsInstance(agp.tableViewObjects, QTableView)
            self.assertEqual(agp.tableViewObjects, agp.layout().itemAt(3).widget())
            with patch(
                "parthenopegui.plotter.AbundancesGenericPanel.cellDoubleClickObjects",
                autospec=True,
            ) as _dc:
                agp.tableViewObjects.doubleClicked.emit(QModelIndex())
                _dc.assert_called_once()
            self.assertIsInstance(agp.objectsModel, PGListLinesModel)
            self.assertEqual(agp.tableViewObjects.model(), agp.objectsModel)

            agp = ExtAGPWoLines(p, self.mainW)
            self.assertIsInstance(agp, AbundancesGenericPanel)
            self.assertIsInstance(agp, QFrame)
            self.assertEqual(agp.mainW, self.mainW)
            self.assertEqual(agp.runner, None)
            self.assertIsInstance(agp.layout(), QVBoxLayout)
            self.assertEqual(agp.dataList, [])

            self.assertIsInstance(agp.layout().itemAt(0).widget(), PGLabel)
            self.assertEqual(
                agp.layout().itemAt(0).widget().text(), PGText.plotAvailablePoints
            )

            self.assertIsInstance(agp.tableViewPts, QTableView)
            self.assertEqual(agp.tableViewPts, agp.layout().itemAt(1).widget())
            with patch(
                "parthenopegui.plotter.AbundancesGenericPanel.cellDoubleClickPoints",
                autospec=True,
            ) as _dc:
                agp.tableViewPts.doubleClicked.emit(QModelIndex())
                _dc.assert_called_once()
            self.assertIsInstance(agp.pointsModel, PGListPointsModel)
            self.assertEqual(agp.tableViewPts.model(), agp.pointsModel)

            self.assertIsInstance(agp.layout().itemAt(2).widget(), PGLabel)
            self.assertEqual(
                agp.layout().itemAt(2).widget().text(), PGText.plotContoursInUse
            )

            self.assertIsInstance(agp.tableViewObjects, QTableView)
            self.assertEqual(agp.tableViewObjects, agp.layout().itemAt(3).widget())
            with patch(
                "parthenopegui.plotter.AbundancesGenericPanel.cellDoubleClickObjects",
                autospec=True,
            ) as _dc:
                agp.tableViewObjects.doubleClicked.emit(QModelIndex())
                _dc.assert_called_once()
            self.assertIsInstance(agp.objectsModel, PGListContoursModel)
            self.assertEqual(agp.tableViewObjects.model(), agp.objectsModel)

        def test_addPlotObject(self):
            """test addPlotObject"""
            p = QWidget()
            agp = ExtAGPWLines(p, self.mainW)
            l = PGLine([0], [1], 0)
            agp.tableViewObjects.resizeColumnsToContents = MagicMock()
            agp.tableViewObjects.resizeRowsToContents = MagicMock()
            with patch(
                "parthenopegui.plotter.AbundancesGenericPanel.optimizePlotSettings",
                autospec=True,
            ) as _ops, patch(
                "parthenopegui.plotter.PGListLinesModel.addLineToData", autospec=True
            ) as _al:
                agp.addPlotObject(l)
                _al.assert_called_once_with(agp.objectsModel, l)
                _ops.assert_called_once_with(agp)
            agp.tableViewObjects.resizeColumnsToContents.assert_called_once_with()
            agp.tableViewObjects.resizeRowsToContents.assert_called_once_with()
            with patch(
                "parthenopegui.plotter.AbundancesGenericPanel.optimizePlotSettings",
                autospec=True,
            ) as _ops, patch(
                "parthenopegui.plotter.PGListLinesModel.replaceLine", autospec=True
            ) as _rl:
                agp.addPlotObject(l, 1)
                _rl.assert_called_once_with(agp.objectsModel, l, 1)
                _ops.assert_called_once_with(agp)

        def test_clearCurrent(self):
            """test clearCurrent"""
            p = QWidget()
            agp = ExtAGPWLines(p, self.mainW)
            l = PGLine([0], [1], 0)
            with patch(
                "parthenopegui.plotter.AbundancesGenericPanel.optimizePlotSettings"
            ):
                agp.addPlotObject(l)
                agp.addPlotObject(l)
            self.assertEqual(agp.objectsModel.rowCount(), 2)
            agp.pointsModel.inUse = [[True, True]]
            agp.clearCurrent()
            self.assertEqual(agp.objectsModel.rowCount(), 0)
            self.assertEqual(agp.pointsModel.inUse, [[]])
            with patch(
                "parthenopegui.plotter.AbundancesGenericPanel.optimizePlotSettings"
            ):
                agp.addPlotObject(l)
                agp.addPlotObject(l)
            self.assertEqual(agp.objectsModel.rowCount(), 2)
            agp.pointsModel.inUse = []
            agp.clearCurrent()
            self.assertEqual(agp.objectsModel.rowCount(), 0)
            self.assertEqual(agp.pointsModel.inUse, [])

        def test_optimizePlotSettings(self):
            """test optimizePlotSettings"""
            p = QWidget()
            ps = self.mainW.plotPanel.plotSettings
            ps.updateAutomaticSettings({"xlims": (-1, 1), "ylims": (-1, 1)})
            agp = ExtAGPWLines(p, self.mainW)
            agp.optimizePlotSettings()
            self.assertEqual(ps.xlims, (-1, 1))
            self.assertEqual(ps.ylims, (-1, 1))
            agp.invert = True
            with patch(
                "parthenopegui.plotter.AbundancesGenericPanel.optimizePlotSettings"
            ):
                agp.addPlotObject(PGLine([0, 1, 2], [1, 2, 3], 0, xl="x", yl="y"))
                agp.addPlotObject(
                    PGLine([0.5, 1, 2.2], [0.8, 2, 3.1], 0, xl="x", yl="y")
                )
            with patch(
                "parthenopegui.plotter.PlotSettings.updateAutomaticSettings"
            ) as _f:
                agp.optimizePlotSettings()
                _f.assert_called_once_with(
                    {
                        "xlims": (2.2, 0),
                        "ylims": (0.8, 3.1),
                        "xlabel": "x",
                        "ylabel": "y",
                        "zlabel": "",
                    }
                )
            # test with lines
            agp.optimizePlotSettings()
            self.assertEqual(ps.xlims, (2.2, 0))
            self.assertEqual(ps.ylims, (0.8, 3.1))
            self.assertEqual(ps.xlabel, "x")
            self.assertEqual(ps.ylabel, "y")
            self.assertEqual(ps.zlabel, "")
            # xl/yl is not the same:
            agp.invert = False
            with patch(
                "parthenopegui.plotter.AbundancesGenericPanel.optimizePlotSettings"
            ):
                agp.addPlotObject(
                    PGLine(
                        [0, 1, 2],
                        [0.8, 2, 3.1],
                        0,
                        xl="x1",
                        yl="y",
                        c2="chi2",
                    )
                )
            agp.optimizePlotSettings()
            self.assertEqual(ps.xlims, (0, 2.2))
            self.assertEqual(ps.ylims, (0.8, 3.1))
            self.assertEqual(ps.xlabel, "")
            self.assertEqual(ps.ylabel, "")

            # test with contours
            agp = ExtAGPWoLines(p, self.mainW)
            ps.updateAutomaticSettings({"xlims": (-1, 1), "ylims": (-1, 1)})
            agp.optimizePlotSettings()
            self.assertEqual(ps.xlims, (-1, 1))
            self.assertEqual(ps.ylims, (-1, 1))
            agp.invert = True
            with patch(
                "parthenopegui.plotter.AbundancesGenericPanel.optimizePlotSettings"
            ):
                agp.addPlotObject(
                    PGContour(
                        [0, 1],
                        [2, 3],
                        np.asarray([[4, 5], [6, 7]]),
                        0,
                        xl="x",
                        yl="y",
                        zl="z",
                    )
                )
                agp.addPlotObject(
                    PGContour(
                        [0, 1],
                        [2, 3],
                        np.asarray([[4, 5], [6, 7]]),
                        0,
                        xl="x",
                        yl="y",
                        zl="z",
                    )
                )
            agp.optimizePlotSettings()
            self.assertEqual(ps.xlims, (1, 0))
            self.assertEqual(ps.ylims, (2, 3))
            self.assertEqual(ps.xlabel, "x")
            self.assertEqual(ps.ylabel, "y")
            self.assertEqual(ps.zlabel, "z")
            # xl/yl/zl is not the same:
            agp.invert = False
            with patch(
                "parthenopegui.plotter.AbundancesGenericPanel.optimizePlotSettings"
            ):
                agp.addPlotObject(
                    PGContour(
                        [0, 1],
                        [2, 3],
                        np.asarray([[4, 5], [6, 7]]),
                        0,
                        xl="x1",
                        yl="y1",
                        zl="z",
                        c2="chi2",
                    )
                )
            agp.optimizePlotSettings()
            self.assertEqual(ps.xlims, (0, 1))
            self.assertEqual(ps.xlabel, "")
            self.assertEqual(ps.ylabel, "")
            self.assertEqual(ps.zlabel, "")

        def test_cellDoubleClickObjects(self):
            """test cellDoubleClickObjects"""
            p = QWidget()
            cpc = self.mainW.plotPanel.currentPlotContent
            l1 = PGLine([0], [1], 0)
            l2 = PGLine([0], [1], 0)
            l3 = PGLine([0], [1], 0)
            c1 = PGContour([0, 1], [2, 3], [[4, 5], [6, 7]], 0)
            c2 = PGContour([0, 1], [2, 3], [[4, 5], [6, 7]], 0)
            c3 = PGContour([0, 1], [2, 3], [[4, 5], [6, 7]], 0)
            agp = ExtAGPWLines(p, self.mainW)
            qmi = QModelIndex()
            qmi.isValid = MagicMock(return_value=False)
            qmi.row = MagicMock(return_value=0)
            qmi.column = MagicMock(return_value=0)
            agp.cellDoubleClickObjects(qmi)
            self.assertEqual(qmi.row.call_count, 0)
            qmi.isValid = MagicMock(return_value=True)
            agp.type = "abc"
            with self.assertRaises(TypeError):
                agp.cellDoubleClickObjects(qmi)
                qmi.row.assert_called_once_with()
                qmi.column.assert_called_once_with()
            agp.type = "evo"

            # test delete
            qmi.column = MagicMock(return_value=agp.objectsModel.header.index("Delete"))
            agp.objectsModel.dataList = [l1, l2, c1]
            cpc.lines = [l1, l2, l3]
            cpc.contours = [c1, c2, c3]
            agp.pointsModel.inUse = [[True], []]
            with patch(
                "parthenopegui.plotter.askYesNo", side_effect=[False, True, True, True]
            ) as _a, patch(
                "parthenopegui.plotter.AbundancesGenericPanel.optimizePlotSettings",
                autospec=True,
            ) as _os, patch(
                "parthenopegui.plotter.MWPlotPanel.refreshPlot", autospec=True
            ) as _rp:
                # answer is no
                agp.cellDoubleClickObjects(qmi)
                _a.assert_called_once_with(PGText.plotAskDeleteLine)
                self.assertEqual(_os.call_count, 0)

                # evo
                agp.cellDoubleClickObjects(qmi)
                _os.assert_called_once_with(agp)
                _rp.assert_called_once_with(self.mainW.plotPanel)
                self.assertEqual(agp.objectsModel.dataList, [l2, c1])
                self.assertEqual(agp.pointsModel.inUse, [[], []])
                self.assertEqual(cpc.lines, [l2, l3])
                self.assertEqual(cpc.contours, [c1, c2, c3])

                # 1d
                agp.type = "1d"
                agp.cellDoubleClickObjects(qmi)
                self.assertEqual(agp.objectsModel.dataList, [c1])
                self.assertEqual(agp.pointsModel.inUse, [[], []])
                self.assertEqual(cpc.lines, [l3])
                self.assertEqual(cpc.contours, [c1, c2, c3])

                # 2d
                agp.type = "2d"
                agp.cellDoubleClickObjects(qmi)
                self.assertEqual(agp.objectsModel.dataList, [])
                self.assertEqual(agp.pointsModel.inUse, [[], []])
                self.assertEqual(cpc.lines, [l3])
                self.assertEqual(cpc.contours, [c2, c3])

            # test move down
            qmi.column = MagicMock(
                return_value=agp.objectsModel.header.index("Move down")
            )
            agp.type = "evo"
            cpc.lines = [l1, l2, l3]
            cpc.contours = [c1, c2, c3]
            with patch(
                "parthenopegui.plotter.AbundancesGenericPanel.addPlotObject",
                autospec=True,
            ) as _ao, patch(
                "parthenopegui.plotter.MWPlotPanel.refreshPlot", autospec=True
            ) as _rp:
                # fail for indexerror
                agp.objectsModel.dataList = [l1, l2]
                qmi.row = MagicMock(return_value=1)
                agp.cellDoubleClickObjects(qmi)
                self.assertEqual(_rp.call_count, 0)
                self.assertEqual(cpc.lines, [l1, l2, l3])
                self.assertEqual(cpc.contours, [c1, c2, c3])
                # evo
                qmi.row = MagicMock(return_value=0)
                agp.cellDoubleClickObjects(qmi)
                _rp.assert_called_once_with(self.mainW.plotPanel)
                _ao.assert_any_call(agp, l2, ix=0)
                _ao.assert_any_call(agp, l1, ix=1)
                self.assertEqual(cpc.lines, [l2, l1, l3])
                self.assertEqual(cpc.contours, [c1, c2, c3])
                _ao.reset_mock()
                # 1d
                agp.type = "1d"
                cpc.lines = [l3, l1, l2]
                agp.cellDoubleClickObjects(qmi)
                _ao.assert_any_call(agp, l2, ix=0)
                _ao.assert_any_call(agp, l1, ix=1)
                self.assertEqual(cpc.lines, [l3, l2, l1])
                self.assertEqual(cpc.contours, [c1, c2, c3])
                # 2d
                agp.objectsModel.dataList = [c1, c2]
                cpc.lines = [l1, l2, l3]
                agp.type = "2d"
                agp.cellDoubleClickObjects(qmi)
                _ao.assert_any_call(agp, c2, ix=0)
                _ao.assert_any_call(agp, c1, ix=1)
                self.assertEqual(cpc.lines, [l1, l2, l3])
                self.assertEqual(cpc.contours, [c2, c1, c3])

            # test move up
            qmi.column = MagicMock(
                return_value=agp.objectsModel.header.index("Move up")
            )
            agp.type = "evo"
            cpc.lines = [l1, l2, l3]
            cpc.contours = [c1, c2, c3]
            with patch(
                "parthenopegui.plotter.AbundancesGenericPanel.addPlotObject",
                autospec=True,
            ) as _ao, patch(
                "parthenopegui.plotter.MWPlotPanel.refreshPlot", autospec=True
            ) as _rp:
                # fail for indexerror
                agp.objectsModel.dataList = [l1, l2]
                qmi.row = MagicMock(return_value=0)
                agp.cellDoubleClickObjects(qmi)
                self.assertEqual(_rp.call_count, 0)
                self.assertEqual(cpc.lines, [l1, l2, l3])
                self.assertEqual(cpc.contours, [c1, c2, c3])
                # evo
                qmi.row = MagicMock(return_value=1)
                agp.cellDoubleClickObjects(qmi)
                _rp.assert_called_once_with(self.mainW.plotPanel)
                _ao.assert_any_call(agp, l2, ix=0)
                _ao.assert_any_call(agp, l1, ix=1)
                self.assertEqual(cpc.lines, [l2, l1, l3])
                self.assertEqual(cpc.contours, [c1, c2, c3])
                _ao.reset_mock()
                # 1d
                agp.type = "1d"
                cpc.lines = [l1, l2, l3]
                agp.cellDoubleClickObjects(qmi)
                _ao.assert_any_call(agp, l2, ix=0)
                _ao.assert_any_call(agp, l1, ix=1)
                self.assertEqual(cpc.lines, [l2, l1, l3])
                self.assertEqual(cpc.contours, [c1, c2, c3])
                # 2d
                agp.objectsModel.dataList = [c1, c2]
                cpc.lines = [l1, l2, l3]
                agp.type = "2d"
                agp.cellDoubleClickObjects(qmi)
                _ao.assert_any_call(agp, c2, ix=0)
                _ao.assert_any_call(agp, c1, ix=1)
                self.assertEqual(cpc.lines, [l1, l2, l3])
                self.assertEqual(cpc.contours, [c2, c1, c3])

            # test else evo
            qmi.column = MagicMock(return_value=0)
            qmi.row = MagicMock(return_value=0)
            agp.type = "evo"
            agp.objectsModel.dataList = [l1, l2]
            cpc.lines = [l1, l2, l3]
            cpc.contours = [c1, c2, c3]
            afp = AddEvoLineFromPoint(self.mainW, None, "", [], [], line=l2)
            afp.exec_ = MagicMock()
            afp.result = MagicMock(side_effect=[False, True])
            afp.getLine = MagicMock(return_value=l2)
            with patch(
                "parthenopegui.plotter.AddEvoLineFromPoint", return_value=afp
            ) as _ap, patch(
                "parthenopegui.plotter.AbundancesGenericPanel.addPlotObject",
                autospec=True,
            ) as _ao, patch(
                "parthenopegui.plotter.MWPlotPanel.refreshPlot", autospec=True
            ) as _rp:
                agp.cellDoubleClickObjects(qmi)
                _ap.assert_called_once_with(self.mainW, None, "", [], [], line=l1)
                self.assertEqual(_rp.call_count, 0)
                afp.exec_.assert_called_once_with()
                afp.result.assert_called_once_with()
                agp.cellDoubleClickObjects(qmi)
                _ao.assert_called_once_with(agp, l2, ix=0)
                _rp.assert_called_once_with(self.mainW.plotPanel)
                afp.getLine.assert_called_once_with()
                self.assertEqual(cpc.lines, [l2, l2, l3])
                self.assertEqual(cpc.contours, [c1, c2, c3])

            # test else 1d
            agp.type = "1d"
            agp.objectsModel.dataList = [l1, l2]
            cpc.lines = [l1, l2, l3]
            cpc.contours = [c1, c2, c3]
            afp = Add1DLineFromPoint(self.mainW, None, "", None, [], [], line=l2)
            afp.exec_ = MagicMock()
            afp.result = MagicMock(side_effect=[False, True])
            afp.getLine = MagicMock(return_value=l2)
            with patch(
                "parthenopegui.plotter.Add1DLineFromPoint", return_value=afp
            ) as _ap, patch(
                "parthenopegui.plotter.AbundancesGenericPanel.addPlotObject",
                autospec=True,
            ) as _ao, patch(
                "parthenopegui.plotter.MWPlotPanel.refreshPlot", autospec=True
            ) as _rp:
                agp.cellDoubleClickObjects(qmi)
                _ap.assert_called_once_with(self.mainW, None, "", None, [], [], line=l1)
                self.assertEqual(_rp.call_count, 0)
                afp.exec_.assert_called_once_with()
                afp.result.assert_called_once_with()
                agp.cellDoubleClickObjects(qmi)
                _ao.assert_called_once_with(agp, l2, ix=0)
                _rp.assert_called_once_with(self.mainW.plotPanel)
                afp.getLine.assert_called_once_with()
                self.assertEqual(cpc.lines, [l2, l2, l3])
                self.assertEqual(cpc.contours, [c1, c2, c3])

            # test else 2d
            agp.type = "2d"
            agp.objectsModel.dataList = [c1, c2]
            cpc.lines = [l1, l2, l3]
            cpc.contours = [c1, c2, c3]
            afp = Add2DContourFromPoint(
                self.mainW, None, "", None, None, [], [], cnt=c3
            )
            afp.exec_ = MagicMock()
            afp.result = MagicMock(side_effect=[False, True])
            afp.getContour = MagicMock(return_value=c2)
            with patch(
                "parthenopegui.plotter.Add2DContourFromPoint", return_value=afp
            ) as _ap, patch(
                "parthenopegui.plotter.AbundancesGenericPanel.addPlotObject",
                autospec=True,
            ) as _ao, patch(
                "parthenopegui.plotter.MWPlotPanel.refreshPlot", autospec=True
            ) as _rp:
                agp.cellDoubleClickObjects(qmi)
                _ap.assert_called_once_with(
                    self.mainW, None, "", None, None, [], [], cnt=c1
                )
                self.assertEqual(_rp.call_count, 0)
                afp.exec_.assert_called_once_with()
                afp.result.assert_called_once_with()
                agp.cellDoubleClickObjects(qmi)
                _ao.assert_called_once_with(agp, c2, ix=0)
                _rp.assert_called_once_with(self.mainW.plotPanel)
                afp.getContour.assert_called_once_with()
                self.assertEqual(cpc.lines, [l1, l2, l3])
                self.assertEqual(cpc.contours, [c2, c2, c3])

        def test_cellDoubleClickPoints(self):
            """test cellDoubleClickPoints"""
            p = QWidget()
            agp = ExtAGPWLines(p, self.mainW)
            with self.assertRaises(NotImplementedError):
                agp.cellDoubleClickPoints(QModelIndex())

        def test_pointString(self):
            """test pointString"""
            p = QWidget()
            agp = ExtAGPWLines(p, self.mainW)
            self.assertEqual(
                agp.pointString(["a", "1.", 1.0, 1]),
                "a, 1., 1.000000000000e+00, 1.000000000000e+00",
            )

        def test_gridPointsString(self):
            """test gridPointsString"""
            p = QWidget()
            agp = ExtAGPWLines(p, self.mainW)
            self.assertEqual(
                agp.gridPointsString([["a", "1.", 1.0, 1], np.asarray([1, 2, 3])]),
                [
                    "a, 1., 1.000000000000e+00, 1.000000000000e+00",
                    "1.000000000000e+00, 2.000000000000e+00, 3.000000000000e+00",
                ],
            )

        def test_reloadModel(self):
            """test reloadModel"""
            p = QWidget()
            agp = ExtAGPWLines(p, self.mainW)
            with self.assertRaises(NotImplementedError):
                agp.reloadModel()

        def test_updatePlotProperties(self):
            """test updatePlotProperties"""
            p = QWidget()
            agp = ExtAGPWLines(p, self.mainW)
            ps = self.mainW.plotPanel.plotSettings
            ps.xlabel = "xlab"
            ps.ylabel = "ylab"
            ps.zlabel = "zlab"
            ps.xscale = "log"
            ps.yscale = "log"
            agp.updatePlotProperties()
            self.assertEqual(ps.xlabel, "")
            self.assertEqual(ps.ylabel, "")
            self.assertEqual(ps.zlabel, "")
            self.assertEqual(ps.xscale, "linear")
            self.assertEqual(ps.yscale, "linear")
            with patch(
                "parthenopegui.plotter.PlotSettings.updateAutomaticSettings",
                autospec=True,
            ) as _f:
                agp.updatePlotProperties()
                _f.assert_called_once_with(
                    ps,
                    {
                        "xlabel": "",
                        "ylabel": "",
                        "zlabel": "",
                        "xscale": "linear",
                        "yscale": "linear",
                    },
                    force=True,
                )

    class TestAddEvoLineFromPoint(PGTestCasewMainW):
        """test the AddEvoLineFromPoint class"""

        def test_init(self):
            """test the init method"""
            l = PGLine(np.asarray([0]), np.asarray([1]), 0, d="d")
            a = AddEvoLineFromPoint(
                self.mainW, 123, "abc", ["a", "b"], [1, 2, 3], line=l
            )
            self.assertIsInstance(a, AddLineFromPoint)
            self.assertEqual(a.headerCut, 1)
            self.assertFalse(a.askChi2)
            self.assertEqual(a.mainW, self.mainW)
            self.assertEqual(a.pointRow, 123)
            self.assertEqual(a.desc, "d")
            self.assertEqual(a.paramix, 0)
            self.assertEqual(a.header, ["a", "b"])
            self.assertEqual(a.output, [1, 2, 3])
            self.assertEqual(a.line, l)

            a = AddEvoLineFromPoint(self.mainW, 123, "abc", ["a", "b"], [1, 2, 3])
            self.assertEqual(a.mainW, self.mainW)
            self.assertEqual(a.pointRow, 123)
            self.assertEqual(a.desc, "abc")
            self.assertEqual(a.paramix, 0)
            self.assertEqual(a.header, ["a", "b"])
            self.assertEqual(a.output, [1, 2, 3])
            self.assertEqual(a.line, None)

    class TestAbundancesEvolutionPanel(PGTestCasewMainW):
        """Test the AbundancesEvolutionPanel class"""

        def test_init(self):
            """test __init__"""
            p = QWidget()
            ep = AbundancesEvolutionPanel(p, self.mainW)
            self.assertTrue(ep.invert)
            self.assertTrue(ep.hasLines)
            self.assertEqual(ep.type, "evo")
            self.assertIsInstance(ep, AbundancesGenericPanel)

        def test_cellDoubleClickPoints(self):
            """test cellDoubleClickPoints"""
            p = QWidget()
            ep = AbundancesEvolutionPanel(p, self.mainW)
            idx = QModelIndex()
            idx.isValid = MagicMock(return_value=False)
            idx.row = MagicMock(return_value=0)
            ep.cellDoubleClickPoints(idx)
            idx.isValid.assert_called_once_with()
            self.assertEqual(idx.row.call_count, 0)
            idx.isValid = MagicMock(return_value=True)
            self.mainW.plotPanel.gridLoader.runner = RunPArthENoPE(
                testConfig.commonParamsGrid, testConfig.gridPointsGrid, self.mainW
            )
            self.mainW.plotPanel.gridLoader.runner.readAllResults()
            ep.reloadModel()
            line = PGLine([0], [1], 0)
            askline = AddEvoLineFromPoint(self.mainW, 0, "abc", ["a"], [0])
            askline.exec_ = MagicMock()
            askline.result = MagicMock(return_value=False)
            askline.getLine = MagicMock(return_value=line)

            with patch(
                "parthenopegui.plotter.AddEvoLineFromPoint", return_value=askline
            ) as _al:
                ep.cellDoubleClickPoints(idx)
                _al.assert_called_once_with(
                    self.mainW,
                    0,
                    ", ".join(
                        [
                            "%s = " % p
                            + Configuration.formatParam % ep.pointsModel.dataList[0, i]
                            for i, p in enumerate(Parameters.paramOrder)
                        ]
                    ),
                    ep.runner.nuclidesHeader,
                    ep.runner.nuclidesEvolution[0],
                )
            idx.row.assert_called_once_with()
            askline.exec_.assert_called_once_with()
            askline.result.assert_called_once_with()
            self.assertEqual(askline.getLine.call_count, 0)

            askline.result = MagicMock(return_value=True)
            self.assertEqual(self.mainW.plotPanel.currentPlotContent.lines, [])
            self.assertEqual(ep.pointsModel.inUse[0], [])
            with patch(
                "parthenopegui.plotter.AddEvoLineFromPoint", return_value=askline
            ) as _al, patch(
                "parthenopegui.plotter.AbundancesGenericPanel.addPlotObject",
                autospec=True,
            ) as _apo, patch(
                "parthenopegui.plotter.MWPlotPanel.refreshPlot", autospec=True
            ) as _rp:
                ep.cellDoubleClickPoints(idx)
                _apo.assert_called_once_with(ep, line)
                _rp.assert_called_once_with(self.mainW.plotPanel)
            self.assertEqual(self.mainW.plotPanel.currentPlotContent.lines, [line])
            self.assertEqual(ep.pointsModel.inUse[0], [True])
            askline.getLine.assert_called_once_with()

        def test_reloadModel(self):
            """test reloadModel"""
            p = QWidget()
            ep = AbundancesEvolutionPanel(p, self.mainW)
            self.assertEqual(ep.runner, None)
            self.assertEqual(ep.pointsModel.dataList, [])
            self.mainW.plotPanel.gridLoader.runner = RunPArthENoPE(
                testConfig.commonParamsGrid, testConfig.gridPointsGrid, self.mainW
            )
            ep.reloadModel()
            self.assertEqual(ep.runner, self.mainW.plotPanel.gridLoader.runner)
            self.assertEqualArray(
                ep.pointsModel.dataList,
                self.mainW.plotPanel.gridLoader.runner.getGridPointsList(),
            )

        def test_updatePlotProperties(self):
            """test updatePlotProperties"""
            p = QWidget()
            ep = AbundancesEvolutionPanel(p, self.mainW)
            ps = self.mainW.plotPanel.plotSettings
            ps.xlabel = "xlab"
            ps.ylabel = "ylab"
            ps.zlabel = "zlab"
            ps.xscale = "linear"
            ps.yscale = "linear"
            ep.updatePlotProperties()
            self.assertEqual(ps.xlabel, "$T$ [MeV]")
            self.assertEqual(ps.ylabel, "")
            self.assertEqual(ps.zlabel, "")
            self.assertEqual(ps.xscale, "log")
            self.assertEqual(ps.yscale, "log")
            with patch(
                "parthenopegui.plotter.PlotSettings.updateAutomaticSettings",
                autospec=True,
            ) as _f:
                ep.updatePlotProperties()
                _f.assert_called_once_with(
                    ps,
                    {
                        "xlabel": "$T$ [MeV]",
                        "ylabel": "",
                        "zlabel": "",
                        "xscale": "log",
                        "yscale": "log",
                    },
                    force=True,
                )

    class TestAdd1DLineFromPoint(PGTestCasewMainW):
        """test the Add1DLineFromPoint class"""

        def test_init(self):
            """test the init method"""
            l = PGLine(np.asarray([0]), np.asarray([1]), 0, d="d")
            a = Add1DLineFromPoint(
                self.mainW, 123, "abc", 11, ["a", "b"], [1, 2, 3], line=l
            )
            self.assertIsInstance(a, AddLineFromPoint)
            self.assertEqual(a.headerCut, 6)
            self.assertTrue(a.askChi2)
            self.assertEqual(a.mainW, self.mainW)
            self.assertEqual(a.pointRow, 123)
            self.assertEqual(a.desc, "d")
            self.assertEqual(a.paramix, 11)
            self.assertEqual(a.header, ["a", "b"])
            self.assertEqual(a.output, [1, 2, 3])
            self.assertEqual(a.line, l)

            a = Add1DLineFromPoint(self.mainW, 123, "abc", 11, ["a", "b"], [1, 2, 3])
            self.assertIsInstance(a, AddLineFromPoint)
            self.assertEqual(a.headerCut, 6)
            self.assertTrue(a.askChi2)
            self.assertEqual(a.mainW, self.mainW)
            self.assertEqual(a.pointRow, 123)
            self.assertEqual(a.desc, "abc")
            self.assertEqual(a.paramix, 11)
            self.assertEqual(a.header, ["a", "b"])
            self.assertEqual(a.output, [1, 2, 3])
            self.assertEqual(a.line, None)

    class TestAbundancesParams1DPanel(PGTestCasewMainW):
        """Test the AbundancesParams1DPanel class"""

        def test_init(self):
            """test __init__"""
            p = QWidget()
            pp = AbundancesParams1DPanel(p, self.mainW)
            self.assertFalse(pp.invert)
            self.assertTrue(pp.hasLines)
            self.assertEqual(pp.type, "1d")
            self.assertIsInstance(pp, AbundancesGenericPanel)

        def test_cellDoubleClickPoints(self):
            """test cellDoubleClickPoints"""
            p = QWidget()
            pp = AbundancesParams1DPanel(p, self.mainW)
            qmi = QModelIndex()
            qmi.isValid = MagicMock(return_value=False)
            qmi.row = MagicMock(return_value=0)
            pp.cellDoubleClickPoints(qmi)
            qmi.isValid.assert_called_once_with()
            self.assertEqual(qmi.row.call_count, 0)
            self.mainW.plotPanel.gridLoader.runner = testConfig.sampleRunner
            pp.reloadModel()
            l1 = PGLine([0, 1], [2, 3], 0)
            l2 = PGLine([0, 1], [2, 3], 0)
            al = Add1DLineFromPoint(self.mainW, None, "", None, [], [], line=l1)
            al.exec_ = MagicMock()
            al.result = MagicMock(side_effect=[False, True])
            al.getLine = MagicMock(return_value=l2)
            self.mainW.plotPanel.currentPlotContent.lines = []
            pp.pointsModel.inUse = [[], [], []]
            qmi.isValid = MagicMock(return_value=True)
            with patch(
                "parthenopegui.plotter.Add1DLineFromPoint",
                return_value=al,
                autospec=USE_AUTOSPEC_CLASS,
            ) as _al, patch(
                "parthenopegui.plotter.AbundancesGenericPanel.addPlotObject",
                autospec=True,
            ) as _ao, patch(
                "parthenopegui.plotter.MWPlotPanel.refreshPlot", autospec=True
            ) as _rp:
                pp.cellDoubleClickPoints(qmi)
                _al.assert_called_once()
                self.assertEqual(_al.call_args[0][0], self.mainW)
                self.assertEqual(_al.call_args[0][1], 0)
                self.assertEqual(
                    _al.call_args[0][2],
                    ", ".join(
                        [
                            "%s = " % Parameters.paramOrder[i]
                            + Configuration.formatParam % pp.pointsModel.dataList[0, i]
                            for i in (0, 2, 3, 4, 5)
                        ]
                    )
                    + ": "
                    + "%s" % Parameters.paramOrderParth[pp.pointsModel.dataList[0, -2]],
                )
                self.assertEqual(_al.call_args[0][3], pp.pointsModel.dataList[0, -2])
                self.assertEqual(
                    _al.call_args[0][4], testConfig.sampleRunner.parthenopeHeader
                )
                self.assertEqualArray(
                    _al.call_args[0][5], pp.pointsModel.dataList[0, -1]
                )
                al.exec_.assert_called_once_with()
                al.result.assert_called_once_with()
                self.assertEqual(al.getLine.call_count, 0)
                pp.cellDoubleClickPoints(qmi)
                _ao.assert_called_once_with(pp, l2)
                _rp.assert_called_once_with(self.mainW.plotPanel)
                self.assertEqual(self.mainW.plotPanel.currentPlotContent.lines, [l2])
                self.assertEqual(pp.pointsModel.inUse, [[True], [], []])

        def test_reloadModel(self):
            """test reloadModel"""
            p = QWidget()
            self.mainW.plotPanel.gridLoader.runner = testConfig.sampleRunner
            pp = AbundancesParams1DPanel(p, self.mainW)
            self.assertEqual(pp.dataList, [])
            self.assertEqual(pp.runner, None)
            with patch(
                "parthenopegui.plotter.PGListPointsModel.replaceDataList", autospec=True
            ) as _r:
                pp.reloadModel()
                _r.assert_called_once()
                self.assertEqual(_r.call_args[0][0], pp.pointsModel)
                dl = _r.call_args[0][1]
            self.assertEqual(len(dl), 16)
            lines = testConfig.sampleRunner.gridPoints[0]
            out = testConfig.sampleRunner.parthenopeOutPoints
            ptlines = {
                1: [[0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11]],
                3: [[0, 3], [1, 4], [2, 5], [6, 9], [7, 10], [8, 11]],
                4: [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
            }
            ptranges = {1: [-3.0, 3.0], 3: [-1.0, 1.0], 4: [-1.0, 0.0, 1.0]}
            gi = 0
            for ix, x in enumerate([1, 3, 4]):
                for b in ptlines[x]:
                    for il, l in enumerate(b):
                        for ip in range(6):
                            if ip == x:
                                self.assertEqualArray(dl[gi][ip], ptranges[x])
                            else:
                                self.assertEqual(dl[gi][ip], lines[l, ip])
                        self.assertEqual(dl[gi][-2], ix)
                        self.assertEqualArray(dl[gi][-1][il], out[l])
                    gi += 1
            self.assertEqual(pp.runner, testConfig.sampleRunner)

    class TestPGListContoursModel(PGTestCase):
        """Test the PGListContoursModel class"""

        def test_class(self):
            """test the class"""
            p = QWidget()
            w = PGListContoursModel([1, 2, 3], p)
            self.assertIsInstance(w, PGListLinesModel)
            self.assertEqual(
                w.header,
                [
                    "description",
                    "label",
                    "filled",
                    "cmap",
                    "levels",
                    "extend",
                    "Delete",
                ],
            )
            self.assertEqual(w.dataList, [1, 2, 3])
            self.assertTrue(hasattr(w, "headerToolTips"))

    class TestCMapMenu(PGTestCase):
        """Test the CMapItems class"""

        def test_init(self):
            """test init"""
            p = QWidget()
            cm = CMapItems(p)
            self.assertIsInstance(cm._cat, QComboBox)
            self.assertIsInstance(cm._cmap, QComboBox)
            fcat = list(cmaps.keys())[0]
            for i, c in enumerate(cmaps.keys()):
                self.assertEqual(cm._cat.itemText(i), c)
            self.assertEqual(cm._cat.currentText(), fcat)
            for i, c in enumerate(cmaps[fcat]):
                self.assertEqual(cm._cmap.itemText(i), c)
            self.assertEqual(cm._cmap.currentText(), cmaps[fcat][0])
            with patch(
                "parthenopegui.plotter.CMapItems.reloadCmapCombo", autospec=True
            ) as _r:
                cm._cat.setCurrentText("Qualitative")
                _r.assert_called_once_with(cm)

            cm = CMapItems(p, "CMRmap")
            for i, c in enumerate(cmaps.keys()):
                self.assertEqual(cm._cat.itemText(i), c)
            self.assertEqual(cm._cat.currentText(), "Miscellaneous")
            for i, c in enumerate(cmaps["Miscellaneous"]):
                self.assertEqual(cm._cmap.itemText(i), c)
            self.assertEqual(cm._cmap.currentText(), "CMRmap")

        def test_cmap(self):
            """test cmap"""
            p = QWidget()
            cm = CMapItems(p)
            self.assertEqual(cm.cmap, cmaps[list(cmaps.keys())[0]][0])
            cm._cat.setCurrentText("Miscellaneous")
            cm._cmap.setCurrentText("CMRmap")
            self.assertEqual(cm.cmap, "CMRmap")
            cm._cat.setCurrentText("Qualitative")
            self.assertEqual(cm.cmap, "Pastel1")

        def test_reloadCmapCombo(self):
            """test reloadCmapCombo"""
            p = QWidget()
            cm = CMapItems(p)
            for c in cmaps.keys():
                prev = cm._cat.currentText()
                with patch("parthenopegui.plotter.CMapItems.reloadCmapCombo") as _r:
                    cm._cat.setCurrentText(c)
                for i, m in enumerate(cmaps[prev]):
                    self.assertEqual(cm._cmap.itemText(i), m)
                cm.reloadCmapCombo()
                for i, m in enumerate(cmaps[c]):
                    self.assertEqual(cm._cmap.itemText(i), m)

    class TestAdd2DContourFromPoint(PGTestCasewMainW):
        """Test the Add2DContourFromPoint class"""

        def test_init(self):
            """test __init__"""
            po = np.asarray(
                [v for k, v in testConfig.sampleRunner.parthenopeOutPoints.items()]
            )
            cp = Add2DContourFromPoint(
                self.mainW,
                4,
                "desc",
                1,
                2,
                testConfig.sampleRunner.parthenopeHeader,
                po,
            )
            self.assertIsInstance(cp.chi2ParamConfigs, list)
            self.assertIsInstance(cp.stackWidget, QStackedWidget)
            self.assertIsInstance(cp.paramWidget, QWidget)
            self.assertIsInstance(cp.paramWidget.layout(), QHBoxLayout)
            self.assertEqual(cp.stackWidget.widget(0), cp.paramWidget)
            self.assertIsInstance(cp.chi2Widget, QWidget)
            self.assertIsInstance(cp.chi2Widget.layout(), QGridLayout)
            self.assertEqual(cp.stackWidget.widget(1), cp.chi2Widget)
            self.assertIsInstance(cp, AddFromPoint)
            self.assertEqual(cp.desc, "desc")
            self.assertEqual(cp.paramix1, 1)
            self.assertEqual(cp.paramix2, 2)
            self.assertEqual(
                cp.parthenopeHeader, testConfig.sampleRunner.parthenopeHeader
            )
            self.assertEqualArray(cp.parthenopeOut, po)
            self.assertEqual(cp.cnt, None)
            self.assertIsInstance(cp.layout(), QGridLayout)
            self.assertEqual(cp.windowTitle(), PGText.plotSelectNuclideContour)
            r = 0
            self.assertIsInstance(cp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(cp.layout().itemAtPosition(r, 0).widget().text(), cp.desc)
            r += 1
            self.assertIsInstance(cp.layout().itemAtPosition(r, 0).widget(), QFrame)
            self.assertEqual(
                cp.layout().itemAtPosition(r, 0).widget().frameShape(), QFrame.HLine
            )
            r += 1
            self.assertIsInstance(cp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                cp.layout().itemAtPosition(r, 0).widget().text(),
                PGText.plotChi2AskLabel,
            )
            self.assertIsInstance(cp.chi2Check, QCheckBox)
            self.assertEqual(cp.chi2Check.toolTip(), PGText.plotChi2AskCheckToolTip)
            self.assertFalse(cp.chi2Check.isChecked())
            with patch(
                "parthenopegui.plotter.AddFromPoint._updateStack", autospec=True
            ) as _f1, patch(
                "parthenopegui.plotter.Add2DContourFromPoint.updateZlabel",
                autospec=True,
            ) as _f2:
                cp.chi2Check.toggle()
                _f2.assert_called_once_with(cp)
                _f1.assert_called_once_with(cp, 2)
                _f1.reset_mock()
                cp.chi2Check.toggle()
                _f1.assert_called_once_with(cp, 0)
            self.assertEqual(cp.layout().itemAtPosition(r, 3).widget(), cp.chi2Check)
            # paramWidget
            self.assertIsInstance(cp.paramWidget.layout().itemAt(0).widget(), PGLabel)
            self.assertEqual(
                cp.paramWidget.layout().itemAt(0).widget().text(),
                PGText.plotSelectNuclide,
            )
            self.assertIsInstance(cp.nuclideCombo, QComboBox)
            for i, f in enumerate(cp.parthenopeHeader[6:]):
                self.assertEqual(cp.nuclideCombo.itemText(i), "%s" % f)
            self.assertEqual(
                cp.paramWidget.layout().itemAt(1).widget(),
                cp.nuclideCombo,
            )
            with patch(
                "parthenopegui.plotter.Add2DContourFromPoint.updateZlabel",
                autospec=True,
            ) as _f:
                cp.nuclideCombo.setCurrentIndex(1)
                _f.assert_called_once_with(cp)
            # chi2Widget
            for j, o in enumerate(cp.chi2ParamConfigs):
                for i, f in enumerate(
                    [
                        "paramLabel",
                        "paramCombo",
                        "meanLabel",
                        "chi2Mean",
                        "stdLabel",
                        "chi2Std",
                    ]
                ):
                    cp.chi2Widget.layout().addWidget(getattr(o, f), j, i)
            self.assertIsInstance(cp.addChi2Line, PGPushButton)
            self.assertEqual(cp.addChi2Line.text(), PGText.plotSelectChi2AddLine)
            self.assertEqual(
                cp.addChi2Line.toolTip(), PGText.plotSelectChi2AddLineToolTip
            )
            with patch(
                "parthenopegui.plotter.AddFromPoint._addChi2Line", autospec=True
            ) as _f:
                QTest.mouseClick(cp.addChi2Line, Qt.LeftButton)
                _f.assert_called_once_with(cp)
            self.assertEqual(
                cp.chi2Widget.layout().itemAtPosition(1, 0).widget(), cp.addChi2Line
            )

            r += 1
            self.assertIsInstance(
                cp.layout().itemAtPosition(r, 0).widget(), QStackedWidget
            )
            self.assertEqual(
                cp.layout().itemAtPosition(r, 0).widget(),
                cp.stackWidget,
            )
            r += 1
            self.assertIsInstance(cp.layout().itemAtPosition(r, 0).widget(), QFrame)
            self.assertEqual(
                cp.layout().itemAtPosition(r, 0).widget().frameShape(), QFrame.HLine
            )
            r += 1
            self.assertIsInstance(cp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                cp.layout().itemAtPosition(r, 0).widget().text(), PGText.plotLineLabel
            )
            self.assertIsInstance(cp.labelInput, QLineEdit)
            self.assertEqual(cp.labelInput.text(), "")
            self.assertEqual(cp.labelInput, cp.layout().itemAtPosition(r, 2).widget())
            r += 1
            self.assertIsInstance(cp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                cp.layout().itemAtPosition(r, 0).widget().text(),
                PGText.plotContourFilled,
            )
            self.assertIsInstance(cp.chi2Check, QCheckBox)
            self.assertEqual(cp.filledCheck.text(), "")
            self.assertTrue(cp.filledCheck.isChecked())
            self.assertEqual(cp.filledCheck, cp.layout().itemAtPosition(r, 2).widget())
            r += 1
            self.assertIsInstance(cp.layout().itemAtPosition(r, 0).widget(), QFrame)
            self.assertEqual(
                cp.layout().itemAtPosition(r, 0).widget().frameShape(), QFrame.HLine
            )
            r += 1
            self.assertIsInstance(cp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                cp.layout().itemAtPosition(r, 0).widget().text(), PGText.plotContourCMap
            )
            r += 1
            self.assertIsInstance(cp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                cp.layout().itemAtPosition(r, 0).widget().text(), PGText.category
            )
            self.assertIsInstance(cp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                cp.layout().itemAtPosition(r, 2).widget().text(), PGText.current
            )
            self.assertIsInstance(cp.cmapItems, CMapItems)
            self.assertEqual(
                cp.cmapItems._cat, cp.layout().itemAtPosition(r, 1).widget()
            )
            self.assertEqual(
                cp.cmapItems._cmap, cp.layout().itemAtPosition(r, 3).widget()
            )
            self.assertEqual(cp.cmapItems._cmap.currentText(), defaultCmap)
            r += 1
            self.assertIsInstance(cp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                cp.layout().itemAtPosition(r, 0).widget().text(),
                PGText.plotContourHasCbar,
            )
            self.assertIsInstance(cp.hasCbarCheck, QCheckBox)
            self.assertEqual(cp.hasCbarCheck.text(), "")
            self.assertTrue(cp.hasCbarCheck.isChecked())
            self.assertEqual(cp.hasCbarCheck, cp.layout().itemAtPosition(r, 2).widget())
            r += 1
            self.assertIsInstance(cp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                cp.layout().itemAtPosition(r, 0).widget().text(),
                PGText.plotContourZLabel,
            )
            self.assertIsInstance(cp.zlabelInput, QLineEdit)
            self.assertEqual(cp.zlabelInput.text(), cp.parthenopeHeader[6])
            self.assertEqual(cp.zlabelInput, cp.layout().itemAtPosition(r, 2).widget())
            r += 1
            self.assertIsInstance(cp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                cp.layout().itemAtPosition(r, 0).widget().text(),
                PGText.plotContourExtend,
            )
            self.assertIsInstance(cp.extendCombo, QComboBox)
            for i, e in enumerate(extendOptions):
                self.assertEqual(cp.extendCombo.itemText(i), e)
            self.assertEqual(cp.extendCombo.currentText(), "neither")
            self.assertEqual(cp.extendCombo, cp.layout().itemAtPosition(r, 2).widget())
            r += 1
            self.assertIsInstance(cp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                cp.layout().itemAtPosition(r, 0).widget().text(),
                PGText.plotContourLevelsLabel,
            )
            self.assertIsInstance(cp.levelsInput, QComboBox)
            self.assertTrue(cp.levelsInput.isEditable())
            self.assertEqual(cp.levelsInput.currentText(), "")
            for i, q in enumerate(
                [
                    "",
                    "%s" % [0.0, 1.0, 4.0, 9.0, 16.0, 25.0],
                    "%s" % [0.0, 2.30, 6.18, 11.83, 19.33, 28.74],
                    "other (edit)",
                ]
            ):
                self.assertEqual(cp.levelsInput.itemText(i), q)
            self.assertEqual(cp.levelsInput, cp.layout().itemAtPosition(r, 2).widget())
            r += 1
            self.assertIsInstance(cp.acceptButton, PGPushButton)
            self.assertEqual(cp.acceptButton.text(), PGText.buttonAccept)
            self.assertEqual(cp.acceptButton, cp.layout().itemAtPosition(r, 0).widget())
            with patch(
                "parthenopegui.plotter.Add2DContourFromPoint.testAccept", autospec=True
            ) as _f:
                QTest.mouseClick(cp.acceptButton, Qt.LeftButton)
                _f.assert_called_once_with(cp)
            self.assertIsInstance(cp.cancelButton, PGPushButton)
            self.assertEqual(cp.cancelButton.text(), PGText.buttonCancel)
            self.assertEqual(cp.cancelButton, cp.layout().itemAtPosition(r, 2).widget())
            cp.reject = MagicMock()
            QTest.mouseClick(cp.cancelButton, Qt.LeftButton)
            cp.reject.assert_called_once_with()

            cnt = PGContour(
                [0, 1],
                [2, 3],
                [[4, 5], [6, 7]],
                0,
                d="mycont",
                c2="chi2",
                l="lab",
                f=False,
                cm="seismic",
                hcb=False,
                zl="zlab",
                ex="min",
                lvs=[1.0, 2.0, 3.0],
            )
            cp = Add2DContourFromPoint(self.mainW, 0, "desc", 1, 2, [], [], cnt=cnt)
            self.assertEqual(cp.desc, "mycont")
            self.assertEqual(cp.cnt, cnt)
            r = 0
            self.assertIsInstance(cp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(cp.layout().itemAtPosition(r, 0).widget().text(), cp.desc)
            self.assertFalse(hasattr(cp, "nuclideCombo"))
            self.assertFalse(hasattr(cp, "addChi2Line"))
            r += 1
            self.assertIsInstance(
                cp.layout().itemAtPosition(r, 0).widget(), QStackedWidget
            )
            self.assertEqual(
                cp.layout().itemAtPosition(r, 0).widget(),
                cp.stackWidget,
            )
            self.assertIsInstance(cp.chi2Check, QCheckBox)
            self.assertEqual(cp.chi2Check.text(), "")
            self.assertTrue(cp.chi2Check.isChecked())
            self.assertTrue(cp.chi2Check.isHidden())
            r += 1
            self.assertIsInstance(cp.layout().itemAtPosition(r, 0).widget(), QFrame)
            self.assertEqual(
                cp.layout().itemAtPosition(r, 0).widget().frameShape(), QFrame.HLine
            )
            r += 1
            self.assertIsInstance(cp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                cp.layout().itemAtPosition(r, 0).widget().text(), PGText.plotLineLabel
            )
            self.assertIsInstance(cp.labelInput, QLineEdit)
            self.assertEqual(cp.labelInput.text(), "lab")
            self.assertEqual(cp.labelInput, cp.layout().itemAtPosition(r, 2).widget())
            r += 1
            self.assertIsInstance(cp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                cp.layout().itemAtPosition(r, 0).widget().text(),
                PGText.plotContourFilled,
            )
            self.assertIsInstance(cp.chi2Check, QCheckBox)
            self.assertEqual(cp.filledCheck.text(), "")
            self.assertFalse(cp.filledCheck.isChecked())
            self.assertEqual(cp.filledCheck, cp.layout().itemAtPosition(r, 2).widget())
            r += 1
            self.assertIsInstance(cp.layout().itemAtPosition(r, 0).widget(), QFrame)
            self.assertEqual(
                cp.layout().itemAtPosition(r, 0).widget().frameShape(), QFrame.HLine
            )
            r += 1
            self.assertIsInstance(cp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                cp.layout().itemAtPosition(r, 0).widget().text(), PGText.plotContourCMap
            )
            r += 1
            self.assertIsInstance(cp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                cp.layout().itemAtPosition(r, 0).widget().text(), PGText.category
            )
            self.assertIsInstance(cp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                cp.layout().itemAtPosition(r, 2).widget().text(), PGText.current
            )
            self.assertIsInstance(cp.cmapItems, CMapItems)
            self.assertEqual(
                cp.cmapItems._cat, cp.layout().itemAtPosition(r, 1).widget()
            )
            self.assertEqual(
                cp.cmapItems._cmap, cp.layout().itemAtPosition(r, 3).widget()
            )
            self.assertEqual(cp.cmapItems._cmap.currentText(), "seismic")
            r += 1
            self.assertIsInstance(cp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                cp.layout().itemAtPosition(r, 0).widget().text(),
                PGText.plotContourHasCbar,
            )
            self.assertIsInstance(cp.hasCbarCheck, QCheckBox)
            self.assertEqual(cp.hasCbarCheck.text(), "")
            self.assertFalse(cp.hasCbarCheck.isChecked())
            self.assertEqual(cp.hasCbarCheck, cp.layout().itemAtPosition(r, 2).widget())
            r += 1
            self.assertIsInstance(cp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                cp.layout().itemAtPosition(r, 0).widget().text(),
                PGText.plotContourZLabel,
            )
            self.assertIsInstance(cp.zlabelInput, QLineEdit)
            self.assertEqual(cp.zlabelInput.text(), "zlab")
            self.assertEqual(cp.zlabelInput, cp.layout().itemAtPosition(r, 2).widget())
            r += 1
            self.assertIsInstance(cp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                cp.layout().itemAtPosition(r, 0).widget().text(),
                PGText.plotContourExtend,
            )
            self.assertIsInstance(cp.extendCombo, QComboBox)
            for i, e in enumerate(extendOptions):
                self.assertEqual(cp.extendCombo.itemText(i), e)
            self.assertEqual(cp.extendCombo.currentText(), "min")
            self.assertEqual(cp.extendCombo, cp.layout().itemAtPosition(r, 2).widget())
            r += 1
            self.assertIsInstance(cp.layout().itemAtPosition(r, 0).widget(), PGLabel)
            self.assertEqual(
                cp.layout().itemAtPosition(r, 0).widget().text(),
                PGText.plotContourLevelsLabel,
            )
            self.assertIsInstance(cp.levelsInput, QComboBox)
            self.assertTrue(cp.levelsInput.isEditable())
            self.assertEqual(cp.levelsInput.currentText(), "%s" % [1.0, 2.0, 3.0])
            for i, q in enumerate(
                [
                    "%s" % [1.0, 2.0, 3.0],
                    "",
                    "%s" % [0.0, 1.0, 4.0, 9.0, 16.0, 25.0],
                    "%s" % [0.0, 2.30, 6.18, 11.83, 19.33, 28.74],
                    "other (edit)",
                ]
            ):
                self.assertEqual(cp.levelsInput.itemText(i), q)
            self.assertEqual(cp.levelsInput, cp.layout().itemAtPosition(r, 2).widget())
            r += 1
            self.assertIsInstance(cp.acceptButton, PGPushButton)
            self.assertEqual(cp.acceptButton.text(), PGText.buttonAccept)
            self.assertEqual(cp.acceptButton, cp.layout().itemAtPosition(r, 0).widget())
            with patch(
                "parthenopegui.plotter.Add2DContourFromPoint.testAccept", autospec=True
            ) as _f:
                QTest.mouseClick(cp.acceptButton, Qt.LeftButton)
                _f.assert_called_once_with(cp)
            self.assertIsInstance(cp.cancelButton, PGPushButton)
            self.assertEqual(cp.cancelButton.text(), PGText.buttonCancel)
            self.assertEqual(cp.cancelButton, cp.layout().itemAtPosition(r, 2).widget())
            cp.reject = MagicMock()
            QTest.mouseClick(cp.cancelButton, Qt.LeftButton)
            cp.reject.assert_called_once_with()

        def test_nuclideCombo(self):
            """test nuclideCombo"""
            po = np.asarray(
                [v for k, v in testConfig.sampleRunner.parthenopeOutPoints.items()]
            )
            afp = Add2DContourFromPoint(
                self.mainW,
                4,
                "desc",
                1,
                2,
                testConfig.sampleRunner.parthenopeHeader,
                po,
            )
            c = afp._nuclideCombo()
            self.assertIsInstance(c, QComboBox)
            self.assertEqual(c.toolTip(), PGText.plotSelectNuclideToolTip)
            for i, f in enumerate(afp.parthenopeHeader[6:]):
                self.assertEqual(c.itemText(i), "%s" % f)

        def test_getContour(self):
            """test getContour"""
            cnt = PGContour(
                [0, 1], [2, 3], [[4, 5], [6, 7]], 0, d="desc", xl="xl", yl="yl", zl="zl"
            )
            cp = Add2DContourFromPoint(
                self.mainW,
                4,
                "desc",
                1,
                2,
                testConfig.sampleRunner.parthenopeHeader,
                np.asarray(
                    [v for k, v in testConfig.sampleRunner.parthenopeOutPoints.items()][
                        0:6
                    ]
                ),
            )
            cp.chi2Check.setChecked(False)
            cp.nuclideCombo.setCurrentIndex(1)
            cp.labelInput.setText("label")
            cp.filledCheck.setChecked(True)
            cp.hasCbarCheck.setChecked(True)
            cp.cmapItems._cat.setCurrentText("Diverging")
            cp.cmapItems._cmap.setCurrentText("seismic")
            cp.levelsInput.setCurrentText("[1., 2., 3.]")
            cp.extendCombo.setCurrentText("max")
            cp.zlabelInput.setText("zlab")
            with patch(
                "parthenopegui.plotter.PGContour", return_value=cnt
            ) as _c, patch(
                "parthenopegui.plotter.AddFromPoint._getChi2Conv",
                return_value=np.array([0.0, 1.0, 2.0]),
                autospec=True,
            ) as _gcc, patch(
                "parthenopegui.plotter.AddFromPoint._getChi2Type",
                return_value=False,
                autospec=True,
            ) as _gct, patch(
                "parthenopegui.plotter.AddFromPoint._getDescLineParam",
                return_value="chi2(%s) ",
                autospec=True,
            ) as _gdp:
                self.assertEqual(cp.getContour(), cnt)
                self.assertEqual(_gcc.call_count, 0)
                _gct.assert_called_once_with(cp, False)
                self.assertEqual(_gdp.call_count, 0)
                _c.assert_called_once()
                self.assertEqualArray(_c.call_args[0][0], [-1.0, 1.0])
                self.assertEqualArray(_c.call_args[0][1], [-1.0, 0.0, 1.0])
                self.assertEqualArray(
                    _c.call_args[0][2],
                    cp.parthenopeOut[:, 6 + cp.nuclideCombo.currentIndex()]
                    .reshape(2, 3)
                    .T,
                )
                self.assertEqual(_c.call_args[0][3], 4)
                self.assertEqual(
                    _c.call_args[1],
                    {
                        "c2": False,
                        "cm": "seismic",
                        "d": "desc - phie",
                        "ex": "max",
                        "f": True,
                        "hcb": True,
                        "l": "label",
                        "lvs": [1.0, 2.0, 3.0],
                        "xl": "xie",
                        "yl": "xix",
                        "zl": "zlab",
                    },
                )
            cp.chi2Check.setChecked(True)
            cp.chi2ParamConfigs[0].paramCombo.setCurrentText("Y_H")
            cp.chi2ParamConfigs[0].chi2Mean.setText("0.4")
            cp.chi2ParamConfigs[0].chi2Std.setText("0.05")
            with patch(
                "parthenopegui.plotter.PGContour", return_value=cnt
            ) as _c, patch(
                "parthenopegui.plotter.AddFromPoint._getChi2Conv",
                return_value=np.array([0.0, 1.0, 2.0, 3, 4, 5]),
                autospec=True,
            ) as _gcc, patch(
                "parthenopegui.plotter.AddFromPoint._getChi2Type",
                return_value=True,
                autospec=True,
            ) as _gct, patch(
                "parthenopegui.plotter.AddFromPoint._getDescLineParam",
                return_value="chi2(%s) ",
                autospec=True,
            ) as _gdp:
                self.assertEqual(cp.getContour(), cnt)
                _gcc.assert_called_once()
                self.assertEqual(_gcc.call_args[0][0], cp)
                self.assertEqualArray(
                    _gcc.call_args[0][1],
                    chi2func(
                        cp.parthenopeOut[
                            :, 6 + cp.chi2ParamConfigs[0].paramCombo.currentIndex()
                        ],
                        0.4,
                        0.05,
                    ),
                )
                _gct.assert_called_once_with(cp, True)
                _gdp.assert_called_once_with(cp)
                _c.assert_called_once()
                self.assertEqualArray(_c.call_args[0][0], [-1.0, 1.0])
                self.assertEqualArray(_c.call_args[0][1], [-1.0, 0.0, 1.0])
                self.assertEqualArray(
                    _c.call_args[0][2],
                    np.array([0.0, 1.0, 2.0, 3, 4, 5]).reshape(2, 3).T,
                )
                self.assertEqual(_c.call_args[0][3], 4)
                self.assertEqual(
                    _c.call_args[1],
                    {
                        "c2": True,
                        "cm": "seismic",
                        "d": "desc - chi2(Y_H) ",
                        "ex": "max",
                        "f": True,
                        "hcb": True,
                        "l": "label",
                        "lvs": [1.0, 2.0, 3.0],
                        "xl": "xie",
                        "yl": "xix",
                        "zl": chi2labels["chi2"],
                    },
                )
            cp._newChi2Widget()
            cp.chi2ParamConfigs[1].paramCombo.setCurrentText("Y_p")
            cp.chi2ParamConfigs[1].chi2Mean.setText("0.25")
            cp.chi2ParamConfigs[1].chi2Std.setText("0.01")
            with patch(
                "parthenopegui.plotter.PGContour", return_value=cnt
            ) as _c, patch(
                "parthenopegui.plotter.AddFromPoint._getChi2Conv",
                return_value=np.array([0.0, 1.0, 2.0, 3, 4, 5]),
                autospec=True,
            ) as _gcc, patch(
                "parthenopegui.plotter.AddFromPoint._getChi2Type",
                return_value=True,
                autospec=True,
            ) as _gct, patch(
                "parthenopegui.plotter.AddFromPoint._getDescLineParam",
                return_value="chi2(%s) ",
                autospec=True,
            ) as _gdp:
                self.assertEqual(cp.getContour(), cnt)
                _gcc.assert_called_once()
                self.assertEqual(_gcc.call_args[0][0], cp)
                self.assertEqualArray(
                    _gcc.call_args[0][1],
                    chi2func(
                        cp.parthenopeOut[
                            :, 6 + cp.chi2ParamConfigs[0].paramCombo.currentIndex()
                        ],
                        0.4,
                        0.05,
                    )
                    + chi2func(
                        cp.parthenopeOut[
                            :, 6 + cp.chi2ParamConfigs[1].paramCombo.currentIndex()
                        ],
                        0.25,
                        0.01,
                    ),
                )
                _gct.assert_called_once_with(cp, True)
                self.assertEqual(_gdp.call_count, 2)
                _c.assert_called_once()
                self.assertEqualArray(_c.call_args[0][0], [-1.0, 1.0])
                self.assertEqualArray(_c.call_args[0][1], [-1.0, 0.0, 1.0])
                self.assertEqualArray(
                    _c.call_args[0][2],
                    np.array([0.0, 1.0, 2.0, 3, 4, 5]).reshape(2, 3).T,
                )
                self.assertEqual(_c.call_args[0][3], 4)
                self.assertEqual(
                    _c.call_args[1],
                    {
                        "c2": True,
                        "cm": "seismic",
                        "d": "desc - chi2(Y_H) chi2(Y_p) ",
                        "ex": "max",
                        "f": True,
                        "hcb": True,
                        "l": "label",
                        "lvs": [1.0, 2.0, 3.0],
                        "xl": "xie",
                        "yl": "xix",
                        "zl": chi2labels["chi2"],
                    },
                )

            cp = Add2DContourFromPoint(self.mainW, 0, "desc", 1, 2, [], [], cnt=cnt)
            cp.chi2Check.setChecked(False)
            cp.labelInput.setText("label")
            cp.filledCheck.setChecked(True)
            cp.hasCbarCheck.setChecked(True)
            cp.cmapItems._cat.setCurrentText("Diverging")
            cp.cmapItems._cmap.setCurrentText("seismic")
            cp.levelsInput.setCurrentText("[1., 2., 3.]")
            cp.extendCombo.setCurrentText("max")
            cp.zlabelInput.setText("zlab")
            with patch("parthenopegui.plotter.PGContour", return_value=cnt) as _c:
                self.assertEqual(cp.getContour(), cnt)
                _c.assert_called_once()
                self.assertEqualArray(_c.call_args[0][0], [0, 1.0])
                self.assertEqualArray(_c.call_args[0][1], [2, 3])
                self.assertEqualArray(_c.call_args[0][2], [[4, 5], [6, 7]])
                self.assertEqual(_c.call_args[0][3], 0)
                self.assertEqual(
                    _c.call_args[1],
                    {
                        "c2": cp.cnt.chi2,
                        "cm": "seismic",
                        "d": "desc",
                        "ex": "max",
                        "f": True,
                        "hcb": True,
                        "l": "label",
                        "lvs": [1.0, 2.0, 3.0],
                        "xl": "xl",
                        "yl": "yl",
                        "zl": "zlab",
                    },
                )
            cp.chi2Check.setChecked(True)
            cp._newChi2Widget()
            cp.chi2ParamConfigs[0].paramCombo.setCurrentText("Y_H")
            cp.chi2ParamConfigs[0].chi2Mean.setText("0.4")
            cp.chi2ParamConfigs[0].chi2Std.setText("0.05")
            with patch("parthenopegui.plotter.PGContour", return_value=cnt) as _c:
                self.assertEqual(cp.getContour(), cnt)
                _c.assert_called_once()
                self.assertEqualArray(_c.call_args[0][0], [0, 1.0])
                self.assertEqualArray(_c.call_args[0][1], [2, 3])
                self.assertEqualArray(_c.call_args[0][2], [[4, 5], [6, 7]])
                self.assertEqual(_c.call_args[0][3], 0)
                self.assertEqual(
                    _c.call_args[1],
                    {
                        "c2": cp.cnt.chi2,
                        "cm": "seismic",
                        "d": "desc",
                        "ex": "max",
                        "f": True,
                        "hcb": True,
                        "l": "label",
                        "lvs": [1.0, 2.0, 3.0],
                        "xl": "xl",
                        "yl": "yl",
                        "zl": "zlab",
                    },
                )

        def test_levels(self):
            """test levels"""
            cnt = PGContour([0, 1], [2, 3], [[4, 5], [6, 7]], 0)
            cp = Add2DContourFromPoint(self.mainW, 0, "desc", 1, 2, [], [], cnt=cnt)
            with self.assertRaises(AttributeError):
                cp.levels = "abc"
            cp.levelsInput.setCurrentText("")
            self.assertEqual(cp.levels, None)
            cp.levelsInput.setCurrentText("None")
            self.assertEqual(cp.levels, None)
            cp.levelsInput.setCurrentText("[1, 2")
            with patch("logging.Logger.info") as _i:
                self.assertEqual(cp.levels, None)
                _i.assert_called_once_with(
                    PGText.plotContourLevelsInvalidSyntax % "[1, 2"
                )
            cp.levelsInput.setCurrentText("abc")
            with patch("logging.Logger.info") as _i:
                self.assertEqual(cp.levels, None)
                _i.assert_called_once_with(
                    PGText.plotContourLevelsInvalidSyntax % "abc"
                )
            cp.levelsInput.setCurrentText("123.456")
            with patch("logging.Logger.info") as _i:
                self.assertEqual(cp.levels, None)
                _i.assert_called_once_with(
                    PGText.plotContourLevelsInvalidType % "123.456"
                )
            cp.levelsInput.setCurrentText("'abc'")
            with patch("logging.Logger.info") as _i:
                self.assertEqual(cp.levels, None)
                _i.assert_called_once_with(PGText.plotContourLevelsInvalidType % "abc")
            cp.levelsInput.setCurrentText("[1.]")
            with patch("logging.Logger.info") as _i:
                self.assertEqual(cp.levels, None)
                _i.assert_called_once_with(PGText.plotContourLevelsInvalidLen % [1.0])
            cp.levelsInput.setCurrentText("[1., 'a', 3.]")
            with patch("logging.Logger.info") as _i:
                self.assertEqual(cp.levels, [1.0, 3.0])
                _i.assert_called_once_with(PGText.plotContourLevelsInvalidLevel % "a")
            cp.levelsInput.setCurrentText("[1., 2., 3.]")
            with patch("logging.Logger.info") as _i:
                self.assertEqual(cp.levels, [1.0, 2.0, 3.0])
                self.assertEqual(_i.call_count, 0)

        def test_testAccept(self):
            """test testAccept"""
            cnt = PGContour([0, 1], [2, 3], [[4, 5], [6, 7]], 0)
            cp = Add2DContourFromPoint(self.mainW, 0, "desc", 1, 2, [], [], cnt=cnt)
            cp.accept = MagicMock()
            cp.chi2Check.setChecked(True)
            cp._newChi2Widget()
            cp._newChi2Widget()
            self.assertEqual(len(cp.chi2ParamConfigs), 2)
            cp.chi2ParamConfigs[0].chi2Mean.setText("abc")
            cp.chi2ParamConfigs[0].chi2Std.setText("def")
            cp.chi2ParamConfigs[1].chi2Mean.setText("ghi")
            cp.chi2ParamConfigs[1].chi2Std.setText("jkl")
            with patch("logging.Logger.warning") as _w:
                cp.testAccept()
                _w.assert_called_once_with(PGText.plotInvalidMean % "abc")
            cp.chi2ParamConfigs[0].chi2Mean.setText("1.23")
            with patch("logging.Logger.warning") as _w:
                cp.testAccept()
                _w.assert_called_once_with(PGText.plotInvalidStddev % "def")
            cp.chi2ParamConfigs[0].chi2Std.setText("-2.34")
            with patch("logging.Logger.warning") as _w:
                cp.testAccept()
                _w.assert_called_once_with(PGText.plotInvalidStddev % "-2.34")
            self.assertEqual(cp.accept.call_count, 0)
            cp.chi2ParamConfigs[0].chi2Std.setText("2.34")
            with patch("logging.Logger.warning") as _w:
                cp.testAccept()
                _w.assert_called_once_with(PGText.plotInvalidMean % "ghi")
            cp.chi2ParamConfigs[1].chi2Mean.setText("1.23")
            with patch("logging.Logger.warning") as _w:
                cp.testAccept()
                _w.assert_called_once_with(PGText.plotInvalidStddev % "jkl")
            cp.chi2ParamConfigs[1].chi2Std.setText("-2.34")
            with patch("logging.Logger.warning") as _w:
                cp.testAccept()
                _w.assert_called_once_with(PGText.plotInvalidStddev % "-2.34")
            self.assertEqual(cp.accept.call_count, 0)
            cp.chi2ParamConfigs[1].chi2Std.setText("2.34")
            with patch("logging.Logger.warning") as _w:
                cp.testAccept()
                self.assertEqual(_w.call_count, 0)
            cp.accept.assert_called_once_with()
            cp.chi2Check.setChecked(False)
            cp.chi2ParamConfigs[0].chi2Mean.setText("abc")
            cp.chi2ParamConfigs[1].chi2Std.setText("jkl")
            cp.testAccept()
            self.assertEqual(cp.accept.call_count, 2)

        def test_updateZlabel(self):
            """test updateZlabel"""
            cp = Add2DContourFromPoint(
                self.mainW,
                4,
                "desc",
                1,
                2,
                testConfig.sampleRunner.parthenopeHeader,
                testConfig.sampleRunner.parthenopeOutPoints,
            )
            cp.nuclideCombo.setCurrentIndex(3)
            cp.chi2Check.setChecked(True)
            cp.zlabelInput.setText("")
            cp.updateZlabel()
            self.assertEqual(cp.zlabelInput.text(), chi2labels["chi2"])
            cp.chi2Check.setChecked(False)
            cp.zlabelInput.setText("")
            cp.updateZlabel()
            self.assertEqual(cp.zlabelInput.text(), cp.parthenopeHeader[9])

            cnt = PGContour([0, 1], [2, 3], [[4, 5], [6, 7]], 0, zl="zl")
            cp = Add2DContourFromPoint(self.mainW, 0, "desc", 1, 2, [], [], cnt=cnt)
            cp.chi2Check.setChecked(True)
            cp.zlabelInput.setText("")
            cp.updateZlabel()
            self.assertEqual(cp.zlabelInput.text(), "")

    class TestAbundancesParams2DPanel(PGTestCasewMainW):
        """Test the AbundancesParams2DPanel class"""

        def test_init(self):
            """test __init__"""
            p = QWidget()
            pp = AbundancesParams2DPanel(p, self.mainW)
            self.assertFalse(pp.invert)
            self.assertFalse(pp.hasLines)
            self.assertEqual(pp.type, "2d")
            self.assertIsInstance(pp, AbundancesGenericPanel)

        def test_cellDoubleClickPoints(self):
            """test cellDoubleClickPoints"""
            p = QWidget()
            pp = AbundancesParams2DPanel(p, self.mainW)
            qmi = QModelIndex()
            qmi.isValid = MagicMock(return_value=False)
            qmi.row = MagicMock(return_value=0)
            pp.cellDoubleClickPoints(qmi)
            qmi.isValid.assert_called_once_with()
            self.assertEqual(qmi.row.call_count, 0)
            self.mainW.plotPanel.gridLoader.runner = testConfig.sampleRunner
            pp.reloadModel()
            c1 = PGContour([0, 1], [2, 3], [[4, 5], [6, 7]], 0)
            c2 = PGContour([0, 1], [2, 3], [[4, 5], [6, 7]], 0)
            ac = Add2DContourFromPoint(self.mainW, None, "", None, None, [], [], cnt=c1)
            ac.exec_ = MagicMock()
            ac.result = MagicMock(side_effect=[False, True])
            ac.getContour = MagicMock(return_value=c2)
            self.mainW.plotPanel.currentPlotContent.contours = []
            pp.pointsModel.inUse = [[], [], []]
            qmi.isValid = MagicMock(return_value=True)
            with patch(
                "parthenopegui.plotter.Add2DContourFromPoint",
                return_value=ac,
                autospec=USE_AUTOSPEC_CLASS,
            ) as _ac, patch(
                "parthenopegui.plotter.AbundancesGenericPanel.addPlotObject",
                autospec=True,
            ) as _ao, patch(
                "parthenopegui.plotter.MWPlotPanel.refreshPlot", autospec=True
            ) as _rp:
                pp.cellDoubleClickPoints(qmi)
                _ac.assert_called_once()
                self.assertEqual(_ac.call_args[0][0], self.mainW)
                self.assertEqual(_ac.call_args[0][1], 0)
                self.assertEqual(
                    _ac.call_args[0][2],
                    ", ".join(
                        [
                            "%s = " % Parameters.paramOrder[i]
                            + Configuration.formatParam % pp.pointsModel.dataList[0, i]
                            for i in (0, 2, 4, 5)
                        ]
                    )
                    + ": "
                    + "%s" % Parameters.paramOrderParth[pp.pointsModel.dataList[0, -3]]
                    + ", "
                    + "%s" % Parameters.paramOrderParth[pp.pointsModel.dataList[0, -2]],
                )
                self.assertEqual(_ac.call_args[0][3], pp.pointsModel.dataList[0, -3])
                self.assertEqual(_ac.call_args[0][4], pp.pointsModel.dataList[0, -2])
                self.assertEqual(
                    _ac.call_args[0][5], testConfig.sampleRunner.parthenopeHeader
                )
                self.assertEqual(_ac.call_args[0][6], pp.pointsModel.dataList[0, -1])
                ac.exec_.assert_called_once_with()
                ac.result.assert_called_once_with()
                self.assertEqual(ac.getContour.call_count, 0)
                pp.cellDoubleClickPoints(qmi)
                _ao.assert_called_once_with(pp, c2)
                _rp.assert_called_once_with(self.mainW.plotPanel)
                self.assertEqual(self.mainW.plotPanel.currentPlotContent.contours, [c2])
                self.assertEqual(pp.pointsModel.inUse, [[True], [], []])

        def test_reloadModel(self):
            """test reloadModel"""
            p = QWidget()
            self.mainW.plotPanel.gridLoader.runner = testConfig.sampleRunner
            pp = AbundancesParams2DPanel(p, self.mainW)
            self.assertEqual(pp.dataList, [])
            self.assertEqual(pp.runner, None)
            with patch(
                "parthenopegui.plotter.PGListPointsModel.replaceDataList", autospec=True
            ) as _r:
                pp.reloadModel()
                _r.assert_called_once()
                self.assertEqual(_r.call_args[0][0], pp.pointsModel)
                dl = _r.call_args[0][1]
            self.assertEqual(len(dl), 7)
            lines = testConfig.sampleRunner.gridPoints[0]
            out = testConfig.sampleRunner.parthenopeOutPoints
            ptlines = {
                13: [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]],
                14: [[0, 1, 2, 6, 7, 8], [3, 4, 5, 9, 10, 11]],
                34: [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]],
            }
            ptranges = {1: [-3.0, 3.0], 3: [-1.0, 1.0], 4: [-1.0, 0.0, 1.0]}
            indexes = {1: 0, 3: 1, 4: 2}
            gi = 0
            for ix, [x, y] in enumerate([[1, 3], [1, 4], [3, 4]]):
                for b in ptlines[10 * x + y]:
                    for il, l in enumerate(b):
                        for ip in range(6):
                            if ip == x:
                                self.assertEqualArray(dl[gi][ip], ptranges[x])
                            elif ip == y:
                                self.assertEqualArray(dl[gi][ip], ptranges[y])
                            else:
                                self.assertEqual(dl[gi][ip], lines[l, ip])
                        self.assertEqual(dl[gi][-3], indexes[x])
                        self.assertEqual(dl[gi][-2], indexes[y])
                        self.assertEqualArray(dl[gi][-1][il], out[l])
                    gi += 1
            self.assertEqual(pp.runner, testConfig.sampleRunner)

    class TestMWPlotPanel(PGTestCasewMainW):
        """Test the MWPlotPanel class"""

        def test_init(self):
            """test the init method"""
            pp = MWPlotPanel(self.mainW)
            self.assertEqual(
                pp.plotTypePanelClass,
                {
                    "none": PlotPlaceholderPanel,
                    "evolution": AbundancesEvolutionPanel,
                    "1Ddependence": AbundancesParams1DPanel,
                    "2Ddependence": AbundancesParams2DPanel,
                },
            )
            self.assertIsInstance(pp.layout(), QGridLayout)

            self.assertIsInstance(pp.gridLoader, GridLoader)
            self.assertIsInstance(pp.showPlotWidget, ShowPlotWidget)
            self.assertIsInstance(pp.plotSettings, PlotSettings)
            with patch(
                "parthenopegui.plotter.MWPlotPanel.refreshPlot", autospec=True
            ) as _f:
                QTest.mouseClick(pp.plotSettings.refreshButton, Qt.LeftButton)
                _f.assert_called_once_with(pp)
            with patch(
                "parthenopegui.plotter.MWPlotPanel.resetPlot", autospec=True
            ) as _f:
                QTest.mouseClick(pp.plotSettings.resetButton, Qt.LeftButton)
                _f.assert_called_once_with(pp, False)
            with patch(
                "parthenopegui.plotter.ShowPlotWidget.saveAction", autospec=True
            ) as _f:
                QTest.mouseClick(pp.plotSettings.saveButton, Qt.LeftButton)
                _f.assert_called_once_with(pp.showPlotWidget)
            with patch(
                "parthenopegui.plotter.MWPlotPanel.exportScript", autospec=True
            ) as _f:
                QTest.mouseClick(pp.plotSettings.exportButton, Qt.LeftButton)
                _f.assert_called_once_with(pp)

            self.assertIsInstance(pp.specificPlotPanelStack, QStackedWidget)
            self.assertEqual(
                pp.specificPlotPanelStack.count(), len(pp.plotTypePanelClass)
            )
            self.assertEqual(pp.stackAttrs, sorted(pp.plotTypePanelClass.keys()))
            self.assertIsInstance(pp.stackWidgets, dict)
            self.assertEqual(len(pp.stackWidgets), len(pp.plotTypePanelClass))
            for i, attr in enumerate(pp.stackAttrs):
                self.assertIn(attr, pp.stackWidgets.keys())
                self.assertIsInstance(
                    pp.stackWidgets[attr], pp.plotTypePanelClass[attr]
                )
                self.assertEqual(
                    pp.specificPlotPanelStack.widget(i), pp.stackWidgets[attr]
                )
            with patch(
                "parthenopegui.plotter.PlotPlaceholderPanel.reloadModel", autospec=True
            ) as _a1, patch(
                "parthenopegui.plotter.AbundancesEvolutionPanel.reloadModel",
                autospec=True,
            ) as _a2, patch(
                "parthenopegui.plotter.AbundancesParams1DPanel.reloadModel",
                autospec=True,
            ) as _a3, patch(
                "parthenopegui.plotter.AbundancesParams2DPanel.reloadModel",
                autospec=True,
            ) as _a4:
                pp.gridLoader.reloadedGrid.emit()
                _a1.assert_called_once_with(pp.stackWidgets["none"])
                _a2.assert_called_once_with(pp.stackWidgets["evolution"])
                _a3.assert_called_once_with(pp.stackWidgets["1Ddependence"])
                _a4.assert_called_once_with(pp.stackWidgets["2Ddependence"])
            self.assertEqual(
                pp.specificPlotPanelStack.currentIndex(), pp.stackAttrs.index("none")
            )
            with patch(
                "parthenopegui.plotter.MWPlotPanel.updatePlotProperties", autospec=True
            ) as _f:
                pp.specificPlotPanelStack.currentChanged.emit(1)
                _f.assert_called_once_with(pp, 1)

            self.assertIsInstance(pp.currentPlotContent, CurrentPlotContent)

            l = pp.layout()
            self.assertEqual(l.itemAtPosition(0, 0).widget(), pp.gridLoader)
            self.assertEqual(l.itemAtPosition(1, 0).widget(), pp.specificPlotPanelStack)
            self.assertEqual(l.itemAtPosition(0, 1).widget(), pp.plotSettings)
            self.assertEqual(l.itemAtPosition(2, 1).widget(), pp.showPlotWidget)

        def test_exportScript(self):
            """test the exportScript method"""
            pp = MWPlotPanel(self.mainW)
            # empty file name
            with patch(
                "parthenopegui.plotter.askSaveFileName", autospec=True, return_value=" "
            ) as _ask, patch("parthenopegui.plotter.ExportPlotCode") as _epc:
                pp.exportScript()
                _ask.assert_called_once_with(
                    pp, PGText.plotWhereToSaveScript, filter="*.py"
                )
                self.assertEqual(_epc.call_count, 0)

            # error in writing
            with patch(
                "parthenopegui.plotter.askSaveFileName",
                autospec=True,
                return_value="abc",
            ) as _ask, patch(
                "parthenopegui.plotter.ExportPlotCode",
                side_effect=parthenopegui.FileNotFoundErrorClass,
            ) as _epc, patch(
                "logging.Logger.exception"
            ) as _e:
                pp.exportScript()
                _ask.assert_called_once_with(
                    pp, PGText.plotWhereToSaveScript, filter="*.py"
                )
                _epc.assert_called_once_with(pp.currentPlotContent, "abc.py")
                _e.assert_called_once_with(PGText.errorCannotWriteFile)

            # file doesn't exist
            with patch(
                "parthenopegui.plotter.askSaveFileName",
                autospec=True,
                return_value="abc.py",
            ) as _ask, patch(
                "parthenopegui.plotter.askYesNo", autospec=True, return_value=True
            ) as _yes, patch(
                "os.path.exists", return_value=False
            ) as _ex, patch(
                "parthenopegui.plotter.ExportPlotCode"
            ) as _epc, patch(
                "logging.Logger.info"
            ) as _i:
                pp.exportScript()
                _ask.assert_called_once_with(
                    pp, PGText.plotWhereToSaveScript, filter="*.py"
                )
                _epc.assert_called_once_with(pp.currentPlotContent, "abc.py")
                _ex.assert_called_once_with("abc.py")
                _i.assert_called_once_with(PGText.plotScriptSaved % "abc.py")
                self.assertEqual(_yes.call_count, 0)

            # file exists, replace
            with patch(
                "parthenopegui.plotter.askSaveFileName",
                autospec=True,
                return_value="abc.py",
            ) as _ask, patch(
                "parthenopegui.plotter.askYesNo", autospec=True, return_value=True
            ) as _yes, patch(
                "os.path.exists", return_value=True
            ) as _ex, patch(
                "parthenopegui.plotter.ExportPlotCode"
            ) as _epc, patch(
                "logging.Logger.info"
            ) as _i:
                pp.exportScript()
                _ask.assert_called_once_with(
                    pp, PGText.plotWhereToSaveScript, filter="*.py"
                )
                _yes.assert_called_once_with(PGText.askReplace)
                _epc.assert_called_once_with(pp.currentPlotContent, "abc.py")
                _ex.assert_called_once_with("abc.py")
                _i.assert_called_once_with(PGText.plotScriptSaved % "abc.py")

            # file exists, don't replace
            with patch(
                "parthenopegui.plotter.askSaveFileName",
                autospec=True,
                return_value="abc.py",
            ) as _ask, patch(
                "parthenopegui.plotter.askYesNo", autospec=True, return_value=False
            ) as _yes, patch(
                "os.path.exists", return_value=True
            ) as _ex, patch(
                "parthenopegui.plotter.ExportPlotCode"
            ) as _epc, patch(
                "logging.Logger.info"
            ) as _i:
                pp.exportScript()
                _ask.assert_called_once_with(
                    pp, PGText.plotWhereToSaveScript, filter="*.py"
                )
                _yes.assert_called_once_with(PGText.askReplace)
                self.assertEqual(_epc.call_count, 0)
                _ex.assert_called_once_with("abc.py")

        def test_refreshPlot(self):
            """test the refreshPlot method"""
            pp = MWPlotPanel(self.mainW)
            fig = plt.figure()
            plt.close()
            with patch(
                "parthenopegui.plotter.CurrentPlotContent.doPlot",
                autospec=True,
                return_value=fig,
            ) as _dp, patch(
                "parthenopegui.plotter.ShowPlotWidget.updatePlots", autospec=True
            ) as _up:
                pp.refreshPlot()
                _dp.assert_called_once_with(pp.currentPlotContent)
                _up.assert_called_once_with(pp.showPlotWidget, fig=fig)

        def test_revertPlotSettings(self):
            """test revertPlotSettings"""
            pp = MWPlotPanel(self.mainW)
            with patch(
                "parthenopegui.plotter.PlotSettings.restoreAutomaticSettings",
                autospec=True,
            ) as _f:
                pp.revertPlotSettings()
                _f.assert_called_once_with(pp.plotSettings)

        def test_resetPlot(self):
            """test the resetPlot method"""
            pp = MWPlotPanel(self.mainW)
            with patch(
                "parthenopegui.plotter.askYesNo",
                autospec=True,
                side_effect=[False, True, False],
            ) as _a, patch(
                "parthenopegui.plotter.PlotPlaceholderPanel.clearCurrent", autospec=True
            ) as _a1, patch(
                "parthenopegui.plotter.AbundancesEvolutionPanel.clearCurrent",
                autospec=True,
            ) as _a2, patch(
                "parthenopegui.plotter.AbundancesParams1DPanel.clearCurrent",
                autospec=True,
            ) as _a3, patch(
                "parthenopegui.plotter.AbundancesParams2DPanel.clearCurrent",
                autospec=True,
            ) as _a4, patch(
                "parthenopegui.plotter.CurrentPlotContent.clearCurrent", autospec=True
            ) as _ac, patch(
                "parthenopegui.plotter.MWPlotPanel.refreshPlot", autospec=True
            ) as _r:
                pp.resetPlot()
                _a.assert_called_once_with(PGText.plotSettAskResetImage)
                self.assertEqual(_ac.call_count, 0)
                self.assertEqual(_r.call_count, 0)
                pp.resetPlot()
                _a1.assert_called_once_with(pp.stackWidgets["none"])
                _a2.assert_called_once_with(pp.stackWidgets["evolution"])
                _a3.assert_called_once_with(pp.stackWidgets["1Ddependence"])
                _a4.assert_called_once_with(pp.stackWidgets["2Ddependence"])
                _ac.assert_called_once_with(pp.currentPlotContent)
                _r.assert_called_once_with(pp)
                _a.reset_mock()
                _a1.reset_mock()
                _a2.reset_mock()
                _a3.reset_mock()
                _a4.reset_mock()
                _ac.reset_mock()
                _r.reset_mock()
                pp.resetPlot(force=True)
                self.assertEqual(_a.call_count, 0)
                _a1.assert_called_once_with(pp.stackWidgets["none"])
                _a2.assert_called_once_with(pp.stackWidgets["evolution"])
                _a3.assert_called_once_with(pp.stackWidgets["1Ddependence"])
                _a4.assert_called_once_with(pp.stackWidgets["2Ddependence"])
                _ac.assert_called_once_with(pp.currentPlotContent)
                _r.assert_called_once_with(pp)

        def test_updatePlotProperties(self):
            """test the updatePlotProperties method"""
            pp = MWPlotPanel(self.mainW)
            cls = [
                "AbundancesParams1DPanel",
                "AbundancesParams2DPanel",
                "AbundancesEvolutionPanel",
                "PlotPlaceholderPanel",
            ]
            for i, attr in enumerate(pp.stackAttrs):
                pp.specificPlotPanelStack.setCurrentIndex(i)
                with patch(
                    "parthenopegui.plotter.%s.updatePlotProperties" % cls[i],
                    autospec=True,
                ) as _f:
                    pp.updatePlotProperties(100)
                    _f.assert_called_once_with(pp.stackWidgets[attr])

        def test_updatePlotTypePanel(self):
            """test the updatePlotTypePanel method"""
            pp = MWPlotPanel(self.mainW)
            for i, attr in enumerate(pp.stackAttrs):
                if attr == "none":
                    continue
                with patch(
                    "parthenopegui.plotter.MWPlotPanel.resetPlot", autospec=True
                ) as _r:
                    getattr(pp.gridLoader, attr).setChecked(True)
                    _r.assert_called_once_with(pp, force=True)
                self.assertEqual(pp.specificPlotPanelStack.currentIndex(), i)

    class TestExportPlotCode(PGTestCase):
        """Test the ExportPlotCode class"""

        def test_init(self):
            """test __init__"""
            ps = PlotSettings()
            cpc = CurrentPlotContent(ps)
            cpc.readSettings = MagicMock()
            with self.assertRaises(TypeError):
                ExportPlotCode("a", "b")
            with patch(
                "parthenopegui.plotter.ExportPlotCode._createFile", autospec=True
            ) as _c:
                epc = ExportPlotCode(cpc, "c")
                self.assertEqual(epc.content, cpc)
                self.assertEqual(epc.settings, ps)
                self.assertEqual(epc.filename, "c")
                self.assertEqual(epc.filecontent, "")
                cpc.readSettings.assert_called_once_with()
                _c.assert_called_once_with(epc)

        def test_createFile(self):
            """test _createFile"""
            ps = PlotSettings()
            cpc = CurrentPlotContent(ps)
            with patch(
                "parthenopegui.plotter.ExportPlotCode._createFile", autospec=True
            ) as _c:
                epc = ExportPlotCode(cpc, "c")
            with patch(
                "parthenopegui.plotter.ExportPlotCode._plotInit", autospec=True
            ) as _pi, patch(
                "parthenopegui.plotter.ExportPlotCode._storeObjects", autospec=True
            ) as _so, patch(
                "parthenopegui.plotter.ExportPlotCode._plotObjects", autospec=True
            ) as _po, patch(
                "parthenopegui.plotter.ExportPlotCode._plotSettings", autospec=True
            ) as _ps, patch(
                "parthenopegui.plotter.ExportPlotCode._write", autospec=True
            ) as _w:
                epc._createFile()
                _pi.assert_called_once_with(epc)
                _so.assert_called_once_with(epc)
                _po.assert_called_once_with(epc)
                _ps.assert_called_once_with(epc)
                _w.assert_called_once_with(epc)

        def test_plotInit(self):
            """test _plotInit"""
            ps = PlotSettings()
            cpc = CurrentPlotContent(ps)
            ps.figsize = (12.0, 4.0)
            with patch(
                "parthenopegui.plotter.ExportPlotCode._createFile", autospec=True
            ) as _c:
                epc = ExportPlotCode(cpc, "c")
            self.assertEqual(epc.filecontent, "")
            epc._plotInit()
            self.assertIn(
                '"""File generated by the PArthENoPE GUI"""\n', epc.filecontent
            )
            self.assertIn("import numpy as np\n", epc.filecontent)
            self.assertIn("import matplotlib\n", epc.filecontent)
            self.assertIn("matplotlib.use('agg')\n", epc.filecontent)
            self.assertIn("import matplotlib.pyplot as plt\n", epc.filecontent)
            self.assertIn(
                "from parthenopegui.plotUtils import PGLine, PGContour\n",
                epc.filecontent,
            )
            self.assertIn(
                "fig = plt.figure(figsize=%s)\n" % str(ps.figsize), epc.filecontent
            )
            self.assertIn("plt.plot(np.nan, np.nan)\n", epc.filecontent)

        def test_plotObjects(self):
            """test _plotObjects"""
            ps = PlotSettings()
            cpc = CurrentPlotContent(ps)
            with patch(
                "parthenopegui.plotter.ExportPlotCode._createFile", autospec=True
            ) as _c:
                epc = ExportPlotCode(cpc, "c")
            self.assertEqual(epc.filecontent, "")
            epc._plotObjects()
            self.assertEqual(epc.filecontent, "")
            cpc.lines.append(PGLine([0], [1], 0))
            epc._plotObjects()
            for l in (
                "for l in lines:\n",
                "    plt.plot(\n",
                "        l.x,\n",
                "        l.y,\n",
                "        label=l.label,\n",
                "        color=l.color,\n",
                "        ls=l.style,\n",
                "        marker=l.marker,\n",
                "        lw=l.width,\n",
                "    )\n",
            ):
                self.assertIn(l, epc.filecontent)
            epc.filecontent = ""
            cpc.lines = []
            cpc.contours.append(PGContour([0, 1], [2, 3], [[4, 5], [6, 7]], 0))
            ps.zlabel = ""
            epc._plotObjects()
            for l in (
                "for c in contours:\n",
                "    if c.filled:\n",
                "        func = plt.contourf\n",
                "    else:\n",
                "        func = plt.contour\n",
                "    options = {}\n",
                "    if c.levels is not None:\n",
                '        options["levels"] = c.levels\n',
                "    if c.extend is not None:\n",
                '        options["extend"] = c.extend\n',
                '    options["cmap"] = matplotlib.cm.get_cmap(c.cmap)\n',
                "    CS = func(c.x, c.y, c.z, **options)\n",
                "    if c.hascbar:\n",
                "        cbar = plt.colorbar(CS)\n",
                "        cbar.ax.set_ylabel(c.zlabel, fontsize='medium')\n",
            ):
                self.assertIn(l, epc.filecontent)
            epc.filecontent = ""
            ps.zlabel = "abc"
            epc._plotObjects()
            for l in (
                "for c in contours:\n",
                "    if c.filled:\n",
                "        func = plt.contourf\n",
                "    else:\n",
                "        func = plt.contour\n",
                "    options = {}\n",
                "    if c.levels is not None:\n",
                '        options["levels"] = c.levels\n',
                "    if c.extend is not None:\n",
                '        options["extend"] = c.extend\n',
                '    options["cmap"] = matplotlib.cm.get_cmap(c.cmap)\n',
                "    CS = func(c.x, c.y, c.z, **options)\n",
                "    if c.hascbar:\n",
                "        cbar = plt.colorbar(CS)\n",
                "        cbar.ax.set_ylabel('abc', fontsize='medium')\n",
            ):
                self.assertIn(l, epc.filecontent)

        def test_plotSettings(self):
            """test _plotSettings"""
            ps = PlotSettings()
            cpc = CurrentPlotContent(ps)
            ps.xlabel = "xlab"
            ps.ylabel = "ylab"
            ps.xscale = "linear"
            ps.yscale = "log"
            ps.xlims = (1.0, 5.0)
            ps.ylims = (1e-2, 1e2)
            ps.title = ""
            ps.tight = False
            ps.legend = False
            ps.axesTextSize = "xx-large"
            with patch(
                "parthenopegui.plotter.ExportPlotCode._createFile", autospec=True
            ) as _c:
                epc = ExportPlotCode(cpc, "c")
            self.assertEqual(epc.filecontent, "")
            epc._plotSettings()
            self.assertIn(
                "plt.xlabel('%s', fontsize='%s')\n" % (ps.xlabel, ps.axesTextSize),
                epc.filecontent,
            )
            self.assertIn(
                "plt.ylabel('%s', fontsize='%s')\n" % (ps.ylabel, ps.axesTextSize),
                epc.filecontent,
            )
            self.assertIn("plt.xscale('%s')\n" % ps.xscale, epc.filecontent)
            self.assertIn("plt.yscale('%s')\n" % ps.yscale, epc.filecontent)
            self.assertIn("plt.xlim(%s)\n" % str(ps.xlims), epc.filecontent)
            self.assertIn("plt.ylim(%s)\n" % str(ps.ylims), epc.filecontent)
            self.assertNotIn("plt.title(", epc.filecontent)
            self.assertNotIn("plt.tight_layout()\n", epc.filecontent)
            self.assertNotIn("plt.legend(loc", epc.filecontent)
            self.assertRegex(
                epc.filecontent, "plt.savefig\('fig_([0-9]{6}_[0-9]{6}).pdf'\)\n"
            )
            self.assertIn("plt.close()\n", epc.filecontent)
            epc.filecontent = ""
            ps.title = "   "
            ps.tight = True
            ps.legend = True
            ps.legendLoc = "center"
            ps.legendNcols = 4
            ps.legendTextSize = "xx-large"
            epc.content.readSettings()
            epc._plotSettings()
            self.assertNotIn("plt.title(", epc.filecontent)
            self.assertIn("plt.tight_layout()\n", epc.filecontent)
            self.assertIn(
                "plt.legend(loc='%s', ncol=%s, fontsize='%s')\n"
                % (ps.legendLoc, ps.legendNcols, ps.legendTextSize),
                epc.filecontent,
            )
            pwm = PGText.PArthENoPEWatermark
            self.assertIn(
                (
                    "plt.text("
                    + '{x:}, {y:}, "{text:}",'
                    + ' color="{color:}", fontsize="{fontsize:}", ha="{ha:}",'
                    + " rotation={rotation:},"
                    + ' transform=plt.gca().transAxes, va="{va:}")\n'
                ).format(x=pwm["x"], y=pwm["y"], text=pwm["text"], **pwm["more"]),
                epc.filecontent,
            )
            epc.filecontent = ""
            ps.title = "mytitle  "
            epc.content.readSettings()
            epc._plotSettings()
            self.assertIn("plt.title('mytitle')\n", epc.filecontent)

        def test_storeObjects(self):
            """test _storeObjects"""
            ps = PlotSettings()
            cpc = CurrentPlotContent(ps)
            with patch(
                "parthenopegui.plotter.ExportPlotCode._createFile", autospec=True
            ) as _c:
                epc = ExportPlotCode(cpc, "a")
            cpc.lines.append(PGLine([0], [1], 0))
            cpc.lines.append(
                PGLine(
                    np.linspace(0, 10, 11),
                    np.linspace(100, 110, 11) / 100.0,
                    1,
                    c="r",
                    c2="chi2",
                    d="desc",
                    l="label",
                    m="o",
                    s=":",
                    w=0.5,
                    xl="xl",
                    yl="yl",
                )
            )
            cpc.contours.append(PGContour([0, 1], [2, 3], [[4, 5], [6, 7]], 0))
            cpc.contours.append(
                PGContour(
                    np.linspace(0, 10, 3),
                    np.linspace(20, 30, 5),
                    np.asarray(
                        [[0.3 * i + 1.0 * j for j in range(3)] for i in range(5)]
                    ),
                    0,
                    c2="lh",
                    cm="rainbow",
                    d="desc",
                    ex="both",
                    f=False,
                    hcb=False,
                    l="label",
                    lvs=[0, 1, 2],
                    xl="xlab",
                    yl="ylab",
                    zl="zlab",
                )
            )
            self.assertEqual(epc.filecontent, "")
            epc._storeObjects()
            self.assertIn("\nlines = []\n", epc.filecontent)
            self.assertIn("\ncontours = []\n", epc.filecontent)
            self.assertIn(
                """lines.append(
    PGLine(
        np.asarray([0]),
        np.asarray([1]),
        0,
        c='k',
        c2=False,
        d='',
        l='',
        m='',
        s='-',
        w=1.0,
        xl='',
        yl='',
    )
)""",
                epc.filecontent,
            )
            self.assertIn(
                """lines.append(
    PGLine(
        np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        np.asarray([1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1]),
        1,
        c='r',
        c2='chi2',
        d='desc',
        l='label',
        m='o',
        s=':',
        w=0.5,
        xl='xl',
        yl='yl',
    )
)""",
                epc.filecontent,
            )
            self.assertIn(
                """contours.append(
    PGContour(
        np.asarray([0, 1]),
        np.asarray([2, 3]),
        [[4, 5], [6, 7]],
        0,
        c2=False,
        cm='CMRmap',
        d='',
        ex=None,
        f=True,
        hcb=True,
        l='',
        lvs=None,
        xl='',
        yl='',
        zl='',
    )
)""",
                epc.filecontent,
            )
            self.assertIn(
                """contours.append(
    PGContour(
        np.asarray([0.0, 5.0, 10.0]),
        np.asarray([20.0, 22.5, 25.0, 27.5, 30.0]),
        np.asarray([[0. , 1. , 2. ], [0.3, 1.3, 2.3], [0.6, 1.6, 2.6], [0.9, 1.9, 2.9], [1.2, 2.2, 3.2]]),
        0,
        c2='lh',
        cm='rainbow',
        d='desc',
        ex='both',
        f=False,
        hcb=False,
        l='label',
        lvs=[0, 1, 2],
        xl='xlab',
        yl='ylab',
        zl='zlab',
    )
)""",
                epc.filecontent,
            )

        def test_write(self):
            """test _write"""
            ps = PlotSettings()
            cpc = CurrentPlotContent(ps)
            fn = "test_file.py"
            fc = "abc\ndef"
            with patch(
                "parthenopegui.plotter.ExportPlotCode._createFile", autospec=True
            ) as _c:
                epc = ExportPlotCode(cpc, fn)
            epc.filecontent = fc
            epc._write()
            with open(fn) as _f:
                c = _f.read()
            self.assertEqual(c, fc)
            os.remove(fn)

        def test_run(self):
            """test that the produced script works properly"""
            ps = PlotSettings()
            cpc = CurrentPlotContent(ps)
            fn = "test_file.py"
            cpc.lines.append(PGLine([0], [1], 0))
            cpc.lines.append(
                PGLine(
                    np.linspace(0, 10, 11),
                    np.linspace(100, 110, 11) / 100.0,
                    1,
                    c="r",
                    c2="chi2",
                    d="desc",
                    l="label",
                    m="o",
                    s=":",
                    w=0.5,
                    xl="xl",
                    yl="yl",
                )
            )
            cpc.contours.append(PGContour([0, 1], [2, 3], [[4, 5], [6, 7]], 0))
            cpc.contours.append(
                PGContour(
                    np.linspace(0, 10, 3),
                    np.linspace(20, 30, 5),
                    np.asarray(
                        [[0.3 * i + 1.0 * j for j in range(3)] for i in range(5)]
                    ),
                    0,
                    c2="chi2",
                    cm="rainbow",
                    d="desc",
                    ex="both",
                    f=False,
                    hcb=False,
                    l="label",
                    lvs=[0, 1, 2],
                    xl="xlab",
                    yl="ylab",
                    zl="zlab",
                )
            )
            epc = ExportPlotCode(cpc, fn)
            self.assertEqual(len(list(glob.iglob("fig_*.pdf"))), 0)
            if six.PY2:
                execfile(fn)
            else:
                with open(fn) as _f:
                    exec(_f.read())
            self.assertEqual(len(list(glob.iglob("fig_*.pdf"))), 1)
            for f in glob.iglob("fig_*.pdf"):
                os.remove(f)
            os.remove(fn)


########## mainWindow
if PGTestsConfig.test_mWi:

    class TestMWDescriptionPanel(PGTestCase):
        """Test the MWDescriptionPanel class"""

        def test_class(self):
            """Test the MWDescriptionPanel structure"""
            dp = MWDescriptionPanel()
            self.assertIsInstance(dp, QFrame)
            self.assertIsInstance(dp.layout(), QVBoxLayout)
            self.assertIsInstance(dp.text, QTextBrowser)
            self.assertIsInstance(dp.document, QTextDocument)
            self.assertEqual(dp.text.document(), dp.document)
            for n in ("infn", "unina", "logo"):
                img = dp.document.resource(
                    QTextDocument.ImageResource, QUrl("mydata://%s.png" % n)
                )
                self.assertIsInstance(img, QImage)
                self.assertIn('src="mydata://%s.png"' % n, dp.text.toHtml())

    class TestMainWindow(PGTestCasewMainW):
        """Test the MainWindow class"""

        def test_init(self):
            """test __init__ and properties"""
            self.assertIsInstance(self.mainW, QMainWindow)
            self.assertIsInstance(self.mainW.errormessage, Signal)
            with patch(
                "parthenopegui.mainWindow.MainWindow.excepthook", autospec=True
            ) as _eh:
                self.mainW.errormessage.emit("a", ValueError, "trcb")
                _eh.assert_called_once_with(self.mainW, "a", ValueError, "trcb")
            self.assertIsInstance(self.mainW.mainStatusBar, QStatusBar)
            self.assertIsInstance(self.mainW.nuclides, Nuclides)
            self.assertIsInstance(self.mainW.parameters, Parameters)
            self.assertIsInstance(self.mainW.reactions, Reactions)
            self.assertFalse(self.mainW.running)
            self.assertEqual(self.mainW.windowTitle(), PGText.appname)
            self.assertGeometry(
                self.mainW,
                0,
                0,
                QGuiApplication.primaryScreen().availableGeometry().width(),
                QGuiApplication.primaryScreen().availableGeometry().height(),
            )
            with patch(
                "parthenopegui.mainWindow.MainWindow.createMainLayout", autospec=True
            ) as _ml, patch(
                "parthenopegui.mainWindow.MainWindow.createMenusAndToolBar",
                autospec=True,
            ) as _mt, patch(
                "parthenopegui.mainWindow.MainWindow.setIcon", autospec=True
            ) as _si:
                m = MainWindow()
                _si.assert_called_once_with(m)
                _mt.assert_called_once_with(m)
                _ml.assert_called_once_with(m)

        def test_setIcon(self):
            """Test setIcon"""
            qi = QIcon()
            with patch("parthenopegui.mainWindow.QIcon", return_value=qi) as _qi, patch(
                "PySide2.QtWidgets.QMainWindow.setWindowIcon"
            ) as _si:
                self.mainW.setIcon()
                _qi.assert_called_once_with(":/images/icon.png")
                _si.assert_called_once_with(qi)

        def test_excepthook(self):
            """Test excepthook"""
            with patch("logging.Logger.error") as _e:
                self.mainW.excepthook("a", "b", "c")
                _e.assert_called_once_with(
                    PGText.errorUnhandled, exc_info=("a", "b", "c")
                )

        def test_closeEvent(self):
            """test closeEvent"""
            qe = MagicMock()
            qe.ignore = MagicMock()
            qe.accept = MagicMock()
            self.mainW.running = False
            self.mainW.closeEvent(qe)
            qe.accept.assert_called_once_with()
            qe.accept.reset_mock()
            self.mainW.running = True
            with patch(
                "parthenopegui.mainWindow.askYesNo", side_effect=[True, False]
            ) as _ayn:
                self.mainW.closeEvent(qe)
                _ayn.assert_called_once_with(PGText.warningRunningProcesses)
                qe.accept.assert_called_once_with()
                self.mainW.closeEvent(qe)
                qe.ignore.assert_called_once_with()
            self.mainW.running = False

        def test_createMenusAndToolBar(self):
            """Test createMenusAndToolBar"""

            def assertAction(act, t, tip, trig, s=None, i=None, p=None, mw=self.mainW):
                """test the properties of a single action

                Parameters:
                    act: the QAction to be tested
                    t: the title/text
                    tip: the status tip
                    trig: the name of the triggered function
                        (must be a MainWindow method)
                    s (default None): the shortcut, if any, or None
                    i (default None): the icon filename, if any, or None
                    p (default None): the mocked triggered function or None
                """
                self.assertIsInstance(act, QAction)
                self.assertEqual(act.text(), t)
                if s is not None:
                    self.assertEqual(act.shortcut(), s)
                if i is not None:
                    img = QImage(i).convertToFormat(QImage.Format_ARGB32_Premultiplied)
                    self.assertEqual(img, act.icon().pixmap(img.size()).toImage())
                self.assertEqual(act.statusTip(), tip)
                if p is None:
                    with patch(
                        "parthenopegui.mainWindow.MainWindow.%s" % trig, autospec=True
                    ) as _f:
                        act.trigger()
                        _f.assert_called_once_with(mw)
                else:
                    act.trigger()
                    self.assertTrue(p.call_count >= 1)

            self.assertIsInstance(self.mainW.exitAct, QAction)
            self.assertIsInstance(self.mainW.aboutAct, QAction)
            self.assertIsInstance(self.mainW.fileMenu, QMenu)
            self.assertEqual(self.mainW.fileMenu.title(), PGText.menuFileTitle)
            macts = self.mainW.fileMenu.actions()
            self.assertEqual(len(macts), 1)
            self.assertEqual(macts[0], self.mainW.exitAct)

            self.assertIsInstance(self.mainW.helpMenu, QMenu)
            self.assertEqual(self.mainW.helpMenu.title(), PGText.menuHelpTitle)
            macts = self.mainW.helpMenu.actions()
            self.assertEqual(len(macts), 1)
            self.assertEqual(macts[0], self.mainW.aboutAct)

            self.assertEqual(
                [a.menu() for a in self.mainW.menuBar().actions()],
                [self.mainW.fileMenu, self.mainW.helpMenu],
            )

            with patch("PySide2.QtWidgets.QMainWindow.close", autospec=True) as _f:
                mw = MainWindow()
                assertAction(
                    mw.exitAct,
                    PGText.menuActionExitTitle,
                    PGText.menuActionExitToolTip,
                    "close",
                    s="Ctrl+Q",
                    i=":/images/application-exit.png",
                    p=_f,
                )
                assertAction(
                    mw.aboutAct,
                    PGText.menuActionAboutTitle,
                    PGText.menuActionAboutToolTip,
                    "showAbout",
                    i=":/images/help-about.png",
                    mw=mw,
                )

            with patch("PySide2.QtWidgets.QMenuBar.addMenu") as _am, patch(
                "parthenopegui.mainWindow.QIcon", side_effect=[QIcon(), QIcon()]
            ) as _qi, patch(
                "parthenopegui.mainWindow.QAction", side_effect=[QAction(), QAction()]
            ) as _qa, patch(
                "PySide2.QtWidgets.QMenu.addAction"
            ) as _aa:
                mw.createMenusAndToolBar()
                self.assertEqual(_qi.call_count, 2)
                self.assertEqual(_qa.call_count, 2)
                self.assertEqual(_am.call_count, 2)
            with patch("PySide2.QtWidgets.QMenu.addAction") as _aa:
                mw.createMenusAndToolBar()
                self.assertEqual(_aa.call_count, 2)

        def test_showAbout(self):
            """test showAbout"""
            qb = QMessageBox()
            qb.setTextFormat = MagicMock()
            qb.setIconPixmap = MagicMock()
            qb.exec_ = MagicMock()
            pm = QPixmap()
            with patch(
                "parthenopegui.mainWindow.QMessageBox", return_value=qb
            ) as _m, patch("parthenopegui.mainWindow.QPixmap", return_value=pm) as _p:
                self.mainW.showAbout()
                _m.assert_called_once_with(
                    _m.Information,
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
                    parent=self.mainW,
                )
                _p.assert_called_once_with(":/images/icon.png")
            qb.setTextFormat.assert_called_once_with(Qt.RichText)
            qb.setIconPixmap.assert_called_once_with(pm)
            qb.exec_.assert_called_once_with()

        def test_createMainLayout(self):
            """Test createMainLayout"""
            with patch(
                "parthenopegui.mainWindow.MainWindow.createMainLayout", autospec=True
            ) as _f:
                mw = MainWindow()
            self.assertFalse(hasattr(mw, "tabWidget"))
            self.assertFalse(hasattr(mw, "descriptionPanel"))
            self.assertFalse(hasattr(mw, "runSettingsTab"))
            self.assertFalse(hasattr(mw, "plotPanel"))

            mw.createMainLayout()
            self.assertIsInstance(mw.tabWidget, QTabWidget)
            self.assertEqual(mw.centralWidget(), mw.tabWidget)
            self.assertFalse(mw.tabWidget.tabsClosable())
            self.assertIsInstance(mw.descriptionPanel, MWDescriptionPanel)
            self.assertEqual(mw.tabWidget.widget(0), mw.descriptionPanel)
            self.assertIsInstance(mw.runSettingsTab, MWRunSettingsPanel)
            self.assertEqual(mw.tabWidget.widget(1), mw.runSettingsTab)
            self.assertIsInstance(mw.plotPanel, MWPlotPanel)
            self.assertEqual(mw.tabWidget.widget(2), mw.plotPanel)


if __name__ == "__main__":
    unittest.main()
