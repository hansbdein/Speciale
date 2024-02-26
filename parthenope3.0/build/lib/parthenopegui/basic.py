import logging
import sys

import six
from appdirs import AppDirs


class PGConfiguration(object):
    """Basic parameters for the configuration of sizes, filenames and so on"""

    fontsize = 13
    fontsizeinfo = 15
    logoHeight = 150
    imageHeaderHeight = 110
    rowImageHeight = 32

    failedRunsInstructionsFilename = "failed_instructions.txt"
    fortranExecutableName = "parthenope3.0"
    fortranOutputFilename = "fortran_output"
    logFileName = "gui.log"
    loggerString = "parthenopelog"
    runSettingsObj = "settings.obj"

    infoFilename = "info"
    nuclidesFilename = "nuclides"
    parthenopeFilename = "parthenope"

    limitNuclides = {"smallNet": 10, "interNet": 19, "complNet": 27}
    limitReactions = {"smallNet": 41, "interNet": 74, "complNet": 101}

    dataPath = AppDirs("PArthENoPE").user_data_dir
    cachePath = AppDirs("PArthENoPE").user_cache_dir

    formatParam = "%.6g"


__paramOrder__ = ["eta10", "DeltaNnu", "taun", "csinue", "csinux", "rhoLambda"]
__paramOrderParth__ = ["DeltaNnu", "csinue", "csinux", "taun", "rhoLambda", "eta10"]
__nuclideOrder__ = {
    "n": 1,
    "p": 2,
    "2H": 3,
    "3H": 4,
    "3He": 5,
    "4He": 6,
    "6Li": 7,
    "7Li": 8,
    "7Be": 9,
    "8Li": 10,
    "8B": 11,
    "9Be": 12,
    "10B": 13,
    "11B": 14,
    "11C": 15,
    "12B": 16,
    "12C": 17,
    "12N": 18,
    "13C": 19,
    "13N": 20,
    "14C": 21,
    "14N": 22,
    "14O": 23,
    "15N": 24,
    "15O": 25,
    "16O": 26,
}


def paramsHidePath(params, dirname):
    """Before saving the settings.obj file, we need to remove
    the name of the folder from all the parameters that contain it,
    so that it will be possible to reuse the content
    after renaming the folder.

    Parameters:
        * the dict with all the parameters
        * dirname: the name of the folder that must be removed
    """
    new = params.copy()
    for k, v in params.items():
        if (
            k != "output_folder"
            and isinstance(v, six.string_types)
            and v.startswith(dirname)
        ):
            new[k] = v.replace(dirname, "FOLDER#")
    return new


def paramsRealPath(params, dirname):
    """After saving the settings.obj file, we need to restore
    the name of the folder in all the parameters that contain it,
    so that it will be possible to reuse the content
    after renaming the folder.

    Parameters:
        * the dict with all the parameters
        * dirname: the name of the folder that must be restored
    """
    new = params.copy()
    for k, v in params.items():
        if k != "output_folder" and isinstance(v, six.string_types):
            new[k] = v.replace("FOLDER#", dirname)
    return new


def runInGUI(argv):
    """Prepare the necessary objects and run the GUI.
    If a command-line argument is present, run the fortran code
    using the inputs from the given folder
    """
    from PySide2.QtWidgets import QApplication

    from parthenopegui.errorManager import pGUIErrorManager
    from parthenopegui.mainWindow import MainWindow

    app = QApplication(argv)
    mainWin = MainWindow()
    sys.excepthook = mainWin.errormessage.emit
    mainWin.show()
    if len(argv) > 1:
        mainWin.tabWidget.setCurrentWidget(mainWin.runSettingsTab)
        pGUIErrorManager.info(
            "Preparing to run with imported grid settings.\n"
            + "Notice that the GUI will not show the run settings properly!\n"
            + "You cannot edit the previous settings nor check them."
        )
        mainWin.runSettingsTab.runPanel.startRunPickle(argv[1])
    mainWin.raise_()
    sys.exit(app.exec_())


def runNoGUI(argv):
    """Run the fortran code using the inputs from the folder
    specified as a command-line argument, if no display is available
    """
    from parthenopegui.errorManager import mainlogger

    mainlogger.setLevel(logging.DEBUG)
    if len(argv) < 2:
        mainlogger.info(
            "Nothing to do: cannot import PySide2 and no folder to run was indicated!"
        )
        sys.exit(1)
        return
    from parthenopegui.runUtils import NoGUIRun

    run = NoGUIRun()
    run.prepareRunFromPickle(argv[1])
    run.run()
