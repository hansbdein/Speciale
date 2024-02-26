"""parthenopegui module with functions related
to running the fortran code
"""
import glob
import os
from multiprocessing import Pool, cpu_count

import numpy as np
import six
from PySide2.QtCore import QObject, QThread, Signal

try:
    from parthenopegui.configuration import (
        Configuration,
        Nuclides,
        Parameters,
        askYesNo,
    )
    from parthenopegui.errorManager import mainlogger, pGUIErrorManager
    from parthenopegui.runUtils import NoGUIRun, runSingle
    from parthenopegui.texts import PGText
except ImportError:
    print("[runner] Necessary parthenopegui submodules not found!")
    raise


class Thread_poolRunner(QThread):
    """Thread that runs the pool.
    Use a thread so that the GUI is not blocked during the execution
    """

    def __init__(self, pool, args, callbackFunc, parent):
        """Store instance parameters

        Parameters:
            pool: the pool that will be used to run the jobs
            args: the list of arguments for map_async
            callbackFunc: the function that will be called at the end
                of each run execution
            parent: the parent widget
        """
        super(Thread_poolRunner, self).__init__(parent)
        self.pool = pool
        self.args = args
        self.callbackFunc = callbackFunc
        self.parentWidget = parent

    def run(self):
        """Map the arguments to the pool,
        wait for completion and then emit some signals
        """
        results = []
        for a in self.args:
            r = self.pool.apply_async(runSingle, a, callback=self.callbackFunc)
            results.append(r)
        self.pool.close()
        for r in results:
            r.wait()
        self.parentWidget.poolHasFinished.emit()
        self.parentWidget.updateStatus.emit(PGText.runnerFinished)


class RunPArthENoPE(QObject, NoGUIRun):
    """Object that prepares the runs and manages the execution"""

    poolHasFinished = Signal()
    runHasFinished = Signal(int)
    updateStatus = Signal(str)
    updateProgressBar = Signal(int)

    def __init__(self, commonParams, gridPoints, parent=None):
        """Check if the PArthENoPE executable exists,
        store the run parameters
        and prepare for the execution

        Parameters:
            commonParams: a dictionary of parameters
                in common to the entire pool
            gridPoints: a numpy array with shape (N, 6)
                with the values of the 6 physical parameters
                in the N required grid points
            parent (default None): the main window instance
        """
        self.mainW = parent
        self.guilogger = pGUIErrorManager
        QObject.__init__(self)
        NoGUIRun.__init__(self)
        self.defineParams(commonParams, gridPoints)
        if not os.path.exists(commonParams["output_folder"]):
            os.makedirs(commonParams["output_folder"])

    def run(self):
        """Create the input cards,
        prepare a list of arguments for runSingle,
        create a Pool and a working thread, execute them
        """
        args = self.prepareArgs()

        self.updateStatus.emit(PGText.runnerRunning)

        self.pool = Pool(processes=cpu_count())
        self.thread = Thread_poolRunner(self.pool, args, self.runHasFinished.emit, self)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def stop(self):
        """Stop the ongoing runs
        and delete the thread that was executing them
        """
        if askYesNo(PGText.runnerAskStop):
            self.pool.terminate()
            try:
                self.thread.quit()
            except RuntimeError:
                mainlogger.debug("", exc_info=True)
            else:
                self.updateStatus.emit(PGText.runnerStopped)
