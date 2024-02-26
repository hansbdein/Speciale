"""Error manager for the GUI for the PArthENoPE BBN code"""

import logging
import logging.handlers
import sys
import traceback

if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

from PySide2.QtWidgets import QMessageBox

try:
    from parthenopegui.basic import PGConfiguration
except ImportError:
    print("[errorManager] Necessary parthenopegui submodules not found!")
    raise


class ErrorStream(StringIO):
    """Define a stream based on StringIO, which opens a `QMessageBox`
    when `write` is called
    """

    def __init__(self, *args, **kwargs):
        """The constructor, which passes the `*args, **kwargs` method
        to `StringIO.__init__`.
        """
        StringIO.__init__(self, *args, **kwargs)
        self.priority = 1
        self.lastMBox = None

    def setPriority(self, prio):
        """Set the priority level to be adopted
        when opening the `QMessageBox`.

        Parameter:
            prio: the priority level. 0, 1, 2+ correspond
                to `Information`, `Warning`, `Error`
        """
        self.priority = prio

    def write(self, text):
        """Override the `write` method of `StringIO`
        to show a `QMessageBox`,
        with icon according to the current priority.
        The priority is set to 1 after execution.

        Parameters:
            text: the text to display. "\n" is replaced with "<br>".

        Output:
            if text is empty or testing is False, return None
        """
        if text.strip() == "":
            return
        text = text.replace("\n", "<br>")
        self.lastMBox = QMessageBox(QMessageBox.Information, "Information", "")
        self.lastMBox.setText(text)
        if self.priority == 0:
            self.lastMBox.setIcon(QMessageBox.Information)
            self.lastMBox.setWindowTitle("Information")
        elif self.priority > 1:
            self.lastMBox.setIcon(QMessageBox.Critical)
            self.lastMBox.setWindowTitle("Error")
        else:
            self.lastMBox.setIcon(QMessageBox.Warning)
            self.lastMBox.setWindowTitle("Warning")
        self.priority = 1
        self.lastMBox.exec_()


class PGErrorManagerClass(object):
    """Class that manages the output of the errors and
    stores the messages into a log file
    """

    def __init__(
        self,
        level=logging.WARNING,
        logfilename=PGConfiguration.logFileName,
        loggerString=PGConfiguration.loggerString,
    ):
        """Define the logger and the verbosity level

        Parameters:
            level: verbosity level of the logger
            logfilename: the file name for the log
            loggerString: the `Logger` identifier string to use
                (default="physbibliolog")
        """
        self.loglevel = level
        # the main logger, will save to stdout and log file
        self.loggerString = loggerString
        self.logger = logging.getLogger(loggerString)
        self.logger.propagate = False
        self.logger.setLevel(logging.DEBUG)

        try:
            fh = logging.handlers.RotatingFileHandler(
                logfilename, maxBytes=5.0 * 2**20, backupCount=5
            )
        except AttributeError:
            self.logger.critical("Cannot create logfile: %s" % logfilename)
            raise
        fh.setLevel(self.loglevel)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)10s : " + "[%(module)s.%(funcName)s] %(message)s"
        )
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.defaultStream = logging.StreamHandler()
        self.defaultStream.setLevel(logging.DEBUG)
        formatter = logging.Formatter("[%(module)s.%(funcName)s] %(message)s")
        self.defaultStream.setFormatter(formatter)
        self.logger.addHandler(self.defaultStream)

    def excepthook(self, cls, exception, trcbk):
        """Function that will replace `sys.excepthook` to log
        any unexpected error that occurs

        Parameters:
            cls, exception, trcbk as in `sys.excepthook`
        """
        self.logger.error("Unhandled exception", exc_info=(cls, exception, trcbk))


class PGErrorManagerClassGui(PGErrorManagerClass):
    """Extends the error manager class to show
    gui messages through ErrorStream
    """

    def __init__(self):
        """Init the class, using PGErrorManagerClass.__init__ and
        the gui logger name,
        then add a new handler which uses ErrorStream
        """
        PGErrorManagerClass.__init__(self, loggerString="parthenopeguilog")
        self.guiStream = ErrorStream()
        self.guiHandler = logging.StreamHandler(self.guiStream)
        self.guiHandler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(message)s")
        self.guiHandler.setFormatter(formatter)
        self.logger.addHandler(self.guiHandler)

        for fn, p in (
            ("debug", 0),
            ("info", 0),
            ("warning", 1),
            ("error", 2),
            ("critical", 2),
            ("exception", 2),
        ):
            self.setDynamicAttr(fn, p)

    def setDynamicAttr(self, fn, p):
        """Wrapper that defines a dynamically named function.
        Needed to create shortcuts for Logger functions
        that set the correct priority in the message dialog

        Parameters:
            fn: the function name
            p: the priority level for defining the message dialog
        """

        def _f(*args, **kwargs):
            self.loggerPriority(p)
            return getattr(self.logger, fn)(*args, **kwargs)

        setattr(self, fn, _f)

    def loggerPriority(self, prio):
        """Define the priority level that must be used by
        the `ErrorStream` class in the first call of its `write` method.

        Parameter:
            prio: the priority level. 0, 1, 2+ correspond
                to Information, Warning, Error
        """
        self.guiStream.priority = prio


pErrorManager = PGErrorManagerClass()
mainlogger = pErrorManager.logger
sys.excepthook = pErrorManager.excepthook

pGUIErrorManager = PGErrorManagerClassGui()
