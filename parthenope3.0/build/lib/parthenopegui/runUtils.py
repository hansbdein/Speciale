import glob
import logging
import os
import pickle
from multiprocessing import Pool, cpu_count

import numpy as np
import six

try:
    from parthenopegui import FileNotFoundErrorClass
    from parthenopegui.basic import (
        PGConfiguration,
        __nuclideOrder__,
        __paramOrder__,
        __paramOrderParth__,
        paramsRealPath,
    )
    from parthenopegui.texts import PGText
except ImportError:
    print("[runUtils] Necessary parthenopegui submodules not found!")
    raise

mainlogger = logging.getLogger(PGConfiguration.loggerString)
shellCommand = "./%s < %s | tee %s > /dev/null"


def runSingle(ix, inputcard, logfilename):
    """Function that executes a single PArthENoPE instance,
    then removes the input files

    Parameters:
        ix: the run index
        inputcard: the name of the input card file
        logfilename: the name of the output log file

    Output:
        the run index
    """
    if not (
        isinstance(ix, int)
        and isinstance(inputcard, six.string_types)
        and isinstance(logfilename, six.string_types)
    ):
        mainlogger.exception(PGText.runSingleArgsType, exc_info=True)
        return -1
    with open(inputcard.replace("card", "in"), "w") as _f:
        _f.write("c\n%s\n" % inputcard)
    mainlogger.info(PGText.startRunI % ix)
    os.system(
        shellCommand
        % (
            PGConfiguration.fortranExecutableName,
            inputcard.replace("card", "in"),
            logfilename,
        )
    )
    return ix


class NoGUIRun(object):
    """Class that contains functions to process input/output files
    for the fortran code and run it outside of the GUI
    """

    def __init__(self):
        """Define basic properties and check that the fortran code
        has been compiled"""
        self.makeFortranExecutable()
        self.finishedRuns = []
        self.failedRuns = []
        self.nuclidesEvolution = {}
        self.nuclidesHeader = []
        self.parthenopeHeader = []
        self.parthenopeOutPoints = {}

    def allHaveFinished(self):
        """Save the grid file with the summary of all parthenope*.out,
        remove the single parthenope*.out files (merged in the grid file),
        print error messages if there are failed runs
        """
        self.fixParthenopeOutPoints()
        np.savetxt(
            self.commonParams["output_file_grid"],
            [
                self.parthenopeOutPoints[ix]
                for ix in sorted(self.parthenopeOutPoints.keys())
            ],
            fmt="%14.7e",
            header="  ".join(self.parthenopeHeader),
        )
        if len(self.failedRuns) == 0:
            for ix, pt in enumerate(self.getGridPointsList()):
                _f = self.commonParams["output_file_parthenope"] % ix
                try:
                    os.remove(_f)
                except IOError:
                    mainlogger.debug(PGText.fileNotFound % _f)
        else:
            failedInstructionsFilename = os.path.join(
                self.commonParams["output_folder"],
                PGConfiguration.failedRunsInstructionsFilename,
            )
            numfailedruns = len(self.failedRuns)
            numtotalruns = len(self.getGridPointsList())
            self.useGuiLogger().warning(
                PGText.failedRunsMessage
                % (
                    numfailedruns,
                    numtotalruns,
                    failedInstructionsFilename,
                )
            )
            with open(failedInstructionsFilename, "w") as _fi:
                _fi.write(
                    PGText.failedRunsInstructions.format(
                        numfailed=numfailedruns,
                        numtotal=numtotalruns,
                        cardslist="\n".join(
                            [
                                self.commonParams["inputcard_filename"] % ix
                                for ix in self.failedRuns
                            ]
                        ),
                        executablename=PGConfiguration.fortranExecutableName,
                        runcommandslist="\n".join(
                            [
                                shellCommand
                                % (
                                    PGConfiguration.fortranExecutableName,
                                    (
                                        self.commonParams["inputcard_filename"] % ix
                                    ).replace("card", "in"),
                                    self.commonParams["output_file_log"] % ix,
                                )
                                for ix in self.failedRuns
                            ]
                        ),
                        parthenopefile=self.commonParams["output_file_grid"],
                    )
                )

    def createInputCard(self, **kwargs):
        """Function that writes the input card to a file.
        The input dictionary should contain all the required fields
        for writing the configuration.
        """
        inputcardContent = """*LINES STARTING WITH AN ASTERISK ARE COMMENTED
TAU       {taun:.7f}                      experimental value of neutron lifetime
DNNU      {DeltaNnu:.7f}                  number of extra neutrinos
XIE       {csinue:.7f}                    nu_e chemical potential
XIX       {csinux:.7f}                    nu_x chemical potential
RHOLMBD   {rhoLambda:.7f}                 value of cosmological constant at the BBN epoch
OVERWRITE {overwrite:}                 option for overwriting the output files
FOLLOW    {onScreen:}                  option for following the evolution on the screen
ETA10     {eta10:.7f}                     value of eta10
OUTPUT    {save_nuclides:} {N_stored_nuclides:} {list_nucl_ix:}     options for customizing the output
NETWORK   {num_nuclides_net:}         number of nuclides in the network
FILES     {output_file_parthenope:} {output_file_nuclides:} {output_file_info:}          names of the three output files
RATES     {N_changed_rates:} {changed_rates:}
EXIT                                          terminates input""".format(
            list_nucl_ix=" ".join(
                [str(__nuclideOrder__[n]) for n in kwargs["stored_nuclides"]]
            ),
            changed_rates=" ".join(
                ["(%s %s %s)" % (tuple(r)) for r in kwargs["changed_rates_list"]]
            ),
            onScreen="T" if kwargs["onScreenOutput"] else "F",
            overwrite="T" if kwargs["output_overwrite"] else "F",
            save_nuclides="T" if kwargs["output_save_nuclides"] else "F",
            **kwargs
        )
        with open(kwargs["inputcard_filename"], "w") as _f:
            _f.write(inputcardContent)

    def defineParams(self, commonParams, gridPoints):
        """Use the provided commonParams and gridPoints to create
        the attributes corresponding attributes for the instance,
        and define the number of totalRuns

        Parameters:
            commonParams: a dict with all the parameters for the runs
            gridPoints: a np.array with all the points to be run
        """
        self.gridPoints = gridPoints
        self.commonParams = commonParams
        self.totalRuns = len(self.getGridPointsList())

    def fixParthenopeOutPoints(self):
        """Fix the parthenopeOutPoints content if points were missing"""
        for k in self.parthenopeOutPoints.keys():
            if len(self.parthenopeOutPoints[k]) > len(self.parthenopeHeader):
                self.parthenopeOutPoints[k] = self.parthenopeOutPoints[k][
                    : len(self.parthenopeHeader)
                ]
            while len(self.parthenopeOutPoints[k]) < len(self.parthenopeHeader):
                self.parthenopeOutPoints[k] = np.append(
                    self.parthenopeOutPoints[k], np.nan
                )

    def getGridPointsList(self):
        """Concatenate the list of points, if it is longer than one"""
        if len(self.gridPoints) > 1:
            gridPoints = np.concatenate(self.gridPoints)
        else:
            gridPoints = self.gridPoints[0]
        return gridPoints

    def getNuclidesHeader(self, fn):
        """Generate the new list of column names from nuclides*.out

        Parameter:
            fn: the name of the file to read
        """
        try:
            with open(fn) as f:
                self.nuclidesHeader = f.readlines()[0].replace("\n", "")
        except IOError:
            mainlogger.debug(PGText.fileNotFound % fn)
        except IndexError:
            mainlogger.debug(PGText.lineNotFound % (0, fn), exc_info=True)
        else:
            if len(self.nuclidesHeader) != 0:
                self.nuclidesHeader = [
                    e
                    for e in self.nuclidesHeader.split(" ")
                    if e.strip() != "" and e.strip() != "#"
                ]
            else:
                mainlogger.debug(PGText.lineNotFound % (0, fn), exc_info=True)

    def getParthenopeHeader(self, fn):
        """Generate the new list of column names from parthenope*.out

        Parameter:
            fn: the name of the file to read
        """
        try:
            with open(fn) as f:
                self.parthenopeHeader = f.readlines()[0].replace("\n", "")
        except IOError:
            mainlogger.debug(PGText.fileNotFound % fn)
        except IndexError:
            mainlogger.debug(PGText.lineNotFound % (0, fn), exc_info=True)
        else:
            if len(self.parthenopeHeader) != 0:
                self.parthenopeHeader = [
                    e.replace("N_nu", "N_eff")
                    for e in self.parthenopeHeader.split(" ")
                    if e.strip() != "" and e.strip() != "#"
                ]
            else:
                mainlogger.debug(PGText.lineNotFound % (0, fn), exc_info=True)
        while len(self.parthenopeHeader) < 6:
            self.parthenopeHeader.append("undefined")

    def makeFortranExecutable(self):
        """If the fortran executable is not present
        in the current directory, try to compile it.
        If compilation fails, raise an IOError
        """
        if not os.path.exists(PGConfiguration.fortranExecutableName):
            os.system("make")
            if not os.path.exists(PGConfiguration.fortranExecutableName):
                self.useGuiLogger().critical(PGText.runnerParthenopeNotFound)
                raise SystemExit(1)

    def oneHasFinished(self, ix):
        """Action to be performed at the end of each run:
        update list of completed runs and update progress bar

        Parameter:
            ix: the run index
        """
        if ix < 0:
            return
        logfile = self.commonParams["output_file_log"] % ix
        try:
            with open(logfile) as _lf:
                lines = _lf.readlines()
        except IOError:
            mainlogger.debug(PGText.fileNotFound % logfile)
            lines = []
        self.readNuclides(ix)
        self.readParthenope(ix)
        if any([PGText.runCompletedSuccessfully in l for l in lines]):
            self.finishedRuns.append(ix)
            if hasattr(self, "guilogger"):
                self.updateProgressBar.emit(len(self.finishedRuns))
            inputcard = self.commonParams["inputcard_filename"] % ix
            os.remove(inputcard.replace("card", "in"))
            os.remove(inputcard)
        else:
            if ix not in self.failedRuns:
                self.failedRuns.append(ix)
            specificErrorMessage = ""
            if any(
                [
                    PGText.runCompletedFailingCheck in l
                    or PGText.runCompletedFailingManual in l
                    for l in lines
                ]
            ):
                lastfail = 0
                lastcheck = 0
                for il, l in enumerate(lines):
                    if PGText.runCompletedFailing in l:
                        lastfail = il
                    if (
                        PGText.runCompletedFailingCheck in l
                        or PGText.runCompletedFailingManual in l
                    ):
                        lastcheck = il
                em = lines[lastfail + 1 : lastcheck]
                specificErrorMessage += PGText.runCompletedErrorMessage + "\n".join(
                    [l.replace("\n", "") for l in em]
                )
                specificErrorMessage += (
                    "\n"
                    + PGText.runCompletedErrorFile
                    % logfile.replace(
                        PGConfiguration.nuclidesFilename, PGConfiguration.infoFilename
                    )
                )
            self.useGuiLogger().warning(
                PGText.runFailed % (ix, logfile) + specificErrorMessage
            )

    def prepareArgs(self):
        """Delete existing output files and
        prepare the list of parameters and create an input card
        for the fortran code for each point to be executed

        Output:
            a list of [int, input filename, log filename] triplets
        """
        args = []
        for f in glob.glob(
            os.path.join(
                self.commonParams["output_folder"],
                "%s_%s_*"
                % (PGConfiguration.parthenopeFilename, self.commonParams["now"]),
            )
        ):
            os.remove(f)
        for ix, pt in enumerate(self.getGridPointsList()):
            params = self.commonParams.copy()
            for ip, p in enumerate(__paramOrder__):
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
            self.createInputCard(**params)
            args.append([ix, params["inputcard_filename"], params["output_file_log"]])
        return args

    def prepareRunFromPickle(self, foldername):
        """Read a previously created settings.obj and
        prepare the parameters for a run of the fortran code

        Parameters:
            foldername: the name of the folder where the file is located
        """
        filename = os.path.join(foldername, PGConfiguration.runSettingsObj)
        if os.path.isfile(filename):
            mainlogger.info(PGText.tryToOpenFolder % foldername)
            try:
                with open(filename, "rb") as _f:
                    commonParams, gridPoints = pickle.load(_f)
            except (
                EOFError,
                FileNotFoundErrorClass,
                TypeError,
                UnicodeDecodeError,
                ValueError,
            ):
                mainlogger.error(PGText.errorCannotLoadGrid)
            else:
                self.defineParams(paramsRealPath(commonParams, foldername), gridPoints)
        else:
            mainlogger.error(PGText.errorCannotLoadGrid)

    def readAllResults(self):
        """Read all the separated nuclides*.out,
        the global parthenope.out and their headers
        """
        _f = self.commonParams["output_file_grid"]
        self.getParthenopeHeader(_f)
        try:
            parthenopeOutPoints = np.loadtxt(_f)
        except (OSError, IOError):
            for ix, pt in enumerate(self.getGridPointsList()):
                self.readParthenope(ix)
            self.allHaveFinished()
            parthenopeOutPoints = np.loadtxt(_f)
        for i, l in enumerate(parthenopeOutPoints):
            self.parthenopeOutPoints[i] = l
        for ix, pt in enumerate(self.getGridPointsList()):
            self.readNuclides(ix)

    def readNuclides(self, ix):
        """Read the content and the header of a single nuclides*.out

        Parameter:
            ix: the index of the file to read in the list of grid points
        """
        _f = self.commonParams["output_file_nuclides"] % ix
        self.getNuclidesHeader(_f)
        try:
            thisData = np.loadtxt(_f, dtype=str, skiprows=1)
        except IOError:
            mainlogger.debug(PGText.fileNotFound % _f)
        except StopIteration:
            mainlogger.debug(PGText.emptyFile % _f)
        else:
            if len(thisData) == 0:
                mainlogger.debug(PGText.emptyFile % _f)
                return
            thisData = np.char.replace(thisData, "D", "e")
            thisData = np.char.replace(thisData, "b", "")
            self.nuclidesEvolution[ix] = thisData.astype(np.float64)

    def readParthenope(self, ix):
        """Read the content and the header of a single parthenope*.out

        Parameter:
            ix: the index of the file to read in the list of grid points
        """
        _f = self.commonParams["output_file_parthenope"] % ix
        if self.parthenopeHeader in ([], ["undefined"] * 6):
            self.getParthenopeHeader(_f)
        while len(self.parthenopeHeader) < 6:
            self.parthenopeHeader.append("undefined")
        thisData = []
        try:
            thisData = np.genfromtxt(_f, dtype=str, skip_header=1)
        except IOError:
            mainlogger.debug(PGText.fileNotFound % _f)
        except StopIteration:
            mainlogger.debug(PGText.emptyFile % _f)
        finally:
            if len(thisData) == 0:
                if ix not in self.failedRuns:
                    self.failedRuns.append(ix)
                mainlogger.debug(PGText.emptyFile % _f)
                if len(self.parthenopeHeader) != 0:
                    self.parthenopeOutPoints[ix] = np.array(
                        [np.nan for x in self.parthenopeHeader]
                    )
                else:
                    self.parthenopeOutPoints[ix] = np.array(
                        [np.nan for ip, p in enumerate(__paramOrderParth__)]
                    )
                try:
                    for ip, p in enumerate(__paramOrderParth__):
                        if p == "eta10":
                            self.parthenopeOutPoints[ix][ip] = float(
                                "{0:.5f}".format(
                                    self.getGridPointsList()[ix][
                                        __paramOrder__.index(p)
                                    ]
                                )
                            )
                        else:
                            self.parthenopeOutPoints[ix][ip] = float(
                                "{0:.2f}".format(
                                    self.getGridPointsList()[ix][
                                        __paramOrder__.index(p)
                                    ]
                                )
                            )
                    self.parthenopeOutPoints[ix][0] += 3  # DeltaNnu to Nnu
                except IndexError:
                    mainlogger.debug("Missing point in the grid: %s" % ix)
            else:
                thisData = np.char.replace(thisData, "D", "e")
                thisData = np.char.replace(thisData, "b", "")
                self.parthenopeOutPoints[ix] = thisData.astype(np.float64)

    def run(self):
        """Prepare the list of arguments, the Pool of parallel jobs,
        submit the jobs for each set of arguments, wait their end
        and perform some final operations
        """
        self.args = self.prepareArgs()
        mainlogger.info(PGText.startRun)
        self.pool = Pool(processes=cpu_count())
        results = []
        for a in self.args:
            r = self.pool.apply_async(runSingle, a, callback=self.oneHasFinished)
            results.append(r)
        self.pool.close()
        for r in results:
            r.wait()
        mainlogger.info(PGText.runnerFinished)
        self.allHaveFinished()

    def useGuiLogger(self):
        """Return self.guilogger if it is a valid logging.Logger,
        or the mainlogger instead

        Output:
            a logging.Logger instance
        """
        return self.guilogger if hasattr(self, "guilogger") else mainlogger
