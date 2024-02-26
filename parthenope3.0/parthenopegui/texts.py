"""Includes the strings that enter the GUI"""
from collections import OrderedDict


class PGText(object):
    """Contains all the strings that appear in GUI elements and messages"""

    appname = "PArthENoPE GUI"
    copyright = "Copyright Naples Astroparticle Group<br>All Rights Reserved"
    contactname = " Ofelia Pisanti "
    contactphone = "+39 081 676914"
    contactemail = "pisanti@na.infn.it"
    parthenopename = "PArthENoPE"
    parthenopenamelong = (
        "Public Algorithm Evaluating the Nucleosynthesis of Primordial Elements"
    )
    parthenopedescription = (
        "PArthENoPE computes the abundances of light nuclides "
        + "produced during Big Bang Nucleosynthesis."
        + "<br>"
        + "Starting from nuclear statistic equilibrium conditions,"
        + "<br>"
        + " the program solves the set of coupled ordinary differential equations,"
        + "<br>"
        + " follows the departure from chemical equilibrium of nuclear species,"
        + "<br>"
        + "and determines their asymptotic abundances as a function of several input"
        + "<br>"
        + "cosmological parameters as the baryon density, the number of effective neutrinos,"
        + "<br>"
        + "the value of cosmological constant and the neutrino chemical potential."
        + "<br>"
        + "<br>For further information and help visit the web page"
    )

    smallAsciiArrow = "->"
    comma = ","

    aboutTitle = "About PArthENoPE GUI"
    askReplace = "File exists. Do you want to replace it?"

    buttonAccept = "OK"
    buttonCancel = "Cancel"
    buttonDelete = "Delete"
    buttonEdit = "Edit"

    cannotWrite = "Cannot write: %s"
    category = "Category:"
    current = "Current:"

    editParametersAdd = "Add a new grid or single point for the physical parameters"
    editParametersColumnLabels = [
        "Type",
        "Default",
        "Minimum",
        "Maximum",
        "Number of points",
    ]
    editParametersCombo = ["single point", "grid"]
    editParametersTitle = "Edit physical parameters"
    editParametersToolTips = {
        "combo": "Select between a 'single point' or "
        + "a list of points ('grid') for this parameter",
        "def": "If 'single point' is selected, enter the parameter value",
        "min": "If 'grid' is selected, enter the lower value of the interval",
        "max": "If 'grid' is selected, enter the upper value of the interval",
        "num": "If 'grid' is selected, enter the number of points to consider",
    }

    editReactionCombo = ["default", "low", "high", "custom factor"]
    editReactionFirstLabel = "Set reaction rate for:"
    editReactionTitle = "Customize reaction"
    editReactionToolTip = (
        "Select the type of correction you want to use.\n"
        + "'Default' will use the standard option in the Fortran code.\n"
        + "'Low' or 'High' will use for the reaction rate the"
        + " lower or upper value at 1 sigma with respect to the 'Default' value.\n"
        + "Finally, you can set a 'Custom factor' that will be used "
        + "as a global normalization of the 'Default' reaction rate."
    )
    editReactionValueToolTip = (
        "When the correction type is 'Custom factor', use the number provided here."
        + " Ignored in the other cases."
    )

    emptyFile = "File is empty: %s"

    errorCannotFindGrid = (
        "Cannot find the settings ('settings.obj') for the selected grid. "
        + " Is the folder correct?"
    )
    errorCannotFindIndex = "Cannot find the requested index!"
    errorCannotLoadGrid = (
        "Cannot import the requested grid.\n"
        + "Does it exist?\n"
        + "Was it created with the current python version?"
    )
    errorCannotWriteFile = "Error! the file cannot be written in the selected location!"
    errorInvalidField = "Invalid %s for %s: %s"
    errorInvalidFieldSet = "Invalid %s for %s: %s. Set to %s"
    errorInvalidParamLimits = (
        "Invalid limits for %s: same value for min (%s) and max (%s)"
    )
    errorLoadingGrid = "Something went wrong while deleting the grid"
    errorNoPhysicalParams = "No physical parameters selected!"
    errorNoOutputFolder = "No output folder selected!"
    errorNoReactionNetwork = "No reaction network selected!"
    errorReadTable = "Error in reading table content"
    errorUnhandled = "Unhandled exception"

    failedRunsInstructions = (
        "{numfailed:} runs over {numtotal:} failed in the last grid run.\n"
        + "In order to solve the problems, we suggest to manually edit the "
        + "input card files listed below, slightly perturb the value "
        + "of some physical parameter and run again the fortran code manually.\n"
        + "DO NOT run the grid again from the GUI, "
        + "as in that case all the points will be repeated and the failed ones "
        + "will most likely fail again.\n\n"
        + "This is the list of input cards that correspond to failed runs:\n"
        + "{cardslist:}\n\n"
        + "After having modified one of the input parameters, "
        + "to run them, you will need to open a command line "
        + "in the same folder of the '{executablename:}' executable "
        + "and use the appropriate line from the following list:\n"
        + "{runcommandslist:}\n\n"
        + "When all the points are completed successfully, you should "
        + "manually remove the summary output file '{parthenopefile:}' "
        + "before opening again the grid from the GUI. "
        + "In this way, all the points will be loaded again and "
        + "the summary file will be properly filled."
    )
    failedRunsMessage = (
        "Some of the grid points (%d over %d) could not be successfully completed.\n"
        + "You can proceed with plots, and in that case some interpolation "
        + "will be used in order to substitute the failed points.\n"
        + "Alternatively, you can check the content of the %s file "
        + "where you will find some instructions on how to run the failed points again."
    )
    fileNotFound = "File not found: %s"

    interpolationPerformedContour = (
        "Warning: NaN detected in the current contour.\n"
        + "Some points have been removed or replaced with a linear interpolation.\n"
        + "Original z values: %s\n"
        + "Modified z values: %s\n"
    )
    interpolationPerformedLine = (
        "Warning: NaN detected in the current line.\n"
        + "Some points have been removed or replaced with a linear interpolation.\n"
        + "Original line: %s\n"
        + "Modified line: %s\n"
    )

    lineNotFound = "Line not found: %d in %s"
    loadGridAsk = "Select the directory that contains the grid"

    mean = "Mean:"
    menuFileTitle = "&File"
    menuHelpTitle = "&Help"

    menuActionAboutTitle = "&About"
    menuActionAboutToolTip = "Show About box"
    menuActionExitTitle = "E&xit"
    menuActionExitToolTip = "Exit application"

    networkCustomizeRate = (
        "Double click on the icon in the Edit column"
        + " to customize a given reaction rate"
    )
    networkDescription = OrderedDict(
        [
            ("smallNet", "SMALL - 9 nuclides, 40 reactions"),
            ("interNet", "INTERMEDIATE - 18 nuclides, 73 reactions"),
            ("complNet", "COMPLETE - 26 nuclides, 100 reactions"),
        ]
    )
    networkToolTips = {
        "smallNet": "Use the smallest network of reactions, "
        + "which includes only the nuclides up to A=7",
        "interNet": "Use an intermediate network of reactions, "
        + "which includes all the nuclides up to A=12",
        "complNet": "Use the complete network of reactions, "
        + "which includes all the nuclides up to A=16",
    }
    networkSelect = "Select network"
    noGrid = (
        "Click here for selecting the directory with the data to consider for the plots"
    )
    nuclidesOthersTitle = "Other available nuclides"
    nuclidesOthersToolTip = (
        "List of nuclides that you may consider to add in the list of outputs"
    )
    nuclidesSelectAllText = "<<"
    nuclidesSelectAllToolTip = "Select all nuclides"
    nuclidesSelectedTitle = "Selected nuclides"
    nuclidesSelectedToolTip = (
        "List of nuclides currently selected for being included in the output"
    )
    nuclidesUnselectAllText = ">>"
    nuclidesUnselectAllToolTip = "Unselect all nuclides"

    outputPanelDirectoryAsk = "Select where to save the output files:"
    outputPanelDirectoryDialogTitle = "Specify output directory"
    outputPanelDirectoryInitialTitle = "Click to select output directory"
    outputPanelDirectoryToolTip = "Select the folder where to save the output files"
    outputPanelNuclidesInOutput = (
        "Check this if you want to save the evolution of nuclide abundances:"
    )
    outputPanelSelectDescription = (
        "By drag&drop you can move nuclides from the left column (selected for the output)"
        + " to the left one (unselected for the output)"
    )

    PArthENoPEWatermark = {
        "text": "Created with PArthENoPE 3.0 GUI",
        "x": 1.0,
        "y": 1.0,
        "more": {
            "color": "#999999",
            "fontsize": "small",
            "ha": "left",
            "rotation": -90,
            "va": "top",
        },
    }

    panelTitleDescription = "Home"
    panelTitleRun = "Run"
    panelTitlePlot = "Plot"
    panelToolTipDescription = "General information on PArthENoPE"
    panelToolTipRun = "Settings for preparing a run and call the Fortran code"
    panelToolTipPlot = "Read existing results and generate plots"

    parameterDescriptions = {
        "eta10": "",
        "DeltaNnu": "",
        "taun": "neutron",
        "csinue": "",
        "csinux": "",
        "rhoLambda": "dark energy",
    }
    parentAttributeMissing = "`parent` has no 'parameters' attribute: cannot proceed"

    physicalParamsAdd = "Add a new point or grid"
    physicalParamsAddToolTip = (
        "Click here to open a window that lets you to configure a new point"
        + " or a list of points that will be used in the PArthENoPE runs"
    )
    physicalParamsDescription = "Configure here the physical parameters:"
    physicalParamsEmpty = "Nothing to list yet..."
    physicalParamsNumberSummary = (
        "<b>%d</b> grids or sparse points currently set, "
        + "for a total of <b>%d points</b>"
    )

    plotAbundancesTypeError = "AbundancesGenericPanel.type must be 'evo', '1d' or '2d'!"
    plotAskDeleteLine = "Do you really want to delete this line from the plot?"
    plotAvailablePoints = "Available output: double click anywhere on a row to use it"
    plotChi2AskLabel = "Do you want to employ one or more abundances for constructing a Chi^2 or likelihood?"
    plotChi2AskCheckToolTip = (
        "If checked, the plot will show a Gaussian chi^2 function "
        + "of the selected nuclide instead of the nuclide itself"
    )
    plotChi2AskMeanToolTip = "The value of the mean for the Gaussian chi^2 function"
    plotChi2AskStdToolTip = (
        "The value of the standard deviation for the Gaussian chi^2 function"
    )
    plotContourCMap = "Select a colormap to use:"
    plotContourExtend = "How do you need to extend the color bar?"
    plotContourExtendToolTip = (
        "If you use some specific contour levels, "
        + "you may want to extend the colorbar above and/or below"
        + " the adopted range"
    )
    plotContourFilled = "Select to use filled contours:"
    plotContourFilledToolTip = (
        "Check if you want filled contours. Only lines will be plotted otherwise"
    )
    plotContourHasCbar = "Check to add a colorbar for this set of contours:"
    plotContourHasCbarToolTip = (
        "If checked, add a colorbar for this set of contours "
        + "on the side of the plot"
    )
    plotContourLevelsInvalidLen = (
        "Invalid length for levels (they must be at least two): %s"
    )
    plotContourLevelsInvalidLevel = (
        "Invalid level (it cannot be converted into a number): %s"
    )
    plotContourLevelsInvalidSyntax = (
        "Invalid syntax for levels (it cannot be parsed): %s"
    )
    plotContourLevelsInvalidType = "Invalid type for levels (it should be a list): %s"
    plotContourLevelsLabel = "Write a list of contours to use in the plot (optional):"
    plotContourLevelsToolTip = (
        "If you want to select manually the levels for the colorbar, "
        + "enter them using the syntax of a python list "
        + "(e.g. '[0, 1, 2]')"
    )
    plotContoursInUse = (
        "This is the list of contours that are currently used in the plot"
        + " (double click to edit the line settings):"
    )
    plotContourZLabel = "Label for the colorbar (if used):"
    plotContourZLabelToolTip = (
        "Insert a label for the colorbar. Latex syntax is allowed using '$...$'"
    )
    plotInvalidColor = "This string has not been recognized as a color: %s"
    plotInvalidMean = "This 'mean' is not a valid number: %s"
    plotInvalidStddev = "This 'standard deviation' is not a valid positive number: %s"
    plotInvalidWidth = "This string is not a valid line width: %s"
    plotLineColor = "Insert line color:"
    plotLineColorToolTip = (
        "The color that will be used for the line in the plot."
        + " Select one of the available or edit to insert a new one."
        + " Valid colors are represented by any string that is allowed by matplotlib.pyplot"
    )
    plotLineLabel = "Insert label for the line:"
    plotLineLabelToolTip = (
        "The legend label for the line in the plot. "
        + "Latex syntax is allowed using '$...$'"
    )
    plotLineMarker = "Select line marker:"
    plotLineMarkerToolTip = (
        "The marker to be used for the points of the line."
        + " It can be any marker allowed by matplotlib.pyplot"
    )
    plotLineStyle = "Select line style:"
    plotLineStyleToolTip = (
        "The line style to be used in the plot."
        + " It can be any style allowed by matplotlib.pyplot"
    )
    plotLineWidth = "Insert line width:"
    plotLineWidthToolTip = "The line width to be used for the line."
    plotLinesInUse = (
        "This is the list of outputs currently included in the plot."
        + " Double click to edit the line settings, "
        + "or double click on the icons in the last columns to delete or reorder the lines."
    )
    plotPlaceholderText = "Select a plot type to start editing"
    plotSaved = "Plot saved."
    plotScriptSaved = "Script to create the plot saved to %s."
    plotSelectChi2AddLine = "Combine with chi2 on other parameter"
    plotSelectChi2AddLineToolTip = (
        "Click to add a line to create a chi2 combination with one more quantity"
    )
    plotSelectChi2Type = (
        "Select the function that you want to plot"
        + " (chi2, likelihood or log likelihood):"
    )
    plotSelectChi2TypeContents = OrderedDict(
        [
            ("chi2", ["Chi2", "Simple Gaussian chi2 function"]),
            (
                "lh",
                [
                    "Likelihood",
                    "Convert the chi2 into a Gaussian likelihood [exp(-chi2/2)]",
                ],
            ),
            (
                "nllh",
                [
                    "- log Likelihood",
                    "Convert the chi2 into a -log likelihood [chi2/2]",
                ],
            ),
            (
                "pllh",
                [
                    "+ log Likelihood",
                    "Convert the chi2 into a log likelihood [-chi2/2]",
                ],
            ),
        ]
    )
    plotSelectGridText = (
        "Select the folder containing the output files that will be used for the plots:"
    )
    plotSelectNuclide = "Select an output quantity to load"
    plotSelectNuclideContour = "Select a contour to add in the plot"
    plotSelectNuclideLine = "Output quantity to load"
    plotSelectNuclideToolTip = (
        "The selected nuclide will be used as y coordinate in the plot"
    )
    plotSelectPlotType = "Select the type of plot that you want to use:"
    plotSettAskResetImage = "Are you sure you want to reset the image content now?"
    plotSettAxesTextSize = "Size of the text in the axis labels:"
    plotSettAxesTextSizeToolTip = (
        "Select the size of the text labels in the legend,"
        + " relative to the default font size"
    )
    plotSettExportImage = "Export image script"
    plotSettExportImageToolTip = (
        "Export the python code used to create the current image in a standalone script"
    )
    plotSettFigSize = "x and y sizes for the figure"
    plotSettFigSizeToolTipH = (
        "Horizontal size of the figure, in inches, for the matplotlib.pyplot commands"
    )
    plotSettFigSizeToolTipV = (
        "Vertical size of the figure, in inches, for the matplotlib.pyplot commands"
    )
    plotSettFigSizeWarning = (
        "(the figure size may be not displayed correctly in the panel above, "
        + "but these settings will be adopted in the saved figure)"
    )
    plotSettLegend = "Add legend?"
    plotSettLegendToolTip = (
        "Select the checkbox if you want to have a legend in the plot"
    )
    plotSettLegendLoc = "Legend location:"
    plotSettLegendLocToolTip = "Select the location for the legend, if present"
    plotSettLegendNumCols = "Number of columns:"
    plotSettLegendNumColsToolTip = (
        "Select the number of columns for the labels in the legend"
    )
    plotSettLegendTextSize = "Size of the text in the legend:"
    plotSettLegendTextSizeToolTip = (
        "Select the size of the text labels in the legend,"
        + " relative to the default font size"
    )
    plotSettRefreshImage = "Refresh the image content"
    plotSettRefreshImageToolTip = "Refresh the image content using the current settings"
    plotSettRevertImage = "Revert changes to plot settings"
    plotSettRevertImageToolTip = (
        "Restore all the automatic values for the plot settings, and discard manual changes."
        + " This will not modify the content of the plot (lines, contours),"
        + " but only labels, scales and limits."
    )
    plotSettResetImage = "Reset the image content"
    plotSettResetImageToolTip = (
        "Cancel the current image content and "
        + "reset all the settings to the default values"
    )
    plotSettSaveImage = "Save the image to file"
    plotSettSaveImageToolTip = "Save the current image in a file"
    plotSettTabTitleAxes = "Axes settings"
    plotSettTabTitleFigure = "Figure settings"
    plotSettTabTitleLegend = "Legend settings"
    plotSettTight = "Use automatic tight layout?"
    plotSettTightToolTip = (
        "Select this checkbox if you want to use the 'matplotlib.pyplot.tight_layout()' "
        + "command to optimize the white spaces around the plot"
        + " and the position of axis labels and ticks"
    )
    plotSettTitleLab = "Figure title"
    plotSettTitleLabToolTip = (
        "Insert a text that will be used as figure title. "
        + "Latex syntax is allowed using '$...$'"
    )
    plotSettXLab = "Label for X axis"
    plotSettXLabToolTip = (
        "Insert a label for the x axis. Latex syntax is allowed using '$...$'"
    )
    plotSettXLims = "X limits"
    plotSettXLimsToolTipLow = "Lower limit for the x coordinate"
    plotSettXLimsToolTipUpp = "Upper limit for the x coordinate"
    plotSettXScale = "X scale"
    plotSettXScaleToolTip = "Select 'linear' or 'logarithmic' scale for the y axis"
    plotSettYLab = "Label for Y axis"
    plotSettYLabToolTip = (
        "Insert a label for the y axis. Latex syntax is allowed using '$...$'"
    )
    plotSettYLims = "Y limits"
    plotSettYLimsToolTipLow = "Lower limit for the y coordinate"
    plotSettYLimsToolTipUpp = "Upper limit for the y coordinate"
    plotSettYScale = "Y scale"
    plotSettYScaleToolTip = "Select 'linear' or 'logarithmic' scale for the y axis"
    plotSettZLab = "Label for Z axis (if used)"
    plotSettZLabToolTip = (
        "Insert a label for the z axis (colorbar). "
        + "Latex syntax is allowed using '$...$'"
    )
    plotTypeDescription = OrderedDict(
        [
            ("evolution", ["Evolution", "Time evolution of nuclide abundances"]),
            (
                "1Ddependence",
                ["1D dependence", "Abundances as a function of one physical parameter"],
            ),
            (
                "2Ddependence",
                [
                    "2D dependence",
                    "Abundances as a function of two physical parameters",
                ],
            ),
        ]
    )
    plotWhereToSave = "Where do you want to save the plot?"
    plotWhereToSaveScript = (
        "Where do you want to save the script to reproduce the current plot?"
    )

    reactionColumnHeaders = ["Reaction", "Type", "Factor", "Value", "Edit"]
    reactionColumnHeaderToolTips = [
        "Description of the reaction",
        "Type of the reaction",
        "Currently selected type for the factor to use for the reaction rate",
        "Value of the current factor to use for the reaction rate",
        "Double click to edit the values to use for this reaction",
    ]
    reactionEditToolTip = "Double click to edit the values to use for this reaction"

    resumeGridAskDelete = (
        "Are you sure you want to delete this grid or single point?"
        + "\nYou will not be able to undo your action."
    )
    resumeGridDeleteToolTip = "Delete this set of parameters"
    resumeGridEditToolTip = "Edit this set of parameters"
    resumeGridHeaders = ["Parameter", "N", "Points"]
    resumeGridHeadersToolTip = [
        "Parameter name",
        "Number of points under consideration for the parameter",
        "List of the points that will be used for the parameter",
    ]

    runCompletedErrorFile = "\nPlease check also the file %s for more information"
    runCompletedErrorMessage = "\n\nError message:\n"
    runCompletedFailing = "Run failed with the following error:"
    runCompletedFailingCheck = "Please, check the info file"
    runCompletedFailingManual = "Please, consult the manual."
    runCompletedSuccessfully = "Run completed successfully"
    runFailed = "Run %d did not end properly! see %s"

    runnerAskStop = "Are you sure you want to stop the runs?"
    runnerDone = "Done!"
    runnerFinished = "Finished!"
    runnerParthenopeNotFound = (
        "Impossible to find the parthenope executable,"
        + " cannot proceed.\nCompile it and try again."
    )
    runnerRunning = "Currently running..."
    runnerStopped = "Stopped!"

    runPanelCustomTitle = "Run with custom parameters"
    runPanelCustomToolTip = (
        "Submit a run using the customized "
        + "grid points, nuclides network and reactions information "
        + "as configured in the current view"
    )
    runPanelDefaultTitle = "Run with default parameters"
    runPanelDefaultToolTip = (
        "Submit a run using the default settings for "
        + "grid points, nuclides network and reactions information"
    )
    runPanelDescription = (
        "When you have selected all the settings"
        + " you can start the PArthENoPE runs"
        + " using one of the following buttons:"
    )
    runPanelProgressBarToolTip = "Current progress of the runs"
    runPanelSaveCustomTitle = "Save custom parameters"
    runPanelSaveCustomToolTip = (
        "Save all the choices configured in this page in the file"
        + " 'settings.obj' in the output directory for running it later"
        + " (launch via command line with '$ ./gui3.0.py %s/')"
    )
    runPanelSaveDefaultTitle = "Save default parameters"
    runPanelSaveDefaultToolTip = (
        "Save all the default choices in the file"
        + " 'settings.obj' in the output directory for running it later"
        + " (launch via command line with '$ ./gui3.0.py %s/')"
    )
    runPanelSpaceInFolderName = (
        "The selected folder name contains a space."
        + " This may break the fortran code during execution."
        + " Please, choose another directory."
    )
    runPanelStopTitle = "Stop run"
    runPanelStopToolTip = "Stop the runs of the current grid"

    runSingleArgsType = "Wrong argument type in runSingle (it must be [int, str, str])"

    selectGridToolTip = "Click here to select a grid to consider for the plots"
    standardDeviation = "Standard deviation:"
    startRun = "Starting run..."
    startRunI = "Starting run %s..."

    tryToOpenFolder = "Try to open a settings.obj file from existing folder: %s"

    warningLargeValue = "The value for %s (%s) is a bit large, is it not?"
    warningMaxEmptyList = "Cannot get max. Empty gridsList?"
    warningMissingLine = "The requested line does not exist!"
    warningRunningProcesses = (
        "There may a running process.\nDo you really want to exit?"
    )
    warningWrongType = "Wrong type for %s. Is it %s?"
