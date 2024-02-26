import os
from collections import OrderedDict

import matplotlib
import numpy as np

matplotlib.use("Qt5Agg")
os.environ["QT_API"] = "pyside2"

# see https://matplotlib.org/tutorials/colors/colormaps.html
cmaps = OrderedDict()
cmaps["Perceptually Uniform Sequential"] = [
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
]
cmaps["Sequential"] = [
    "Greys",
    "Purples",
    "Blues",
    "Greens",
    "Oranges",
    "Reds",
    "YlOrBr",
    "YlOrRd",
    "OrRd",
    "PuRd",
    "RdPu",
    "BuPu",
    "GnBu",
    "PuBu",
    "YlGnBu",
    "PuBuGn",
    "BuGn",
    "YlGn",
]
cmaps["Sequential (2)"] = [
    "binary",
    "gist_yarg",
    "gist_gray",
    "gray",
    "bone",
    "pink",
    "spring",
    "summer",
    "autumn",
    "winter",
    "cool",
    "Wistia",
    "hot",
    "afmhot",
    "gist_heat",
    "copper",
]
cmaps["Diverging"] = [
    "PiYG",
    "PRGn",
    "BrBG",
    "PuOr",
    "RdGy",
    "RdBu",
    "RdYlBu",
    "RdYlGn",
    "Spectral",
    "coolwarm",
    "bwr",
    "seismic",
]
cmaps["Cyclic"] = ["twilight", "twilight_shifted", "hsv"]
cmaps["Qualitative"] = [
    "Pastel1",
    "Pastel2",
    "Paired",
    "Accent",
    "Dark2",
    "Set1",
    "Set2",
    "Set3",
    "tab10",
    "tab20",
    "tab20b",
    "tab20c",
]
cmaps["Miscellaneous"] = [
    "flag",
    "prism",
    "ocean",
    "gist_earth",
    "terrain",
    "gist_stern",
    "gnuplot",
    "gnuplot2",
    "CMRmap",
    "cubehelix",
    "brg",
    "gist_rainbow",
    "rainbow",
    "jet",
    "nipy_spectral",
    "gist_ncar",
]
defaultCmap = "CMRmap"
for c in cmaps.keys():
    new = []
    for m in cmaps[c]:
        try:
            if isinstance(matplotlib.cm.get_cmap(m), matplotlib.colors.Colormap):
                new.append(m)
        except ValueError:
            pass
    cmaps[c] = new

extendOptions = ["neither", "both", "min", "max"]
markerOptions = [
    "",
    ".",
    ",",
    "o",
    "v",
    "^",
    "<",
    ">",
    "1",
    "2",
    "3",
    "4",
    "8",
    "s",
    "p",
    "P",
    "*",
    "h",
    "H",
    "+",
    "x",
    "X",
    "D",
    "d",
    "|",
    "_",
]
styleOptions = ["-", "--", "-.", ":"]

chi2labels = {
    "chi2": r"$\chi^2$",
    "lh": r"$\mathcal{L}$",
    "nllh": r"$-\log \mathcal{L}$",
    "pllh": r"$\log \mathcal{L}$",
}


def chi2func(v, mean, stddev):
    """Compute a Gaussian chi2 from the given number or vector

    Parameter:
        v: a float or a np.array
        mean: a float or a np.array, the mean for the chi2 function
        stddev: a float or a np.array, the standard deviation

    Output:
        a float or a np.array, depending on the input
    """
    if isinstance(v, list):
        v = np.asarray(v)
    return ((v - mean) / stddev) ** 2


######### functions to read output from previous grid, select type of plots, variables, ...
class PGPlotObject(object):
    """Class that contains useful common functions
    for the PGLine and PGContour classes
    """

    def getYlabel(self):
        """return the ylabel of the object.
        It may return the saved ylabel or a chi2labels entry,
        depending on whether self.z exists and the value of self.chi2
        """
        if hasattr(self, "z"):
            return self.ylabel
        return chi2labels[self.chi2] if self.chi2 else self.ylabel

    def getZlabel(self):
        """return the zlabel of the object, if it has one"""
        if hasattr(self, "z") and hasattr(self, "zlabel"):
            return chi2labels[self.chi2] if self.chi2 else self.zlabel
        return ""


class PGLine(PGPlotObject):
    """Contains basic properties of lines that will be inserted in plots"""

    def __init__(
        self,
        xvec,
        yvec,
        ptr,
        c="k",
        c2=False,
        d="",
        l="",
        m="",
        s="-",
        w=1.0,
        xl="",
        yl="",
    ):
        """Save the properties of the line object

        Parameters:
            xvec: a np.array with the x points to use in the plot
            yvec: a np.array with the y points to use in the plot
            ptr: the index of the point in the model dataList
            c (default "k"): the color for the line
            c2 (default False): if True, convert the y points
                using a Gaussian chi2
            d (default ""): a description for the line
            l (default ""): the legend label for the line
            m (default ""): a marker for the points, if required
            s (default "-"): the line style
            w (default 1): the line width
            xl (default ""): the label for the x axis
            yl (default ""): the label for the y axis
        """
        self.x = xvec
        self.y = yvec
        self.pointRow = ptr
        self.chi2 = c2
        self.color = c
        self.description = d
        self.label = l
        self.marker = m
        self.style = s
        self.width = w
        self.xlabel = xl
        self.ylabel = yl


class PGContour(PGPlotObject):
    """Contains basic properties of contours that will be inserted in plots"""

    def __init__(
        self,
        xvec,
        yvec,
        zmat,
        ptr,
        c2=False,
        cm=defaultCmap,
        d="",
        ex=None,
        f=True,
        hcb=True,
        l="",
        lvs=None,
        xl="",
        yl="",
        zl="",
    ):
        """Save the properties of the object

        Parameters:
            xvec: a np.array with the x points to use in the plot
            yvec: a np.array with the y points to use in the plot
            zmat: a (2D) np.array with the z values to use in the plot
            ptr: the index of the point in the model dataList
            c2 (default False): if True, convert the z points
                using a Gaussian chi2 function
            cm (default `defaultCmap`): the name of the colormap to use
            d (default ""): a description for the contour
            ex (default None): if needed, the extension for the colorbar
                (it can be "neither", "min", "max" or "both")
            f (default True): if True, use filled contours
                (plt.contourf instead of plt.contour)
            hcb (default True): if False, do not add a colorbar to the plot
            l (default ""): the legend label for the contour
            lvs (default None): a list with the levels to use
                in the contour plot
            xl (default ""): the label for the x axis
            yl (default ""): the label for the y axis
            zl (default ""): the label for the z axis (colorbar)
        """
        self.x = xvec
        self.y = yvec
        self.z = zmat
        self.pointRow = ptr
        self.chi2 = c2
        self.cmap = cm
        self.description = d
        self.extend = ex
        self.filled = f
        self.hascbar = hcb
        self.label = l
        self.levels = lvs
        self.xlabel = xl
        self.ylabel = yl
        self.zlabel = zl
