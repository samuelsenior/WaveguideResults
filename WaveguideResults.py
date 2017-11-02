"""
@author: Samuel Senior
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.mlab import griddata
from scipy.integrate import simps
from scipy.integrate import odeint


class ReadInData():
    """
    Reads in the data and header from a file assuming it is a text file
    exported from Comsol.
    """
    def __init__(self, file, kwargs_get_raw_data={}, print_details='Minimal'):
        """
        Initialises the ReadInData class instance. __init__ creates
        instances of internal variables such as __file__, __header__
        and __raw_data__, and then reads in the raw data from the file.

        Parameters:
        file : (string) The file name to be read in.

        **kwargs : (options, dict) Additional arguments that can be
                   passed in, such as a bool to specify whether to
                   ignore the header or not and a bool to specify
                   the data file contains complex data types.

        Returns:
        (none) : Does not return any variables.
        """
        self.__kwargs_get_raw_data__ = kwargs_get_raw_data
        self.__print_details__ = print_details
        self.__file__ = file
        self.__header__ = {'Model': '',
                           'Version': '',
                           'Date': '',
                           'Dimension': '',
                           'Nodes': '',
                           'Expressions': '',
                           'Description': '',
                           'Length unit': '',
                           'Data Structure': ''}
        self.__raw_data__ = None

        self.get_raw_data(**self.__kwargs_get_raw_data__)

    def get_number_of_lines(self):
        """
        Determines the number of lines in the data file, as well as the
        number of header lines, the number of entries per line, and the
        number of data lines.

        Parameters:
        (none) : Does not take any input parameters.

        Returns:
        (none) : Does not return any variables.
        """
        with open(self.__file__) as input_file:
            self.n_lines = 0
            self.n_data_per_line = 0
            self.n_header_lines = 0
            header_check = True
            for line in input_file:
                line = line.strip()
                self.n_data_per_line = 0
                if (line[0] == '%' and header_check is True):
                    self.n_header_lines += 1
                else:
                    header_check = False
                for number in line.split():
                    self.n_data_per_line += 1
                self.n_lines += 1
        if self.__print_details__ is True:
            print("File: {}".format(self.__file__))
            print("      n_lines: {}".format(self.n_lines))
            print("      n_data_per_line: {}".format(self.n_data_per_line))
            print("      n_header_lines: {}".format(self.n_header_lines))
            print("      (n_data_lines: {} (inferred))\n".format(self.n_lines - self.n_header_lines))

    def get_raw_data(self, ignore_header=False, complexData=False):
        """
        Reads in the raw data from the data file. It firstly loops over the
        header and either ignores it or sets the values of the header dict,
        and then loops over the remaining lines and sets the __raw_data__
        varaible to the values on each line.

        If the ignore_header flag is set to False then the values of the header
        dict are set to their respective entry from the data file.

        If the the complexData flag is True then the __raw_data__ variable is
        initialised as a np.complex128 type rather than np.float64.

        Parameters:
        ignore_header : (optional, bool) The flag to specify whether to ignore
                        the header or not. By default this is True and the
                        header is ignored.

        complexData : (optional, bool) The flag to specifiy if complex data
                      types are being read in. By default this is False and the
                      elements of __raw_data__ are set as np.float64.

        Returns:
        (none) : Does not return any variables.
        """
        self.get_number_of_lines()
        with open(self.__file__) as input_file:
            if complexData is False:
                self.__raw_data__ = np.zeros([self.n_data_per_line,
                                              self.n_lines - self.n_header_lines])
            elif complexData is True:
                self.__raw_data__ = np.zeros([self.n_data_per_line,
                                              self.n_lines - self.n_header_lines], dtype=np.complex128)
            i = 0
            for line in input_file:
                if (i < self.n_header_lines):
                    if ignore_header is True and self.__print_details__ is True:
                        print("get_data(): Skipping header line {} ...".format(i))
                    else:
                        line = line.split()
                        line.pop(0)
                        if line[0] == 'Model:':
                            if self.__print_details__ is True:
                                print("get_data(): In header, Model value found")
                            self.__header__['Model'] = ' '.join(line[1:])
                        elif line[0] == 'Version:':
                            if self.__print_details__ is True:
                                print("get_data(): In header, Version value found")
                            self.__header__['Version'] = ' '.join(line[1:])
                        elif line[0] == 'Date:':
                            if self.__print_details__ is True:
                                print("get_data(): In header, Date value found")
                            self.__header__['Date'] = ' '.join(line[1:])
                        elif line[0] == 'Dimension:':
                            if self.__print_details__ is True:
                                print("get_data(): In header, Dimension value found")
                            self.__header__['Dimension'] = ' '.join(line[1:])
                        elif line[0] == 'Nodes:':
                            if self.__print_details__ is True:
                                print("get_data(): In header, Nodes value found")
                            self.__header__['Nodes'] = ' '.join(line[1:])
                        elif line[0] == 'Expressions:':
                            if self.__print_details__ is True:
                                print("get_data(): In header, Expressions value found")
                            self.__header__['Expressions'] = ' '.join(line[1:])
                        elif line[0] == 'Description:':
                            if self.__print_details__ is True:
                                print("get_data(): In header, Description value found")
                            self.__header__['Description'] = ' '.join(line[1:])
                        elif line[0] == 'Length' and line[1] == 'unit:':
                            if self.__print_details__ is True:
                                print("get_data(): In header, Length Unit value found")
                            self.__header__['Length unit'] = ' '.join(line[2:])
                        elif i == self.n_header_lines - 1:
                            if self.__print_details__ is True:
                                print("get_data(): In header, Data Structure value assumed as last line of header")
                            tmp = ' '.join(line[2:]).split(" @ ")
                            tmp2 = [tmp[0]]
                            tmp.pop(0)
                            for j in range(len(tmp)):
                                tmp2 += [tmp[j].replace(tmp2[0], "").strip()]
                            tmp3 = [tmp2[0], tmp2[1:]]
                            self.__header__['Data Structure'] = [line[0]] + [line[1]] + tmp3
                else:
                    line = line.strip()
                    j = 0
                    for number in line.split():
                        if complexData is False:
                            self.__raw_data__[j, i - self.n_header_lines] = number
                        elif complexData is True:
                            self.__raw_data__[j, i - self.n_header_lines] = complex(number.replace('i', 'j'))
                        j += 1
                i += 1
            if self.__print_details__ == 'Minimal':
                print("get_data(): Read in data of shape {} from file '{}'".format(np.shape(self.__raw_data__), self.__file__))

    def header_details(self):
        """
        Prints the header dict keys and values.
        Parameters:
        (none) : Does not take any input parameters.

        Returns:
        (none) : Does not return any variables.
        """
        print("\n{} header details:".format(self.header()['Description']))
        for key, value in self.header().items():
            if key != 'Data Structure':
                print ("      {}: {}".format(key, value))
        print ("Data Structure: {}\n".format(self.header()['Data Structure']))

    def raw_data(self):
        """
        Function to return the encapsulated variable __raw_data__.

        Parameters:
        (none) : Does not take any input parameters.

        Returns:
        self.__raw_data__ : (numpy array) The raw data.
        """
        return self.__raw_data__

    def file(self):
        """
        Function to return the encapsulated variable __file__.

        Parameters:
        (none) : Does not take any input parameters.

        Returns:
        self.__raw_data__ : (numpy array) The file name.
        """
        return self.__file__

    def header(self):
        """
        Function to return the encapsulated variable __header__.

        Parameters:
        (none) : Does not take any input parameters.

        Returns:
        self.__raw_data__ : (dict) The header.
        """
        return self.__header__


class Plot():
    """
    Interfaces to and plots various pyplotlib plots for the given data.
    """
    def __init__(self, X, Y):
        """
        Initialises the Plot class instance. __init__ creates
        instances of internal variables such as __X__, __Y__, xlim,
        and ylim.

        Parameters:
        X : (array of np.float64) The grided values of the X values.

        Y : (array of np.float64) The grided values of the Y values.

        Returns:
        (none) : Does not return any variables.
        """
        self.__X__ = X
        self.__Y__ = Y
        self.__xlim__ = [np.min(self.__X__), np.max(self.__X__)]
        self.__ylim__ = [np.min(self.__Y__), np.max(self.__Y__)]

    def pcolor(self, Z, n_plots, cmap="hot",
               sub_title=None, title=None, axis_titles=None,
               xlim=None, ylim=None,
               figsize=(12, 6), dpi=80, tight_layout=True):
        """
        Interfaces to and plots the pyplotlib.pcolor plot.

        Parameters:
        Z : (array of np.float64) The grided values of the data points.

        n_plots : (int) The number of plots to make.

        cmap : (optional, string) The colour map.

        sub_title : (optional, list of string) The subtitles for each
                    individual plot. The shape taken is, for the example of two
                    plots:
                    sub_title = [title_1, title_2]

        title : (optional, string) The title of the overall plots

        axis_titles : (optional, list of list of string) The axis titles for
                      each plot. The shape taken is, for the example of the
                      axis titles for two plots is:
                      axis_titles = [[x_axis_title_1, y_axis_title_1],
                                     [x_axis_title_2, y_axis_title_2]]

        xlim : (optional, list of list of float) The x limits for each plot.
               The shape taken is, for the example of the axis titles for two
               plots is:
               x_lim = [[xlim_min_1, xlim_max_1],
                        [xlim_min_2, xlim_max_2]]

        ylim : (optional, list of list of float) The y limits for each plot.
               The shape taken is, for the example of the axis titles for two
               plots is:
               y_lim = [[ylim_min_1, ylim_max_1],
                        [ylim_min_2, ylim_max_2]]

        figsize : (optional, tuple of float) The figure size of the overall
                  plot. The shape taken is:
                  figsize = (x, y)

        dpi : (optional, float) The dpi of the overall plot.

        tight_layout : (optional, bool) A switch to enable tight_layout.

        Returns:
        (none) : Does not return any variables.
        """
        if sub_title is None:
            if n_plots == 1:
                sub_title = ""
            elif n_plots > 1:
                sub_title = [""]*n_plots

        if title is None:
            title = ""

        if axis_titles is None:
            if n_plots == 1:
                axis_titles = ["", ""]
            elif n_plots > 1:
                axis_titles = [["", ""]]*n_plots

        if xlim is None:
            if n_plots == 1:
                xlim = self.__xlim__
            elif n_plots > 1:
                xlim = [self.__xlim__]*n_plots
        if ylim is None:
            if n_plots == 1:
                ylim = self.__ylim__
            elif n_plots > 1:
                ylim = [self.__ylim__]*n_plots

        fig = plt.figure(figsize=figsize, dpi=dpi)
        fig.suptitle(title)
        if n_plots == 1:
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.pcolor(self.__X__, self.__Y__, Z, cmap=cmap)
            ax1.set_xlim([xlim[0], xlim[1]])
            ax1.set_ylim([ylim[0], ylim[1]])
            ax1.set_xlabel(axis_titles[0])
            ax1.set_ylabel(axis_titles[1])
            ax1.set_title(sub_title)
        else:
            for i in range(n_plots):
                ax1 = fig.add_subplot(1, n_plots, i+1)
                ax1.pcolor(self.__X__, self.__Y__, Z[i], cmap=cmap)
                ax1.set_xlim([xlim[i][0], xlim[i][1]])
                ax1.set_ylim([ylim[i][0], ylim[i][1]])
                ax1.set_xlabel(axis_titles[i][0])
                ax1.set_ylabel(axis_titles[i][1])
                ax1.set_title(sub_title[i])
        if tight_layout is True:
            fig.tight_layout()
        plt.show()

    def contourf(self, Z, n_plots, n_lines=10,
                 sub_title=None, title=None, axis_titles=None,
                 xlim=None, ylim=None,
                 figsize=(12, 6), dpi=80, tight_layout=True):
        """
        Interfaces to and plots the pyplotlib.contourf plot.

        Parameters:
        Z : (array of np.float64) The grided values of the data points.

        n_plots : (int) The number of plots to make.

        n_lines : (int) The number of lines in the contour plot.

        sub_title : (optional, list of string) The subtitles for each
                    individual plot. The shape taken is, for the example of two
                    plots:
                    sub_title = [title_1, title_2]

        title : (optional, string) The title of the overall plots

        axis_titles : (optional, list of list of string) The axis titles for
                      each plot. The shape taken is, for the example of the
                      axis titles for two plots is:
                      axis_titles = [[x_axis_title_1, y_axis_title_1],
                                     [x_axis_title_2, y_axis_title_2]]

        xlim : (optional, list of list of float) The x limits for each plot.
               The shape taken is, for the example of the axis titles for two
               plots is:
               x_lim = [[xlim_min_1, xlim_max_1],
                        [xlim_min_2, xlim_max_2]]

        ylim : (optional, list of list of float) The y limits for each plot.
               The shape taken is, for the example of the axis titles for two
               plots is:
               y_lim = [[ylim_min_1, ylim_max_1],
                        [ylim_min_2, ylim_max_2]]

        figsize : (optional, tuple of float) The figure size of the overall
                  plot.
                  The shape taken is:
                  figsize = (x, y)

        dpi : (optional, float) The dpi of the overall plot.

        tight_layout : (optional, bool) A switch to enable tight_layout.

        Returns:
        (none) : Does not return any variables.
        """
        if sub_title is None:
            if n_plots == 1:
                sub_title = ""
            elif n_plots > 1:
                sub_title = [""]*n_plots

        if title is None:
            title = ""

        if axis_titles is None:
            if n_plots == 1:
                axis_titles = ["", ""]
            elif n_plots > 1:
                axis_titles = [["", ""]]*n_plots

        if xlim is None:
            if n_plots == 1:
                xlim = self.__xlim__
            elif n_plots > 1:
                xlim = [self.__xlim__]*n_plots
        if ylim is None:
            if n_plots == 1:
                ylim = self.__ylim__
            elif n_plots > 1:
                ylim = [self.__ylim__]*n_plots

        fig = plt.figure(figsize=figsize, dpi=dpi)
        fig.suptitle(title)
        if n_plots == 1:
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.contourf(self.__X__, self.__Y__, Z, n_lines)
            ax1.set_xlim([xlim[0], xlim[1]])
            ax1.set_ylim([ylim[0], ylim[1]])
            ax1.set_xlabel(axis_titles[0])
            ax1.set_ylabel(axis_titles[1])
            ax1.set_title(sub_title)
        else:
            for i in range(n_plots):
                ax1 = fig.add_subplot(1, n_plots, i+1)
                ax1.contourf(self.__X__, self.__Y__, Z[i], n_lines)
                ax1.set_xlim([xlim[i][0], xlim[i][1]])
                ax1.set_ylim([ylim[i][0], ylim[i][1]])
                ax1.set_xlabel(axis_titles[i][0])
                ax1.set_ylabel(axis_titles[i][1])
                ax1.set_title(sub_title[i])
        if tight_layout is True:
            fig.tight_layout()
        plt.show()

    def plot(self, X, Y, n_plots,
             sub_title=None, title=None, axis_titles=None,
             xlim=None, ylim=None,
             figsize=(12, 6), dpi=80, tight_layout=True):
        """
        Interfaces to and plots the pyplotlib.plot plot.

        Parameters:
        X : (array of np.float64) The X values.

        Y : (array of np.float64) The Y values.

        n_plots : (int) The number of plots to make.

        sub_title : (optional, list of string) The subtitles for each
                    individual plot. The shape taken is, for the example of two
                    plots:
                    sub_title = [title_1, title_2]

        title : (optional, string) The title of the overall plots

        axis_titles : (optional, list of list of string) The axis titles for
                      each plot. The shape taken is, for the example of the
                      axis titles for two plots is:
                      axis_titles = [[x_axis_title_1, y_axis_title_1],
                                     [x_axis_title_2, y_axis_title_2]]

        xlim : (optional, list of list of float) The x limits for each plot.
               The shape taken is, for the example of the axis titles for two
               plots is:
               x_lim = [[xlim_min_1, xlim_max_1],
                        [xlim_min_2, xlim_max_2]]

        ylim : (optional, list of list of float) The y limits for each plot.
               The shape taken is, for the example of the axis titles for two
               plots is:
               y_lim = [[ylim_min_1, ylim_max_1],
                        [ylim_min_2, ylim_max_2]]

        figsize : (optional, tuple of float) The figure size of the overall
                  plot. The shape taken is:
                  figsize = (x, y)

        dpi : (optional, float) The dpi of the overall plot.

        tight_layout : (optional, bool) A switch to enable tight_layout.

        Returns:
        (none) : Does not return any variables.
        """
        if sub_title is None:
            if n_plots == 1:
                sub_title = ""
            elif n_plots > 1:
                sub_title = [""]*n_plots

        if title is None:
            title = ""

        if axis_titles is None:
            if n_plots == 1:
                axis_titles = ["", ""]
            elif n_plots > 1:
                axis_titles = [["", ""]]*n_plots

        if xlim is None:
            if n_plots == 1:
                xlim = [np.min(X), np.max(X)]
            elif n_plots > 1:
                xlim = [[np.min(X), np.max(X)]]*n_plots
        if ylim is None:
            if n_plots == 1:
                ylim = [np.min(Y), np.max(Y)]
            elif n_plots > 1:
                ylim = [[np.min(Y), np.max(Y)]]*n_plots

        fig = plt.figure(figsize=figsize, dpi=dpi)
        fig.suptitle(title)
        if n_plots == 1:
            ax1 = fig.add_subplot(1, 1, 1)
            if np.ndim(Y) == 2:
                for j in range(np.shape(Y)[0]):
                    ax1.plot(X, Y[j])
            else:
                ax1.plot(X, Y)
            ax1.set_xlim([xlim[0], xlim[1]])
            ax1.set_ylim([ylim[0], ylim[1]])
            ax1.set_xlabel(axis_titles[0])
            ax1.set_ylabel(axis_titles[1])
            ax1.set_title(sub_title)
        else:
            for i in range(n_plots):
                ax1 = fig.add_subplot(1, n_plots, i+1)
                ax1.plot(X, Y[i])
                ax1.set_xlim([xlim[i][0], xlim[i][1]])
                ax1.set_ylim([ylim[i][0], ylim[i][1]])
                ax1.set_xlabel(axis_titles[i][0])
                ax1.set_ylabel(axis_titles[i][1])
                ax1.set_title(sub_title[i])
        if tight_layout is True:
            fig.tight_layout()
        plt.show()


class Data(ReadInData, Plot):
    """
    Formats the data passed in from the ReadInData class, grids the data,
    as well X and Y arrays.

    When reading in the data, it is assumed that there is an even number of
    modes present, half of which are for the pump wavelength and the other
    half for the signal wavelength.

    Base Class:
    ReadInData : (class) The ReadInData class.

    Plot : (class) The Plot class.
    """
    def __init__(self, file,
                 kwargs_Data={}, kwargs_ReadInData={}, kwargs_grid={}):
        """
        Initialises the Data class instance. __init__ creates instances of
        internal variables such as kwargs_ReadInData, kwargs_grid and __Data__,
        and then initialises the ReadInData class and the Plot class.

        Parameters:
        file : (string) The file name to be read in.

        kwargs : (optional, dict) Additional arguments that can be passed in,
                 currently unused.

        kwargs_ReadInData : (optional, dict) Additional arguments to be passed
                            in to the ReadInData class.

        kwargs_grid : (optional, dict) Additional arguments to be passed in to
                      the grid function.

        Returns:
        (none) : Does not return any variables.
        """
        # self.kwargs_ReadInData = kwargs_ReadInData
        self.__kwargs_Data__ = kwargs_Data
        self.__kwargs_ReadInData__ = kwargs_ReadInData
        self.__kwargs_grid__ = kwargs_grid
        self.__x__ = None
        self.__y__ = None
        self.__data__ = None
        self.__X__ = None
        self.__Y__ = None
        self.__Data__ = None

        ReadInData.__init__(self, file, self.__kwargs_ReadInData__)
        self.format_raw_data()
        self.grid(**self.__kwargs_grid__)
        self.format_data()

        Plot.__init__(self, self.X(), self.Y())

    def format_raw_data(self):
        """
        Formats the raw data into the seperate x, y, data variables.

        Parameters:
        (none) : Does not take any input parameters.

        Returns:
        (none) : Does not return any variables.
        """
        self.__x__ = self.__raw_data__[0, :]
        self.__y__ = self.__raw_data__[1, :]
        self.__data__ = self.__raw_data__[2:, :]

    def format_data(self):
        """
        Formats the data into the X, Y, Data variables.

        Parameters:
        (none) : Does not take any input parameters.

        Returns:
        (none) : Does not return any variables.
        """
        self.__X__ = np.nan_to_num(self.__X__)
        self.__Y__ = np.nan_to_num(self.__Y__)
        self.__Data__ = np.nan_to_num(self.__Data__)

    def grid(self, res=1, resX=None, resY=None):
        """
        Convert 3 column data to matplotlib grid.

        Use either res, or resX and resY, as using just res causing resX and resY
        to be calculated from res.

        Parameters:
        res : (optional, int) The number of grid divisions per unit length.

        resX : (optional, int) The number of grid divisions in the X direction.

        resY : (optional, int) The number of grid divisions in the Y direction.

        Returns:
        (none) : Does not return any variables.
        """
        if 'complexData' in self.__kwargs_Data__ and self.__kwargs_Data__['complexData'] == True:
            print("grid(): Warning, complex data type present, unable to grid data...\n")
            self.__X__ = self.__x__
            self.__Y__ = self.__y__
            self.__Data__ = self.__data__
        else:
            if resX == None or resY == None:
                resX = (np.max(self.__x__) - np.min(self.__x__))*res
                resY = (np.max(self.__y__) - np.min(self.__y__))*res
            n_modes = np.shape(self.__data__)[0]
            xi = np.linspace(np.min(self.__x__), np.max(self.__x__), resX)
            yi = np.linspace(np.min(self.__y__), np.max(self.__y__), resY)
            if n_modes == 1:
                self.__Data__ = griddata(self.__x__, self.__y__, self.__data__, xi, yi, interp='linear')
            else:
                self.__Data__ = np.zeros([n_modes, int(resY), int(resX)])
                for i in range(n_modes):
                    self.__Data__[i] = griddata(self.__x__, self.__y__, self.__data__[i], xi, yi, interp='linear')
            self.__X__, self.__Y__ = np.meshgrid(xi, yi)

    def data_details(self):
        """
        Prints the x, y, data, X, Y, Data details to the screen.

        Parameters:
        (none) : Does not take any input parameters.

        Returns:
        (none) : Does not return any variables.
        """
        print("\n{} data details:".format(self.header()['Description']))
        print ("      x: {}".format(np.shape(self.__x__)))
        print ("      y: {}".format(np.shape(self.__y__)))
        print ("      data: {}".format(np.shape(self.__data__)))
        print ("      X: {}".format(np.shape(self.__X__)))
        print ("      Y: {}".format(np.shape(self.__Y__)))
        print ("      Data: {}\n".format(np.shape(self.__Data__)))

    def x(self):
        """
        Function to return the encapsulated variable __x__.

        Parameters:
        (none) : Does not take any input parameters.

        Returns:
        self.__x__ : (numpy array) The x variable.
        """
        return self.__x__

    def y(self):
        """
        Function to return the encapsulated variable __y__.

        Parameters:
        (none) : Does not take any input parameters.

        Returns:
        self.__y__ : (numpy array) The y variable.
        """
        return self.__y__

    def data(self):
        """
        Function to return the encapsulated variable __data__.

        Parameters:
        (none) : Does not take any input parameters.

        Returns:
        self.__data__ : (numpy array) The data variable.
        """
        return self.__data__

    def X(self):
        """
        Function to return the encapsulated variable __X__.

        Parameters:
        (none) : Does not take any input parameters.

        Returns:
        self.__X__ : (numpy array) The X variable.
        """
        return self.__X__

    def Y(self):
        """
        Function to return the encapsulated variable __Y__.

        Parameters:
        (none) : Does not take any input parameters.

        Returns:
        self.__Y__ : (numpy array) The Y variable.
        """
        return self.__Y__

    def Data(self):
        """
        Function to return the encapsulated variable __Data__.

        Parameters:
        (none) : Does not take any input parameters.

        Returns:
        self.__Data__ : (numpy array) The Data variable.
        """
        return self.__Data__


class NonlinearOptics():

    def __init__(self, X, Y, E_w, E_2w, effective_mode_index_w, effective_mode_index_2w, fundamental_mode_index=0, lamda=1550.0E-9, T_Op=20.0, solve=True):
        self.__X__ = X
        self.__Y__ = Y
        self.__E_w__ = E_w    # Fundamental mode of source
        if np.ndim(E_2w) == 1:
            self.__E_2w__ = [E_2w, ]  # Single mode of the second harmonic
        else:
            self.__E_2w__ = E_2w  # Multiple modes of the second harmonic
        self.__effective_mode_index_w__ = effective_mode_index_w


        if np.ndim(effective_mode_index_2w) == 1:
            self.__effective_mode_index_2w__ = [effective_mode_index_2w, ]
        else:
            self.__effective_mode_index_2w__ = effective_mode_index_2w
        self.__effective_mode_index_2w__ = effective_mode_index_2w

        self.__fundamental_mode_index__ = fundamental_mode_index


        self.__lamda_w__ = lamda
        self.__lamda_2w__ = self.__lamda_w__ / 2.0

        self.__T_Op__ = T_Op


        self.__c__ = 299792458               # Speed of light
        self.__mu_0__ = 4*np.pi*1.0E-7       # Permeability of free space
        self.__epsilon_0__ = 8.85418782e-12  # Permitivity of free space
        self.__omega_w__ = 2*np.pi * self.__c__ / self.__lamda_w__
        self.__omega_2w__ = 2*np.pi * self.__c__ / self.__lamda_2w__
        self.__d_eff__ = 14.0E-12             # deff

        # Default values for solving in the z direction of the waveguide
        self.__z_max__ = 50E-3
        self.__z_points__ = 500
        self.__z__ = np.arange(0, self.__z_max__, self.__z_max__/self.__z_points__)
        self.__ODE_rtol__ = 1.0E-4

        self.__amplitude_solutions__ = None
        self.__conversion_efficiency__ = None

        self.__propagation_constant_w__ = None
        self.__normalisation_factor_w__ = None
        self.__propagation_constant_2w__ = None
        self.__normalisation_factor_2w__ = None
        self.__E_w_norm__ = None
        self.__E_2w_norm__ = None
        self.__nonlinear_coupling_coefficient__ = None

        self.__effective_nonlinear_coefficient__ = None
        self.__delta_k__ = (4 * np.pi / self.__lamda_w__) * (self.__sellmeier_equation__(self.__lamda_2w__, self.__T_Op__) - self.__sellmeier_equation__(self.__lamda_w__, self.__T_Op__))
        self.__coherent_length__ = self.__lamda_w__ / (4*(np.real(self.__effective_mode_index_2w__) - np.real(self.__effective_mode_index_w__)))
        self.__inversion_period__ = 2*self.__coherent_length__
        self.__effective_mode_overlap__ = None
        self.__effective_cross_section__ = None

        self.__set_propagation_constant_w__()
        self.__set_propagation_constant_2w__()
        self.__set_normalisation_factor_w__()
        self.__set_normalisation_factor_2w__()
        self.__set_E_w_norm__()
        self.__set_E_2w_norm__()

        self.__set_effective_cross_section__()
        self.__set_nonlinear_coupling_coefficient__()

        if solve is True:
            self.__solve_amplitude_coupled_ODEs__()

        assert not isinstance(self.__propagation_constant_w__, type(None))
        assert not isinstance(self.__normalisation_factor_w__, type(None))
        assert not isinstance(self.__propagation_constant_2w__, type(None))
        assert not isinstance(self.__normalisation_factor_2w__, type(None))

        assert not isinstance(self.__E_w_norm__, type(None)), \
            ("NonLinearOptics.__init__(): __E_w_norm__ of type %r" % (type(self.__E_w_norm__ )))
        assert not isinstance(self.__E_2w_norm__, type(None)), \
            ("NonLinearOptics.__init__(): __E_w_norm__ of type %r" % (type(self.__E_2w_norm__ )))

    def __sellmeier_equation__(self, lam, T):
        """
        Remember the operational temperature, T, is in oC not K here!
        """
        lam = lam*1.0E6
        a1 = 5.756
        a2 = 0.0983
        a3 = 0.2020
        a4 = 189.32
        a5 = 12.52
        a6 = 1.32E-2
        b1 = 2.860E-6
        b2 = 4.700E-8
        b3 = 6.113E-8
        b4 = 1.516E-4

        f = (T - 24.5)*(T + 570.82)

        return np.sqrt(a1 + b1*f + (a2 + b2*f)/(lam**2 - (a3 + b3*f)**2) + (a4 + b4*f)/(lam**2 - a5**2) - a6*lam**2)

    def __set_propagation_constant_w__(self):
        self.__propagation_constant_w__ = self.__effective_mode_index_w__ * 2*np.pi / self.__lamda_w__
        print("__set_propagation_constant_w__(): Propagation constant set.")

    def __set_propagation_constant_2w__(self):
        self.__propagation_constant_2w__ = np.zeros(np.shape(self.__effective_mode_index_2w__)[0], dtype=np.complex128)
        for i in range(len(self.__propagation_constant_2w__)):
            self.__propagation_constant_2w__[i] = self.__effective_mode_index_2w__[i] * 2*np.pi / self.__lamda_2w__
        print("__set_propagation_constant_2w__(): Propagation constant set.")

    def __set_normalisation_factor_w__(self):
        self.__normalisation_factor_w__ = np.power(self.__propagation_constant_w__ /(2*self.__omega_w__*self.__mu_0__) *
            simps(simps(np.abs(self.__E_w__)**2, self.__X__[0,:]), self.__Y__[:,0]), -0.5)

    def __set_normalisation_factor_2w__(self):
        self.__normalisation_factor_2w__ = np.zeros(np.shape(self.__E_2w__)[0], dtype=np.complex128)
        for i in range(len(self.__normalisation_factor_2w__)):
            self.__normalisation_factor_2w__[i] = np.power(self.__propagation_constant_2w__[i] /(2*2*self.__omega_w__*self.__mu_0__) *
                simps(simps(np.abs(self.__E_2w__[i])**2, self.__X__[0,:]), self.__Y__[:,0]), -0.5)

    def __set_E_w_norm__(self):
        assert not isinstance(self.__normalisation_factor_w__, type(None)), \
            ("NonLinearOptics.__set_E_w_norm__(): __normalisation_factor_w__ of type %r"
             % (type(self.__normalisation_factor_w__ )))
        assert not isinstance(self.__E_w__, type(None)), \
            ("NonLinearOptics.__set_E_w_norm__(): __E_w__ of type %r" % (type(self.__E_w__ )))

        self.__E_w_norm__ = self.__normalisation_factor_w__ * self.__E_w__

    def __set_E_2w_norm__(self):
        assert not isinstance(self.__normalisation_factor_2w__, type(None)), \
            ("NonLinearOptics.__set_E_2w_norm__(): __normalisation_factor_2w__ of type %r"
             % (type(self.__normalisation_factor_2w__ )))
        assert not isinstance(self.__E_2w__, type(None)), \
            ("NonLinearOptics.__set_E_2w_norm__(): __E_2w__ of type %r" % (type(self.__E_2w__ )))

        self.__E_2w_norm__ = np.zeros_like(self.__E_2w__, dtype=np.complex128)
        for i in range(np.shape(self.__normalisation_factor_2w__)[0]):
            self.__E_2w_norm__[i] = self.__normalisation_factor_2w__[i] * self.__E_2w__[i]

    def __set_effective_cross_section__(self):
        assert not isinstance(self.__E_2w_norm__, type(None)), \
            ("NonLinearOptics.__set_effective_cross_section__(): __E_2w_norm__ of type %r"
             % (type(self.__E_2w_norm__ )))
        assert not isinstance(self.__E_w_norm__, type(None)), \
            ("NonLinearOptics.__set_effective_cross_section__(): __E_w_norm__ of type %r" % (type(self.__E_w_norm__ )))

        assert not isinstance(self.__X__, type(None)), \
            ("NonLinearOptics.__set_effective_cross_section__(): __X__ of type %r"
             % (type(self.__X__ )))
        assert not isinstance(self.__Y__, type(None)), \
            ("NonLinearOptics.__set_effective_cross_section__(): __Y__ of type %r" % (type(self.__Y__ )))

        self.__effective_cross_section__ = np.zeros(np.shape(self.__E_2w_norm__)[0])
        for i in range(len(self.__effective_cross_section__)):
            self.__effective_cross_section__[i] = np.real(simps(simps(np.abs(self.__E_2w_norm__[i])**2, self.__X__[0, :]), self.__Y__[:, 0]) *
                (simps(simps(np.abs(self.__E_w_norm__)**2, self.__X__[0, :]), self.__Y__[:, 0]))**2 /
                (simps(simps(np.conjugate(self.__E_2w_norm__[i])*self.__E_w_norm__**2, self.__X__[0, :]), self.__Y__[:, 0]))**2)

    def __set_nonlinear_coupling_coefficient__(self):
        self.__nonlinear_coupling_coefficient__ = np.zeros(np.shape(self.__E_2w_norm__)[0], dtype=np.complex128)
        for i in range(len(self.__nonlinear_coupling_coefficient__)):
            self.__nonlinear_coupling_coefficient__[i] = self.__epsilon_0__ * np.sqrt(((2*self.__omega_w__)**2 /
                                                      (2 * (self.__effective_mode_index_w__)**2 * self.__effective_mode_index_2w__[i])) *
                                                      np.power(self.__mu_0__/self.__epsilon_0__, 3/2) *
                                                      (self.__d_eff__**2/self.__effective_cross_section__[i]))

    def __set_normalised_conversion_efficiency__(self):
        assert not isinstance(self.__nonlinear_coupling_coefficient__, type(None)), \
            ("NonLinearOptics.__set_normalised_conversion_efficiency__(): __nonlinear_coupling_coefficient__ of type %r" % (type(self.__nonlinear_coupling_coefficient__ )))
        assert not isinstance(self.__propagation_constant_2w__, type(None)), \
            ("NonLinearOptics.__set_normalised_conversion_efficiency__(): __propagation_constant_2w__ of type %r" % (type(self.__propagation_constant_2w__ )))
        assert not isinstance(self.__propagation_constant_w__, type(None)), \
            ("NonLinearOptics.__set_normalised_conversion_efficiency__(): __propagation_constant_w__ of type %r" % (type(self.__propagation_constant_w__ )))
        assert not isinstance(self.delta_k, type(None)), \
            ("NonLinearOptics.__set_normalised_conversion_efficiency__(): delta_k of type %r" % (type(self.delta_k )))
        assert not isinstance(z, type(None)), \
            ("NonLinearOptics.__set_normalised_conversion_efficiency__(): z of type %r" % (type(z )))

        self.__normalised_conversion_efficiency__ = np.zeros([np.shape(self.__E_2w_norm__)[0], len(self.__z__)])
        for i in range(len(self.__normalised_conversion_efficiency__)):
            self.__normalised_conversion_efficiency__[i] = (self.__nonlinear_coupling_coefficient__[i]**2 *
                                                         np.sinc((self.__propagation_constant_2w__[i] - 2*self.__propagation_constant_w__)
                                                         - 2*np.pi/(2*np.pi/self.__delta_k__) * self.__z__/2)**2)
        return self.__normalised_conversion_efficiency__

    def set_ODE_rtol_(self, rtol):
        self.__ODE_rtol__ = rtol

    def set_z_max(self, z_max):
        self.__z_max__ = z_max

    def set_z_points(self, z_points):
        self.__z_points__ = z_points

    def __solve_amplitude_coupled_ODEs__(self):
        """
        Defines the differential equations for the system.
        """
        print("__solve_amplitude_coupled_ODEs__(): Solving system...")

        P = (0.5*self.__effective_mode_index_w__*self.__epsilon_0__*self.__c__ *
                np.real(simps(simps(np.abs(self.__E_w_norm__)**2, self.__X__[0,:]), self.__Y__[:,0])))
        AB0 = np.array([np.sqrt(P), 0.0], dtype='complex')
        k = self.__nonlinear_coupling_coefficient__
        def system(AB, z, k, Lamda, Delta):
            dABdz=np.zeros(2,dtype='complex')
            A = AB[0] + AB[1]*1j
            B = AB[2] + AB[3]*1j
            kk =  k*np.sign(np.sin(2*np.pi*z/Lamda))
            dABdz[0] = -1j * np.conjugate(kk) * np.conjugate(A) * B * np.exp(-1j*Delta*z)
            dABdz[1] = -1j * kk * (np.abs(A))**2 * np.exp(1j*Delta*z)
            return dABdz.view(np.float64)

        AB = np.zeros([len(self.__z__),2])
        AB_sol = np.zeros([np.shape(self.__E_2w_norm__)[0], len(self.__z__), 2], dtype=np.complex128)
        eta = np.zeros([np.shape(self.__E_2w_norm__)[0], len(self.__z__)])
        for i in range(len(self.__E_2w_norm__)):
            k = np.real(self.__nonlinear_coupling_coefficient__[i])
            Lamda = self.__coherent_length__[self.__fundamental_mode_index__] * 2
            Delta = np.real(self.__propagation_constant_2w__[i] - 2*self.__propagation_constant_w__)
            AB = odeint(system, AB0.view(np.float64), self.__z__, args=(k, Lamda, Delta), rtol=self.__ODE_rtol__)
            AB = AB.view(np.complex128)

            eta[i] = (np.abs(AB[:, 1])/np.abs(AB0[0]))**2

            AB_sol[i] = AB

        self.__amplitude_solutions__ = AB_sol
        self.__conversion_efficiency__ = eta

        print("__solve_amplitude_coupled_ODEs__(): Solution to system computed.")

    def update_values(self):
        print("update_values(): Updating values...")
        self.__set_propagation_constant_w__()
        self.__set_propagation_constant_2w__()
        self.__set_normalisation_factor_w__()
        self.__set_normalisation_factor_2w__()
        self.__set_E_w_norm__()
        self.__set_E_2w_norm__()
        self.__set_effective_cross_section__()
        self.__set_nonlinear_coupling_coefficient__()
        print("update_values(): Values updated.")

    def update_amplitude_coupled_ODEs_solution(self):
        self.__solve_amplitude_coupled_ODEs__()

    def X(self):
        return self.__X__

    def Y(self):
        return self.__Y__

    def E_w(self):
        return self.__E_w__

    def E_2w(self):
        return self.__E_2w__

    def effective_mode_index_w(self):
        return self.__effective_mode_index_w__

    def effective_mode_index_2w(self):
        return self.__effective_mode_index_2w__

    def lamda_w(self):
        return self.__lamda_w__

    def lamda_2w(self):
        return self.__lamda_2w__

    def c(self):
        return self.__c__

    def mu_0(self):
        return self.__mu_0__

    def epsilon_0(self):
        return self.__epsilon_0__

    def omega_w(self):
        return self.__omega_w__

    def omega_2w(self):
        return self.__omega_2w__

    def d_eff(self):
        return self.__d_eff__

    def z_max(self):
        return self.__z_max__

    def z_points(self):
        return self.__z_points__

    def z(self):
        return self.__z__

    def ODE_rtol(self):
        return self.__ODE_rtol__

    def coherent_length(self):
        return self.__coherent_length__

    def inversion_period(self):
        return self.__inversion_period__

    def propagation_constant_w(self):
        return self.__propagation_constant_w__

    def normalisation_factor_w(self):
        return self.__normalisation_factor_w__

    def propagation_constant_2w(self):
        return self.__propagation_constant_2w__

    def normalisation_factor_2w(self):
        return self.__normalisation_factor_2w__

    def E_w_norm(self):
        return self.__E_w_norm__

    def E_2w_norm(self):
        return self.__E_2w_norm__

    def effective_cross_section(self):
        return self.__effective_cross_section__

    def nonlinear_coupling_coefficient(self):
        return self.__nonlinear_coupling_coefficient__

    def amplitude_solutions(self):
        return self.__amplitude_solutions__

    def conversion_efficiency(self):
        return self.__conversion_efficiency__


class WaveguideResults(ReadInData, NonlinearOptics, Plot):

    def __init__(self, files, pump_fundamental_index=0, kwargs_NonLinearOptics={}, kwargs_ReadInData={},
                 kwargs_Data={}, kwargs_grid={}):
        self.__files__ = files
        self.__pump_fundamental_index__ = pump_fundamental_index
        self.__kwargs_NonLinearOptics__ = kwargs_NonLinearOptics
        self.__kwargs_ReadInData__ = kwargs_ReadInData
        self.__kwargs_Data__ = self.__kwargs_ReadInData__
        self.__kwargs_grid__ = kwargs_grid

        assert np.shape(self.__files__) == (3, )

        for i in range(len(files)):
            ReadInData.__init__(self, self.__files__[i],
                                self.__kwargs_ReadInData__[i], print_details=False)
            if self.header()['Description'] == 'Electric field norm':
                self.electricFieldNorm = Data(self.__files__[i],
                                              self.__kwargs_ReadInData__[i],
                                              self.__kwargs_Data__[i],
                                              self.__kwargs_grid__)
            elif self.header()['Description'] == 'Concentration':
                self.concentration = Data(self.__files__[i],
                                          self.__kwargs_ReadInData__[i],
                                          self.__kwargs_Data__[i],
                                          self.__kwargs_grid__)
            elif self.header()['Description'] == 'Effective mode index':
                self.effectiveModeIndex = Data(self.__files__[i],
                                               self.__kwargs_ReadInData__[i],
                                               self.__kwargs_Data__[i],
                                               self.__kwargs_grid__)
        
        Plot.__init__(self, self.electricFieldNorm.X(),
                      self.electricFieldNorm.Y())

        self.E = self.electricFieldNorm.Data()[::-1]
        self.E = self.E*1.0E-6
        self.E_w = self.E[int(np.shape(self.electricFieldNorm.Data())[0]/2):]
        self.E_w = self.E_w[self.__pump_fundamental_index__]

        self.E_2w = self.E[:int(np.shape(self.electricFieldNorm.Data())[0]/2)]


        self.effective_mode_index_tmp = self.effectiveModeIndex.Data()[::-1]
        self.effective_mode_index_w = self.effective_mode_index_tmp[int(np.shape(self.effectiveModeIndex.Data())[0]/2):]
        self.effective_mode_index_w = self.effective_mode_index_w[self.__pump_fundamental_index__, 0]

        self.effective_mode_index_2w = self.effective_mode_index_tmp[:int(np.shape(self.effectiveModeIndex.Data())[0]/2)]
        self.effective_mode_index_2w = self.effective_mode_index_2w[:,0]

        self.nonLinearOptics = NonlinearOptics(self.electricFieldNorm.X()*1.0E-6,
                                               self.electricFieldNorm.Y()*1.0E-6,
                                               self.E_w, self.E_2w,
                                               self.effective_mode_index_w,
                                               self.effective_mode_index_2w,
                                               **self.__kwargs_NonLinearOptics__)
