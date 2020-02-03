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
        instances of internal variables such as _file, _header
        and _raw_data, and then reads in the raw data from the file.

        Parameters:
        file : (string) The file name to be read in.

        **kwargs : (options, dict) Additional arguments that can be
                   passed in, such as a bool to specify whether to
                   ignore the header or not and a bool to specify
                   the data file contains complex data types.

        Returns:
        (none) : Does not return any variables.
        """
        self._kwargs_get_raw_data = kwargs_get_raw_data
        self._print_details = print_details
        self._file = file
        self._header = {'Model': '',
                           'Version': '',
                           'Date': '',
                           'Dimension': '',
                           'Nodes': '',
                           'Expressions': '',
                           'Description': '',
                           'Length unit': '',
                           'Data Structure': ''}
        self._raw_data = None

        self.get_raw_data(**self._kwargs_get_raw_data)

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
        with open(self._file) as input_file:
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
        if self._print_details is True:
            print("File: {}".format(self._file))
            print("      n_lines: {}".format(self.n_lines))
            print("      n_data_per_line: {}".format(self.n_data_per_line))
            print("      n_header_lines: {}".format(self.n_header_lines))
            print("      (n_data_lines: {} (inferred))\n".format(self.n_lines - self.n_header_lines))

    def get_raw_data(self, ignore_header=False, complexData=False):
        """
        Reads in the raw data from the data file. It firstly loops over the
        header and either ignores it or sets the values of the header dict,
        and then loops over the remaining lines and sets the _raw_data
        varaible to the values on each line.

        If the ignore_header flag is set to False then the values of the header
        dict are set to their respective entry from the data file.

        If the the complexData flag is True then the _raw_data variable is
        initialised as a np.complex128 type rather than np.float64.

        Parameters:
        ignore_header : (optional, bool) The flag to specify whether to ignore
                        the header or not. By default this is True and the
                        header is ignored.

        complexData : (optional, bool) The flag to specifiy if complex data
                      types are being read in. By default this is False and the
                      elements of _raw_data are set as np.float64.

        Returns:
        (none) : Does not return any variables.
        """
        self.get_number_of_lines()
        with open(self._file) as input_file:
            if complexData is False:
                self._raw_data = np.zeros([self.n_data_per_line,
                                              self.n_lines - self.n_header_lines])
            elif complexData is True:
                self._raw_data = np.zeros([self.n_data_per_line,
                                              self.n_lines - self.n_header_lines], dtype=np.complex128)
            i = 0
            for line in input_file:
                if (i < self.n_header_lines):
                    if ignore_header is True and self._print_details is True:
                        print("get_data(): Skipping header line {} ...".format(i))
                    else:
                        line = line.split()
                        line.pop(0)
                        if line[0] == 'Model:':
                            if self._print_details is True:
                                print("get_data(): In header, Model value found")
                            self._header['Model'] = ' '.join(line[1:])
                        elif line[0] == 'Version:':
                            if self._print_details is True:
                                print("get_data(): In header, Version value found")
                            self._header['Version'] = ' '.join(line[1:])
                        elif line[0] == 'Date:':
                            if self._print_details is True:
                                print("get_data(): In header, Date value found")
                            self._header['Date'] = ' '.join(line[1:])
                        elif line[0] == 'Dimension:':
                            if self._print_details is True:
                                print("get_data(): In header, Dimension value found")
                            self._header['Dimension'] = ' '.join(line[1:])
                        elif line[0] == 'Nodes:':
                            if self._print_details is True:
                                print("get_data(): In header, Nodes value found")
                            self._header['Nodes'] = ' '.join(line[1:])
                        elif line[0] == 'Expressions:':
                            if self._print_details is True:
                                print("get_data(): In header, Expressions value found")
                            self._header['Expressions'] = ' '.join(line[1:])
                        elif line[0] == 'Description:':
                            if self._print_details is True:
                                print("get_data(): In header, Description value found")
                            self._header['Description'] = ' '.join(line[1:])
                        elif line[0] == 'Length' and line[1] == 'unit:':
                            if self._print_details is True:
                                print("get_data(): In header, Length Unit value found")
                            self._header['Length unit'] = ' '.join(line[2:])
                        elif i == self.n_header_lines - 1:
                            if self._print_details is True:
                                print("get_data(): In header, Data Structure value assumed as last line of header")
                            tmp = ' '.join(line[2:]).split(" @ ")
                            tmp2 = [tmp[0]]
                            tmp.pop(0)
                            for j in range(len(tmp)):
                                tmp2 += [tmp[j].replace(tmp2[0], "").strip()]
                            tmp3 = [tmp2[0], tmp2[1:]]
                            self._header['Data Structure'] = [line[0]] + [line[1]] + tmp3
                else:
                    line = line.strip()
                    j = 0
                    for number in line.split():
                        if complexData is False:
                            self._raw_data[j, i - self.n_header_lines] = number
                        elif complexData is True:
                            self._raw_data[j, i - self.n_header_lines] = complex(number.replace('i', 'j'))
                        j += 1
                i += 1
            if self._print_details == 'Minimal':
                print("get_data(): Read in data of shape {} from file '{}'".format(np.shape(self._raw_data), self._file))

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
        Function to return the encapsulated variable _raw_data.

        Parameters:
        (none) : Does not take any input parameters.

        Returns:
        self._raw_data : (numpy array) The raw data.
        """
        return self._raw_data

    def file(self):
        """
        Function to return the encapsulated variable _file.

        Parameters:
        (none) : Does not take any input parameters.

        Returns:
        self._raw_data : (numpy array) The file name.
        """
        return self._file

    def header(self):
        """
        Function to return the encapsulated variable _header.

        Parameters:
        (none) : Does not take any input parameters.

        Returns:
        self._raw_data : (dict) The header.
        """
        return self._header


class Plot():
    """
    Interfaces to and plots various pyplotlib plots for the given data.
    """
    def __init__(self, X, Y):
        """
        Initialises the Plot class instance. __init__ creates
        instances of internal variables such as _X, _Y, xlim,
        and ylim.

        Parameters:
        X : (array of np.float64) The grided values of the X values.

        Y : (array of np.float64) The grided values of the Y values.

        Returns:
        (none) : Does not return any variables.
        """
        self._X = X
        self._Y = Y
        self._xlim = [np.min(self._X), np.max(self._X)]
        self._ylim = [np.min(self._Y), np.max(self._Y)]

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
                xlim = self._xlim
            elif n_plots > 1:
                xlim = [self._xlim]*n_plots
        if ylim is None:
            if n_plots == 1:
                ylim = self._ylim
            elif n_plots > 1:
                ylim = [self._ylim]*n_plots

        fig = plt.figure(figsize=figsize, dpi=dpi)
        fig.suptitle(title)
        if n_plots == 1:
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.pcolor(self._X, self._Y, Z, cmap=cmap)
            ax1.set_xlim([xlim[0], xlim[1]])
            ax1.set_ylim([ylim[0], ylim[1]])
            ax1.set_xlabel(axis_titles[0])
            ax1.set_ylabel(axis_titles[1])
            ax1.set_title(sub_title)
        else:
            for i in range(n_plots):
                ax1 = fig.add_subplot(1, n_plots, i+1)
                ax1.pcolor(self._X, self._Y, Z[i], cmap=cmap)
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
                xlim = self._xlim
            elif n_plots > 1:
                xlim = [self._xlim]*n_plots
        if ylim is None:
            if n_plots == 1:
                ylim = self._ylim
            elif n_plots > 1:
                ylim = [self._ylim]*n_plots

        fig = plt.figure(figsize=figsize, dpi=dpi)
        fig.suptitle(title)
        if n_plots == 1:
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.contourf(self._X, self._Y, Z, n_lines)
            ax1.set_xlim([xlim[0], xlim[1]])
            ax1.set_ylim([ylim[0], ylim[1]])
            ax1.set_xlabel(axis_titles[0])
            ax1.set_ylabel(axis_titles[1])
            ax1.set_title(sub_title)
        else:
            for i in range(n_plots):
                ax1 = fig.add_subplot(1, n_plots, i+1)
                ax1.contourf(self._X, self._Y, Z[i], n_lines)
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
        internal variables such as kwargs_ReadInData, kwargs_grid and _Data,
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
        self._kwargs_Data = kwargs_Data
        self._kwargs_ReadInData = kwargs_ReadInData
        self._kwargs_grid = kwargs_grid
        self._x = None
        self._y = None
        self._data = None
        self._X = None
        self._Y = None
        self._Data = None

        ReadInData.__init__(self, file, self._kwargs_ReadInData)
        self.format_raw_data()
        self.grid(**self._kwargs_grid)
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
        self._x = self._raw_data[0, :]
        self._y = self._raw_data[1, :]
        self._data = self._raw_data[2:, :]

    def format_data(self):
        """
        Formats the data into the X, Y, Data variables.

        Parameters:
        (none) : Does not take any input parameters.

        Returns:
        (none) : Does not return any variables.
        """
        self._X = np.nan_to_num(self._X)
        self._Y = np.nan_to_num(self._Y)
        self._Data = np.nan_to_num(self._Data)

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
        if 'complexData' in self._kwargs_Data and self._kwargs_Data['complexData'] == True:
            print("grid(): Warning, complex data type present, unable to grid data...\n")
            self._X = self._x
            self._Y = self._y
            self._Data = self._data
        else:
            if resX == None or resY == None:
                resX = (np.max(self._x) - np.min(self._x))*res
                resY = (np.max(self._y) - np.min(self._y))*res
            n_modes = np.shape(self._data)[0]
            xi = np.linspace(np.min(self._x), np.max(self._x), resX)
            yi = np.linspace(np.min(self._y), np.max(self._y), resY)
            if n_modes == 1:
                self._Data = griddata(self._x, self._y, self._data, xi, yi, interp='linear')
            else:
                self._Data = np.zeros([n_modes, int(resY), int(resX)])
                for i in range(n_modes):
                    self._Data[i] = griddata(self._x, self._y, self._data[i], xi, yi, interp='linear')
            self._X, self._Y = np.meshgrid(xi, yi)

    def data_details(self):
        """
        Prints the x, y, data, X, Y, Data details to the screen.

        Parameters:
        (none) : Does not take any input parameters.

        Returns:
        (none) : Does not return any variables.
        """
        print("\n{} data details:".format(self.header()['Description']))
        print ("      x: {}".format(np.shape(self._x)))
        print ("      y: {}".format(np.shape(self._y)))
        print ("      data: {}".format(np.shape(self._data)))
        print ("      X: {}".format(np.shape(self._X)))
        print ("      Y: {}".format(np.shape(self._Y)))
        print ("      Data: {}\n".format(np.shape(self._Data)))

    def x(self):
        """
        Function to return the encapsulated variable _x.

        Parameters:
        (none) : Does not take any input parameters.

        Returns:
        self._x : (numpy array) The x variable.
        """
        return self._x

    def y(self):
        """
        Function to return the encapsulated variable _y.

        Parameters:
        (none) : Does not take any input parameters.

        Returns:
        self._y : (numpy array) The y variable.
        """
        return self._y

    def data(self):
        """
        Function to return the encapsulated variable _data.

        Parameters:
        (none) : Does not take any input parameters.

        Returns:
        self._data : (numpy array) The data variable.
        """
        return self._data

    def X(self):
        """
        Function to return the encapsulated variable _X.

        Parameters:
        (none) : Does not take any input parameters.

        Returns:
        self._X : (numpy array) The X variable.
        """
        return self._X

    def Y(self):
        """
        Function to return the encapsulated variable _Y.

        Parameters:
        (none) : Does not take any input parameters.

        Returns:
        self._Y : (numpy array) The Y variable.
        """
        return self._Y

    def Data(self):
        """
        Function to return the encapsulated variable _Data.

        Parameters:
        (none) : Does not take any input parameters.

        Returns:
        self._Data : (numpy array) The Data variable.
        """
        return self._Data


class NonlinearOptics():

    def __init__(self, X, Y, E_w, E_2w, effective_mode_index_w, effective_mode_index_2w, fundamental_mode_index=0, lamda=1550.0E-9, T_Op=20.0, solve=True):
        self._X = X
        self._Y = Y
        self._E_w = E_w    # Fundamental mode of source
        if np.ndim(E_2w) == 1:
            self._E_2w = [E_2w, ]  # Single mode of the second harmonic
        else:
            self._E_2w = E_2w  # Multiple modes of the second harmonic
        self._effective_mode_index_w = effective_mode_index_w


        if np.ndim(effective_mode_index_2w) == 1:
            self._effective_mode_index_2w = [effective_mode_index_2w, ]
        else:
            self._effective_mode_index_2w = effective_mode_index_2w
        self._effective_mode_index_2w = effective_mode_index_2w

        self._fundamental_mode_index = fundamental_mode_index


        self._lamda_w = lamda
        self._lamda_2w = self._lamda_w / 2.0

        self._T_Op = T_Op


        self._c = 299792458               # Speed of light
        self._mu_0 = 4*np.pi*1.0E-7       # Permeability of free space
        self._epsilon_0 = 8.85418782e-12  # Permitivity of free space
        self._omega_w = 2*np.pi * self._c / self._lamda_w
        self._omega_2w = 2*np.pi * self._c / self._lamda_2w
        self._d_eff = 14.0E-12            # deff

        # Default values for solving in the z direction of the waveguide
        self._z_max = 50E-3
        self._z_points = 500
        self._z = np.arange(0, self._z_max, self._z_max/self._z_points)
        self._ODE_rtol = 1.0E-4

        self._amplitude_solutions = None
        self._conversion_efficiency = None

        self._propagation_constant_w = None
        self._normalisation_factor_w = None
        self._propagation_constant_2w = None
        self._normalisation_factor_2w = None
        self._E_w_norm = None
        self._E_2w_norm = None
        self._nonlinear_coupling_coefficient = None

        self._effective_nonlinear_coefficient = None
        self._delta_k = (4 * np.pi / self._lamda_w) * (self._sellmeier_equation(self._lamda_2w, self._T_Op) - self._sellmeier_equation(self._lamda_w, self._T_Op))
        self._coherent_length = self._lamda_w / (4*(np.real(self._effective_mode_index_2w) - np.real(self._effective_mode_index_w)))
        self._inversion_period = 2*self._coherent_length
        self._effective_mode_overlap = None
        self._effective_cross_section = None

        self._set_propagation_constant_w()
        self._set_propagation_constant_2w()
        self._set_normalisation_factor_w()
        self._set_normalisation_factor_2w()
        self._set_E_w_norm()
        self._set_E_2w_norm()

        self._set_effective_cross_section()
        self._set_nonlinear_coupling_coefficient()

        if solve is True:
            self._solve_amplitude_coupled_ODEs()

        assert not isinstance(self._propagation_constant_w, type(None))
        assert not isinstance(self._normalisation_factor_w, type(None))
        assert not isinstance(self._propagation_constant_2w, type(None))
        assert not isinstance(self._normalisation_factor_2w, type(None))

        assert not isinstance(self._E_w_norm, type(None)), \
            ("NonLinearOptics.__init__(): _E_w_norm of type %r" % (type(self._E_w_norm )))
        assert not isinstance(self._E_2w_norm, type(None)), \
            ("NonLinearOptics.__init__(): _E_w_norm of type %r" % (type(self._E_2w_norm )))

    def _sellmeier_equation(self, lam, T):
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

    def _set_propagation_constant_w(self):
        self._propagation_constant_w = self._effective_mode_index_w * 2*np.pi / self._lamda_w
        print("_set_propagation_constant_w(): Propagation constant set.")

    def _set_propagation_constant_2w(self):
        self._propagation_constant_2w = np.zeros(np.shape(self._effective_mode_index_2w)[0], dtype=np.complex128)
        for i in range(len(self._propagation_constant_2w)):
            self._propagation_constant_2w[i] = self._effective_mode_index_2w[i] * 2*np.pi / self._lamda_2w
        print("_set_propagation_constant_2w(): Propagation constant set.")

    def _set_normalisation_factor_w(self):
        self._normalisation_factor_w = np.power(self._propagation_constant_w /(2*self._omega_w*self._mu_0) *
            simps(simps(np.abs(self._E_w)**2, self._X[0,:]), self._Y[:,0]), -0.5)

    def _set_normalisation_factor_2w(self):
        self._normalisation_factor_2w = np.zeros(np.shape(self._E_2w)[0], dtype=np.complex128)
        for i in range(len(self._normalisation_factor_2w)):
            self._normalisation_factor_2w[i] = np.power(self._propagation_constant_2w[i] /(2*2*self._omega_w*self._mu_0) *
                simps(simps(np.abs(self._E_2w[i])**2, self._X[0,:]), self._Y[:,0]), -0.5)

    def _set_E_w_norm(self):
        assert not isinstance(self._normalisation_factor_w, type(None)), \
            ("NonLinearOptics._set_E_w_norm(): _normalisation_factor_w of type %r"
             % (type(self._normalisation_factor_w )))
        assert not isinstance(self._E_w, type(None)), \
            ("NonLinearOptics._set_E_w_norm(): _E_w of type %r" % (type(self._E_w )))

        self._E_w_norm = self._normalisation_factor_w * self._E_w

    def _set_E_2w_norm(self):
        assert not isinstance(self._normalisation_factor_2w, type(None)), \
            ("NonLinearOptics._set_E_2w_norm(): _normalisation_factor_2w of type %r"
             % (type(self._normalisation_factor_2w )))
        assert not isinstance(self._E_2w, type(None)), \
            ("NonLinearOptics._set_E_2w_norm(): _E_2w of type %r" % (type(self._E_2w )))

        self._E_2w_norm = np.zeros_like(self._E_2w, dtype=np.complex128)
        for i in range(np.shape(self._normalisation_factor_2w)[0]):
            self._E_2w_norm[i] = self._normalisation_factor_2w[i] * self._E_2w[i]

    def _set_effective_cross_section(self):
        assert not isinstance(self._E_2w_norm, type(None)), \
            ("NonLinearOptics._set_effective_cross_section(): _E_2w_norm of type %r"
             % (type(self._E_2w_norm )))
        assert not isinstance(self._E_w_norm, type(None)), \
            ("NonLinearOptics._set_effective_cross_section(): _E_w_norm of type %r" % (type(self._E_w_norm )))

        assert not isinstance(self._X, type(None)), \
            ("NonLinearOptics._set_effective_cross_section(): _X of type %r"
             % (type(self._X )))
        assert not isinstance(self._Y, type(None)), \
            ("NonLinearOptics._set_effective_cross_section(): _Y of type %r" % (type(self._Y )))

        self._effective_cross_section = np.zeros(np.shape(self._E_2w_norm)[0])
        for i in range(len(self._effective_cross_section)):
            self._effective_cross_section[i] = np.real(simps(simps(np.abs(self._E_2w_norm[i])**2, self._X[0, :]), self._Y[:, 0]) *
                (simps(simps(np.abs(self._E_w_norm)**2, self._X[0, :]), self._Y[:, 0]))**2 /
                (simps(simps(np.conjugate(self._E_2w_norm[i])*self._E_w_norm**2, self._X[0, :]), self._Y[:, 0]))**2)

    def _set_nonlinear_coupling_coefficient(self):
        self._nonlinear_coupling_coefficient = np.zeros(np.shape(self._E_2w_norm)[0], dtype=np.complex128)
        for i in range(len(self._nonlinear_coupling_coefficient)):
            self._nonlinear_coupling_coefficient[i] = self._epsilon_0 * np.sqrt(((2*self._omega_w)**2 /
                                                      (2 * (self._effective_mode_index_w)**2 * self._effective_mode_index_2w[i])) *
                                                      np.power(self._mu_0/self._epsilon_0, 3/2) *
                                                      (self._d_eff**2/self._effective_cross_section[i]))

    def _set_normalised_conversion_efficiency(self):
        assert not isinstance(self._nonlinear_coupling_coefficient, type(None)), \
            ("NonLinearOptics._set_normalised_conversion_efficiency(): _nonlinear_coupling_coefficient of type %r" % (type(self._nonlinear_coupling_coefficient )))
        assert not isinstance(self._propagation_constant_2w, type(None)), \
            ("NonLinearOptics._set_normalised_conversion_efficiency(): _propagation_constant_2w of type %r" % (type(self._propagation_constant_2w )))
        assert not isinstance(self._propagation_constant_w, type(None)), \
            ("NonLinearOptics._set_normalised_conversion_efficiency(): _propagation_constant_w of type %r" % (type(self._propagation_constant_w )))
        assert not isinstance(self.delta_k, type(None)), \
            ("NonLinearOptics._set_normalised_conversion_efficiency(): delta_k of type %r" % (type(self.delta_k )))
        assert not isinstance(z, type(None)), \
            ("NonLinearOptics._set_normalised_conversion_efficiency(): z of type %r" % (type(z )))

        self._normalised_conversion_efficiency = np.zeros([np.shape(self._E_2w_norm)[0], len(self._z)])
        for i in range(len(self._normalised_conversion_efficiency)):
            self._normalised_conversion_efficiency[i] = (self._nonlinear_coupling_coefficient[i]**2 *
                                                         np.sinc((self._propagation_constant_2w[i] - 2*self._propagation_constant_w)
                                                         - 2*np.pi/(2*np.pi/self._delta_k) * self._z/2)**2)
        return self._normalised_conversion_efficiency

    def set_ODE_rtol_(self, rtol):
        self._ODE_rtol = rtol

    def set_z_max(self, z_max):
        self._z_max = z_max

    def set_z_points(self, z_points):
        self._z_points = z_points

    def _solve_amplitude_coupled_ODEs(self):
        """
        Defines the differential equations for the system.
        """
        print("_solve_amplitude_coupled_ODEs(): Solving system...")

        P = (0.5*self._effective_mode_index_w*self._epsilon_0*self._c *
                np.real(simps(simps(np.abs(self._E_w_norm)**2, self._X[0,:]), self._Y[:,0])))
        AB0 = np.array([np.sqrt(P), 0.0], dtype='complex')
        k = self._nonlinear_coupling_coefficient
        def system(AB, z, k, Lamda, Delta):
            dABdz=np.zeros(2,dtype='complex')
            A = AB[0] + AB[1]*1j
            B = AB[2] + AB[3]*1j
            kk =  k*np.sign(np.sin(2*np.pi*z/Lamda))
            dABdz[0] = -1j * np.conjugate(kk) * np.conjugate(A) * B * np.exp(-1j*Delta*z)
            dABdz[1] = -1j * kk * (np.abs(A))**2 * np.exp(1j*Delta*z)
            return dABdz.view(np.float64)

        AB = np.zeros([len(self._z),2])
        AB_sol = np.zeros([np.shape(self._E_2w_norm)[0], len(self._z), 2], dtype=np.complex128)
        eta = np.zeros([np.shape(self._E_2w_norm)[0], len(self._z)])
        for i in range(len(self._E_2w_norm)):
            k = np.real(self._nonlinear_coupling_coefficient[i])
            Lamda = self._coherent_length[self._fundamental_mode_index] * 2
            Delta = np.real(self._propagation_constant_2w[i] - 2*self._propagation_constant_w)
            AB = odeint(system, AB0.view(np.float64), self._z, args=(k, Lamda, Delta), rtol=self._ODE_rtol)
            AB = AB.view(np.complex128)

            eta[i] = (np.abs(AB[:, 1])/np.abs(AB0[0]))**2

            AB_sol[i] = AB

        self._amplitude_solutions = AB_sol
        self._conversion_efficiency = eta

        print("_solve_amplitude_coupled_ODEs(): Solution to system computed.")

    def update_values(self):
        print("update_values(): Updating values...")
        self._set_propagation_constant_w()
        self._set_propagation_constant_2w()
        self._set_normalisation_factor_w()
        self._set_normalisation_factor_2w()
        self._set_E_w_norm()
        self._set_E_2w_norm()
        self._set_effective_cross_section()
        self._set_nonlinear_coupling_coefficient()
        print("update_values(): Values updated.")

    def update_amplitude_coupled_ODEs_solution(self):
        self._solve_amplitude_coupled_ODEs()

    def X(self):
        return self._X

    def Y(self):
        return self._Y

    def E_w(self):
        return self._E_w

    def E_2w(self):
        return self._E_2w

    def effective_mode_index_w(self):
        return self._effective_mode_index_w

    def effective_mode_index_2w(self):
        return self._effective_mode_index_2w

    def lamda_w(self):
        return self._lamda_w

    def lamda_2w(self):
        return self._lamda_2w

    def c(self):
        return self._c

    def mu_0(self):
        return self._mu_0

    def epsilon_0(self):
        return self._epsilon_0

    def omega_w(self):
        return self._omega_w

    def omega_2w(self):
        return self._omega_2w

    def d_eff(self):
        return self._d_eff

    def z_max(self):
        return self._z_max

    def z_points(self):
        return self._z_points

    def z(self):
        return self._z

    def ODE_rtol(self):
        return self._ODE_rtol

    def coherent_length(self):
        return self._coherent_length

    def inversion_period(self):
        return self._inversion_period

    def propagation_constant_w(self):
        return self._propagation_constant_w

    def normalisation_factor_w(self):
        return self._normalisation_factor_w

    def propagation_constant_2w(self):
        return self._propagation_constant_2w

    def normalisation_factor_2w(self):
        return self._normalisation_factor_2w

    def E_w_norm(self):
        return self._E_w_norm

    def E_2w_norm(self):
        return self._E_2w_norm

    def effective_cross_section(self):
        return self._effective_cross_section

    def nonlinear_coupling_coefficient(self):
        return self._nonlinear_coupling_coefficient

    def amplitude_solutions(self):
        return self._amplitude_solutions

    def conversion_efficiency(self):
        return self._conversion_efficiency


class WaveguideResults(ReadInData, NonlinearOptics, Plot):

    def __init__(self, files, pump_fundamental_index=0, kwargs_NonLinearOptics={}, kwargs_ReadInData={},
                 kwargs_Data={}, kwargs_grid={}):
        self._files = files
        self._pump_fundamental_index = pump_fundamental_index
        self._kwargs_NonLinearOptics = kwargs_NonLinearOptics
        self._kwargs_ReadInData = kwargs_ReadInData
        self._kwargs_Data = self._kwargs_ReadInData
        self._kwargs_grid = kwargs_grid

        assert np.shape(self._files) == (3, )

        for i in range(len(files)):
            ReadInData.__init__(self, self._files[i],
                                self._kwargs_ReadInData[i], print_details=False)
            if self.header()['Description'] == 'Electric field norm':
                self.electricFieldNorm = Data(self._files[i],
                                              self._kwargs_ReadInData[i],
                                              self._kwargs_Data[i],
                                              self._kwargs_grid)
            elif self.header()['Description'] == 'Concentration':
                self.concentration = Data(self._files[i],
                                          self._kwargs_ReadInData[i],
                                          self._kwargs_Data[i],
                                          self._kwargs_grid)
            elif self.header()['Description'] == 'Effective mode index':
                self.effectiveModeIndex = Data(self._files[i],
                                               self._kwargs_ReadInData[i],
                                               self._kwargs_Data[i],
                                               self._kwargs_grid)
        
        Plot.__init__(self, self.electricFieldNorm.X(),
                      self.electricFieldNorm.Y())

        self.E = self.electricFieldNorm.Data()[::-1]
        self.E = self.E*1.0E-6
        self.E_w = self.E[int(np.shape(self.electricFieldNorm.Data())[0]/2):]
        self.E_w = self.E_w[self._pump_fundamental_index]

        self.E_2w = self.E[:int(np.shape(self.electricFieldNorm.Data())[0]/2)]


        self.effective_mode_index_tmp = self.effectiveModeIndex.Data()[::-1]
        self.effective_mode_index_w = self.effective_mode_index_tmp[int(np.shape(self.effectiveModeIndex.Data())[0]/2):]
        self.effective_mode_index_w = self.effective_mode_index_w[self._pump_fundamental_index, 0]

        self.effective_mode_index_2w = self.effective_mode_index_tmp[:int(np.shape(self.effectiveModeIndex.Data())[0]/2)]
        self.effective_mode_index_2w = self.effective_mode_index_2w[:,0]

        self.nonLinearOptics = NonlinearOptics(self.electricFieldNorm.X()*1.0E-6,
                                               self.electricFieldNorm.Y()*1.0E-6,
                                               self.E_w, self.E_2w,
                                               self.effective_mode_index_w,
                                               self.effective_mode_index_2w,
                                               **self._kwargs_NonLinearOptics)
