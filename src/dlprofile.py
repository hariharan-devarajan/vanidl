"""
 Copyright (C) 2020  Argonne, Hariharan Devarajan <hdevarajan@anl.gov>
 This file is part of DLProfile
 HFetch is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as
 published by the Free Software Foundation, either version 3 of the published by the Free Software Foundation, either
 version 3 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 details.

 You should have received a copy of the GNU General Public License along with this program.
 If not, see <http://www.gnu.org/licenses/>.
"""

"""
System Includes
"""
import os
import subprocess
import shlex
import ntpath
import numpy
import pathlib
import math

"""
Pip Packages
"""
import pandas as pd
from progressbar import ProgressBar, FormatLabel

"""
Local Includes
"""
from src.error_code import ErrorCodes
from src.constants import *


def _exec_cmd(command):
    """
    Executes a command on Shell and returns stdout and stderr from the command.
    :param command: the string of the command to be executed
    :return: stdout: standard output of command , stderr  standard error of command
    """
    command_array = shlex.split(command)
    out = subprocess.Popen([command_array],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    return stdout, stderr


class DLProfile(object):
    """
    DLProfile is a Deep Learning profiling tool.

    Methods
    -------
    load(darshan_file=None, preprocessed_dir="./temp_analysis")
        Loads the tool with the respective darshan file and processes the trace to load various structures for
        DLProfile tool.

    GetDXTAsDF()
        Returns the processed DXT Trace as a Pandas Dataframe.

    GetJobTime()
        Get the total execution time of the job

    GetIOTime(filepath=None, rank=None)
        Get the total time spent on I/O in the job

    GetIOSize(filepath=None, rank=None)
        Get the total I/O performed by the job in bytes

    GetAccessPattern(filepath=None)
        Get I/O access pattern per file within the job

    GetFileSizes(filepath=None)
        Get map of sizes of the files accessed by the job

    IOPerRank()
        Get the total I/O time per rank in an array

    CreateIOTimeline(filepath=None, rank=None, time_step=None)
        Create a timeline of execution with timesteps, number of operations, and total I/O performed in bytes

    GetIORequestDistribution(self, filepath=None, bins=100, threshold=AUTO)
        Get a histogram of request sizes overall or per-file (if passed).

    GetSummary()
        Get a dictionary of summary results from the DLProfiler.
    """

    def __init__(self):
        """
        Initializes the internal variables of DLProfiler.
        """
        self._loaded = False
        self._darshan_file = None
        self._errors = []
        self._darshan_bin_dir = None
        self._dl_profile_bin_path = None
        self._dxt_df = None
        self._file_access_pattern = None
        self._preprocessed_dir = None
        return

    """
    Private Functions
    """

    def _error_str(self):
        """
        Converts error into string concatenated by space.
        :return: concatenated error with space.
        """
        return ' '.join(self._errors)

    def _verify_env_path(self):
        """
        Verifies if the environment variable and darshan file exists.
        Environment Variables checked are DARSHAN_BIN_DIR and DLPROFILE_BIN_DIR
        DARSHAN_BIN_DIR : defines the bin directory of the darshan installation.
        DLPROFILE_BIN_DIR : defines the bin director of DLProfile installation.

        If returns false, the errors are appended to _errors class attribute
        :return: True if all variables are exists and False otherwise.
        """
        dxt_file = os.path.exists(self._darshan_file)
        darshan_path = True
        dlprofile_bin_path = True
        if DARSHAN_BIN_DIR not in os.environ:
            darshan_path = False
        else:
            darshan_path = os.path.exists(os.environ[DARSHAN_BIN_DIR])
        if DLPROFILE_BIN_DIR not in os.environ:
            dlprofile_bin_path = False
        else:
            dlprofile_bin_path = os.path.exists(os.environ[DLPROFILE_BIN_DIR])
        is_valid = True
        if not dxt_file:
            self._errors.append(str(ErrorCodes.EC1002))
            is_valid = False
        if not darshan_path:
            self._errors.append(str(ErrorCodes.EC1003))
            is_valid = False
        if not dlprofile_bin_path:
            self._errors.append(str(ErrorCodes.EC1004))
            is_valid = False
        if is_valid:
            self._darshan_bin_dir = darshan_path
            self._dl_profile_bin_path = dlprofile_bin_path
        return is_valid

    def _check_loaded(self):
        """
        Check if the Load() function was called.
        If it is called the internal attribute _loaded is set.
        :return: True, if Load() function was called, and False, otherwise.
        """
        if self._loaded:
            return True
        return False

    def _throw_if_not_loaded(self):
        """
        Throws an exception with Error Code 1001 if the Load() function is not called.
        :return: Exception if Load() not called.
        """
        if not self._check_loaded():
            raise Exception(str(ErrorCodes.EC1001))

    def _get_darshan_dxt_exe(self):
        """
        Returns path of the Darshan DXT parser executable
        :return: string of darshan-dxt-parser executable.
        """
        return "{}/darshan-dxt-parser".format(self._darshan_bin_dir)

    def _get_darshan_exe(self):
        """
        Returns path of the Darshan parser executable
        :return: string of darshan-parser executable.
        """
        return "{}/darshan-parser".format(self._darshan_bin_dir)

    def _get_darshan_convert_exe(self):
        """
        Returns path of the Darshan convert executable
        :return: string of darshan-convert executable.
        """
        return "{}/darshan-convert".format(self._darshan_bin_dir)

    def _get_darshan_job_summary_exe(self):
        """
        Returns path of the custom Darshan Job Summary script
        :return: string of darshan-job-summary.pl executable.
        """
        return "{}/darshan-job-summary.pl".format(self._dl_profile_bin_path)

    def _parse_dxt_trace(self):
        """
        Parses the dxt trace and creates a Pandas Dataframe out of it with all the DXT columns.
        The dataframe returned has the following columns:
            ['Module', 'Filename', 'Rank', 'Operation', 'Segment', 'Offset', 'Length', 'Start', 'End']
        :return: a dataframe of DXT values.
        """
        cmd = "{} {}".format(self._get_darshan_dxt_exe(), self._darshan_file)
        lines, stderr = _exec_cmd(cmd)
        io_lines = False
        df = pd.DataFrame(
            columns=['Module', 'Filename', 'Rank', 'Operation', 'Segment', 'Offset', 'Length', 'Start', 'End'])
        temp_filename = ""
        widgets = [FormatLabel('Processed: %(value)d lines of {} (in: %(elapsed)s)'.format(len(lines)))]
        pbar = ProgressBar(widgets=widgets)
        for line in pbar(lines):
            if line == '':
                io_lines = False
                continue
            elif "DXT, file_id" in line:
                temp_filename = line.split(" ")[5]
                io_lines = False
                continue
            elif "Module" in line:
                io_lines = True
            elif io_lines:
                # Module,Rank, Wt/Rd, Segment,Offset,Length,Start(s),End(s)
                vals = line.split()
                new_row = {'Module': vals[0],
                           'Filename': temp_filename,
                           'Rank': int(vals[1]),
                           'Operation': vals[2],
                           'Segment': int(vals[3]),
                           'Offset': int(vals[4]),
                           'Length': int(vals[5]),
                           'Start': float(vals[6]),
                           'End': float(vals[7])}
                df = df.append(new_row, ignore_index=True)
        return df

    def _pre_process_dxt_df(self):
        """
        Processes the DXT Dataframe and computes additional columns. Main transformations are:
            - Change Filename into categorical column.
            - Compute I/O time based on Start and End.
            - Compute per operation Bandwidth achieved.
            - Extract Extension of the filename.
            - Remove python files which were traced.
        :return: A Processed DXT dataframe
        """
        # make Filename categorical
        self._dxt_df["Filename"] = self._dxt_df["Filename"].astype('category')
        # Compute I/O time
        self._dxt_df['io_time'] = self._dxt_df['End'] - self._dxt_df['Start']
        # Default erroneous io_time
        self._dxt_df.loc[self._dxt_df['io_time'] == 0, 'io_time'] = 0.001
        # Compute I/O Bandwidth
        self._dxt_df['bandwidth'] = self._dxt_df['Length'] / self._dxt_df['io_time']
        # Compute ext
        self._dxt_df['ext'] = self._dxt_df.Filename.apply(lambda x: x.split('.')[-1])
        # If no extension then use TFRecord
        self._dxt_df.loc[self._dxt_df['Filename'].str.contains("\.") == False, 'ext'] = "tfrecord"
        # remove .py files
        self._dxt_df = self._dxt_df[~self._dxt_df['Filename'].str.contains("py")]
        return self._dxt_df

    def _analyze_access_pattern(self):
        """
        This function extracts file access pattern using the darshan utilities.
        It specifically uses a modified perl script called darshan_job_summary.pl
        which calculates the access pattern to be sequential, consecutive, or random.
        :return: a file map containing per file access pattern observed by darshan.
        """
        cmd = "{}  --file-list {} | egrep -v '^(#|$)' | cut -f 1-2 | sort -n | uniq | \
                while read -r hash filepath stuff ; do \
                echo $hash $filepath; \
                done".format(self._get_darshan_exe(), self._darshan_file)
        lines, stderr = _exec_cmd(cmd)
        file_hash_map = {}
        self._dxt_df['hash'] = 0
        widgets = [FormatLabel('Processed: %(value)d lines of {} (in: %(elapsed)s)'.format(len(lines)))]
        pbar = ProgressBar(widgets=widgets)
        for line in pbar(lines):
            vals = line.split()
            # print(vals)
            self._dxt_df.loc[self._dxt_df['Filename'] == vals[1], 'hash'] = vals[0]
            file_hash_map[vals[1]] = vals[0]
        pattern_file_map = {}
        for key in file_hash_map:
            hash_val = file_hash_map[key]
            if key.find(".py") == -1 and key.find("cpython") == -1 and key.find("STDOUT") == -1:
                file = os.path.splitext(ntpath.basename(key))[0]
                dest_file = "{}{}.darshan".format(self.preprocessing_dir_, file)
                if os.path.exists(dest_file):
                    os.remove(dest_file)
                cmd = "{} --file {} {} {}".format(self._get_darshan_convert_exe(), hash_val, self._darshan_file,
                                                  dest_file)
                output, stderr = _exec_cmd(cmd)
                cmd = "{} {} --verbose --output {}{}.pdf".format(self._get_darshan_job_summary_exe(),
                                                                 dest_file,
                                                                 self.preprocessing_dir_,
                                                                 file,
                                                                 self.preprocessing_dir_,
                                                                 file)

                lines, stderr = _exec_cmd(cmd)
                map_value = {"name": file,
                             "read": [0, 0, 0],
                             "write": [0, 0, 0],
                             "io_bytes": 0.0,
                             "io_time": 0.0}
                for line in lines:
                    if line.find("Total bytes read and written") == 0:
                        vals = line.split(": ")
                        map_value["io_bytes"] = int(vals[1])
                    elif line.find("Total absolute I/O time") == 0:
                        vals = line.split(": ")
                        map_value["io_time"] = float(vals[1])
                    elif line.find("Read") == 0:
                        vals = line.split(", ")
                        map_value["read"][0] = int(vals[1])
                        map_value["read"][1] = int(vals[2])
                        map_value["read"][2] = int(vals[3])
                    elif line.find("Write") == 0:
                        vals = line.split(", ")
                        map_value["write"][0] = int(vals[1])
                        map_value["write"][1] = int(vals[2])
                        map_value["write"][2] = int(vals[3])
                pattern_file_map[key] = map_value
        return pattern_file_map

    """
    Public Functions
    """

    def Load(self, darshan_file, preprocessed_dir="./temp_analysis"):
        """
        This functions bootstraps the DLProfiler with the given darshan filename
        :param darshan_file: Darshan's DXT trace file.
        :param preprocessed_dir: full path where post processing checkpoints can be made for faster loading.
        :return: Exception with Error code 1000, if darshan file is not passed.
                 Exception with Error code 1002, if darshan file is invalid.
                 Exception with Error code 1003, if environment variable DARSHAN_BIN_DIR is not set correctly.
                 Exception with Error code 1004, if environment variable DLPROFILE_BIN_DIR is not set correctly.
                 True, if loading was successful
        """
        if darshan_file is None:
            raise SystemExit(str(ErrorCodes.EC1000))
        self._darshan_file = darshan_file
        self._loaded = True
        self._preprocessed_dir = preprocessed_dir
        if not self._verify_env_path():
            raise SystemExit(self._error_str())
        if not os.path.exists(preprocessed_dir):
            os.mkdir(preprocessed_dir)
        io_df_filename = "{}/io_df.csv".format(preprocessed_dir)
        if not os.path.exists(io_df_filename):
            self._dxt_df = self._parse_dxt_trace()
            self._dxt_df.to_csv(index=False, path_or_buf=io_df_filename)
        else:
            self._dxt_df = pd.read_csv(io_df_filename)
        self._pre_process_dxt_df()
        self._file_access_pattern = self._analyze_access_pattern()
        return True

    def GetDXTAsDF(self):
        """
        Get the processed DXT traced as a Pandas Dataframe
        :return: Pandas Dataframe
        """
        self._throw_if_not_loaded()
        return self._dxt_df

    def GetJobTime(self):
        """
        Get the total time spent in the job.
        :return: time in seconds.
        """
        self._throw_if_not_loaded()
        cmd = "{} {} | \
        egrep 'run time' | \
        cut -d' ' -f4".format(self._get_darshan_dxt_exe(), self._darshan_file)
        return_val, errors = _exec_cmd(cmd)
        job_time = float(return_val[0])
        return job_time

    def GetIOTime(self, filepath=None, rank=None):
        """
        Returns the total time spent by job spent in I/O.
        If file path and rank are passed, data is further filtered.
        :param filepath: Filters data by filename
        :param rank: Filters data by rank
        :return: Returns I/O time in seconds.
        """
        self._throw_if_not_loaded()
        temp_df = self._dxt_df
        if filepath is not None:
            temp_df = temp_df[temp_df['Filename'].eq(filepath)]
        if rank is not None:
            temp_df = temp_df[temp_df['Rank'].eq(rank)]
        return self.temp_df['io_time'].sum()

    def GetIOSize(self, filepath=None, rank=None):
        """
        Returns the total I/O in bytes performed by job spent in I/O.
        If file path and rank are passed, data is further filtered.
        :param filepath: Filters data by filename
        :param rank: Filters data by rank
        :return: Returns I/O in bytes.
        """
        self._throw_if_not_loaded()
        temp_df = self._dxt_df
        if filepath is not None:
            temp_df = temp_df[temp_df['Filename'].eq(filepath)]
        if rank is not None:
            temp_df = temp_df[temp_df['Rank'].eq(rank)]
        return self.temp_df['Length'].sum()

    def GetAccessPattern(self, filepath=None):
        """
        Computes the file access pattern for the job.
        If filepath is passed data is further filtered.
        If not then all files are aggregated to get the overall access pattern.
        :param filepath: Filters data by filename
        :return: Returns Object of I/O access pattern for reads and write
        """
        self._throw_if_not_loaded()
        if filepath is not None:
            return {"read": {"total_ops": self._file_access_pattern[filepath]["read"][0],
                             "sequential": self._file_access_pattern[filepath]["read"][1],
                             "consecutive": self._file_access_pattern[filepath]["read"][2]
                             },
                    "write": {"total_ops": self._file_access_pattern[filepath]["write"][0],
                              "sequential": self._file_access_pattern[filepath]["write"][1],
                              "consecutive": self._file_access_pattern[filepath]["write"][2]
                              }
                    }
        else:
            total_seq = [0, 0]
            total_consecutive = [0, 0]
            total_ops = [0, 0]
            for key in self._file_access_pattern:
                total_ops[0] += int(self._file_access_pattern[key]["read"][0])
                total_seq[0] += int(self._file_access_pattern[key]["read"][1])
                total_consecutive[0] += int(self._file_access_pattern[key]["read"][2])
                total_ops[1] += int(self._file_access_pattern[key]["write"][0])
                total_seq[1] += int(self._file_access_pattern[key]["write"][1])
                total_consecutive[1] += int(self._file_access_pattern[key]["write"][2])
            total_ops = numpy.array(total_ops)
            total_seq = numpy.array(total_seq)
            total_consecutive = numpy.array(total_consecutive)
            return {"read": {"total_ops": total_ops[0],
                             "sequential": total_seq[0],
                             "consecutive": total_consecutive[0]
                             },
                    "write": {"total_ops": total_ops[1],
                              "sequential": total_seq[1],
                              "consecutive": total_consecutive[1]
                              }
                    }

    def GetFileSizes(self, filepath=None):
        """
        Get size of the files used in the job.
        :param filepath: Filters by filename
        :return: returns a map of filenames and size.
        """
        self._throw_if_not_loaded()
        if filepath is not None:
            file = self._file_access_pattern[filepath]
            size = pathlib.Path(file).stat().st_size
            return {filepath: size}
        else:
            file_size_map = {}
            for file in self._file_access_pattern:
                size = pathlib.Path(file).stat().st_size
                file = os.path.splitext(ntpath.basename(file))[0]
                file_size_map[file] = float(size)
            return file_size_map

    def IOPerRank(self):
        """
        Returns a array of I/O per Rank for the job.
        :return: a array of I/O per rank with index = rank
        """
        self._throw_if_not_loaded()
        io_time_array = []
        for rank in self._dxt_df['Rank'].unique():
            io_time_array.append(self.GetIOSize(rank=rank))
        return numpy.array(io_time_array)

    def CreateIOTimeline(self, filepath=None, rank=None, time_step=None):
        """
        Create a timeline for I/O where per timestep, we calculate number of operations and amount of I/O.
        if filepath is set, data is further filtered by filepath
        if rank is set, data is further filtered by rank
        if timestep is not set, it is auto tuned to be the mean io_time within the data
        :param filepath: filters the data by filename
        :param rank: filters the data by rank
        :param time_step: creates size of each timestep.
        :return: A dataframe consisting of ['timestep','operation_count','io_bytes']
        """
        self._throw_if_not_loaded()
        temp_df = self._dxt_df
        tm_df_filename = "{}/tm_df".format(self._preprocessed_dir)
        if filepath is not None:
            temp_df = temp_df[temp_df['Filename'].eq(filepath)]
            filename = os.path.splitext(ntpath.basename(filepath))[0]
            tm_df_filename = "{}_{}".format(tm_df_filename, filename)
        if rank is not None:
            temp_df = temp_df[temp_df['Rank'].eq(rank)]
            tm_df_filename = "{}_{}".format(tm_df_filename, rank)
        tm_df_filename = "{}.csv".format(tm_df_filename)
        min_time = round(0, 3)
        max_time = round(self.GetJobTime(), 3)
        if temp_df['End'].max() > max_time:
            max_time = temp_df['End'].max()
        if time_step is None:
            time_step = round(temp_df['io_time'].mean(), 3)
        data_points = math.ceil((max_time - min_time) / time_step)
        data_points_series = numpy.arange(0, data_points, 1)
        count_series = numpy.zeros(data_points)
        sum_series = numpy.zeros(data_points)
        widgets = [FormatLabel('Processed: %(value)d lines of {} (in: %(elapsed)s)'.format(temp_df.count()['Module']))]
        pbar = ProgressBar(widgets=widgets)
        for index, row in pbar(temp_df.iterrows()):
            start_index = math.floor(float(row['Start']) / time_step)
            end_index = math.ceil(float(row['End']) / time_step)
            # print(row['Start'],row['End'],start_index,end_index)
            for n in numpy.arange(start_index, end_index, 1):
                count_series[n] += 1
                sum_series[n] += float(row['Length'])
        df_time = pd.DataFrame(
            {'time_step': data_points_series, 'operation_count': count_series, 'io_bytes': sum_series})
        return df_time

    def GetIORequestDistribution(self, filepath=None, bins=100, threshold=AUTO):
        """
        Returns a 2d series of value counts for given bins of io sizes.
        if filepath is passed, data is filtered by filename
        bins decide the number of points on x axis of the histogram
        and threshold can be used to ignore points less than the given threshold.
        By default threshold is set to 1/1000 of the total sum of counts.
        :param filepath: filters the data by filepath
        :param bins: sets the bins for the histogram
        :param threshold: sets the threshold to ignore on histogram
        :return: a dataframe object which can be plotted using plot function.
        """
        self._throw_if_not_loaded()
        temp_df = self._dxt_df
        if filepath is not None:
            temp_df = temp_df[temp_df['Filename'].eq(filepath)]
        counts = temp_df['Length'].value_counts(bins=bins)
        if threshold is AUTO:
            threshold = temp_df['Length'].count() * .001
            counts = counts[counts > threshold]
        return counts.sort_index()

    def GetSummary(self):
        """
        Gets the overall summary of the job.
        Fields extracted are:
        - Total Job Execution Time: job_time (in seconds)
        - Total time spent on I/O: total_io_time (in seconds)
        - Total Bytes written/read by the job: total_io_bytes (in bytes)
        - Types of interfaces used: io_interface_used (array of all interfaces used)
        - Types of I/O operations: io_operations_used (read or write)
        - # of files operated upon: files_used
        - Number of ranks in the job: num_ranks
        - Data Transfer Size per operation: descriptive stats (min,max,mean, and median) (in bytes)
        - Data Transfer Bandwidth per operation: descriptive stats (min,max,mean, and median) (in bytes/s)
        - Overall Access Pattern:   sequential: An I/O op issued at an offset greater than where the previous I/O op ended. (%)
                                    consecutive: An I/O op issued at the offset immediately after the end of the previous I/O. (%)
                                    total_operations: total Operations (Count)
        - Summary of files used:    types: extensions of the file
                                    size: descriptive stats (min,max,mean, and median) (in bytes)
        :return: map of job summary.
        """
        self._throw_if_not_loaded()
        pattern = self.GetAccessPattern()
        total_ops = pattern["read"]["total_ops"] + pattern["write"]["total_ops"]
        total_seq = pattern["read"]["sequential"] + pattern["write"]["sequential"]
        total_cons = pattern["read"]["consecutive"] + pattern["write"]["consecutive"]
        file_size_map = self.GetFileSizes()
        file_sizes = []
        for key in file_size_map:
            file_sizes.append(file_size_map[key])
        file_sizes = numpy.array(file_sizes)
        return {
            "job_time": self.GetJobTime(),
            "total_io_time": self.IOPerRank().max(),
            "total_io_bytes": self.GetIOSize(),
            "io_interface_used": self._dxt_df['Module'].unique(),
            "io_operations_used": self._dxt_df['Operation'].unique(),
            "files_used": self._dxt_df["Filename"].unique(),
            "num_ranks": self._dxt_df["Rank"].nunique(),
            "data_transfer_size": {
                "min": self._dxt_df["Length"].min(),
                "max": self._dxt_df["Length"].max(),
                "mean": self._dxt_df["Length"].mean(),
                "median": self._dxt_df["Length"].median(),
            },
            "data_transfer_bandwidth": {
                "min": self._dxt_df["bandwidth"].min(),
                "max": self._dxt_df["bandwidth"].max(),
                "mean": self._dxt_df["bandwidth"].mean(),
                "median": self._dxt_df["bandwidth"].median(),
            },
            "access_pattern": {
                "total_operations": total_ops,
                "sequential": float(total_seq) * 100.0 / total_ops,
                "consecutive": float(total_cons) * 100.0 / total_ops
            },
            "file_used_summary": {
                "types": self._dxt_df['ext'].unique(),
                "size": {
                    "total": file_sizes.sum(),
                    "min": file_sizes.min(),
                    "max": file_sizes.max(),
                    "mean": file_sizes.mean(),
                    "median": file_sizes.median()
                }
            }
        }
