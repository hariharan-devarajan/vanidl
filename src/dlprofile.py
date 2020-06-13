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
import json

"""
Pip Packages
"""
import pandas as pd
import h5py
import tensorflow as tf

"""
Local Includes
"""
from src.error_code import ErrorCodes
from src.constants import *
from src.configuraton import *

"""
Global Methods
"""


def _exec_cmd(command):
    """
    Executes a command on Shell and returns stdout and stderr from the command.
    :param command: the string of the command to be executed
    :return: stdout: standard output of command , stderr  standard error of command
    """
    command_array = shlex.split(command)
    out = subprocess.Popen(command_array,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)

    stdout, stderr = out.communicate()
    lines_b = stdout.splitlines()
    lines = []
    for line in lines_b:
        lines.append(line.decode("utf-8"))
    return lines, stderr


def _exec_cmds(commands):
    """
    Executes a command on Shell and returns stdout and stderr from the command.
    :param commands: the string of the command to be executed
    :return: stdout: standard output of command , stderr  standard error of command
    """
    out = [None] * len(commands)
    prev = None

    for i, command in enumerate(commands):
        command_array = shlex.split(command)
        if i == 0:
            out[i] = subprocess.Popen(command_array,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT)
        else:
            out[i] = subprocess.Popen(command_array,
                                      stdin=out[i - 1].stdout,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT)
    #for i in range(len(commands)):
    #    stdout, stderr = out[i].communicate()
    stdout, stderr = out[len(commands) - 1].communicate()
    lines_b = stdout.splitlines()
    lines = []
    for line in lines_b:
        lines.append(line.decode("utf-8"))
    return lines, stderr


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    if count == 1:
        print("")
    print("\r[{}] {}% {} of {} {} ".format(bar, percents, count, total, status), end='')
    if count == total:
        print("")
    os.sys.stdout.flush()


class DLProfile(object):
    """
    DLProfile is a Deep Learning profiling tool.

    Methods
    -------
    load(darshan_file=None, preprocessed_dir="./temp_analysis", data_paths_include=[])
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

    GetIOPerRank()
        Get the total I/O time per rank in an array

    CreateIOTimeline(filepath=None, rank=None, time_step=None)
        Create a timeline of execution with timesteps, number of operations, and total I/O performed in bytes

    GetIORequestDistribution(self, filepath=None, rank=None, bins=100, threshold=AUTO)
        Get a histogram of request sizes overall or per-file or per-rank (if passed).

    GetSummary()
        Get a dictionary of summary results from the DLProfiler.

    GetHDF5FileSummary(filepath, only_special_summary=False)
        Get a summary of HDF5 File

    GetTFRecordSummary(filepath, features, only_special_summary=False)
        Get a summary of TFRecord File

    GetFileSummary(self, filepath, ext=UNKNOWN, tf_record_features=[])
        Get a summary of File
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
        self._tf_features = None
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
        if DARSHAN_DIR not in os.environ:
            darshan_path = False
        else:
            darshan_path = os.path.exists("{}/bin".format(os.environ[DARSHAN_DIR]))
        if DLPROFILE_DIR not in os.environ:
            dlprofile_bin_path = False
        else:
            dlprofile_bin_path = os.path.exists("{}/bin".format(os.environ[DLPROFILE_DIR]))
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
            self._darshan_bin_dir = "{}/bin".format(os.environ[DARSHAN_DIR])
            self._dl_profile_bin_path = "{}/bin".format(os.environ[DLPROFILE_DIR])
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
        pb_total = len(lines)
        df = pd.DataFrame(index=numpy.arange(pb_total),
                          columns=['Module', 'Filename', 'Rank', 'Operation', 'Segment', 'Offset', 'Length', 'Start',
                                   'End'])
        temp_filename = ""
        i = 1
        index = 0
        for line in lines:
            if i % 100 == 0 or i == pb_total:
                progress(i, pb_total, status='Parsing DXT File')
            i += 1
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
                df.loc[index] = {'Module': vals[0],
                                 'Filename': temp_filename,
                                 'Rank': int(vals[1]),
                                 'Operation': vals[2],
                                 'Segment': int(vals[3]),
                                 'Offset': int(vals[4]),
                                 'Length': int(vals[5]),
                                 'Start': float(vals[6]),
                                 'End': float(vals[7])}
                index += 1
        df = df.drop(df.index[index:])
        return df

    def _pre_process_dxt_df(self, data_paths_include):
        """
        Processes the DXT Dataframe and computes additional columns. Main transformations are:
            - Change Filename into categorical column.
            - Compute I/O time based on Start and End.
            - Compute per operation Bandwidth achieved.
            - Extract Extension of the filename.
            - Remove python files which were traced.
        :param: data_paths_include: paths to include
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
        # If no extension then use blank
        self._dxt_df.loc[self._dxt_df['Filename'].str.contains("\.") == False, 'ext'] = ""
        # remove .py files
        self._dxt_df = self._dxt_df[~self._dxt_df['Filename'].str.contains("py")]
        if len(data_paths_include) > 0:
            #print(len(data_paths_include))
            for data_path in data_paths_include:
                self._dxt_df = self._dxt_df[self._dxt_df['Filename'].str.contains(data_path)]
        return self._dxt_df

    def _analyze_access_pattern(self):
        """
        This function extracts file access pattern using the darshan utilities.
        It specifically uses a modified perl script called darshan_job_summary.pl
        which calculates the access pattern to be sequential, consecutive, or random.
        :return: a file map containing per file access pattern observed by darshan.
        """
        cmds = ["{}  --file-list {}".format(self._get_darshan_exe(),
                                            self._darshan_file),
                "egrep -v '^(#|$)'",
                "cut -f 1-2",
                "sort -n",
                "uniq"]
        lines, stderr = _exec_cmds(cmds)
        file_hash_map = {}
        self._dxt_df['hash'] = 0
        pb_total = len(lines)
        i = 1
        for line in lines:
            if i % 100 == 0 or i == pb_total:
                progress(i, pb_total, status='Analyzing Access Pattern')
            i += 1
            vals = line.split()
            #print(vals)
            if vals[1] in self._dxt_df['Filename'].unique():
                self._dxt_df.loc[self._dxt_df['Filename'] == vals[1], 'hash'] = vals[0]
                file_hash_map[vals[1]] = vals[0]
        pattern_file_map = {}
        pb_total = len(lines)
        i = 1
        #print(file_hash_map)
        for key in file_hash_map:
            if i % 100 == 0 or i == pb_total:
                progress(i, pb_total, status='Running Darshan Perl Script')
            i += 1
            hash_val = file_hash_map[key]
            if key.find(".py") == -1 and key.find("cpython") == -1 and key.find("STDOUT") == -1:
                file = os.path.splitext(ntpath.basename(key))[0]
                dest_file = "{}/{}.darshan".format(self._preprocessed_dir, file)
                if os.path.exists(dest_file):
                    os.remove(dest_file)
                cmd = "{} --file {} {} {}".format(self._get_darshan_convert_exe(), hash_val, self._darshan_file,
                                                  dest_file)
                output, stderr = _exec_cmd(cmd)
                cmd = "perl {} {} --verbose --output {}/{}.pdf".format(self._get_darshan_job_summary_exe(),
                                                                  dest_file,
                                                                  self._preprocessed_dir,
                                                                  file,
                                                                  self._preprocessed_dir,
                                                                  file)

                lines, stderr = _exec_cmd(cmd)
                #print(lines)
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

    def _parse_tf_record(self, example_proto):
        return tf.io.parse_single_example(example_proto, self._tf_features)

    def _explore_hdf5(self, h5object, name):
        """
        Explores the hdf5 file hierarchically and retrieves all dataset information
        Parameters
        ----------
        h5object: actual h5 object
        name: name for the object

        Returns
        -------
        map of information about the hdf5 object.
        """
        is_dataset = isinstance(h5object, h5py.Dataset)
        is_group = isinstance(h5object, h5py.Group)
        is_file = isinstance(h5object, h5py.File)

        if is_group:
            group_map = {"type": "group",
                         "name": name}
            key_maps = []
            for key in h5object.keys():
                key_map = self._explore_hdf5(h5object[key], key)
                key_maps.append(key_map)
            group_map["keys"] = key_maps
            return group_map
        elif is_file:
            file_map = {"type": "file",
                        "name": name}
            key_maps = []
            for key in h5object.keys():
                key_map = self._explore_hdf5(h5object[key], key)
                key_maps.append(key_map)
            file_map["keys"] = key_maps
            return file_map
        elif is_dataset:
            dataset_map = {"type": "dataset",
                           "name": name,
                           "size": h5object.size,
                           "shape": h5object.shape,
                           "obj": h5object}
            return dataset_map
        else:
            return None

    """
    Public Functions
    """

    def Load(self, darshan_file, preprocessed_dir="/tmp/temp_analysis", data_paths_include=[]):
        """
        This functions bootstraps the DLProfiler with the given darshan filename

        Parameters
        ----------
        :param darshan_file: Darshan's DXT trace file.
        :param preprocessed_dir: full path where post processing checkpoints can be made for faster loading.
        :param data_paths_include: paths to include for I/O Analysis
        :return: Exception with Error code 1000, if darshan file is not passed.
                 Exception with Error code 1002, if darshan file is invalid.
                 Exception with Error code 1003, if environment variable DARSHAN_BIN_DIR is not set correctly.
                 Exception with Error code 1004, if environment variable DLPROFILE_BIN_DIR is not set correctly.
                 True, if loading was successful
        """
        if darshan_file is None:
            raise SystemExit(str(ErrorCodes.EC1000))
        self._darshan_file = darshan_file
        self._preprocessed_dir = preprocessed_dir
        if not self._verify_env_path():
            return False
        if not os.path.exists(preprocessed_dir):
            os.mkdir(preprocessed_dir)
        filename = os.path.splitext(ntpath.basename(darshan_file))[0]
        io_df_filename = "{}/{}_io_df.csv".format(preprocessed_dir,filename)
        if not os.path.exists(io_df_filename):
            self._dxt_df = self._parse_dxt_trace()
            self._dxt_df.to_csv(index=False, path_or_buf=io_df_filename)
        else:
            self._dxt_df = pd.read_csv(io_df_filename)
            print("Loaded Pre-processed DXT from file: {}".format(io_df_filename))
        self._pre_process_dxt_df(data_paths_include)
        pattern_json = "{}/{}_pattern.json".format(preprocessed_dir,filename)
        if not os.path.exists(pattern_json):
            self._file_access_pattern = self._analyze_access_pattern()
            with open(pattern_json, 'w') as outfile:
                json.dump(self._file_access_pattern, outfile)
        else:
            with open(pattern_json) as json_file:
                self._file_access_pattern = json.load(json_file)
            print("Loaded Pre-processed Pattern file: {}".format(pattern_json))
        self._errors = []
        self._loaded = True
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
        cmds = ["{} {}".format(self._get_darshan_dxt_exe(), self._darshan_file),
                "egrep 'run time'",
                "cut -d' ' -f4"]
        return_val, stderr = _exec_cmds(cmds)
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
        return temp_df['io_time'].sum()

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
        return temp_df['Length'].sum()

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
            if not os.path.exists(filepath):
                print("file not found {}".format(filepath))
                raise SystemExit(str(ErrorCodes.EC1009))
            file = self._file_access_pattern[filepath]
            size = pathlib.Path(filepath).stat().st_size
            return {filepath: size}
        else:
            file_size_map = {}
            for file in self._file_access_pattern:
                if not os.path.exists(file):
                    print("file not found {}".format(file))
                    raise SystemExit(str(ErrorCodes.EC1009).format(file))
                size = pathlib.Path(file).stat().st_size
                file = os.path.splitext(ntpath.basename(file))[0]
                file_size_map[file] = float(size)
            return file_size_map

    def GetIOPerRank(self):
        """
        Returns a array of I/O per Rank for the job.
        :return: a array of I/O per rank with index = rank
        """
        self._throw_if_not_loaded()
        io_time_array = []
        for rank in self._dxt_df['Rank'].unique():
            io_time_array.append(self.GetIOTime(rank=rank))
        return numpy.array(io_time_array)

    def CreateIOTimeline(self, filepath=None, rank=None, time_step=None, save=True):
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
        trace_filename = os.path.splitext(ntpath.basename(self._darshan_file))[0]
        tm_df_filename = "{}/{}_tm_df".format(self._preprocessed_dir, trace_filename)
        if filepath is not None:
            temp_df = temp_df[temp_df['Filename'].eq(filepath)]
            filename = os.path.splitext(ntpath.basename(filepath))[0]
            tm_df_filename = "{}_{}".format(tm_df_filename, filename)
        if rank is not None:
            temp_df = temp_df[temp_df['Rank'].eq(rank)]
            tm_df_filename = "{}_{}".format(tm_df_filename, rank)
        tm_df_filename = "{}.csv".format(tm_df_filename)
        if os.path.exists(tm_df_filename):
            df_time = pd.read_csv(tm_df_filename)
            print("Loaded Pre-processed Timeline from file: {}".format(tm_df_filename))
            return df_time
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
        pb_total = temp_df.count()['Module'];
        i = 1
        for index, row in temp_df.iterrows():
            if i % 100 == 0 or i == pb_total:
                progress(i, pb_total, status='Creating Timeline')
            i += 1
            start_index = math.floor(float(row['Start']) / time_step)
            end_index = math.ceil(float(row['End']) / time_step)
            # print(row['Start'],row['End'],start_index,end_index)
            for n in numpy.arange(start_index, end_index, 1):
                count_series[n] += 1
                sum_series[n] += float(row['Length'])
        df_time = pd.DataFrame(
            {'time_step': data_points_series, 'operation_count': count_series, 'io_bytes': sum_series})
        if save:
            df_time.to_csv(index=False, path_or_buf=tm_df_filename)
        return df_time

    def GetIORequestDistribution(self, filepath=None, rank=None, bins=100, threshold=AUTO):
        """
        Returns a 2d series of value counts for given bins of io sizes.
        if filepath is passed, data is filtered by filename
        bins decide the number of points on x axis of the histogram
        and threshold can be used to ignore points less than the given threshold.
        By default threshold is set to 1/1000 of the total sum of counts.
        :param filepath: filters the data by filepath
        :param rank: filters the data by rank
        :param bins: sets the bins for the histogram
        :param threshold: sets the threshold to ignore on histogram
        :return: a dataframe object which can be plotted using plot function.
        """
        self._throw_if_not_loaded()
        temp_df = self._dxt_df
        if filepath is not None:
            temp_df = temp_df[temp_df['Filename'].eq(filepath)]
        if rank is not None:
            temp_df = temp_df[temp_df['Rank'].eq(rank)]
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
            "total_io_time": self.GetIOPerRank().max(),
            "total_io_bytes": self.GetIOSize(),
            "io_interface_used": self._dxt_df['Module'].unique(),
            "io_operations_used": self._dxt_df['Operation'].unique(),
            "files_used": self._dxt_df["Filename"].unique().tolist(),
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
                    "median": numpy.median(file_sizes)
                }
            }
        }

    def GetHDF5FileSummary(self, filepath, only_special_summary=False):
        """
        Create a summary of the HDF5 file.
        General summary includes:
            - path: full path of the file
            - filename: name of the file
            - size: size of file in bytes
            - ext: format of the file
            - io_time: time spent by job in performing I/O on this file (in seconds)
            - io_size: amount of I/O (in bytes) performed on this file
            - special: A special summary is generate for HDF5 and TFRecord dataset
                - for HDF5
                    - a map of hiearchical structure of the file with dataset information
                        - name: Name of the dataset
                        - size: Size of the dataset
                        - shape: shape of the dataset
                        - obj: hdf5 dataset object for future processing
        Parameters
        ----------
        filepath: full path of the file.
        only_special_summary: if set to true only special summary is returned

        Returns
        -------
        map of summary of file
        """
        self._throw_if_not_loaded()
        if filepath is None:
            raise Exception(str(ErrorCodes.EC1005))
        if not os.path.exists(filepath):
            raise SystemExit(str(ErrorCodes.EC1009))
        file_ext_array = os.path.splitext(ntpath.basename(filepath))
        filename = file_ext_array[0]
        file_ext = filepath.split('.')[-1]
        if file_ext == filename or file_ext != 'h5':
            raise Exception(str(ErrorCodes.EC1006))
        file_obj = h5py.File(filepath, "r")
        special = self._explore_hdf5(file_obj, filename)
        if only_special_summary:
            return special
        else:
            temp_df = self._dxt_df[self._dxt_df['Filename'].eq(filepath)]
            file_size = pathlib.Path(filepath).stat().st_size
            return {
                "path": filepath,
                "filename": filename,
                "size": file_size,
                "ext": file_ext,
                "io_time": temp_df['io_time'].sum(),
                "io_size": temp_df['Length'].sum(),
                "special": special
            }

    def GetTFRecordSummary(self, filepath, features, only_special_summary=False):
        """
        Create a summary of TFRecord file.
        General summary includes:
            - path: full path of the file
            - filename: name of the file
            - size: size of file in bytes
            - ext: format of the file
            - io_time: time spent by job in performing I/O on this file (in seconds)
            - io_size: amount of I/O (in bytes) performed on this file
            - special: A special summary is generate for HDF5 and TFRecord dataset
                - for TFRecord:
                    - Input: tf_record_features is required.
                    - an list of processed records based on the features passed.
        Parameters
        ----------
        filepath: full path of the file.
        features: features to read TFRecord file
        only_special_summary: if set to true only special summary is returned

        Returns
        -------
        map of summary of TFRecord file
        """
        self._throw_if_not_loaded()
        if filepath is None:
            raise Exception(str(ErrorCodes.EC1005))
        if not os.path.exists(filepath):
            raise SystemExit(str(ErrorCodes.EC1009))
        file_ext_array = os.path.splitext(ntpath.basename(filepath))
        filename = file_ext_array[0]
        file_ext = filepath.split('.')[-1]
        if file_ext == filename or file_ext != 'tfrecord':
            raise Exception(str(ErrorCodes.EC1007))
        filenames = [filepath]
        raw_dataset = tf.data.TFRecordDataset(filenames)
        if len(features) == 0:
            raise Exception(str(ErrorCodes.EC1008))
        self._tf_features = features
        special = raw_dataset.map(self._parse_tf_record, num_parallel_calls=POOL_SIZE)
        if only_special_summary:
            return special
        else:
            temp_df = self._dxt_df[self._dxt_df['Filename'].eq(filepath)]
            file_size = pathlib.Path(filepath).stat().st_size
            return {
                "path": filepath,
                "filename": filename,
                "size": file_size,
                "ext": file_ext,
                "io_time": temp_df['io_time'].sum(),
                "io_size": temp_df['Length'].sum(),
                "special": special
            }

    def GetFileSummary(self, filepath, ext=UNKNOWN, tf_record_features=[]):
        """
        Create a summary of the file.
        General summary includes:
            - path: full path of the file
            - filename: name of the file
            - size: size of file in bytes
            - ext: format of the file
            - io_time: time spent by job in performing I/O on this file (in seconds)
            - io_size: amount of I/O (in bytes) performed on this file
            - special: A special summary is generate for HDF5 and TFRecord dataset
                - for HDF5
                    - a map of hiearchical structure of the file with dataset information
                        - name: Name of the dataset
                        - size: Size of the dataset
                        - shape: shape of the dataset
                        - obj: hdf5 dataset object for future processing
                - for TFRecord:
                    - Input: tf_record_features is required.
                    - an list of processed records based on the features passed.
        Parameters
        ----------
        filepath: full path of the file.
        ext: recommended format of the file (Supported are h5 and tfrecord).
        tf_record_features: if ext is tfrecord then tf_record_features are required.

        Returns
        -------
        map of summary of file
        """
        self._throw_if_not_loaded()
        if filepath is None:
            raise Exception(str(ErrorCodes.EC1005))
        if not os.path.exists(filepath):
            raise SystemExit(str(ErrorCodes.EC1009))
        temp_df = self._dxt_df[self._dxt_df['Filename'].eq(filepath)]
        file_ext_array = os.path.splitext(ntpath.basename(filepath))
        filename = file_ext_array[0]
        if ext == UNKNOWN:
            file_ext = filepath.split('.')[-1]
        else:
            file_ext = ext
        special_summary = {}
        if file_ext == filename:
            file_ext = ""
        elif file_ext == 'h5':
            special_summary = self.GetHDF5FileSummary(filepath, only_special_summary=True)
        elif file_ext == 'tfrecord':
            if len(tf_record_features) == 0:
                raise Exception(str(ErrorCodes.EC1008))
            special_summary = self.GetTFRecordSummary(filepath, tf_record_features, only_special_summary=True)
        file_size = pathlib.Path(filepath).stat().st_size
        return {
            "path": filepath,
            "filename": filename,
            "size": file_size,
            "ext": file_ext,
            "io_time": temp_df['io_time'].sum(),
            "io_size": temp_df['Length'].sum(),
            "special": special_summary
        }
