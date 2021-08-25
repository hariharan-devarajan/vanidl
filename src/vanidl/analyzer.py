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
import glob
import gzip

"""
Pip Packages
"""
import pandas as pd
"""
Local Includes
"""
from vanidl.error_code import ErrorCodes
from vanidl.constants import *
from vanidl.configuraton import *

"""
Global Methods
"""

"""
 Copyright (C) 2020  Argonne, Hariharan Devarajan <hdevarajan@anl.gov>
 This file is part of VaniDL
 HFetch is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as
 published by the Free Software Foundation, either version 3 of the published by the Free Software Foundation, either
 version 3 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 details.

 You should have received a copy of the GNU General Public License along with this program.
 If not, see <http://www.gnu.org/licenses/>.
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
    # for i in range(len(commands)):
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


class VaniDL(object):
    """
    VaniDL is a Deep Learning profiling tool.
    """

    def __init__(self):
        """
        Initializes the internal variables of VaniDLr.
        """
        self._loaded = False
        self._darshan_file = None
        self._errors = []
        self._darshan_bin_dir = None
        self._dxt_df = None
        self._df = None
        self._file_access_pattern = None
        self._preprocessed_dir = None
        self._tf_features = None
        self._tb_input_pipeline = None
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
        Environment Variables checked are DARSHAN_BIN_DIR and VaniDL_BIN_DIR
        DARSHAN_BIN_DIR : defines the bin directory of the darshan installation.
        VaniDL_BIN_DIR : defines the bin director of VaniDL installation.

        If returns false, the errors are appended to _errors class attribute
        :return: True if all variables are exists and False otherwise.
        """
        dxt_file = os.path.exists(self._darshan_file)
        darshan_path = True
        if DARSHAN_DIR not in os.environ:
            darshan_path = False
        else:
            darshan_path = os.path.exists("{}/bin".format(os.environ[DARSHAN_DIR]))
        is_valid = True
        if not dxt_file:
            self._errors.append(str(ErrorCodes.EC1002))
            is_valid = False
        if not darshan_path:
            self._errors.append(str(ErrorCodes.EC1003))
            is_valid = False
        if is_valid:
            self._darshan_bin_dir = "{}/bin".format(os.environ[DARSHAN_DIR])
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

    def _parse_trace(self):
        """
        Parses the darshan trace to get aggregate values. This is useful when we want aggregate values only.

        Returns
        -------
        dataframe with values of darshan counters.
        """
        cmds = ["{} {}".format(self._get_darshan_exe(), self._darshan_file),
                "egrep -v '^(#|$)'"]
        lines, stderr = _exec_cmds(cmds)
        darshan_map = {}
        i = 1
        pb_total = len(lines)
        for line in lines:
            if i % 100 == 0 or i == pb_total:
                progress(i, pb_total, status='Parsing Darshan File')
            i += 1
            values = line.split()
            if values[2] not in darshan_map:
                # create a new record
                darshan_map[values[2]] = {}
                darshan_map[values[2]]["Module"] = values[0]
                darshan_map[values[2]]["Rank"] = values[1]
                darshan_map[values[2]][values[3]] = values[4]
                darshan_map[values[2]]["Filename"] = values[5]
                darshan_map[values[2]]["Mount"] = values[6]
                darshan_map[values[2]]["FS"] = values[7]
            else:
                # update existing
                darshan_map[values[2]][values[3]] = values[4]
        df = pd.DataFrame.from_dict(darshan_map, orient='index')
        return df

    def _parse_dxt_trace(self):
        """
        Parses the dxt trace and creates a Pandas Dataframe out of it with all the DXT columns.
        The dataframe returned has the following columns:
            ['Module', 'Filename', 'Rank', 'Operation', 'Segment', 'Offset', 'Length', 'Start', 'End']
        :return: a dataframe of DXT values.
        """
        cmd = "{} --show-incomplete {}".format(self._get_darshan_dxt_exe(), self._darshan_file)
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

    def _pre_process_df(self, data_paths_include):
        """
        Processes the DXT Dataframe and computes additional columns. Main transformations are:
            - Change Filename into categorical column.
            - Compute I/O time based on Start and End.
            - Compute per operation Bandwidth achieved.
            - Extract Extension of the filename.
            - Remove python files which were traced.
        :param: data_paths_include: paths to include
        """
        # make Filename categorical
        self._df = self._df.fillna('0')
        if "POSIX" in self._df['Module'].unique():
            self._df = self._df.astype({'POSIX_READS': 'int64',
                                        'POSIX_SEQ_READS': 'int64',
                                        'POSIX_CONSEC_READS': 'int64',
                                        'POSIX_WRITES': 'int64',
                                        'POSIX_SEQ_WRITES': 'int64',
                                        'POSIX_CONSEC_WRITES': 'int64',
                                        'POSIX_BYTES_WRITTEN': 'int64',
                                        'POSIX_BYTES_READ': 'int64',
                                        'POSIX_F_WRITE_TIME': 'float64',
                                        'POSIX_F_READ_TIME': 'float64',
                                        'POSIX_F_META_TIME': 'float64'
                                        }, errors='ignore')
        if "MPIIO" in self._df['Module'].unique():
            self._df = self._df.astype({'MPIIO_BYTES_READ': 'int64',
                                        'MPIIO_BYTES_WRITTEN': 'int64',
                                        'MPIIO_F_READ_TIME': 'float64',
                                        'MPIIO_F_WRITE_TIME': 'float64',
                                        'MPIIO_F_META_TIME': 'float64'
                                        }, errors='ignore')
        if "STDIO" in self._df['Module'].unique():
            self._df = self._df.astype({'STDIO_BYTES_READ': 'int64',
                                        'STDIO_BYTES_WRITTEN': 'int64',
                                        'STDIO_F_READ_TIME': 'float64',
                                        'STDIO_F_WRITE_TIME': 'float64',
                                        'STDIO_F_META_TIME': 'float64'
                                        }, errors='ignore')
        if "H5D" in self._df['Module'].unique():
            self._df = self._df.astype({'H5D_BYTES_READ': 'int64',
                                        'H5D_BYTES_WRITTEN': 'int64',
                                        'H5D_F_READ_TIME': 'float64',
                                        'H5D_F_WRITE_TIME': 'float64',
                                        'H5D_F_META_TIME': 'float64'
                                        }, errors='ignore')
        self._dxt_df["io_time"] = 0
        self._dxt_df["bandwidth"] = 0
        self._dxt_df["ext"] = 0
        if self._dxt_df['Module'].count() > 0:
            self._dxt_df["Filename"] = self._dxt_df["Filename"].astype('category')
            self._dxt_df['io_time'] = self._dxt_df['End'] - self._dxt_df['Start']
            self._dxt_df.loc[self._dxt_df['io_time'] == 0, 'io_time'] = 0.001
            self._dxt_df['bandwidth'] = self._dxt_df['Length'] / self._dxt_df['io_time']
            self._dxt_df['ext'] = self._dxt_df.Filename.apply(lambda x: x.split('.')[-1])
            self._dxt_df = self._dxt_df[~self._dxt_df['Filename'].str.contains(".py",regex=False)]

        self._df["Filename"] = self._df["Filename"].astype('category')
        self._df['ext'] = self._df.Filename.apply(lambda x: x.split('.')[-1])
        # remove .py files
        self._df = self._df[~self._df['Filename'].str.contains(".py",regex=False)]
        self._df = self._df[~self._df['Filename'].str.contains("<STDERR>")]
        self._df = self._df[~self._df['Filename'].str.contains("<STDOUT>")]
        if len(data_paths_include) > 0:
            # print(len(data_paths_include))
            for data_path in data_paths_include:
                self._dxt_df = self._dxt_df[self._dxt_df['Filename'].str.contains(data_path)]
                self._df = self._df[self._df['Filename'].str.contains(data_path)]

    def _analyze_access_pattern(self):
        """
        This function extracts file access pattern using the darshan utilities.
        It specifically uses a modified perl script called darshan_job_summary.pl
        which calculates the access pattern to be sequential, consecutive, or random.
        :return: a file map containing per file access pattern observed by darshan.
        """
        pattern_file_map = {}
        for index, row in self._df.iterrows():
            file = os.path.splitext(ntpath.basename(row['Filename']))[0]
            if row['Module'] == "POSIX":
                pattern_file_map[row['Filename']] = {"name": file,
                                                     "read": [int(row['POSIX_READS']),
                                                              int(row['POSIX_SEQ_READS']),
                                                              int(row['POSIX_CONSEC_READS'])],
                                                     "write": [int(row['POSIX_WRITES']),
                                                               int(row['POSIX_SEQ_WRITES']),
                                                               int(row['POSIX_CONSEC_WRITES'])],
                                                     "io_bytes": int(row['POSIX_BYTES_WRITTEN']) +
                                                                 int(row['POSIX_BYTES_READ']),
                                                     "io_time": float(row['POSIX_F_WRITE_TIME']) +
                                                                float(row['POSIX_F_READ_TIME'])}
            elif row['Module'] == "MPIIO":
                pattern_file_map[row['Filename']] = {"name": file,
                                                     "read": [0, 0, 0],
                                                     "write": [0, 0, 0],
                                                     "io_bytes": int(row['MPIIO_BYTES_READ']) +
                                                                 int(row['MPIIO_BYTES_WRITTEN']),
                                                     "io_time": float(row['MPIIO_F_READ_TIME']) +
                                                                float(row['MPIIO_F_WRITE_TIME'])}
            elif row['Module'] == "STDIO":
                pattern_file_map[row['Filename']] = {"name": file,
                                                     "read": [0, 0, 0],
                                                     "write": [0, 0, 0],
                                                     "io_bytes": int(row['STDIO_BYTES_READ']) +
                                                                 int(row['STDIO_BYTES_WRITTEN']),
                                                     "io_time": float(row['STDIO_F_READ_TIME']) +
                                                                float(row['STDIO_F_WRITE_TIME'])}
            elif row['Module'] == "H5D":
                pattern_file_map[row['Filename']] = {"name": file,
                                                     "read": [0, 0, 0],
                                                     "write": [0, 0, 0],
                                                     "io_bytes": int(row['H5D_BYTES_READ']) +
                                                                 int(row['H5D_BYTES_WRITTEN']),
                                                     "io_time": float(row['H5D_F_READ_TIME']) +
                                                                float(row['H5D_F_WRITE_TIME'])}
        return pattern_file_map

    def _parse_tf_record(self, example_proto):
        import tensorflow as tf
        return tf.io.parse_single_example(example_proto, self._tf_features)

    def _explore_hdf5(self, h5object, name):
        import h5py
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

    def _parse_tb_logs(self, tensorflow_logs_dir):
        """
        :param tensorflow_logs_dir, log directory for tensorboard logs.
        :return JSON of IPA
        """
        from tensorboard_plugin_profile.convert import input_pipeline_proto_to_gviz 
        from tensorboard_plugin_profile.protobuf import input_pipeline_pb2
        fileExt = "*input_pipeline.pb"
        input_pipeline_files = list(pathlib.Path(tensorflow_logs_dir).rglob(fileExt))
        ipa_hosts = {}
        for file in input_pipeline_files:
            filename = os.path.basename(file)
            hostname = filename.split(".")[0]
            ipa_hosts[hostname] = {}
            ipa_hosts[hostname]['log_file'] = str(file)
        pb_total = len(ipa_hosts.keys())
        i = 1
        for key in ipa_hosts.keys():
            if i % 100 == 0 or i == pb_total:
                progress(i, pb_total, status='Reading TB IPA log file')
            i += 1
            ipa = input_pipeline_pb2.InputPipelineAnalysisResult()
            f = open(ipa_hosts[key]['log_file'], "rb")
            ipa.ParseFromString(f.read())
            (table_description, data, custom_properties) = input_pipeline_proto_to_gviz.get_step_breakdown_table_args(
                ipa)
            # print(custom_properties)
            # print(table_description)
            ipa_hosts[key]['step_data'] = {}
            ipa_hosts[key]['step_data']['data'] = []
            ipa_hosts[key]['step_data']['custom_properties'] = custom_properties
            for index, step_data_val in enumerate(data):
                step_data = {}
                # print(step_data_val)
                step_data['stepnum'] = int(step_data_val[0])
                step_data['deviceComputeTimeMs'] = float(step_data_val[1])
                step_data['deviceToDeviceTimeMs'] = float(step_data_val[2])
                step_data['hostComputeTimeMs'] = float(step_data_val[3])
                step_data['kernelLaunchTimeMs'] = float(step_data_val[4])
                step_data['infeedTimeMs'] = float(step_data_val[5])
                step_data['hostComputeTimeMs'] = float(step_data_val[6])
                step_data['outfeedTimeMs'] = float(step_data_val[7])
                step_data['compileTimeMs'] = float(step_data_val[8])
                ipa_hosts[key]['step_data']['data'].append(step_data)
            (table_description, data, custom_properties) = input_pipeline_proto_to_gviz.get_input_op_table_args(ipa)
            ipa_hosts[key]['op_data'] = {}
            ipa_hosts[key]['op_data']['data'] = []
            ipa_hosts[key]['op_data']['custom_properties'] = custom_properties
            for index, op_data_val in enumerate(data):
                op_data = {}
                # print(step_data_val)
                op_data['opName'] = op_data_val[0]
                op_data['count'] = int(op_data_val[1])
                op_data['timeInMs'] = float(op_data_val[2])
                op_data['timeInPercent'] = float(op_data_val[3])
                op_data['selfTimeInMs'] = float(op_data_val[4])
                op_data['selfTimeInPercent'] = float(op_data_val[5])
                op_data['category'] = op_data_val[6]
                ipa_hosts[key]['op_data']['data'].append(op_data)
        self._tb_input_pipeline = ipa_hosts
        return self._tb_input_pipeline

    """
    Public Functions
    """

    def Load(self, darshan_file, preprocessed_dir="/tmp/temp_analysis", data_paths_include=[],
             tensorflow_logs_dir=None):
        """
        This functions bootstraps the VaniDLr with the given darshan filename

        Parameters
        ----------
        :param darshan_file: Darshan's DXT trace file.
        :param preprocessed_dir: full path where post processing checkpoints can be made for faster loading.
        :param data_paths_include: paths to include for I/O Analysis.
        :param tensorflow_logs_dir: directory where tensorflow logs are present.
        :return: Exception with Error code 1000, if darshan file is not passed.
                 Exception with Error code 1002, if darshan file is invalid.
                 Exception with Error code 1003, if environment variable DARSHAN_BIN_DIR is not set correctly.
                 Exception with Error code 1004, if environment variable VaniDL_BIN_DIR is not set correctly.
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
        io_df_dxt_filename = "{}/{}_io_dxt_df.csv".format(preprocessed_dir, filename)
        if not os.path.exists(io_df_dxt_filename):
            self._dxt_df = self._parse_dxt_trace()
            self._dxt_df.to_csv(index=False, path_or_buf=io_df_dxt_filename)
        else:
            self._dxt_df = pd.read_csv(io_df_dxt_filename)
            print("Loaded Pre-processed DXT DF from file: {}".format(io_df_dxt_filename))
        io_df_filename = "{}/{}_io_df.csv".format(preprocessed_dir, filename)
        if not os.path.exists(io_df_filename):
            self._df = self._parse_trace()
            self._df.to_csv(index=False, path_or_buf=io_df_filename)
        else:
            self._df = pd.read_csv(io_df_filename)
            print("Loaded Pre-processed DF from file: {}".format(io_df_filename))
        self._pre_process_df(data_paths_include)
        pattern_json = "{}/{}_pattern.json".format(preprocessed_dir, filename)
        if not os.path.exists(pattern_json):
            self._file_access_pattern = self._analyze_access_pattern()
            with open(pattern_json, 'w') as outfile:
                json.dump(self._file_access_pattern, outfile)
        else:
            with open(pattern_json) as json_file:
                self._file_access_pattern = json.load(json_file)
            print("Loaded Pre-processed Pattern file: {}".format(pattern_json))
        if tensorflow_logs_dir is not None:
            ipa_json = "{}/tb_ipa.json".format(self._preprocessed_dir)
            if not os.path.exists(ipa_json):
                self._tb_input_pipeline = self._parse_tb_logs(tensorflow_logs_dir)
                with open(ipa_json, 'w') as outfile:
                    json.dump(self._tb_input_pipeline, outfile)
            else:
                with open(ipa_json) as json_file:
                    self._tb_input_pipeline = json.load(json_file)
                print("Loaded Pre-processed Input Analyzer file: {}".format(ipa_json))

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

    def GetTraceAsDF(self):
        """
        Get the processed trace as a Pandas Dataframe
        :return: Pandas Dataframe
        """
        self._throw_if_not_loaded()
        return self._df

    def GetTFInputPipeline(self):
        """
        Get the processed IPA information as a JSON object.
        :return: JSON object with IPA information.
        """
        self._throw_if_not_loaded()
        return self._tb_input_pipeline

    def GetJobTime(self):
        """
        Get the total time spent in the job.
        :return: time in seconds.
        """
        self._throw_if_not_loaded()
        cmds = ["{} {}".format(self._get_darshan_exe(), self._darshan_file),
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
        temp_df = self._df
        if self._dxt_df.count()['Module'] == 0:
            if filepath is not None and rank is None:
                temp_df = temp_df[temp_df['Filename'].eq(filepath)]
            if rank is None:
                val = 0
                if "POSIX" in temp_df['Module'].unique():
                    val += temp_df['POSIX_F_READ_TIME'].sum() + \
                           temp_df['POSIX_F_WRITE_TIME'].sum() + \
                           temp_df['POSIX_F_META_TIME'].sum()
                if "MPIIO" in temp_df['Module'].unique():
                    val += temp_df['MPIIO_F_READ_TIME'].sum() + \
                           temp_df['MPIIO_F_WRITE_TIME'].sum() + \
                           temp_df['MPIIO_F_META_TIME'].sum()
                if "STDIO" in temp_df['Module'].unique():
                    val += temp_df['STDIO_F_META_TIME'].sum() + \
                           temp_df['STDIO_F_WRITE_TIME'].sum() + \
                           temp_df['STDIO_F_READ_TIME'].sum()
                if "H5D" in temp_df['Module'].unique():
                    val += temp_df['H5D_F_READ_TIME'].sum() + \
                           temp_df['H5D_F_WRITE_TIME'].sum() + \
                           temp_df['H5D_F_META_TIME'].sum()
                return val
        if self._dxt_df.count()['Module'] == 0:
            raise Exception(str(ErrorCodes.EC1010))
        temp_df = self._dxt_df
        if filepath is not None:
            temp_df = temp_df[temp_df['Filename'].eq(filepath)]
        if rank is not None:
            temp_df = temp_df[temp_df['Rank'].eq(rank)]
        return temp_df['io_time'].sum() / temp_df['Rank'].nunique()

    def GetIOSize(self, filepath=None, rank=None):
        """
        Returns the total I/O in bytes performed by job spent in I/O.
        If file path and rank are passed, data is further filtered.
        :param filepath: Filters data by filename
        :param rank: Filters data by rank
        :return: Returns I/O in bytes.
        """
        self._throw_if_not_loaded()
        temp_df = self._df
        if filepath is not None and rank is None:
            temp_df = temp_df[temp_df['Filename'].eq(filepath)]
        if rank is None:
            val = 0
            if "POSIX" in temp_df['Module'].unique():
                val += temp_df['POSIX_BYTES_WRITTEN'].sum() + \
                       temp_df['POSIX_BYTES_READ'].sum()
            if "MPIIO" in temp_df['Module'].unique():
                val += temp_df['MPIIO_BYTES_READ'].sum() + \
                       temp_df['MPIIO_BYTES_WRITTEN'].sum()
            if "STDIO" in temp_df['Module'].unique():
                val += temp_df['STDIO_BYTES_WRITTEN'].sum() + \
                       temp_df['STDIO_BYTES_READ'].sum()
            if "H5D" in temp_df['Module'].unique():
                val += temp_df['H5D_BYTES_READ'].sum() + \
                       temp_df['H5D_BYTES_WRITTEN'].sum()
            return val
        if self._dxt_df.count()['Module'] == 0:
            raise Exception(str(ErrorCodes.EC1010))
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
            size = pathlib.Path(filepath).stat().st_size
            return {filepath: size}
        else:
            file_size_map = {}
            for file in self._file_access_pattern:
                if os.path.exists(file):
                    size = pathlib.Path(file).stat().st_size
                    file = os.path.splitext(ntpath.basename(file))[0]
                    file_size_map[file] = float(size)
            return file_size_map

    def CreateIOTimeline(self, filepath=None, rank=None, time_step=None, save=True, is_print=True):
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
        if self._dxt_df.count()['Module'] == 0:
            raise Exception(str(ErrorCodes.EC1010))
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
            if is_print:
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
        read_series = numpy.zeros(data_points)
        read_bytes_series = numpy.zeros(data_points)
        sum_series = numpy.zeros(data_points)
        write_series = numpy.zeros(data_points)
        write_bytes_series = numpy.zeros(data_points)
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
                if row['Operation'] == "read":
                    read_series[n] += 1
                    read_bytes_series[n] += float(row['Length'])
                elif row['Operation'] == "write":
                    write_series[n] += 1
                    write_bytes_series[n] += float(row['Length'])
                count_series[n] += 1
                sum_series[n] += float(row['Length'])
        df_time = pd.DataFrame(
            {'time_step': data_points_series, 'operation_count': count_series, 'io_bytes': sum_series,
             'read_bytes': read_bytes_series, 'read_count': read_series, 'write_bytes': write_bytes_series,
             'write_count': write_series})
        if save:
            df_time.to_csv(index=False, path_or_buf=tm_df_filename)
        return df_time

    def GetIORequestDistribution(self, filepath=None, rank=None, operation=None, bins=100, threshold=AUTO):
        """
        Returns a 2d series of value counts for given bins of io sizes.
        if filepath is passed, data is filtered by filename
        bins decide the number of points on x axis of the histogram
        and threshold can be used to ignore points less than the given threshold.
        By default threshold is set to 1/1000 of the total sum of counts.
        :param filepath: filters the data by filepath
        :param rank: filters the data by rank
        :param operation: filters the data by operation
        :param bins: sets the bins for the histogram
        :param threshold: sets the threshold to ignore on histogram
        :return: a dataframe object which can be plotted using plot function.
        """
        self._throw_if_not_loaded()
        if self._dxt_df.count()['Module'] == 0:
            raise Exception(str(ErrorCodes.EC1010))
        temp_df = self._dxt_df
        if filepath is not None:
            temp_df = temp_df[temp_df['Filename'].eq(filepath)]
        if rank is not None:
            temp_df = temp_df[temp_df['Rank'].eq(rank)]
        if operation is not None:
            temp_df = temp_df[temp_df['Operation'].eq(operation)]
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
        if len(file_sizes) == 0:
            file_sizes.append(0)
        file_sizes = numpy.array(file_sizes)
        if self._dxt_df.count()['Module'] != 0:
            return {
                "type": "full",
                "job_time": self.GetJobTime(),
                "total_io_time": self.GetIOTime(),
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
        else:
            operations = []
            write = 0
            read = 0
            if "POSIX" in self._df['Module'].unique():
                write += self._df['POSIX_BYTES_WRITTEN'].sum()
                read += self._df['POSIX_BYTES_READ'].sum()
            if "MPIIO" in self._df['Module'].unique():
                write += self._df['MPIIO_BYTES_WRITTEN'].sum()
                read += self._df['MPIIO_BYTES_READ'].sum()
            if "STDIO" in self._df['Module'].unique():
                write += self._df['STDIO_BYTES_WRITTEN'].sum()
                read += self._df['STDIO_BYTES_READ'].sum()
            if "H5D" in self._df['Module'].unique():
                write += self._df['H5D_BYTES_WRITTEN'].sum()
                read += self._df['H5D_BYTES_READ'].sum()
            if write > 0:
                operations.append("write")
            if read > 0:
                operations.append("read")

            return {
                "type": "basic",
                "job_time": self.GetJobTime(),
                "total_io_time": self.GetIOTime(),
                "total_io_bytes": self.GetIOSize(),
                "io_interface_used": self._df['Module'].unique(),
                "io_operations_used": operations,
                "files_used": self._df["Filename"].unique().tolist(),
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
        import h5py
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
            file_size = pathlib.Path(filepath).stat().st_size
            return {
                "path": filepath,
                "filename": filename,
                "size": file_size,
                "ext": file_ext,
                "io_time": self.GetIOTime(filepath=filepath),
                "io_size": self.GetIOSize(filepath=filepath),
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
        import tensorflow as tf
        self._throw_if_not_loaded()
        if filepath is None:
            raise Exception(str(ErrorCodes.EC1005))
        if not os.path.exists(filepath):
            raise SystemExit(str(ErrorCodes.EC1009))
        file_ext_array = os.path.splitext(ntpath.basename(filepath))
        filename = file_ext_array[0]
        file_ext = 'tfrecord'
        filenames = [filepath]
        raw_dataset = tf.data.TFRecordDataset(filenames)
        if len(features) == 0:
            raise Exception(str(ErrorCodes.EC1008))
        self._tf_features = features
        special = raw_dataset.map(self._parse_tf_record, num_parallel_calls=POOL_SIZE)
        if only_special_summary:
            return special
        else:
            file_size = pathlib.Path(filepath).stat().st_size
            return {
                "path": filepath,
                "filename": filename,
                "size": file_size,
                "ext": file_ext,
                "io_time": self.GetIOTime(filepath=filepath),
                "io_size": self.GetIOSize(filepath=filepath),
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
            "io_time": self.GetIOTime(filepath=filepath),
            "io_size": self.GetIOSize(filepath=filepath),
            "special": special_summary
        }

    def CreateChromeTimeline(self, location="/tmp/temp_analysis", filename="timeline.json", save=True, timeshift=0):
        """
        This functions build a timeline from the darshan traces to be analyzed using chrome://tracing
        It puts the darshan dxt trace on thread id 0 and put the darshan normal trace on thread id 1.

        :param location: folder where timeline should be stored
        :param filename: filename of the timeline
        :param save: if the timeline returned should be persisted
        :param timeshift: how much time darshan trace should be shifted for alignment. (negetive value would move the timeline backwards)
        :return: the timeline returned in json format.
        """
        self._throw_if_not_loaded()
        if self._dxt_df.count()['Module'] == 0 and self._df.count()['Module'] == 0:
            raise Exception(str(ErrorCodes.EC1010))
        chromeTimeline = {
            "traceEvents": [],
            "displayTimeUnit": "ms",
            "systemTraceEvents": "SystemTraceData",
            "otherData": {
                "version": "VaniDL v1.0"
            },
            "stackFrames": {},
            "samples": []
        }
        timestamps = []
        data = []
        # 'Module', 'Filename', 'Rank', 'Operation', 'Segment', 'Offset', 'Length', 'Start', 'End'
        pb_total = self._dxt_df.count()['Module'];
        i = 1
        ranks_set = set()
        for index, row in self._dxt_df.iterrows():
            if i % 100 == 0 or i == pb_total:
                progress(i, pb_total, status='Creating DXT Timeline')
            i += 1
            ranks_set.add(int(row['Rank']))
            event_start = {"name": row['Filename'], "cat": row['Module'], "ph": "B",
                           "ts": int((float(row['Start']) + float(timeshift)) * 1e6), "pid": int(row['Rank']), "tid": 0,
                           "args": {
                               "Module": row['Module'],
                               "Filename": row['Filename'],
                               "Rank": row['Rank'],
                               "Operation": row['Operation'],
                               "Segment": row['Segment'],
                               "Offset": row['Offset'],
                               "Length": row['Length'],
                               "Start": row['Start'],
                               "End": row['End']
                           }
                           }
            event_end = {"ph": "E", "ts": int((float(row['End']) + float(timeshift)) * 1e6), "pid": int(row['Rank']), "tid": 0,
                         "args": {
                             "Module": row['Module'],
                             "Filename": row['Filename'],
                             "Rank": row['Rank'],
                             "Operation": row['Operation'],
                             "Segment": row['Segment'],
                             "Offset": row['Offset'],
                             "Length": row['Length'],
                             "Start": row['Start'],
                             "End": row['End']
                         }
                         }

            timestamps.append(int(float(row['Start']) * 1e6))
            data.append(event_start)
            timestamps.append(int(float(row['End']) * 1e6))
            data.append(event_end)
        pb_total = self._df.count()['Module'];
        i = 1
        for index, row in self._df.iterrows():
            if i % 100 == 0 or i == pb_total:
                progress(i, pb_total, status='Creating DXT Timeline')
            i += 1
            if row['Module'] == "STDIO":
                args_val = {
                    "Module": row['Module'],
                    "Rank": row['Rank'],
                    "Filename": row['Filename'],
                    "STDIO_OPENS": row['STDIO_OPENS'],
                    "STDIO_FDOPENS": row['STDIO_FDOPENS'],
                    "STDIO_READS": row['STDIO_READS'],
                    "STDIO_WRITES": row['STDIO_WRITES'],
                    "STDIO_SEEKS": row['STDIO_SEEKS'],
                    "STDIO_FLUSHES": row['STDIO_FLUSHES'],
                    "STDIO_BYTES_WRITTEN": row['STDIO_BYTES_WRITTEN'],
                    "STDIO_BYTES_READ": row['STDIO_BYTES_READ'],
                    "STDIO_MAX_BYTE_READ": row['STDIO_MAX_BYTE_READ'],
                    "STDIO_MAX_BYTE_WRITTEN": row['STDIO_MAX_BYTE_WRITTEN'],
                    "STDIO_FASTEST_RANK": row['STDIO_FASTEST_RANK'],
                    "STDIO_FASTEST_RANK_BYTES": row['STDIO_FASTEST_RANK_BYTES'],
                    "STDIO_SLOWEST_RANK": row['STDIO_SLOWEST_RANK'],
                    "STDIO_SLOWEST_RANK_BYTES": row['STDIO_SLOWEST_RANK_BYTES'],
                    "STDIO_F_META_TIME": row['STDIO_F_META_TIME'],
                    "STDIO_F_WRITE_TIME": row['STDIO_F_WRITE_TIME'],
                    "STDIO_F_READ_TIME": row['STDIO_F_READ_TIME'],
                    "STDIO_F_OPEN_START_TIMESTAMP": row['STDIO_F_OPEN_START_TIMESTAMP'],
                    "STDIO_F_CLOSE_START_TIMESTAMP": row['STDIO_F_CLOSE_START_TIMESTAMP'],
                    "STDIO_F_WRITE_START_TIMESTAMP": row['STDIO_F_WRITE_START_TIMESTAMP'],
                    "STDIO_F_READ_START_TIMESTAMP": row['STDIO_F_READ_START_TIMESTAMP'],
                    "STDIO_F_OPEN_END_TIMESTAMP": row['STDIO_F_OPEN_END_TIMESTAMP'],
                    "STDIO_F_CLOSE_END_TIMESTAMP": row['STDIO_F_CLOSE_END_TIMESTAMP'],
                    "STDIO_F_WRITE_END_TIMESTAMP": row['STDIO_F_WRITE_END_TIMESTAMP'],
                    "STDIO_F_READ_END_TIMESTAMP": row['STDIO_F_READ_END_TIMESTAMP'],
                    "STDIO_F_FASTEST_RANK_TIME": row['STDIO_F_FASTEST_RANK_TIME'],
                    "STDIO_F_SLOWEST_RANK_TIME": row['STDIO_F_SLOWEST_RANK_TIME'],
                    "STDIO_F_VARIANCE_RANK_TIME": row['STDIO_F_VARIANCE_RANK_TIME'],
                    "STDIO_F_VARIANCE_RANK_BYTES": row['STDIO_F_VARIANCE_RANK_BYTES']
                }
                ranks = []
                if row['Rank'] != -1:
                    ranks.append(int(row['Rank']))
                else:
                    ranks.extend(list(ranks_set))
                for rank in ranks:
                    if row['STDIO_BYTES_READ'] != 0:
                        event_start = {"name": row['Filename'], "cat": row['Module'], "ph": "B",
                                       "ts": int((float(row['STDIO_F_READ_START_TIMESTAMP']) + float(timeshift)) * 1e6),
                                       "pid": rank, "tid": 1,
                                       "args": args_val
                                       }
                        event_end = {"ph": "E", "ts": int((float(row['STDIO_F_READ_END_TIMESTAMP']) + float(timeshift)) * 1e6),
                                     "pid": rank, "tid": 1,
                                     "args": args_val
                                     }
                        timestamps.append(int((float(row['STDIO_F_READ_START_TIMESTAMP']) + float(timeshift)) * 1e6))
                        data.append(event_start)
                        timestamps.append(int((float(row['STDIO_F_READ_END_TIMESTAMP']) + float(timeshift)) * 1e6))
                        data.append(event_end)
                    if row['STDIO_BYTES_WRITTEN'] != 0:
                        event_start = {"name": "darshan", "cat": row['Module'], "ph": "B",
                                       "ts": int((float(row['STDIO_F_WRITE_START_TIMESTAMP']) + float(timeshift)) * 1e6),
                                       "pid": rank, "tid": 1,
                                       "args": args_val
                                       }
                        event_end = {"ph": "E", "ts": int((float(row['STDIO_F_WRITE_END_TIMESTAMP']) + float(timeshift)) * 1e6),
                                     "pid": rank, "tid": 1,
                                     "args": args_val
                                     }
                        timestamps.append(int((float(row['STDIO_F_WRITE_START_TIMESTAMP']) + float(timeshift)) * 1e6))
                        data.append(event_start)
                        timestamps.append(int((float(row['STDIO_F_WRITE_END_TIMESTAMP']) + float(timeshift)) * 1e6))
                        data.append(event_end)
        data.sort(key=lambda x: x['ts'])
        chromeTimeline["traceEvents"] = data
        if save:
            json_str = json.dumps(chromeTimeline) + "\n"
            json_bytes = json_str.encode('utf-8')
            with gzip.GzipFile("{}/{}.gz".format(location, filename), 'w') as fout:  # 4. gzip
                fout.write(json_bytes)
        return chromeTimeline

    def CreateMergedTimeline(self, tensorboard_dir, merged_timeline_output_dir, merged_timeline_file_prefix,
                             save=True, split_by_ranks=False, split_by_time=False, time_slice=None, timeshift=0):
        """
        This method merges all tracing files from tensorboard_dir with the darshan traces.
        It first converts hostnames and process id to ranks. (Assumption: hostname and pids are ordered by MPI
        and then merges the darshan trace with tf logs.
        :param tensorboard_dir: The log directory where tensorboard logs are present.
        :param merged_timeline_output_dir: directory where merged timeline should be output.
        :param merged_timeline_file_prefix: prefix for out files to be written.
        :param save: if the timeline should be saved
        :param split_by_ranks: should the timeline be split by ranks.
        :param split_by_time: should the timeline be split by time.
        :param time_slice: if timeline is split by time then what is the timeslice.
        :param timeshift: shifts the darshan timeline.
        :return: the generated timeline which is merged between darshan and td logs files.
        """
        if tensorboard_dir == None or merged_timeline_file_prefix == None or merged_timeline_output_dir == None:
            raise Exception(str(ErrorCodes.EC1011))
        if not (os.path.exists(tensorboard_dir) and os.path.exists(merged_timeline_output_dir)):
            raise Exception(str(ErrorCodes.EC10112))
        fileExt = "*.trace.json.gz"
        posix_path_files = list(pathlib.Path(tensorboard_dir).rglob(fileExt))
        files = []
        for path in posix_path_files:
            files.append(str(path))
        hosts = {}
        pb_total = len(files);
        i = 1
        for file in files:
            progress(i, pb_total, status='Parsing TF logs for hostname and Rank')
            i += 1
            # with open(file) as json_file:
            with gzip.open(file, 'rb') as json_file:
                data = json.load(json_file)
            trace_events = list(data["traceEvents"])
            filename = os.path.basename(file)
            hostname = filename.split(".")[0]
            if hostname not in hosts:
                hosts[hostname] = {}
            for trace_event in trace_events:
                if 'pid' in trace_event and trace_event['pid'] not in hosts[hostname]:
                    hosts[hostname][trace_event['pid']] = {'rank': 0, 'threads': set()}
                if 'pid' in trace_event and 'tid' in trace_event and trace_event['tid'] not in \
                        hosts[hostname][trace_event['pid']]['threads']:
                    hosts[hostname][trace_event['pid']]['threads'].add(trace_event['tid'])
        rank = 0
        for hostname in sorted(hosts.keys()):
            for pid in sorted(hosts[hostname].keys()):
                hosts[hostname][pid]['rank'] = rank
                rank += 1
        base_json = self.CreateChromeTimeline(save=False, timeshift=timeshift)
        b_base_json = base_json
        print("merging")
        merged_events = []
        max_ts = 0
        trace_events = list(b_base_json["traceEvents"])
        for trace_event in trace_events:
            if 'pid' in trace_event:
                pid = trace_event['pid']
                merged_events.append(trace_event)
                if max_ts < trace_event["ts"]:
                    max_ts = trace_event["ts"]
        pb_total = len(files);
        i = 1
        for file in files:
            progress(i, pb_total, status='Merging darshan with tf timeline')
            i += 1
            with gzip.open(file, 'rb') as json_file:
                data = json.load(json_file)
            trace_events = list(data["traceEvents"])
            filename = os.path.basename(file)
            hostname = filename.split(".")[0]
            final_traces = []
            for trace_event in trace_events:
                if 'pid' in trace_event and "ts" in trace_event :
                    trace_event['pid'] = hosts[hostname][trace_event['pid']]['rank']
                    pid = trace_event['pid']
                    merged_events.append(trace_event)
                    if max_ts < trace_event["ts"]:
                        max_ts = trace_event["ts"]
        merged_timeline_json = base_json
        merged_timeline_json["traceEvents"] = merged_events
        json_file = "{}/{}_complete.json.gz".format(merged_timeline_output_dir, merged_timeline_file_prefix)
        json_str = json.dumps(merged_timeline_json) + "\n"
        json_bytes = json_str.encode('utf-8')
        if save:
            with gzip.GzipFile(json_file, 'w') as fout:  # 4. gzip
                fout.write(json_bytes)
            print("written {}".format(json_file))
        if split_by_ranks:
            trace_data_proc = [None] * 8
            pb_total = len(merged_events)
            i = 1
            for merged_event in merged_events:
                if i % 100 == 0 or i == pb_total:
                    progress(i, pb_total, status='Splitting timeline by rank')
                i += 1
                if 'pid' in merged_event:
                    pid = merged_event['pid']
                    if trace_data_proc[pid] is None:
                        trace_data_proc[pid] = []
                    trace_data_proc[pid].append(merged_event)
            for i, trace_data in enumerate(trace_data_proc):
                b_base_json = base_json
                b_base_json["traceEvents"] = trace_data
                json_file = "{}/{}_r{}.json.gz".format(merged_timeline_output_dir, merged_timeline_file_prefix, i)
                json_str = json.dumps(b_base_json) + "\n"
                json_bytes = json_str.encode('utf-8')
                if save:
                    with gzip.GzipFile(json_file, 'w') as fout:  # 4. gzip
                        fout.write(json_bytes)
                    print("written {}".format(json_file))
        elif split_by_time:
            if time_slice is None:
                time_slice=100*1e6
            num_pieces = math.ceil(max_ts/time_slice)
            trace_data_time = [None]*num_pieces
            pb_total = len(merged_events);
            i = 1
            for merged_event in merged_events:
                if i % 100 == 0 or i == pb_total:
                    progress(i, pb_total, status='Splitting timeline by time')
                i += 1
                time_piece = int(merged_event["ts"]/time_slice)
                while time_piece > len(trace_data_time):
                    trace_data_time.append(None)
                if trace_data_time[time_piece] is None:
                    trace_data_time[time_piece]=[]
                trace_data_time[time_piece].append(merged_event)
            for i, trace_data in enumerate(trace_data_time):
                b_base_json = base_json
                b_base_json["traceEvents"] = trace_data
                json_file = "{}/{}_t{}.json.gz".format(merged_timeline_output_dir, merged_timeline_file_prefix, i)
                json_str = json.dumps(b_base_json) + "\n"
                json_bytes = json_str.encode('utf-8')
                if save:
                    with gzip.GzipFile(json_file, 'w') as fout:  # 4. gzip
                        fout.write(json_bytes)
                    print("written {}".format(json_file))
        return merged_timeline_json

    def MergeTimelines(self, timeline_file1, timeline_file2, merged_timeline_file):
        """
        This method merges two timeline files.
        :param timeline_file1: The first timeline to which the second would be merged
        :param timeline_file2: The timeline file which will be merged into the first
        :param merged_timeline_file: The output file for the resultant merged timeline.
        :return: The merged timeline.
        """
        if timeline_file1 == None or timeline_file2 == None or merged_timeline_file == None:
            raise Exception(str(ErrorCodes.EC1011))
        if not os.path.exists(timeline_file1) or not os.path.exists(timeline_file2):
            raise Exception(str(ErrorCodes.EC10112))
        file_1_json = {}
        with open(timeline_file1) as f:
            file_1_json = json.load(f)
        file_2_json = {}
        with open(timeline_file2) as f:
            file_2_json = json.load(f)
        new_trace_values = file_1_json["traceEvents"]
        new_trace_values.extend(file_2_json["traceEvents"])
        file_1_json["traceEvents"] = new_trace_values
        with open(merged_timeline_file, 'w') as outfile:
            json.dump(file_1_json, outfile)
        return file_1_json
