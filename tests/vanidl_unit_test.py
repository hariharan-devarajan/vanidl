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
import unittest
import os
import sys
import pathlib

# sys.path.insert(0, pathlib.Path(__file__).parent.parent.absolute())

"""
Local Includes
"""
from src.vanidl import VaniDL
from src.constants import *
import tensorflow as tf

DARSHAN_DIR_PATH = "/home/hariharan/install"
VANIDL_DIR_PATH = "/home/hariharan/PycharmProjects/vanidl"

ORIGINAL_DATASET_DIR = "/projects/datascience/dhari/datasets/cosmic_tagger"
TARGET_DATASET_DIR = "{}/tests".format(VANIDL_DIR_PATH)
FILE_PATH = "{}/cosmic_tagging_train.h5".format(TARGET_DATASET_DIR)
DATAPATH_INCLUDES = [ORIGINAL_DATASET_DIR]
RANK = 0
TIMESTEP_SEC = 1
PROCESSED_DIR = "{}/tests/temp_analysis".format(VANIDL_DIR_PATH)
DARSHAN_FILE = "{}/tests/test.darshan".format(VANIDL_DIR_PATH)


def LoadEnv():
    os.environ[DARSHAN_DIR] = DARSHAN_DIR_PATH
    os.environ[VANIDL_DIR] = VANIDL_DIR_PATH


def rectify_paths(profile):
    df_normal = profile._df
    df = profile._dxt_df
    df['Filename'] = df['Filename'].str.replace(ORIGINAL_DATASET_DIR, TARGET_DATASET_DIR)
    df_normal['Filename'] = df_normal['Filename'].str.replace(ORIGINAL_DATASET_DIR, TARGET_DATASET_DIR)
    profile._df = df_normal
    profile._dxt_df = df
    keys = list(profile._file_access_pattern.keys())
    for key in keys:
        new_key = key.replace(ORIGINAL_DATASET_DIR, TARGET_DATASET_DIR)
        if new_key != key:
            profile._file_access_pattern[new_key] = profile._file_access_pattern.pop(key)
    return profile


class MyTestCase(unittest.TestCase):
    def test_Load(self):
        LoadEnv()
        profile = VaniDL()
        status = profile.Load(DARSHAN_FILE, data_paths_include=DATAPATH_INCLUDES, preprocessed_dir=PROCESSED_DIR)
        profile = rectify_paths(profile)
        self.assertEqual(status, True)

    def test_GetDXTAsDF(self):
        LoadEnv()
        profile = VaniDL()
        status = profile.Load(DARSHAN_FILE, data_paths_include=DATAPATH_INCLUDES, preprocessed_dir=PROCESSED_DIR)
        profile = rectify_paths(profile)
        df = profile.GetDXTAsDF()
        print(df['Filename'].unique()[1])
        self.assertEqual(True, True)

    def test_GetJobTime(self):
        LoadEnv()
        profile = VaniDL()
        status = profile.Load(DARSHAN_FILE, data_paths_include=DATAPATH_INCLUDES, preprocessed_dir=PROCESSED_DIR)
        profile = rectify_paths(profile)
        job_time = profile.GetJobTime()
        self.assertNotEqual(job_time, 0)

    def test_GetIOTime(self):
        LoadEnv()
        profile = VaniDL()
        status = profile.Load(DARSHAN_FILE, data_paths_include=DATAPATH_INCLUDES, preprocessed_dir=PROCESSED_DIR)
        profile = rectify_paths(profile)
        io_time = profile.GetIOTime()
        self.assertNotEqual(io_time, 0)

    def test_GetIOTimeFilepath(self):
        LoadEnv()
        profile = VaniDL()
        status = profile.Load(DARSHAN_FILE, data_paths_include=DATAPATH_INCLUDES, preprocessed_dir=PROCESSED_DIR)
        profile = rectify_paths(profile)
        io_time = profile.GetIOTime(filepath=FILE_PATH)
        self.assertNotEqual(io_time, 0)

    def test_GetIOTimeRank(self):
        LoadEnv()
        profile = VaniDL()
        status = profile.Load(DARSHAN_FILE, data_paths_include=DATAPATH_INCLUDES, preprocessed_dir=PROCESSED_DIR)
        profile = rectify_paths(profile)
        io_time = profile.GetIOTime(rank=RANK)
        self.assertNotEqual(io_time, 0)

    def test_GetIOTimeFilepathAndRank(self):
        LoadEnv()
        profile = VaniDL()
        status = profile.Load(DARSHAN_FILE, data_paths_include=DATAPATH_INCLUDES, preprocessed_dir=PROCESSED_DIR)
        profile = rectify_paths(profile)
        io_time = profile.GetIOTime(filepath=FILE_PATH, rank=RANK)
        self.assertNotEqual(io_time, 0)

    def test_GetIOSize(self):
        LoadEnv()
        profile = VaniDL()
        status = profile.Load(DARSHAN_FILE, data_paths_include=DATAPATH_INCLUDES, preprocessed_dir=PROCESSED_DIR)
        profile = rectify_paths(profile)
        io_size = profile.GetIOSize()
        self.assertNotEqual(io_size, 0)

    def test_GetIOSizeFilepath(self):
        LoadEnv()
        profile = VaniDL()
        status = profile.Load(DARSHAN_FILE, data_paths_include=DATAPATH_INCLUDES, preprocessed_dir=PROCESSED_DIR)
        profile = rectify_paths(profile)
        io_size = profile.GetIOSize(filepath=FILE_PATH)
        self.assertNotEqual(io_size, 0)

    def test_GetIOSizeRank(self):
        LoadEnv()
        profile = VaniDL()
        status = profile.Load(DARSHAN_FILE, data_paths_include=DATAPATH_INCLUDES, preprocessed_dir=PROCESSED_DIR)
        profile = rectify_paths(profile)
        io_size = profile.GetIOSize(rank=RANK)
        self.assertNotEqual(io_size, 0)

    def test_GetIOSizeFilepathAndRank(self):
        LoadEnv()
        profile = VaniDL()
        status = profile.Load(DARSHAN_FILE, data_paths_include=DATAPATH_INCLUDES, preprocessed_dir=PROCESSED_DIR)
        profile = rectify_paths(profile)
        io_size = profile.GetIOSize(filepath=FILE_PATH, rank=RANK)
        self.assertNotEqual(io_size, 0)

    def test_GetAccessPattern(self):
        LoadEnv()
        profile = VaniDL()
        status = profile.Load(DARSHAN_FILE, data_paths_include=DATAPATH_INCLUDES, preprocessed_dir=PROCESSED_DIR)
        profile = rectify_paths(profile)
        pattern = profile.GetAccessPattern()
        self.assertNotEqual(len(pattern), 0)

    def test_GetAccessPatternFilepath(self):
        LoadEnv()
        profile = VaniDL()
        status = profile.Load(DARSHAN_FILE, data_paths_include=DATAPATH_INCLUDES, preprocessed_dir=PROCESSED_DIR)
        profile = rectify_paths(profile)
        pattern = profile.GetAccessPattern(filepath=FILE_PATH)
        self.assertNotEqual(len(pattern), 0)

    def test_GetFileSizes(self):
        LoadEnv()
        profile = VaniDL()
        status = profile.Load(DARSHAN_FILE, data_paths_include=DATAPATH_INCLUDES, preprocessed_dir=PROCESSED_DIR)
        profile = rectify_paths(profile)
        file_sizes = profile.GetFileSizes()
        self.assertNotEqual(len(file_sizes), 0)

    def test_GetFileSizesFilepath(self):
        LoadEnv()
        profile = VaniDL()
        status = profile.Load(DARSHAN_FILE, data_paths_include=DATAPATH_INCLUDES, preprocessed_dir=PROCESSED_DIR)
        profile = rectify_paths(profile)
        file_sizes = profile.GetFileSizes(filepath=FILE_PATH)
        self.assertNotEqual(len(file_sizes), 0)

    def test_CreateIOTimeline(self):
        LoadEnv()
        profile = VaniDL()
        status = profile.Load(DARSHAN_FILE, data_paths_include=DATAPATH_INCLUDES, preprocessed_dir=PROCESSED_DIR)
        profile = rectify_paths(profile)
        df = profile.CreateIOTimeline()
        print(df.count()['time_step'])
        self.assertNotEqual(df.count()['time_step'], 0)

    def test_CreateIOTimelineFilepath(self):
        LoadEnv()
        profile = VaniDL()
        status = profile.Load(DARSHAN_FILE, data_paths_include=DATAPATH_INCLUDES, preprocessed_dir=PROCESSED_DIR)
        profile = rectify_paths(profile)
        df = profile.CreateIOTimeline(filepath=FILE_PATH)
        self.assertNotEqual(df.count()['time_step'], 0)

    def test_CreateIOTimelineRank(self):
        LoadEnv()
        profile = VaniDL()
        status = profile.Load(DARSHAN_FILE, data_paths_include=DATAPATH_INCLUDES, preprocessed_dir=PROCESSED_DIR)
        profile = rectify_paths(profile)
        df = profile.CreateIOTimeline(rank=RANK)
        self.assertNotEqual(df.count()['time_step'], 0)

    def test_CreateIOTimelineTimestep(self):
        LoadEnv()
        profile = VaniDL()
        status = profile.Load(DARSHAN_FILE, data_paths_include=DATAPATH_INCLUDES, preprocessed_dir=PROCESSED_DIR)
        profile = rectify_paths(profile)
        df = profile.CreateIOTimeline(time_step=TIMESTEP_SEC)
        self.assertNotEqual(df.count()['time_step'], 0)

    def test_CreateIOTimelineFilepathRank(self):
        LoadEnv()
        profile = VaniDL()
        status = profile.Load(DARSHAN_FILE, data_paths_include=DATAPATH_INCLUDES, preprocessed_dir=PROCESSED_DIR)
        profile = rectify_paths(profile)
        df = profile.CreateIOTimeline(filepath=FILE_PATH, rank=RANK)
        self.assertNotEqual(df.count()['time_step'], 0)

    def test_GetIORequestDistribution(self):
        LoadEnv()
        profile = VaniDL()
        status = profile.Load(DARSHAN_FILE, data_paths_include=DATAPATH_INCLUDES, preprocessed_dir=PROCESSED_DIR)
        profile = rectify_paths(profile)
        df = profile.GetIORequestDistribution()
        print(df)
        self.assertNotEqual(df.count(), 0)

    def test_GetIORequestDistributionFilepath(self):
        LoadEnv()
        profile = VaniDL()
        status = profile.Load(DARSHAN_FILE, data_paths_include=DATAPATH_INCLUDES, preprocessed_dir=PROCESSED_DIR)
        profile = rectify_paths(profile)
        df = profile.GetIORequestDistribution(filepath=FILE_PATH)
        self.assertNotEqual(df.count(), 0)

    def test_GetIORequestDistributionRank(self):
        LoadEnv()
        profile = VaniDL()
        status = profile.Load(DARSHAN_FILE, data_paths_include=DATAPATH_INCLUDES, preprocessed_dir=PROCESSED_DIR)
        profile = rectify_paths(profile)
        df = profile.GetIORequestDistribution(rank=RANK)
        self.assertNotEqual(df.count(), 0)

    def test_GetSummary(self):
        LoadEnv()
        profile = VaniDL()
        status = profile.Load(DARSHAN_FILE, data_paths_include=DATAPATH_INCLUDES, preprocessed_dir=PROCESSED_DIR)
        profile = rectify_paths(profile)
        summary = profile.GetSummary()
        self.assertNotEqual(len(summary), 0)

    def test_GetFileSummaryHDF5WithExt(self):
        LoadEnv()
        profile = VaniDL()
        status = profile.Load(DARSHAN_FILE, data_paths_include=DATAPATH_INCLUDES, preprocessed_dir=PROCESSED_DIR)
        profile = rectify_paths(profile)
        summary = profile.GetFileSummary(FILE_PATH, ext='h5')
        self.assertNotEqual(len(summary), 0)

    def test_GetFileSummaryHDF5(self):
        LoadEnv()
        profile = VaniDL()
        status = profile.Load(DARSHAN_FILE, data_paths_include=DATAPATH_INCLUDES, preprocessed_dir=PROCESSED_DIR)
        profile = rectify_paths(profile)
        summary = profile.GetFileSummary(FILE_PATH)
        self.assertNotEqual(len(summary), 0)

    def test_CreateChromeTimeline(self):
        LoadEnv()
        profile = VaniDL()
        status = profile.Load(DARSHAN_FILE, data_paths_include=DATAPATH_INCLUDES, preprocessed_dir=PROCESSED_DIR)
        profile = rectify_paths(profile)
        summary = profile.CreateChromeTimeline(location=PROCESSED_DIR, filename="timeline.json")
        self.assertNotEqual(len(summary), 0)

    def test_MergeTimelines(self):
        LoadEnv()
        profile = VaniDL()
        merged_trace = profile.MergeTimelines(timeline_file1="./compute_trace.json",timeline_file2="./io_timeline.json", merged_timeline_file="./merged_trace.json")
        self.assertNotEqual(len(merged_trace), 0)

if __name__ == '__main__':
    unittest.main()
