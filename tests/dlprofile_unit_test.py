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

#sys.path.insert(0, pathlib.Path(__file__).parent.parent.absolute())

"""
Local Includes
"""
from src.dlprofile import DLProfile
from src.constants import *
import tensorflow as tf


def LoadEnv():
    os.environ[DARSHAN_DIR] = "/home/hdevarajan/software/install"
    os.environ[DLPROFILE_DIR] = "/home/hdevarajan/dlprofile"


FILE_PATH = "/projects/datascience/dhari/datasets/cosmic_tagger/cosmic_tagging_train.h5"
TF_FILE_PATH = "/projects/datascience/rzamora/data/imagenet/count.48.size.8m/train-00263-of-01024"
DATAPATH_INCLUDES = ["/projects/datascience/dhari/datasets/cosmic_tagger/"]
RANK = 0
TIMESTEP_SEC = 1


class MyTestCase(unittest.TestCase):
    def test_Load(self):
        LoadEnv()
        profile = DLProfile()
        status = profile.Load("./test.darshan", data_paths_include=DATAPATH_INCLUDES)
        self.assertEqual(status, True)

    def test_GetDXTAsDF(self):
        LoadEnv()
        profile = DLProfile()
        status = profile.Load("./test.darshan", data_paths_include=DATAPATH_INCLUDES)
        df = profile.GetDXTAsDF()
        print(df['Filename'].unique()[1])
        self.assertEqual(True, True)

    def test_GetJobTime(self):
        LoadEnv()
        profile = DLProfile()
        status = profile.Load("./test.darshan", data_paths_include=DATAPATH_INCLUDES)
        job_time = profile.GetJobTime()
        self.assertNotEqual(job_time, 0)

    def test_GetIOTime(self):
        LoadEnv()
        profile = DLProfile()
        status = profile.Load("./test.darshan", data_paths_include=DATAPATH_INCLUDES)
        io_time = profile.GetIOTime()
        self.assertNotEqual(io_time, 0)

    def test_GetIOTimeFilepath(self):
        LoadEnv()
        profile = DLProfile()
        status = profile.Load("./test.darshan", data_paths_include=DATAPATH_INCLUDES)
        io_time = profile.GetIOTime(filepath=FILE_PATH)
        self.assertNotEqual(io_time, 0)

    def test_GetIOTimeRank(self):
        LoadEnv()
        profile = DLProfile()
        status = profile.Load("./test.darshan", data_paths_include=DATAPATH_INCLUDES)
        io_time = profile.GetIOTime(rank=RANK)
        self.assertNotEqual(io_time, 0)

    def test_GetIOTimeFilepathAndRank(self):
        LoadEnv()
        profile = DLProfile()
        status = profile.Load("./test.darshan", data_paths_include=DATAPATH_INCLUDES)
        io_time = profile.GetIOTime(filepath=FILE_PATH, rank=RANK)
        self.assertNotEqual(io_time, 0)

    def test_GetIOSize(self):
        LoadEnv()
        profile = DLProfile()
        status = profile.Load("./test.darshan", data_paths_include=DATAPATH_INCLUDES)
        io_size = profile.GetIOSize()
        self.assertNotEqual(io_size, 0)

    def test_GetIOSizeFilepath(self):
        LoadEnv()
        profile = DLProfile()
        status = profile.Load("./test.darshan", data_paths_include=DATAPATH_INCLUDES)
        io_size = profile.GetIOSize(filepath=FILE_PATH)
        self.assertNotEqual(io_size, 0)

    def test_GetIOSizeRank(self):
        LoadEnv()
        profile = DLProfile()
        status = profile.Load("./test.darshan", data_paths_include=DATAPATH_INCLUDES)
        io_size = profile.GetIOSize(rank=RANK)
        self.assertNotEqual(io_size, 0)

    def test_GetIOSizeFilepathAndRank(self):
        LoadEnv()
        profile = DLProfile()
        status = profile.Load("./test.darshan", data_paths_include=DATAPATH_INCLUDES)
        io_size = profile.GetIOSize(filepath=FILE_PATH, rank=RANK)
        self.assertNotEqual(io_size, 0)

    def test_GetAccessPattern(self):
        LoadEnv()
        profile = DLProfile()
        status = profile.Load("./test.darshan", data_paths_include=DATAPATH_INCLUDES)
        pattern = profile.GetAccessPattern()
        self.assertNotEqual(len(pattern), 0)

    def test_GetAccessPatternFilepath(self):
        LoadEnv()
        profile = DLProfile()
        status = profile.Load("./test.darshan", data_paths_include=DATAPATH_INCLUDES)
        pattern = profile.GetAccessPattern(filepath=FILE_PATH)
        self.assertNotEqual(len(pattern), 0)

    def test_GetFileSizes(self):
        LoadEnv()
        profile = DLProfile()
        status = profile.Load("./test.darshan", data_paths_include=DATAPATH_INCLUDES)
        file_sizes = profile.GetFileSizes()
        self.assertNotEqual(len(file_sizes), 0)

    def test_GetFileSizesFilepath(self):
        LoadEnv()
        profile = DLProfile()
        status = profile.Load("./test.darshan", data_paths_include=DATAPATH_INCLUDES)
        file_sizes = profile.GetFileSizes(filepath=FILE_PATH)
        self.assertNotEqual(len(file_sizes), 0)

    def test_GetIOPerRank(self):
        LoadEnv()
        profile = DLProfile()
        status = profile.Load("./test.darshan", data_paths_include=DATAPATH_INCLUDES)
        io_sizes = profile.GetIOPerRank()
        self.assertNotEqual(len(io_sizes), 0)

    def test_CreateIOTimeline(self):
        LoadEnv()
        profile = DLProfile()
        status = profile.Load("./test.darshan", data_paths_include=DATAPATH_INCLUDES)
        df = profile.CreateIOTimeline()
        print(df.count()['time_step'])
        self.assertNotEqual(df.count()['time_step'], 0)

    def test_CreateIOTimelineFilepath(self):
        LoadEnv()
        profile = DLProfile()
        status = profile.Load("./test.darshan", data_paths_include=DATAPATH_INCLUDES)
        df = profile.CreateIOTimeline(filepath=FILE_PATH)
        self.assertNotEqual(df.count()['time_step'], 0)

    def test_CreateIOTimelineRank(self):
        LoadEnv()
        profile = DLProfile()
        status = profile.Load("./test.darshan", data_paths_include=DATAPATH_INCLUDES)
        df = profile.CreateIOTimeline(rank=RANK)
        self.assertNotEqual(df.count()['time_step'], 0)

    def test_CreateIOTimelineTimestep(self):
        LoadEnv()
        profile = DLProfile()
        status = profile.Load("./test.darshan", data_paths_include=DATAPATH_INCLUDES)
        df = profile.CreateIOTimeline(time_step=TIMESTEP_SEC)
        self.assertNotEqual(df.count()['time_step'], 0)

    def test_CreateIOTimelineFilepathRank(self):
        LoadEnv()
        profile = DLProfile()
        status = profile.Load("./test.darshan", data_paths_include=DATAPATH_INCLUDES)
        df = profile.CreateIOTimeline(filepath=FILE_PATH, rank=RANK)
        self.assertNotEqual(df.count()['time_step'], 0)

    def test_GetIORequestDistribution(self):
        LoadEnv()
        profile = DLProfile()
        status = profile.Load("./test.darshan", data_paths_include=DATAPATH_INCLUDES)
        df = profile.GetIORequestDistribution()
        print(df)
        self.assertNotEqual(df.count(), 0)

    def test_GetIORequestDistributionFilepath(self):
        LoadEnv()
        profile = DLProfile()
        status = profile.Load("./test.darshan", data_paths_include=DATAPATH_INCLUDES)
        df = profile.GetIORequestDistribution(filepath=FILE_PATH)
        self.assertNotEqual(df.count(), 0)

    def test_GetIORequestDistributionRank(self):
        LoadEnv()
        profile = DLProfile()
        status = profile.Load("./test.darshan", data_paths_include=DATAPATH_INCLUDES)
        df = profile.GetIORequestDistribution(rank=RANK)
        self.assertNotEqual(df.count(), 0)

    def test_GetSummary(self):
        LoadEnv()
        profile = DLProfile()
        status = profile.Load("./test.darshan", data_paths_include=DATAPATH_INCLUDES)
        summary = profile.GetSummary()
        self.assertNotEqual(len(summary), 0)

    def test_GetFileSummaryHDF5WithExt(self):
        LoadEnv()
        profile = DLProfile()
        status = profile.Load("./test.darshan", data_paths_include=DATAPATH_INCLUDES)
        summary = profile.GetFileSummary(FILE_PATH,ext='h5')
        self.assertNotEqual(len(summary), 0)

    def test_GetFileSummaryHDF5(self):
        LoadEnv()
        profile = DLProfile()
        status = profile.Load("./test.darshan", data_paths_include=DATAPATH_INCLUDES)
        summary = profile.GetFileSummary(FILE_PATH)
        self.assertNotEqual(len(summary), 0)

    def test_GetFileSummaryTFRecord(self):
        LoadEnv()
        profile = DLProfile()
        status = profile.Load("./test.darshan", data_paths_include=DATAPATH_INCLUDES)
        summary = profile.GetFileSummary(TF_FILE_PATH, ext="tfrecord", tf_record_features={
              'image/encoded': tf.io.FixedLenFeature((), dtype=tf.string, default_value=''),
              'image/source_id': tf.io.FixedLenFeature((), tf.string, default_value=''),
              'image/height': tf.io.FixedLenFeature((), tf.int64, default_value=1),
              'image/width': tf.io.FixedLenFeature((), tf.int64, default_value=1),
              'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),
              'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
              'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
              'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32),
              'image/object/class/label': tf.io.VarLenFeature(dtype=tf.int64),
              'image/object/class/text': tf.io.FixedLenFeature((), dtype=tf.string, default_value=''),
              'image/colorspace': tf.io.FixedLenFeature((), tf.string, default_value=''),
              'image/channels': tf.io.FixedLenFeature((), tf.int64, default_value=1),
              'image/format': tf.io.FixedLenFeature((), tf.string, default_value=''),
              'image/filename': tf.io.FixedLenFeature((), tf.string, default_value='')
          })
        self.assertNotEqual(len(summary), 0)


if __name__ == '__main__':
    unittest.main()
