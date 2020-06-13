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


class ErrorCode(object):
    def __init__(self, error_code, error_message):
        self.error_code_ = error_code
        self.error_message_ = error_message

    def __repr__(self):
        return {'error_code': self.error_code_, 'error_message': self.error_message_}

    def __str__(self):
        return self.error_message_.format(self.error_code_)


class ErrorCodes:
    EC0000 = {0, "SUCCESSFUL"}
    EC1000 = {1000, "ERROR:{} A darshan File is required."}
    EC1001 = {1001, "ERROR:{} DL Profiler is not loaded."}
    EC1002 = {1002, "ERROR:{} A darshan File Path is not existing."}
    EC1003 = {1003, "ERROR:{} A darshan BIN Path is not existing."}
    EC1004 = {1004, "ERROR:{} A DLProfile BIN Path is not existing."}
    EC1005 = {1005, "ERROR:{} filepath not passed."}
    EC1006 = {1006, "ERROR:{} HDF5 file not passed. If file is HDF5 use ext='h5' argument"}
    EC1007 = {1007, "ERROR:{} TFRecord not passed. If file is TFRecord use ext='tfrecord' argument"}
    EC1008 = {1008, "ERROR:{} Features are required for TF dataset"}
    EC1009 = {1009, "ERROR:{} filepath does not exists."}
