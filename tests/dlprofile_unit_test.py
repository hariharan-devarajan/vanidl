import unittest
import os
from src.dlprofile import DLProfile, DARSHAN_BIN_DIR


class MyTestCase(unittest.TestCase):
    def test_load(self):
        #os.environ[DARSHAN_BIN_DIR] = ""
        try:
            profile = DLProfile()
            status = profile.Load("./test.darshan")
            self.assertEqual(status, True)
        except:
            self.assertEqual(False, True, " Exception Occurred in processing")


if __name__ == '__main__':
    unittest.main()
