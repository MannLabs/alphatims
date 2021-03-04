#!python
"""This module provides unit tests for alphatims.utils."""

# builtin
import unittest

# external
import numpy as np

# local
import os
import alphatims.utils
BASE_PATH = os.path.dirname(__file__)
alphatims.utils.set_logger(
    stream=None,
    log_file_name=os.path.join(
        BASE_PATH,
        "sandbox_data",
    )
)
import alphatims.bruker


class TestSlicing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        file_name = os.path.join(
            BASE_PATH,
            "sandbox_data",
            "20201016_tims03_Evo03_PS_MA_HeLa_200ng_DDA_06-15_5_6min_4cm_S1-A1_1_21717.hdf"
        )
        if not os.path.exists(file_name):
            file_name = os.path.join(
                BASE_PATH,
                "sandbox_data",
                "20201016_tims03_Evo03_PS_MA_HeLa_200ng_DDA_06-15_5_6min_4cm_S1-A1_1_21717.d"
            )
        if os.path.exists(file_name):
            try:
                cls.data = alphatims.bruker.TimsTOF(file_name)
            except:
                assert False, "Test data set is invalid..."
        else:
            assert False, "No test data found..."

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_convert_slice_key_to_float_array(self):
        test_cases = [
            (
                slice(100, 300),
                np.array([[100., 300.]])
            ), (
                slice(300),
                np.array([[-np.inf, 300.]])
            ), (
                slice(None, 300),
                np.array([[-np.inf, 300.]])
            ), (
                slice(300, None),
                np.array([[300., np.inf]])
            ), (
                100,
                np.array([[100., 100.]])
            ), (
                100.8,
                np.array([[100.8, 100.8]])
            ), (
                [100, 300],
                np.array([[100., 100.], [300., 300.]])
            ), (
                np.array([[100., 100.], [300., 300.]]),
                np.array([[100., 100.], [300., 300.]])
            ),
        ]
        for key, expected_result in test_cases:
            actual_result = alphatims.bruker.convert_slice_key_to_float_array(
                key
            )
            assert np.array_equal(
                actual_result,
                expected_result
            ), (
                f"{key} was wrongly converted to {actual_result} "
                f"instead of {expected_result}"
            )

    def test_convert_slice_key_to_int_array(self):
        pass

    def test_parse_keys(self):
        pass

    def test_filter_indices(self):
        pass


if __name__ == "__main__":
    unittest.main()
