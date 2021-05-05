#!python -m unittest tests.test_bruker
"""This module provides unit tests for alphatims.utils."""

# builtin
import unittest
import logging
import os

# external
import numpy as np

# local
import alphatims.utils
import alphatims.bruker
alphatims.utils.set_progress_callback(None)


class TestSlicing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(alphatims.utils.DEMO_FILE_NAME):
            logging.info("Downloading sample...")
            import urllib.request
            import urllib.error
            import zipfile
            import io
            with urllib.request.urlopen(
                alphatims.utils.DEMO_FILE_NAME_GITHUB
            ) as sample_file:
                sample_byte_stream = io.BytesIO(sample_file.read())
                with zipfile.ZipFile(sample_byte_stream, 'r') as zip_ref:
                    zip_ref.extractall(
                        os.path.dirname(alphatims.utils.DEMO_FILE_NAME)
                    )
        if os.path.exists(alphatims.utils.DEMO_FILE_NAME):
            try:
                cls.data = alphatims.bruker.TimsTOF(
                    alphatims.utils.DEMO_FILE_NAME
                )
            except:
                assert False, "Test data set is invalid..."
        else:
            assert False, "No test data found..."
            # TODO: fetch from URL?

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
            result = alphatims.bruker.convert_slice_key_to_float_array(
                key
            )
            assert np.array_equal(
                result,
                expected_result
            ), (
                f"{key} was wrongly converted to {result} "
                f"instead of {expected_result}"
            )

    def test_convert_slice_key_to_int_array(self):
        for dimension, (float_low, float_high, max_int) in {
            "frame_indices": [110., 150., self.data.frame_max_index],
            "tof_indices": [500., 500.5, self.data.tof_max_index],
        }.items():
            int_low, int_high = self.data.convert_to_indices(
                [float_low, float_high],
                return_frame_indices=dimension == "frame_indices",
                return_scan_indices=dimension == "scan_indices",
                return_tof_indices=dimension == "tof_indices",
            )
            for key, expected_result in [
                (
                    slice(int_low, int_high),
                    np.array([[int_low, int_high, 1]])
                ), (
                    slice(int_low, int_high, 8),
                    np.array([[int_low, int_high, 8]])
                ), (
                    slice(None, int_high, 8),
                    np.array([[0, int_high, 8]])
                ), (
                    slice(int_low, None, 8),
                    np.array([[int_low, max_int, 8]])
                ), (
                    slice(None, None, 8),
                    np.array([[0, max_int, 8]])
                ), (
                    int_low,
                    np.array([[int_low, int_low + 1, 1]])
                ), (
                    [int_low, int_high],
                    np.array(
                        [
                            [int_low, int_low + 1, 1],
                            [int_high, int_high + 1, 1]
                        ]
                    )
                ), (
                    slice(None, float_high, 8),
                    np.array([[0, int_high, 8]])
                ), (
                    slice(float_low, None, 8),
                    np.array([[int_low, max_int, 8]])
                ), (
                    slice(float_low, float_high, 8),
                    np.array([[int_low, int_high, 8]])
                ), (
                    [float_low, float_high],
                    np.array(
                        [
                            [int_low, int_low + 1, 1],
                            [int_high, int_high + 1, 1]
                        ]
                    )
                ),
            ]:
                result = alphatims.bruker.convert_slice_key_to_int_array(
                    self.data,
                    key,
                    dimension
                )
                assert np.array_equal(
                    result,
                    expected_result
                ), (
                    f"Key '{key}' in dimension '{dimension}' was wrongly "
                    f"converted to {result} instead of "
                    f"{expected_result}"
                )
        dimension, float_low, float_high, max_int = [
            "scan_indices", 1., 1.2, self.data.scan_max_index
        ]
        # NOTE: order high and low is reversed
        int_high, int_low = self.data.convert_to_indices(
            [float_low, float_high],
            return_frame_indices=dimension == "frame_indices",
            return_scan_indices=dimension == "scan_indices",
            return_tof_indices=dimension == "tof_indices",
        )
        for key, expected_result in [
            (
                slice(int_low, int_high),
                np.array([[int_low, int_high, 1]])
            ), (
                slice(int_low, int_high, 8),
                np.array([[int_low, int_high, 8]])
            ), (
                slice(None, int_high, 8),
                np.array([[0, int_high, 8]])
            ), (
                slice(int_low, None, 8),
                np.array([[int_low, max_int, 8]])
            ), (
                slice(None, None, 8),
                np.array([[0, max_int, 8]])
            ), (
                int_low,
                np.array([[int_low, int_low + 1, 1]])
            ), (
                [int_low, int_high],
                np.array(
                    [
                        [int_low, int_low + 1, 1],
                        [int_high, int_high + 1, 1]
                    ]
                )
            ), (
                slice(None, float_high, 8),
                np.array([[int_low, max_int, 8]])
            ), (
                slice(float_low, None, 8),
                np.array([[0, int_high, 8]])
            ), (
                slice(float_low, float_high, 8),
                np.array([[int_low, int_high, 8]])
            ), (
                [float_high, float_low],
                np.array(
                    [
                        [int_low - 1, int_low, 1],
                        [int_high - 1, int_high, 1]
                    ]
                )
            ),
        ]:
            result = alphatims.bruker.convert_slice_key_to_int_array(
                self.data,
                key,
                dimension
            )
            assert np.array_equal(result, expected_result), (
                f"Key '{key}' in dimension '{dimension}' was wrongly "
                f"converted to {result} instead of "
                f"{expected_result}"
            )

    def test_parse_keys(self):
        # TODO: expand testing?
        key = (1,)
        result = alphatims.bruker.parse_keys(self.data, key)
        expected_result = {
            "frame_indices": np.array(
                [[1, 2, 1]],
                dtype=np.int64
            ),
            "scan_indices": np.array(
                [[0, self.data.scan_max_index, 1]],
                dtype=np.int64
            ),
            "tof_indices": np.array(
                [[0, self.data.tof_max_index, 1]],
                dtype=np.int64
            ),
            "precursor_indices": np.array(
                [[0, self.data.precursor_max_index, 1]],
                dtype=np.int64
            ),
            "quad_values": np.array(
                [[-np.inf, np.inf]],
                dtype=np.float64
            ),
            "intensity_values": np.array(
                [[-np.inf, np.inf]],
                dtype=np.float64
            ),
        }
        for dimension in [
            "frame_indices",
            "scan_indices",
            "tof_indices",
            "precursor_indices",
            "quad_values",
            "intensity_values",
        ]:
            assert np.array_equal(
                result[dimension],
                expected_result[dimension]
            ), (
                f"Key '{key}' in dimension '{dimension}' was wrongly "
                f"converted to {result[dimension]} instead of "
                f"{expected_result[dimension]}"
            )

    def test_data_slicing(self):
        df = self.data[1, :700, 0, 500.1:600.5, [100, 200]]
        min_values = df.min()
        assert min_values.frame_indices >= 1
        assert min_values.mz_values >= 500.1
        max_values = df.max()
        assert max_values.frame_indices < 2
        assert max_values.scan_indices < 700
        assert max_values.mz_values < 600.5
        assert len(
            set(np.unique(df.intensity_values)) - set([100, 200])
        ) == 0
        df = self.data[:100., 1.:]
        assert np.min(df.mobility_values) >= 1
        assert np.min(df.rt_values) < 100.
        df = self.data[:100., :1.]
        assert np.min(df.mobility_values) < 1
        assert np.min(df.rt_values) < 100.
        # TEST


if __name__ == "__main__":
    unittest.main()
