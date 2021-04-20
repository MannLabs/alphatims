#!python -m unittest tests.test_utils
"""This module provides unit tests for alphatims.utils."""

# builtin
import unittest

# local
import alphatims.utils
alphatims.utils.set_progress_callback(None)


class TestLogging(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_set_logger(self):
        pass


class TestStacks(unittest.TestCase):

    def test_option_stack(self):
        value1 = (1, 920)
        value2 = (300, 920)
        value3 = (500, 600)
        scan_stack = alphatims.utils.Option_Stack('scans_slider', value1)
        assert str(scan_stack) == f'0 scans_slider [{value1}]', (
            "The instance of the Option_Stack class is generated wrongly."
        )
        scan_stack.update(value2)
        assert str(scan_stack) == f'1 scans_slider [{value1}, {value2}]', (
            "The update of the class instance with a new value doesn't work."
        )
        scan_stack.update(value2)
        assert str(scan_stack) == f'1 scans_slider [{value1}, {value2}]', (
            "The update of the class instance with the same value doesn't work."
        )
        scan_stack.undo()
        assert str(scan_stack) == f'0 scans_slider [{value1}, {value2}]', (
            "Undo function doesn't work correctly."
        )
        scan_stack.update(value3)
        assert str(scan_stack) == f'1 scans_slider [{value1}, {value3}]', (
            "Update of the values doesn't work after Undo function."
        )
        scan_stack.undo()
        scan_stack.redo()
        assert str(scan_stack) == f'1 scans_slider [{value1}, {value3}]', (
            "Redo function doesn't work correctly."
        )

    def test_global_stack(self):
        value1 = (1, 920)
        value2 = (1, 5000)
        value3 = (300, 920)
        value4 = (1000, 4000)
        value5 = (500, 600)
        value6 = (1000, 2000)
        stack = alphatims.utils.Global_Stack(
            {
                "scans_slider": value1,
                "frames_slider": value2,
            }
        )
        assert str(stack) == f"0 scans_slider [{value1}] 0 frames_slider [{value2}] 0 global [None]", (
            "The instance of the Global_Stack class is generated wrongly."
        )
        stack.update("scans_slider", value3)
        assert str(stack) == f"1 scans_slider [{value1}, {value3}] 0 frames_slider [{value2}] 1 global [None, 'scans_slider']", (
            "The update of the first option inside the class instance with a new value doesn't work."
        )
        stack.update("scans_slider", value3)
        assert str(stack) == f"1 scans_slider [{value1}, {value3}] 0 frames_slider [{value2}] 1 global [None, 'scans_slider']", (
            "The update of the same option with the same value doesn't work."
        )
        stack.update("frames_slider", value4)
        assert str(stack) == f"1 scans_slider [{value1}, {value3}] 1 frames_slider [{value2}, {value4}] 2 global [None, 'scans_slider', 'frames_slider']", (
            "The update of the second option with a new value doesn't work."
        )
        stack.update("scans_slider", value5)
        assert str(stack) == f"2 scans_slider [{value1}, {value3}, {value5}] 1 frames_slider [{value2}, {value4}] 3 global [None, 'scans_slider', 'frames_slider', 'scans_slider']", (
            "The update of the first option with a new value doesn't work."
        )
        stack.undo()
        assert str(stack) == f"1 scans_slider [{value1}, {value3}, {value5}] 1 frames_slider [{value2}, {value4}] 2 global [None, 'scans_slider', 'frames_slider', 'scans_slider']", (
            "Undo function doesn't work correctly."
        )
        stack.redo()
        assert str(stack) == f"2 scans_slider [{value1}, {value3}, {value5}] 1 frames_slider [{value2}, {value4}] 3 global [None, 'scans_slider', 'frames_slider', 'scans_slider']", (
            "Redo function doesn't work correctly."
        )
        stack.undo()
        stack.update("frames_slider", value6)
        assert str(stack) == f"1 scans_slider [{value1}, {value3}] 2 frames_slider [{value2}, {value4}, {value6}] 3 global [None, 'scans_slider', 'frames_slider', 'frames_slider']", (
            "Update of the values doesn't work after Undo function."
        )


if __name__ == "__main__":
    unittest.main()
