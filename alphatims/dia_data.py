# builtin
import os
import dataclasses

# external
import alphatims.bruker
import numpy as np

# local
import alphatims.compiling


@alphatims.compiling.njit_class
@dataclasses.dataclass(kw_only=True, frozen=True)
class DiaData:

    file_name: str = dataclasses.field(repr=False)
    sample_name: str = dataclasses.field(init=False)

    def __init__(
        self,
        *,
        dia_data: alphatims.bruker.TimsTOF,
    ):
        file_name = self.parse_file_name(dia_data.bruker_d_folder_name)
        object.__setattr__(self, "cycle", dia_data._cycle)
        object.__setattr__(self, "im_values", dia_data._mobility_values)
        object.__setattr__(self, "rt_values", dia_data._rt_values)
        object.__setattr__(self, "mz_values", dia_data._mz_values)
        object.__setattr__(self, "frames", dia_data._frames)
        object.__setattr__(self, "fragment_frames", dia_data._fragment_frames)
        object.__setattr__(self, "zeroth_frame", dia_data._zeroth_frame)
        object.__setattr__(self, "tof_indptr", dia_data._push_indptr)
        object.__setattr__(self, "tof_indices", dia_data._tof_indices)
        object.__setattr__(
            self,
            "intensity_values",
            dia_data._intensity_values.astype(np.float32)
        )
        object.__setattr__(self, "as_dataframe", dia_data.as_dataframe)
        object.__setattr__(self, "bin_intensities", dia_data.bin_intensities)
        object.__setattr__(self, "_dia_data", dia_data)

        root_directory = os.path.basename(file_name)
        sample_name = '.'.join(
            os.path.basename(file_name).split('.')[:-1]
        )
        object.__setattr__(self, "file_name", file_name)
        object.__setattr__(self, "sample_name", sample_name)
        object.__setattr__(self, "directory", root_directory)

    def __getitem__(self, keys):
        return self._dia_data[keys]

    @staticmethod
    def parse_file_name(file_name):
        file_name = file_name.strip()
        while file_name[-1] in ["'", '"']:
            file_name = file_name[:-1]
        while file_name[0] in ["'", '"']:
            file_name = file_name[1:]
        if not file_name.endswith(".d"):
            raise ValueError(
                f"File {file_name} is not a valid Bruker .d file."
            )
        return file_name

    def __len__(self):
        return self.size
