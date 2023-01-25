from __future__ import annotations

from src.helpers.timer import Timer
from src.helpers.general_helpers import create_path_if_not_exists, path_exists, find_file_by_name

from abc import abstractmethod
import logging
import os
from datasets import concatenate_datasets

logging.basicConfig()

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


class TrainingDataSet:

    def __init__(self, save_dir=None, local_file_name=None):
        self.timer = Timer()
        self.local_data_path = save_dir
        self.data = None
        self.name = None
        self.file_name = local_file_name
        self.init_data_path()
        self.f_extension = ""
        self.timer = Timer()

    def set_data_path(self, path_to_dir):
        self.local_data_path = path_to_dir
        self.init_data_path()

    def set_file_name(self, name):
        self.file_name = name

    def init_data_path(self):
        if self.local_data_path:
            create_path_if_not_exists(self.local_data_path)

    @abstractmethod
    def show_info(self):
        pass

    @abstractmethod
    def load_data(self, percentage=None, try_local=True, save=True, overwrite=False):
        pass

    def save_to_disk(self, overwrite=False):
        print(f"\nSaving dataset {self.name} ...")
        if self.local_data_path is None:
            _logger.warning("No data path was set, use the __.set_data_path method before trying to save a dataset.")
            return
        print(f"Saving to {self.local_data_path} ...")
        if not self.data:
            print("Nothing to save")
        found = find_file_by_name(self.local_data_path, self.file_name)
        if found:
            if not overwrite:
                print(
                    f"File for dataset {self.name} already exists for the specified path, aborting. Use overwrite=True to overwrite the existing file.")
                return
            else:
                print(f"File for dataset {self.name} already exists at the specified path, overwriting.")
        self.save(os.path.join(self.local_data_path, self.file_name))

    @abstractmethod
    def save(self, path):
        pass

    def load_from_disk_if_exists(self):
        if self.local_data_path is None:
            _logger.warning("No data path was set, use the __.set_data_path method before trying to load from disk.")
            return None
        print(f"\nSearching for existing file for dataset {self.name}...")
        if "." in self.file_name:
            path = os.path.join(self.local_data_path, self.file_name)
        else:
            path = os.path.join(self.local_data_path, f"{self.file_name}.{self.f_extension}")
        if path_exists(path):
            print("File found, loading data ...")
            self.load_from_disk(path)
            return True
        else:
            return False

    @abstractmethod
    def load_from_disk(self, path):
        pass

    def concat_with(self, ds: __class__):
        ds_list = concatenate_datasets([self.data, ds.data])
        return ds_list
