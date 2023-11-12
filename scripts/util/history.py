import os
import datetime
import pyexcel

from modules.scripts import basedir
from scripts.util.auto_mbw_rt_logger import logger_autombwrt as logger

__location__ = basedir()
history_path = os.path.join(__location__, "csv", "history.tsv")

class MergeHistory():
    def __init__(self):
        if os.path.isfile(history_path):
            self.data = {**pyexcel.get_book_dict(file_name=history_path)}
        else:
            self.data = {os.path.split(history_path)[1]: [["datetime", "current_pass", "model_A", "model_B", "model_O", "weights", "clamp_lower", "clamp_upper", "search_type_A", "search_type_B", "classifier", "tally_type", "init_grid", "init_vertices", "init_random", "warm_start"]]}
    def add_history(self, current_pass, model_A, model_B, model_O, weights, clamp_lower, clamp_upper, search_type_A, search_type_B, classifier, tally_type, init_grid, init_vertices, init_random, warm_start):
        self.data[os.path.split(history_path)[1]].append([f"{datetime.datetime.now()}", current_pass, model_A, model_B, model_O, weights, clamp_lower, clamp_upper, search_type_A, search_type_B, classifier, tally_type, init_grid, init_vertices, init_random, warm_start])
    def write_history(self):
        logger.info(f"Writing merge history to ${history_path}")
        pyexcel.save_book_as(bookdict=self.data, dest_file_name=history_path)


