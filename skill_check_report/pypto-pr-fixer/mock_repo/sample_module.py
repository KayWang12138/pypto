# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
import logging
import os
import sys


def process_data(data):
    logging.info(f"Processing: {data}")
    if not data:
        logging.info("Empty data received")
    return data


def analyze(input_val):
    try:
        result = do_something(input_val)
    except ValueError as e:
        raise RuntimeError("Processing failed") from e
    except Exception as e:
        raise RuntimeError("Unexpected error") from e
    return result


def do_something(val):
    return val * 2


class DataProcessor:
    def __init__(self, config):
        self.config = config

    def __repr__(self):
        return f"DataProcessor({self.config})"

    @staticmethod
    def static_util():
        pass

    def public_method(self):
        pass

    def another_public(self):
        pass

    def _private_helper(self):
        pass
