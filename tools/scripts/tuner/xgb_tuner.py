#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""
import os
import csv
import random 
import datetime
from scipy import optimize
import xgboost as xgb
import tuner


class RandomParameterMutator():
    '''
    Takes the current combination of parameters -> 
    selects one parameter -> 
    get next value in candidates list or previous.
    '''
    def __init__(self, json_config):
        self.json_config = json_config 
        self.tune_params = make_flat(json_config)

        #how many possible values have one parameter
        self.value_ranges = [] 
        for param_name in self.tune_params:
            candidates_size = len(self.tune_params[param_name])
            self.value_ranges.append(candidates_size)

        variables_count = len(self.tune_params)
        self.prev = [0] * variables_count
        self.id = 0

    def comb_vector_to_params(self, vector):
        vec_idx = 0
        run_values = []
        run_params = []

        for file_conf in self.json_config["files"]:
            path_to_file, lines_conf = list(file_conf.items())[0]
            run_param = {"file": path_to_file}
            generated_lines_conf = dict()
            line_info = flat_line_info(self.json_config)

            for param_name, param_values in self.tune_params.items():
                candidate_idx = vector[vec_idx]
                param_value = param_values[candidate_idx]

                run_value = {param_name: param_value}
                run_values.append(run_value)

                line_no = line_info[param_name]["line"]
                line_value = line_info[param_name]["string"].format(**run_value)
                generated_lines_conf[line_no] = line_value

                vec_idx += 1

            run_param["lines"] = generated_lines_conf
            run_params.append(run_param)

        return run_params, run_values 

    def generate_combinations(self):
        if self.id != 0:   
            random_idx = random.randint(0, len(self.prev) - 1)
            if self.prev[random_idx] == 0:
                self.prev[random_idx] += min(1, self.value_ranges[random_idx] - 1)
            elif self.prev[random_idx] == self.value_ranges[random_idx] - 1:
                self.prev[random_idx] -= min(1, self.value_ranges[random_idx] - 1)
            else:
                self.prev[random_idx] += random.choice([-1, 1])

        self.id += 1
        run_params, run_values = self.comb_vector_to_params(self.prev)
        return run_params, run_values


def run_values_to_vec(run_values):
    flat_list = [item for d in run_values for item in list(d.values())[0]]
    return flat_list


def vec_to_run_values(vec, original_run_values):
    run_values = original_run_values.copy()
    pointer = 0

    vec = vec.copy()
    vec = list(map(int, vec))

    for run_value in run_values:
        for key in run_value.keys():
            part = vec[pointer: pointer + len(run_value[key])]
            run_value[key] = part
            pointer += len(run_value[key])

    return run_values


def run_values_to_run_params(run_values, json_config):
    vec_idx = 0
    run_params = []

    for file_conf in json_config["files"]:
        path_to_file, lines_conf = list(file_conf.items())[0]
        run_param = {"file": path_to_file}
        generated_lines_conf = dict()

        for line in lines_conf:
            run_value = run_values[vec_idx]
            vec_idx += 1

            line_no = line["line"]
            line_value = line["string"].format(**run_value)
            generated_lines_conf[line_no] = line_value

        run_param["lines"] = generated_lines_conf
        run_params.append(run_param)

    return run_params 


class MeasurePerf:
    def __init__(self, model, real_rate, original_run_values, result_folder_path, json_config):
        self.original_run_values = original_run_values 
        self.json_config = json_config
        self.executor = tuner.Execution("single_run")
        self.model = model 
        self.idx = 0 
        self.real_rate = real_rate

        self.result_folder = result_folder_path

        self.train_x = []
        self.train_y = []


    def __call__(self, vector):
        if self.idx % self.real_rate == 0:
            run_values = vec_to_run_values(vector, self.original_run_values)
            run_params = run_values_to_run_params(run_values, self.json_config)
            self.executor.result_folder = f"{self.result_folder}/combination_{self.idx}"
            time = self.executor.measure_perf(self.json_config, run_params)

            self.train_x.append(vector)
            self.train_y.append(time)
            print(self.train_x)
            print(self.train_y)
            self.model.fit(self.train_x, self.train_y) 
        else:
            time = self.model.predict([vector])[0]
        self.idx += 1

        return time 


class Storage:
    def __init__(self, save_folder, real_rate):
        self.iter = 0
        self.save_folder = save_folder
        self.real_rate = real_rate

    def __call__(self, x, f, accept):
        result = {"combination": f"combination_{self.iter}",
                  "time(us)": float(f), 
                  "is_real": self.iter % self.real_rate == 0, 
                  "accepted": accept
                  }
        
        with open(os.path.join(self.save_folder, "results.csv"), "a", newline='') as res_csv:
            writer = csv.DictWriter(res_csv, fieldnames=list(result.keys()))
            if self.iter == 0:
                writer.writeheader()
            writer.writerow(result)
        
        self.iter += 1


def make_flat(json_confg):
    """
    Flatten configuration values from a nested JSON‑style structure.

    This function return dictionary with format 
    variable_name:candidates 
    
    Example:
        {
            "my_tile1" : [tile_1, ... , tile_n],
            "my_flag" : [value_1, ... , value_n],
            ....
        }
    """
    for file_conf in json_confg["files"]:
        values = [
            candidates
            for filename, line_conf in file_conf.items()
            for line in line_conf
            for candidates in line.values()
            if isinstance(candidates, list)
        ]

        keys = [
            variable_name
            for filename, line_conf in file_conf.items()
            for line in line_conf
            for variable_name in line.keys()
            if not variable_name.startswith("string") and not variable_name.startswith("line")
        ]
    
    candidates = dict(zip(keys, values))
    return candidates


def flat_line_info(json_confg):
    """
    Flatten configuration values from a nested JSON‑style structure.

    This function return dictionary with format 
    variable_name: {apth_to_file, line, format_string}
    
    Example:
        {
            "my_tile1" : {file: "path_to_file", line: 12, string: "format_string"},
            "my_flag" : {file: "path_to_file", line: 113, string: "format_string"},
            ....
        }
    """
    for file_conf in json_confg["files"]:
        values = [
            {
                "file": filename,
                "line": line["line"],
                "string": line["string"]
            }
            for filename, line_conf in file_conf.items()
            for line in line_conf
        ]

        keys = [
            variable_name
            for filename, line_conf in file_conf.items()
            for line in line_conf
            for variable_name in line.keys()
            if not variable_name.startswith("string") and not variable_name.startswith("line")
        ]
    
    line_info = dict(zip(keys, values))
    return line_info



def main():
    json_config = tuner.parse_json()



    mutator = RandomParameterMutator(json_config)
    executor = tuner.Execution("single_run")
    run_params, run_values = mutator.generate_combinations()    
    original_run_values = run_values


    def dummy_minizer(f, x0, args, **kwargs):
        #we don't implement any local optimization
        dummy_result = optimize.OptimizeResult(
            x=x0,
            fun=f(x0),
            success=True,
            message="Dummy local optimization",
        )
        return dummy_result


    def take_step(vector):
        run_params, run_values = mutator.generate_combinations()
        return run_values_to_vec(run_values)

    initial_vector = run_values_to_vec(run_values)

    now = datetime.datetime.now(datetime.timezone.utc)
    today_folder_name = now.strftime("%d_%b_%H_%M_%S")
    os.makedirs(json_config["results_folder"], exist_ok=True)  
    result_folder_path = json_config["results_folder"] + "/" + today_folder_name
    os.makedirs(result_folder_path, exist_ok=True)

    real_perf_limit = 100
    real_rate = 10

    model = xgb.XGBRegressor()

    mp = MeasurePerf(
        model=model, 
        real_rate=real_rate, 
        original_run_values=original_run_values,
        result_folder_path=result_folder_path,
        json_config=json_config)

    storage = Storage(result_folder_path, real_rate)
    temperatures = [10, 5, 1, 0.25]

    for temp in temperatures:
        result = optimize.basinhopping(
            mp,
            initial_vector,                
            take_step=take_step,
            niter=real_rate * real_perf_limit // len(temperatures),
            minimizer_kwargs={"method": dummy_minizer},
            callback=storage, 
            T=temp
        )

if __name__ == '__main__':
    main()