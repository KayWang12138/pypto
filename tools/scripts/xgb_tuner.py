import tuner
import xgboost as xgb
import os
import csv
from scipy import optimize
import random 


class RandomParameterMutator():
    '''
    Takes the current combination of parameters -> randomly selects one parameter -> randomly changes its value.
    '''
    def __init__(self, json_config):
        self.json_config = json_config 

        #how many possible values have one parameter
        self.value_ranges = [] 

        variables_count = 0
        for file_conf in json_config["files"]:
            file, lines = next(iter(file_conf.items()))
            for line in lines:
                value_range = len(self._select_values(line))
                self.value_ranges.append(value_range)
            variables_count += len(lines)

        self.prev = [0] * variables_count
        self.id = 0


    def _select_values(self, line):
        for param_name, values in line.items():
            if param_name not in ("line", "string"):
                return values 

    def _select_value(self, line, idx):
        for param_name, values in line.items():
            if param_name not in ("line", "string"):
                return param_name, values[idx]


    def comb_vector_to_params(self, vector):
        vec_idx = 0
        run_values = []
        run_params = []

        for file_conf in self.json_config["files"]:
            path_to_file, lines_conf = list(file_conf.items())[0]
            run_param = {"file": path_to_file}
            generated_lines_conf = dict()

            for line in lines_conf:
                param_name, param_value = self._select_value(line, vector[vec_idx])
                run_value = {param_name: param_value}
                run_values.append(run_value)
                vec_idx += 1

                line_no = line["line"]
                line_value = line["string"].format(**run_value)
                generated_lines_conf[line_no] = line_value

            run_param["lines"] = generated_lines_conf
            run_params.append(run_param)

        return run_params, run_values 


    def generate_combinations(self):
        if self.id != 0:   
            random_idx = random.randint(0, len(self.prev)-1)
            if self.prev[random_idx] == 0:
                self.prev[random_idx] += min(1, self.value_ranges[random_idx]-1)
            elif self.prev[random_idx] == self.value_ranges[random_idx]-1:
                self.prev[random_idx] -= min(1, self.value_ranges[random_idx]-1)
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
    def __init__(self, model, real_rate, original_run_values, json_config):
        self.original_run_values = original_run_values 
        self.json_config = json_config
        self.executor = tuner.Execution("single_run")
        self.model = model 
        self.idx = 0 
        self.real_rate = real_rate

        self.train_X = []
        self.train_y = []


    def __call__(self, vector):
        if self.idx % self.real_rate == 0:
            run_values = vec_to_run_values(vector, self.original_run_values)
            run_params = run_values_to_run_params(run_values, self.json_config)
            self.executor.result_folder = f"single_run/combination_{self.idx}"
            time = self.executor.measure_perf(self.json_config, run_params)

            self.train_X.append(vector)
            self.train_y.append(time)
            self.model.fit(self.train_X, self.train_y) 

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
        result = {"combination":f"combination_{self.iter}",
                  "time(us)":float(f), 
                  "is_real": self.iter%self.real_rate == 0, 
                  "accepted": accept
                  }
        
        with open(os.path.join(self.save_folder, "results.csv"), "a", newline='') as res_csv:
            writer = csv.DictWriter(res_csv, fieldnames=list(result.keys()))
            if self.iter == 0:
                writer.writeheader()
            writer.writerow(result)
        
        self.iter += 1



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


    REAL_PERF_LIMIT = 100
    REAL_RATE = 10

    model = xgb.XGBRegressor()
    mp = MeasurePerf(
        model=model, 
        real_rate=2, 
        original_run_values=original_run_values, 
        json_config=json_config)

    now = datetime.datetime.now(datetime.timezone.utc)
    today_folder_name = now.strftime("%d_%b_%H_%M_%S")
    os.makedirs(json_config["results_folder"], exist_ok=True)  
    result_folder_path = json_config["results_folder"] + "/" + today_folder_name
    os.makedirs(result_folder_path, exist_ok=True)

    storage = Storage(result_folder_path, REAL_RATE)

    result = optimize.basinhopping(
        mp,
        initial_vector,                
        take_step=take_step,
        niter = REAL_RATE*REAL_PERF_LIMIT//len(temperatures),
        minimizer_kwargs={"method": dummy_minizer},
        callback=storage, 
    )

main()