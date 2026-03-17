import re
import os 
import sys 
import argparse
import json
import subprocess
import shutil
import random
# import xgboost as xgb
from pathlib import Path
from itertools import product
from scipy.stats import gmean
from scipy import optimize

script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = Path(script_dir).parents[1]
target_dir = os.path.join(base_path, '.agents', 'skills', 'pypto-operator-perf-analyzer', 'scripts')
sys.path.insert(0, target_dir)
import analyze_perf

class Patcher:
    '''
    Modify files according to run_params
    '''
    def __init__(self, run_params):
        self.run_params = run_params
        self.backups = []
    
    def __enter__(self):
        for item in self.run_params:
            self._make_backup(item["file"])
            self.backups.append(item["file"])
        return self
    
    def apply_changes(self):
        for item in self.run_params:
            sorted_by_lines = dict(sorted(item["lines"].items(), reverse=True))
            for line_num, text in sorted_by_lines.items():
                self._replace_line(item["file"], line_num, text)
    
    def _replace_line(self, file_path, line_num, new_content):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            prev_line = lines[line_num-1].rstrip()
            
            leading_spaces = re.match(r'^\s*', prev_line).group(0) 
            if prev_line.endswith(":"):
                leading_spaces += "\t"

            new_content = leading_spaces+new_content
            lines.insert(line_num, new_content + "\n")
        
        with open(file_path, "w") as file:
            file.writelines(lines)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for filepath in self.backups:
            self._restore_backup(filepath)
        return False
    
    def _make_backup(self, path):
        shutil.copy(path, path + ".backup")
    
    def _restore_backup(self, path):
        backup = path + ".backup"
        if os.path.exists(backup):
            os.replace(backup, path)


class Execution:
    def __init__(self, result_folder):
        self.result_folder = result_folder

    def _get_execution_time(self):
        dirs = list(filter(os.path.isdir, [os.path.join("output", f) for f in os.listdir("output")]))
        newest_folder = max(dirs, key=os.path.getmtime)
        bubble_log_path = os.path.join(newest_folder, 'bubble_analysis.log')
        cores = analyze_perf.parse_bubble_analysis(bubble_log_path)
        metrics = analyze_perf.calculate_performance_metrics(cores)
        return metrics['max_work_time']

    def _run_test(self, json_config):
        device = json_config["device_number"]
        test = json_config["test_name"]
        run_command = f"python3 build_ci.py -j 80 -d {device} -f python3 -s {test}"
        print(run_command)
        log_path = f"{self.result_folder}/tiling.log"
        print(f"Build logs: {log_path}")

        env = dict(os.environ)
        with open(log_path, "w") as f:
            test = subprocess.run(run_command.split(), stdout=f, stderr=subprocess.STDOUT, env=env)
        return test.returncode

    def measure_perf(self, json_config, run_params):
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)

        with Patcher(run_params) as patcher:
            patcher.apply_changes()
            self._save_run_params(self.result_folder,
                run_params, self.result_folder + "/combination_params.json")
            test_result = self._run_test(json_config)

        if test_result != 0:
            return None
        
        time = self._get_execution_time()
        return time

    def _save_run_params(self, name, run_params, file_name):
        run_params = run_params.copy()
        with open(file_name, 'w') as run_file:
            run_params.append({"name": name})
            json_txt = json.dumps(run_params, indent=4)
            run_file.write(json_txt)


class HeuristicTile:
    def __init__(self, 
                L1_SIZE = 131072, 
                L0A_SIZE = 65536, 
                L0B_SIZE = 65536, 
                L0C_SIZE = 524288, 
                OUTPUT_DT_BYTES = 4):

        self.L1_SIZE = L1_SIZE
        self.L0A_SIZE = L0A_SIZE
        self.L0B_SIZE = L0B_SIZE
        self.L0C_SIZE = L0C_SIZE
        self.OUTPUT_DT_BYTES = OUTPUT_DT_BYTES



    def is_good_tiling(self, tile, input_dt_bytes):
        m, m_big, k, k_big, n, n_big = tile
        r1 = k_big % k == 0 and n_big >= n
        r2 = m * n * self.OUTPUT_DT_BYTES <= self.L0C_SIZE
        r3 = n * k * input_dt_bytes <= self.L0B_SIZE
        r4 = m * k * input_dt_bytes <= self.L0A_SIZE
        return r1 and r2 and r3 and r4


    def greatest_bit(self, x):
        n = 0
        power = 1 
        while power <= x:
            power *= 2 
            n += 1
        return n - 1


    def generate_tiles(self, shape, input_dt_bytes):
        shape_m, shape_k, shape_n = shape
        num_of_variants = 0

        min_m = min(3, max(self.greatest_bit(shape_m), 4))
        min_k = min(3, max(self.greatest_bit(shape_k), 4))
        min_n = min(3, max(self.greatest_bit(shape_n), 4))

        values_m = [2**i for i in range(min_m, self.greatest_bit(shape_m) + 1)]
        values_k = [2**i for i in range(min_k, self.greatest_bit(shape_k) + 1)]
        values_n = [2**i for i in range(min_n, self.greatest_bit(shape_n) + 1)]

        for m, k, n in product(values_m, values_k, values_n):
            m_big = m
            k_big = k
            n_big = n

            if self.is_good_tiling([m, m_big, k, k_big, n, n_big], input_dt_bytes):
                num_of_variants += 1
                yield [m, m_big, k, k_big, n, n_big]

            if self.is_good_tiling([m, m_big, k, 2 * k_big, n, n_big], input_dt_bytes):
                num_of_variants += 1
                yield [m, m_big, k, 2 * k_big, n, n_big]
            
            if self.is_good_tiling([m, m_big, k, k_big, n, 2 * n_big], input_dt_bytes):
                num_of_variants += 1
                yield [m, m_big, k, k_big, n, 2 * n_big]


    def get_score_for_tiling(self, tile, mx_shape, input_type_size):
        """
        Estimate score for tiling. The higher the score, the better it is.
        """
        m_dim = 0
        k_dim = 2
        n_dim = 4
        
        m = tile[m_dim]
        k = tile[k_dim]
        n = tile[n_dim]

        min_tile = 16
        min_tile = 200

        balance_weight = 1

        whole_m_score = 2
        whole_k_score = 2
        whole_n_score = 2

        score = 0
        
        #If the tiling size = shape size -> the preferred option
        score = (score + whole_m_score) if tile[m_dim] == max(m, min_tile) else score
        score = (score + whole_k_score) if tile[k_dim] == max(k, min_tile) else score
        score = (score + whole_n_score) if tile[n_dim] == max(n, min_tile) else score

        #The more filled L0A, L0B, L0C is better
        utilization_l0a = (tile[m_dim] * tile[k_dim] * input_type_size) / self.L0A_SIZE
        utilization_l0b = (tile[k_dim] * tile[n_dim] * input_type_size) / self.L0B_SIZE 
        utilization_l0c = (tile[m_dim] * tile[n_dim] * self.OUTPUT_DT_BYTES) / self.L0C_SIZE
        score += min_tile * gmean([utilization_l0a, utilization_l0b, utilization_l0c])

        #The closer the ratio's is to 1, the better
        ratio_mk = (m / k) if (m > k) else (k / m)
        ratio_kn = (k / n) if (k > n) else (n / k)
        ratio_mn = (m / n) if (m > n) else (n / m)

        #Penalty for bad balance
        score -= balance_weight * gmean([ratio_mk, ratio_kn, ratio_mn])
        return score


    def get_best_k_tiles(self, tiles, shape, dt_bytes, best_k_tiles):
        tiles.sort(key=lambda tile: -self.get_score_for_tiling(tile, shape, dt_bytes))

        best_tiles = []
        d = set()
        i = 0

        while len(d) < best_k_tiles:
            shape = tiles[i]
            if (shape[0], shape[2], shape[4]) not in d:
                d.add((shape[0], shape[2], shape[4]))
                best_tiles.append(shape)
            i += 1

        return best_tiles


    def preproc_line_conf(self, line_conf):
        for key in line_conf.keys():
            if key != "string" and isinstance(line_conf[key], str):
                #param set like Matmul_int08_32_1536_783
                operation, datatype, m, k, n = line_conf[key].split("_")
                m = int(m)
                k = int(k)
                n = int(n)
    
                dt_bytes = int(re.search(r"\d+", datatype).group(0)) // 8
                tiles = list(self.generate_tiles([m, k, n], input_dt_bytes=dt_bytes))
                best_tiles = self.get_best_k_tiles(tiles, [m, k, n], dt_bytes, best_k_tiles=5)

                print(f"Autogenerated shape for {line_conf[key]}")
                for tile in best_tiles:
                    print(tile)
                print("--------------------------")

                line_conf[key] = best_tiles


def check_json(json_config):
    param_names = dict()
    for file_conf in json_config["files"]:
        path_to_file, lines_conf = list(file_conf.items())[0]
        for line_conf in lines_conf:
            line_param_names = list(line_conf.keys())[2:] # except params "string" and "line"
            for name in line_param_names:
                if name in param_names:
                    raise NameError(f"Name \"{name}\" in json config are same " \
                                    f"for lines {param_names['name']} and {line_conf['line']}")
                param_names[name] = line_conf["line"] 


def preproc_json(json_config):
    """
    Replaces parameters that are set using strings  
    """
    ht = HeuristicTile()
    check_json(json_config)
    for file_conf in json_config["files"]:
        path_to_file, lines_conf = list(file_conf.items())[0]
        for line_conf in lines_conf:
            best_tile = ht.preproc_line_conf(line_conf)

class ExhaustiveGenerator:
    def __init__(self, json_config):
        self.json_config = json_config
    
    def _parse_file_conf(self, lines_conf):
        """
        Generates permutations for single line of code
        """
        lines_comb = []
        for line_conf in lines_conf:
            params = line_conf.copy()

            del params["line"]
            del params["string"]

            string_param_comb = []
            for e in product(*params.values()):
                param_comb = dict(zip(params.keys(), e))
                result = line_conf["string"].format(**param_comb)
                string_param_comb.append([(line_conf["line"], result), param_comb])
            lines_comb.append(string_param_comb)
        
        return list(product(*lines_comb))

    def generate_combinations(self):
        """
        Generates all possible permutations of tiles inside config.json
        """       
        comb_inside_file = dict()
        for file_conf in self.json_config["files"]:
            path_to_file, lines_conf = list(file_conf.items())[0]
            comb_inside_file[path_to_file] = self._parse_file_conf(lines_conf)
        
        for e in product(*comb_inside_file.values()):
            single_run_params = []
            single_param_values = []

            for cpp_file, line_params in zip(comb_inside_file.keys(), e):
                d = dict()
                d["file"] = cpp_file
                num_lines = dict()

                for line_param in line_params:
                    line, string = line_param[0]
                    names = line_param[1]
                    single_param_values.append(names)
                    num_lines[line] = string

                d["lines"] = num_lines   
                single_run_params.append(d)

            yield single_run_params, single_param_values


class XGBGenerator(ExhaustiveGenerator):
    def _parse_line_conf(self, line_conf):
        res = {"line":None, "string":None, "tile":None, "name":None}

        for key in line_conf.keys():
            if key == "line" or key=="string":
                res[key] = line_conf[key]
            else:
                res["name"] = key
                res["tile"] = line_conf[key]
        return res 

    def get_initial(self):
        parent = super().generate_combinations()
        initial_combination = next(iter(parent))
        run_params, run_values = initial_combination

    def generate_combinations(self):
        #Выбрать первую комбинацию
        #Сгенерировать ближайшую
        #Если для нее можно предсказать время
        # model = xgb.XGBRegressor()

        # print(initial_combination)
        # print(len(initial_combination))

        for p in run_params:
            print(p)



#         # ind = 0
       

#         # for file in self.json_config["files"]:
#         #     filename, lines_conf = next(iter(file.items()))
#         #     print(filename)
#         #     print("-----------")
#         #     print(lines_conf)

#         #     random_ind = random.randint(0, len(run_values))

#         #     for line_conf in lines_conf:
#         #         if ind == random_ind:
#         #             tiles = self._parse_line_conf(line_conf)["tile"]
#         #             random_variable = random.choice(tiles)
#         #             print("Random tile: ", random_tile)





def main():
    parser = argparse.ArgumentParser(description='Iterate through the vector and cube tiles ' \
        'in the specified test and measure performance \n' \
        'To specify which tiles and tile values to iterate through, use config.json \n' \
        'To start the iteration: \n\n' \
        'python tools/scripts/tiling_tool.py --json_path /path/to/config.json', 
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--json_path", 
                        help="path to config.json where described tiling configs", 
                        type=str,
                        required=True)
    
    args = parser.parse_args()

    with open(args.json_path) as f:
        json_config = json.load(f)
        preproc_json(json_config)

    
    if json_config["tune_type"] == "exhaustive":
        executor = Execution("single_run")
        generator = ExhaustiveGenerator(json_config)
        idx = 0

        for run_params, run_values in generator.generate_combinations():
            executor.result_folder = f"single_run/combination_{idx}"
            time = executor.measure_perf(json_config, run_params)
            print(idx, time)
            idx += 1
            
    elif json_config["tune_type"] == "xgb":
        


        # def dummy_minizer(f, x0, args, **kwargs):
        #     #we don't implement any local optimization
        #     dummy_result = optimize.OptimizeResult(
        #         x=x0,
        #         fun=f(x0),
        #         success=True,
        #         message="Dummy local optimization",
        #     )
        #     return dummy_result


        # temperatures = [10, 5, 1, 0.25]
        # REAL_PERF_LIMIT = 100
        # REAL_RATE = 30

        # for temp in temperatures:
        #     result = optimize.basinhopping(
        #         perf_measure,
        #         initial_vector,                
        #         take_step=changer,
        #         niter = REAL_RATE*REAL_PERF_LIMIT//len(temperatures),
        #         minimizer_kwargs={"method": dummy_minizer},
        #         callback=storage, 
        #     )
        #     initial_vector = result.x  


        # generator = XGBGenerator(json_config)
        # generator.generate_combinations()

# main()




