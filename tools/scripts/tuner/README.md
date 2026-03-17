# Tune tool

## Usage 

```
python tools/tuner/tuner.py --json_path tools/tuner/config.json
```

## Description of config.json 

*results_folder* - path where all results will be stored

*save_best_k* - allows saving folder with launch information only for best combinations

*files* - list of files where replacement is necessary. For each file you must specify the line number and format string, as well as the parameter names. 

For each parameter, a list of values must be made.

**Parameter names must be unique within the config.json file.** It useful for representation results in results.csv file, these names will be column names
in table.

*line* - the line number you want to change in the source code file. For example you want to check different tiles on 
test glm_moe_fusion. You need to open `models/glm_v4_5/glm_moe_fusion.py` in your text editor and find line thats start `pypto.set_vec_tile` or `pypto.set_cube_tile`. Line numbers of this string it is what you need. 

*string* - write a format string as well as for the .format() function in the Python language. The values for these format string are parameter names. Using format strings allow to you change different types of parameters. 

*parameters* - just set name for your parameter. It will be used in format string and will display in final performance csv table. Enumerate in *list* possible values.

## Example of heuristic tiles

Alternatively, you can ask the script to generate the tiling automatically for the matmul with shapes you specify. Just set parameter in json (Matmul_datatype_m_k_n)

```
{
    "line": 384, 
    "string": "pypto.set_cube_tile_shapes([{mm1[0]},{mm1[0]}],[{mm1[1]},{mm1[2]}], [{mm1[3]},{mm1[4]}])"
    "mm1": Matmul_int8_48_1536_24576 
}
```

The tiling is generated in such a way that it does not exceed L0A, L0B, L0C, L1 caches and pick top 5 best. 

## Output

After finish the script, you will get a folder with a similar structure

```
results_folder
└── 26_Mar_18_57_43
    ├── combination_0 <- result of run for one combination
    │   ├── combination_params.json <- combination description
    │   └── tiling.log <- logs of run
    ├── combination_1 
    │   ├── combination_params.json 
    │   └── tiling.log
    ├── combination_2 
    │   ├── combination_params.json 
    │   └── tiling.log
    ├── combination_3 
    │   ├── combination_params.json 
    │   └── tiling.log 
    └── configs_perf.csv
```

In config_perf.csv all results will be sorted by time(us)

### config_perf.csv example

|combination|amax1|mm1|time(us)
|-------|-------|-------|-------| 
combination_0|[1,1,160]|[1,1,512,1024,16,16]|47.3|
combination_1|[1,1,160]|[1,1,512,1024,16,16]|58.3|
combination_2|[1,1,256]|[32,32,128,128,256,256]|62.4|
combination_3|[1,1,256]|[32,32,128,128,256,256]|68.8|

## XGBTune

To speed up the search, we can use the XGBoostRegressor model, which is capable of predicting performance on tiling combination.

The results and usage of xgb_tuner.py is the same as tuner.py. However, not every combination is run on the device, some of them are evaluated during the search using the trained model. The model is trained during real runs.

in config_perf.csv you can find new column `is_real`. 
* `true` if result was achieved by running on device
* `false` if result was estimated by xgboost model

### config_perf.csv example

|combination|amax1|mm1|is_real|time(us)
|-------|-------|-------|-------|-------| 
combination_0|[1,1,160]|[1,1,512,1024,16,16]|true|47.3|
combination_1|[1,1,160]|[1,1,512,1024,16,16]|true|58.3|
combination_2|[1,1,256]|[32,32,128,128,256,256]|false|62.4|
combination_3|[1,1,256]|[32,32,128,128,256,256]|false|68.8|