# pypto.set\_verify\_options

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series / Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series / Atlas A2 Inference Series |    √     |

## Description

Sets the self-check enable switch and the corresponding feature options for the precision debugging Verify feature.

## Function Prototype

```python
set_verify_options(*,
                   enable_pass_verify: Optional[bool] = None,
                   pass_verify_save_tensor: Optional[bool] = None,
                   pass_verify_save_tensor_dir: Optional[str] = None,
                   pass_verify_pass_filter: Optional[List[str]] = None,
                   pass_verify_error_tol: Optional[List[float]] = None,
                   ) -> None
```

## Parameters


| Parameter                       | Input/Output | Description                                                                 |
|---------------------------------|--------------|-----------------------------------------------------------------------------|
| enable_pass_verify              | Input        | Meaning: Master enable switch that determines whether all *pass_verify_* options and interfaces are active. <br> Note: True means enabled. <br> Type: bool <br> Value range: True/False <br> Default: False |
| pass_verify_save_tensor         | Input        | Meaning: Configures whether to save simulation computation data to disk. <br> Note: True means save to disk. <br> Type: bool <br> Value range: True/False <br> Default: False |
| pass_verify_save_tensor_dir     | Input        | Meaning: Configures the save path for detection results and data. <br> Note: A string specifying an absolute path. <br> Type: str <br> Default: <br> "{RUNNING_DIR}/output/output_{TS}" |
| pass_verify_pass_filter         | Input        | Meaning: Configures the list of pass names to self-check. <br> Note: Must be valid pass names. <br> If not specified, the following passes are verified by default: ["ExpandFunction", "SplitK", "L1CopyInReuseMerge", "InferDynShape", "InferParamIndex", "CodegenPreproc"]; specifying "all" verifies all passes; specifying [] skips pass verification and only verifies the tensor_graph; invalid names are ignored. <br> Type: List[str] <br> Default: empty |
| pass_verify_error_tol           | Input        | Meaning: Configures the rtol and atol values used by the precision tool for comparison. <br> Note: The first value in the list is rtol and the second is atol. If the list length is not 2, the default values are used. <br> Type: List[float] <br> Default: [1e-3, 1e-3] |

## Return Value

void: Set method has no return value. The setting takes effect immediately upon success.

## Constraints

## Example

```python
verify_options = {
        "enable_pass_verify": True,
        "pass_verify_save_tensor": True,
        "pass_verify_save_tensor_dir": "/LARGE/DRIVE/DIR",
        }
pypto.set_verify_options(**verify_options)
```

