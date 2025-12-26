import numpy as np
import os
import json


X_HOME = os.path.dirname(__file__)
CASE_ROOT_PATH = os.path.join(X_HOME, "testcase", "tensorflow_case")

rtols = {
    np.float16: 0.001,
    np.float32: 0.0001,
    np.int32: 0.0001,
    np.int16: 0.0001
}
atols = {
    np.float16: 0.001,
    np.float32: 0.0001,
    np.int32: 0.0001,
    np.int16: 0.0001
}

dtype_map = {
    "fp16": np.float16,
    "fp32": np.float32,
    "int32": np.int32,
    "int16": np.int16
}


def get_all_op(test_path):
    return os.listdir(test_path)


def get_all_case(op_path):
    cases = []
    for case in os.listdir(op_path):
        if os.path.isfile(os.path.join(op_path, case)) and case.endswith(".cs"):
            cases.append(case[:-3])
    return cases


def get_output_dtypes(op_path, case_name):
    cs_file_tp = case_name + ".cs"
    with open(os.path.join(op_path, cs_file_tp)) as cs:
        cs_json = json.load(cs)
        output_dtypes = cs_json.get("dtype_output")

    return [dtype_map.get(dtype) for dtype in output_dtypes]


def get_tols(op_path, case_name):
    cs_file_tp = case_name + ".cs"
    with open(os.path.join(op_path, cs_file_tp)) as cs:
        cs_json = json.load(cs)
        rtol = cs_json.get("diff_thd")
        atol = cs_json.get("pct_thd")

    return rtol, atol


def get_cpu_out(op_path, case_name, dtypes):
    cpu_outs = []
    cpu_out_file_tp = case_name + "_cpu_output_{}.bin"
    index = 0
    while True:
        cpu_out_file_name = cpu_out_file_tp.format(index)
        if os.path.exists(os.path.join(op_path, cpu_out_file_name)):
            cpu_outs.append(np.fromfile(os.path.join(op_path, cpu_out_file_name), dtype=dtypes[index]))
        else:
            break
        index = index + 1
    return cpu_outs


def get_npu_out(op_path, case_name, dtypes):
    npu_outs = []
    npu_out_file_tp = "output_" + case_name + "_input_0_0{}.bin"
    index = 0
    while True:
        npu_out_file_name = npu_out_file_tp.format(index)
        if os.path.exists(os.path.join(op_path, case_name, npu_out_file_name)):
            npu_outs.append(np.fromfile(os.path.join(op_path, case_name, npu_out_file_name), dtype=dtypes[index]))
        else:
            break
        index = index + 1
    return npu_outs


def is_close(cpu_out, npu_out, dtype, rtol, atol):
    if rtol is None or atol is None:
        rtol = rtols.get(dtype)
        atol = atols.get(dtype)
    return np.allclose(cpu_out, npu_out, rtol=rtol, atol=atol)


def data_compare_np(npu_output, cpu_output, diff_thd=0.01, pct_thd=0.05, max_diff_hd=0.1, is_display=False):
    real_data = npu_output.flatten()
    data_compe = cpu_output.flatten()
    max_error_idx = len(data_compe)
    if real_data.size == 0 and real_data.size == data_compe.size:
        print('The npu_output is [],and it is same as bm_output, the result of data_compare is \"Pass\"')
        return "Pass", 100.0, 0
    start = 0
    end = real_data.size - 1
    if end < start:
        end = start
    max_error = 0
    result = "Failed"
    if real_data.size != data_compe.size:
        print(
            'Error,the size of npu output[%s] and benchmark[%s] is not equal.' % (real_data.size, data_compe.size))
        return result, 0.0, max_error

    overflows_count = data_compe[np.isinf(data_compe)].size + data_compe[np.isnan(data_compe)].size
    split_count = int(end - start + 1) if end != start else 1

    try:
        diff_abs = np.abs(np.subtract(real_data.astype(np.float32), data_compe.astype(np.float32)))
    except MemoryError:
        return result, 0.0, max_error
    diff_index = np.where(diff_abs > 0)
    rdiff = cal_relative_diff_np(real_data[diff_index].astype(np.float32), data_compe[diff_index].astype(np.float32),
                                 diff_thd)
    err_diff = rdiff[rdiff > diff_thd]
    diff_idx_list = diff_index[0]
    err_idx = diff_idx_list[np.where(rdiff > diff_thd)]
    fulfill_percent = float(split_count - err_diff.size) / float(split_count) * 100.0

    pct_thd = (1 - pct_thd) * 100.0
    result = "Pass" if (fulfill_percent >= pct_thd) else "Failed"
    fulfill_percent = float('%.4f' % fulfill_percent)
    if len(err_diff) > 0:
        max_error = max(err_diff[0:max_error_idx])
        if max_error >= max_diff_hd:
            result = "Failed"

    if is_display:
        if overflows_count > 0:
            print('Overflow,size:%s,benchmark_output:%s, %s' % (
                overflows_count, data_compe[np.isinf(data_compe)][0:10], data_compe[np.isnan(data_compe)][0:10]))
        print('split_count:%s; max_diff_hd:%s;' % (float(split_count), max_diff_hd))

        display_output(real_data, data_compe, start, end, diff_thd)
        print('---------------------------------------------------------------------------------------')
        print('DiffThd  \t PctThd   \t PctRlt   \t Result')
        print('---------------------------------------------------------------------------------------')
        print('%.4f     \t %.2f%%   \t %.6f%%   \t %s' % (diff_thd, pct_thd, fulfill_percent, result))
        if len(err_diff) > 0:
            print('Max-RelativeError is: %s. Threshold is: %s.' % (max_error, max_diff_hd))
        if result == "Failed":
            display_error_output(real_data, data_compe, err_idx, err_diff[0:max_error_idx])

    return result, fulfill_percent, max_error


def cal_relative_diff_np(real_data, expect_data, diff_thd):
    a = np.abs(np.subtract(real_data, expect_data))
    b1 = np.maximum(np.abs(real_data), (np.abs(expect_data)))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    result = np.where(a < diff_thd, a, a / b)
    return result


def display_output(real_data, expect_data, start, end, diff_thd):
    print('---------------------------------------------------------------------------------------')
    print('Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff')
    print('---------------------------------------------------------------------------------------')
    split_count = int(end - start)
    if split_count <= 20:
        for i in range(split_count + 1):
            j = i + start
            print('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                start + i + 1, expect_data[j], real_data[j], abs(np.float64(expect_data[j]) - np.float64(real_data[j])),
                cal_relative_diff(expect_data[j], real_data[j], diff_thd)))
    else:
        for i in range(10):
            j = i + start
            print('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                start + i + 1, expect_data[j], real_data[j], abs(np.float64(expect_data[j]) - np.float64(real_data[j])),
                cal_relative_diff(expect_data[j], real_data[j], diff_thd)))
        print('...   \t   ...   \t   ...   \t   ...    \t   ...')
        for i in range(split_count - 10 + 1, split_count + 1):
            j = i + start
            print('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                start + i + 1, expect_data[j], real_data[j], abs(np.float64(expect_data[j]) - np.float64(real_data[j])),
                cal_relative_diff(expect_data[j], real_data[j], diff_thd)))


def cal_relative_diff(real_data, expect_data, diff_thd, type_str='fp16'):
    if 'nan' in str(expect_data) or 'inf' in str(expect_data):
        if type_str.lower() == 'fp16':
            expect_data = 65504
        else:
            expect_data = 3.4028e38
    diff = abs(float(real_data) - float(expect_data))
    if abs(float(real_data) - float(expect_data)) < diff_thd:
        result = diff
    else:
        result = diff / (float(max(abs(real_data), abs(expect_data))) + 10e-10)
    return result


def display_error_output(real_data, expect_data, err_idx, relative_diff):
    print('Error Line-----------------------------------------------------------------------------')
    print('Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff')
    print('---------------------------------------------------------------------------------------')
    count = 0
    len_err = len(err_idx)
    for i in err_idx:
        count += 1
        # if len_err <= 200 or count < 100 or count > len_err - 100:
        if 0 <= i and i <= 10000:
            print('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                i, expect_data[i], real_data[i], abs(np.float64(expect_data[i]) - np.float64(real_data[i])),
                relative_diff[count - 1]))
        elif count == 100:
            dot_3 = '...'
            print('%08s \t %07s \t %07s \t %07s \t %07s \t %07s ' % (dot_3, dot_3, dot_3, dot_3, dot_3, dot_3))
    print('---------------------------------------------------------------------------------------')


def write_result(file, result_lines):
    with open(file, mode="w") as f:
        f.writelines(result_lines)


# if __name__ == '__main__':
#     ops = get_all_op(CASE_ROOT_PATH)
#     print("Case,Result")
#     result_lines = ["Case,Result\n"]
#     for op in ops:
#         op_path = os.path.join(CASE_ROOT_PATH, op)
#         for case in get_all_case(op_path):
#             dtypes = get_output_dtypes(op_path, case)
#             cpu_outs = get_cpu_out(op_path, case, dtypes)
#             npu_outs = get_npu_out(op_path, case, dtypes)
#             if len(cpu_outs) == 0 or len(cpu_outs) != len(npu_outs):
#                 line = f"{case},no_result"
#                 print(line)
#                 result_lines.append(line + "\n")
#                 continue
#             compare_result = True
#             rtol, atol = get_tols(op_path, case)
#             for i in range(len(cpu_outs)):
#                 result, fulfill_percent, max_error = data_compare_np(cpu_outs[i], npu_outs[i], rtol, atol)
#                 compare_result = compare_result and result == "Pass"
#             status = "success" if compare_result else "failed"
#             line = f"{case},{status}"
#             result_lines.append(line + "\n")
#             print(line)

#     import time
#     time_suffix = time.strftime("%Y-%m-%d", time.localtime())
#     result_file = "benchmark_" + time_suffix + ".csv"
#     write_result(result_file, result_lines)
import sys

if __name__ == '__main__':
    compare_result = True
    rtol = 0.001
    atol = 0.001
    if len(sys.argv) > 2:
        input_file1 = sys.argv[1]
        input_file2 = sys.argv[2]
        print("input_file1 : ", input_file1)
        print("input_file2 : ", input_file2)
    else:
        print("Error! please input  filename1 && filename2")
        exit

    input1 = np.fromfile(input_file1, dtype=np.float32)
    input2 = np.fromfile(input_file2, dtype=np.float32)

    result, fulfill_percent, max_error = data_compare_np(input1, input2, rtol, atol, is_display=True)
    print(result, fulfill_percent, max_error)
