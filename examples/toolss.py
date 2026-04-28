import numpy as np
import functools
import random
import time
import multiprocessing as mp

"""
@author: s00417895 for binaryDataCompare
@author: l00469792 for integration
"""


def getDataType(type_str):
    type_dict = {
        'fp32': np.float32,
        'fp16': np.float16,
        'int8': np.int8,
        'uint8': np.uint8,
        'int16':np.int16,
        'uint16':np.uint16,
        'int32': np.int32,
        'uint32':np.uint32,
        'int64': np.int64,
        'uint64':np.uint64,
        'bool':np.bool,
        'fp64':np.float64,
        'complex64':np.complex64,
        'complex128':np.complex128,
    }
    return type_dict[type_str]


def getStrDataType(type_id):
    type_dict = {
        0: "fp32",
        1: "fp16",
        2: "int8",
        3: "int32",
        4: "uint8",
        6: "int16",
        7: "uint16",
        8: "uint32",
        9: "int64",
        10: "uint64",
        11: "double",
        12: "bool",
        13: "string",
        14: "sint8",  # DT_DUAL_SUB_INT8 = 14,    /**< dual output int8 type */
        15: "suint8",  # DT_DUAL_SUB_UINT8 = 15,    /**< dual output uint8 type */
        16: "complex64",  # complex64 type
        17: "complex128",  # complex128 type
        18: "qint8",
        19: "qint16",
        20: "qint32",
        21: "quint8",
        22: "quint16",
        23: "resource",  # DT_RESOURCE = 23,          // resource type
        24: "string_ref",  # DT_STRING_REF = 24,        // string ref type
        25: "dual",  # /**< dual output type */
        26: "undefined"

    }
    return type_dict[int(float(type_id))]

    
def getStrDtype(dtype):
    type_dict = {
        np.float32: "fp32",
        np.float32: "fp16",
        np.int8: "int16",
    }
    return type_dict[dtype]
    
    
def gen_data(data_size, min, max, dtype):
    # return np.random.uniform(min, max, size=data_size).astype(getDataType(dtype))
    return np.random.uniform(min, max, size=data_size).astype(dtype)

def getData(data_path, shape, dtype):
    #data_pool = np.fromfile(data_path, dtype=getDataType(dtype))
    data_pool = np.fromfile(data_path, dtype)
    data_len = functools.reduce(lambda x,y:x*y, shape)
    # print('data_len: ', data_len)
    return data_pool[0:data_len].reshape(shape)

def dump_file(data, path, dtype):
    #path = os.path.join(self.data_path, path)
    # dtype = getStrDataType(dtype)
    # print("getStrDataType:",dtype)
    if dtype in ('fp16', 'FP16', 'Fp16'):
        np.array(data).astype(np.float16).tofile(path)
    elif dtype in ('fp32', 'FP32', 'Fp32'):
        np.array(data).astype(np.float32).tofile(path)
    elif dtype in ('int8', 'INT8', 'Int8'):
        np.array(data).astype(np.int8).tofile(path)
    elif dtype in ('int16', 'INT16', 'Int16'):
        np.array(data).astype(np.int16).tofile(path)
    elif dtype in ('int32', 'INT32', 'Int32'):
        np.array(data).astype(np.int32).tofile(path)
    elif dtype in ('int64', 'INT64', 'Int64'):
        np.array(data).astype(np.int64).tofile(path)
    elif dtype in ('uint8', 'UINT8','Uint8'):
        np.array(data).astype(np.uint8).tofile(path)
    elif dtype in ('uint16', 'UINT16','Uint16'):
        np.array(data).astype(np.uint16).tofile(path)
    elif dtype in ('uint32', 'UINT32','Uint32'):
        np.array(data).astype(np.uint32).tofile(path)
    elif dtype in ('uint64', 'UINT64','Uint64'):
        np.array(data).astype(np.uint64).tofile(path)
    elif dtype in ('fp64'):
        np.array(data).astype(np.float64).tofile(path)
    elif dtype in ('complex64'):
        np.array(data).astype(np.complex64).tofile(path)
    elif dtype in ('complex128'):
        np.array(data).astype(np.complex128).tofile(path)
    elif dtype in ('bool'):
        np.array(data).astype(np.bool).tofile(path)

def dump_data(inputData, path, dtype):
    # fileName=os.path.join(self.data_path,filename)
    outDataFmt= getStrDataType(dtype)
    fOutput = open(path, "wb")
    if outDataFmt == "fp16":
        for elem in inputData:
            fOutput.write(np.float16(elem).tobytes())
    elif outDataFmt == "fp32":
        for elem in inputData:
            fOutput.write(np.float32(elem).tobytes())
    elif outDataFmt == "int8":
        for elem in inputData:
            fOutput.write(np.int8(elem).tobytes())
    elif outDataFmt == "int32":
        for elem in inputData:
            fOutput.write(np.int32(elem).tobytes())
    elif outDataFmt == "uint8":
        for elem in inputData:
            fOutput.write(np.uint8(elem).tobytes())
    elif outDataFmt == "uint32":
        for elem in inputData:
            fOutput.write(np.uint32(elem).tobytes())
    elif outDataFmt == "uint16":
        for elem in inputData:
            fOutput.write(np.uint16(elem).tobytes())
    fOutput.close()
    print('file:%s dump success.' %path)

def read_file(path, dtype):
        dtype = getDataType(dtype)
        return np.fromfile(path, dtype=dtype)

# def calRelativediff(x,y):
#     return abs(float(x)-float(y))/max((float(max(abs(x),abs(y)))+10e-10),2**(-12))

def calRelativediff(x,y,diffThd):
    if float(max(abs(x), abs(y))) < float((1.0/(1<<14))/diffThd):
        result = abs(float(x) - float(y))/(float((1.0/(1<<14))/diffThd) + 10e-10)
    else:
        result = abs(float(x)-float(y))/(float(max(abs(x),abs(y)))+10e-10)

    return result

def calRelativediffNumpy(dataCheck, dataExpect, diffThd):
    a = np.abs(np.subtract(dataCheck, dataExpect))
    b1 = np.maximum(np.abs(dataCheck), (np.abs(dataExpect)))
    b2 = float((1.0 / (1 << 14)) / diffThd)
    b = np.maximum(b1, b2)
    result = np.divide(a, np.add(b, 10e-10))
    return result

def displayOutput(dr, de, start, end,diffThd):

    print ('\n--------------------------------------------------------------------')
    print ('Loop \t  ExpectOut \t RealOut \t FpDiff \t RateDiff')
    print ('--------------------------------------------------------------------')
    dataCount = dr.size
    splitCount = int(end - start)

    if splitCount <= 20:
        for i in range(splitCount+1):
            j = i + start
            print ('%d \t %.7f \t %.7f \t %.7f \t %.7f' % (start+i+1, de[j], dr[j], abs(np.float64(de[j])-np.float64(dr[j])), calRelativediff(de[j],dr[j],diffThd)))
    else:
        for i in range(10):
            j = i + start
            print ('%d \t %.7f \t %.7f \t %.7f \t %.7f' % (start+i+1, de[j], dr[j], abs(np.float64(de[j])-np.float64(dr[j])), calRelativediff(de[j],dr[j],diffThd)))
        print ('...   \t   ...   \t   ...   \t   ...    \t   ...')
        for i in range(splitCount-10+1,splitCount+1):
            j = i + start
            print ('%d \t %.7f \t %.7f \t %.7f \t %.7f' % (start+i+1, de[j], dr[j], abs(np.float64(de[j])-np.float64(dr[j])), calRelativediff(de[j],dr[j],diffThd)))

def displayOutput_new(dr, de, de32, start, end,diffThd):

    print ('\n--------------------------------------------------------------------')
    print ('Loop \t  ExpFP32Out \t ExpFP16Out \t  NPUOut \t FpDiff(NPU-ExpFP16) \t RateDiff')
    print ('--------------------------------------------------------------------')
    dataCount = dr.size
    splitCount = int(end - start)

    if splitCount <= 20:
        for i in range(splitCount+1):
            j = i + start
            print ('%d \t %.7f \t %.7f \t %.7f \t %.7f \t %.7f' % (start+i+1, de32[j], de[j],dr[j], abs(np.float64(de[j])-np.float64(dr[j])), calRelativediff(de[j],dr[j],diffThd)))
    else:
        for i in range(10):
            j = i + start
            print ('%d \t %.7f \t %.7f \t %.7f \t %.7f \t %.7f' % (start+i+1, de32[j], de[j], dr[j], abs(np.float64(de[j])-np.float64(dr[j])), calRelativediff(de[j],dr[j],diffThd)))
        print ('...   \t   ...   \t   ...   \t   ...    \t   ...')
        for i in range(splitCount-10+1,splitCount+1):
            j = i + start
            print ('%d \t %.7f \t %.7f \t %.7f \t %.7f \t %.7f' % (start+i+1, de32[j], de[j], dr[j], abs(np.float64(de[j])-np.float64(dr[j])), calRelativediff(de[j],dr[j],diffThd)))


def displayErrorOutput(dr, de, rdiff ,start, end, diffThd):
    print ('\n----------------------------Error Line------------------------------')
    print ('--------------------------------------------------------------------')
    print ('Loop \t  ExpectOut \t RealOut \t FpDiff \t RateDiff')
    print ('--------------------------------------------------------------------')
    dataCount = dr.size
    splitCount = int(end - start)
    count = 0
    for i in range(len(rdiff)):
        j = i + start
        if rdiff[j] > diffThd:
            count += 1
            print ('%d \t %.7f \t %.7f \t %.7f \t %.7f' % (start+i+1, de[j], dr[j], abs(np.float64(de[j])-np.float64(dr[j])), calRelativediff(de[j],dr[j],diffThd)))
        if count == 300:
            break

def displayErrorOutput_new(datareal, datacomp, dataexp, err_idx , rdiff, start, end, diffThd):
    print ('\n----------------------------Error Line------------------------------')
    print ('--------------------------------------------------------------------')
    print ('Loop \t  ExpFp32Out \t ExpFp16Out \t NPUOut \t FpDiff(NPU- ExpFp16) \t Fp16rateDiff')
    print ('--------------------------------------------------------------------')
    dataCount = datareal.size
    splitCount = int(end - start)
    count = 0
    for i in err_idx:
        count += 1
        print ('%d \t %.8f \t %.8f \t %.8f \t %.8f  \t %.8f ' % (i, dataexp[i],datacomp[i], datareal[i],rdiff[count-1],calRelativediff(datacomp[i], datareal[i], diffThd)))
        if count == 100:
            break


#ɾ��ǰ�����
def broadcast1(shape):
    newlist=shape.copy()
    if len(newlist)>1:
        for i in range(0,random.randint(1,len(newlist)-1)):
            newlist.pop(0)
    return newlist

#�������ֵ��Ϊ1
def broadcast2(shape):
    newlist=shape.copy()
    indexlist=[]
    if len(newlist) > 1:
        for i in range(0,random.randint(0,len(newlist)-1)):
            axis=random.randint(0,len(newlist)-1)
            if np.sum(list(map(lambda  x:x>1,[abs(x-axis) for x in indexlist]))) <= 1:
               newlist[axis]=1
               indexlist.append(axis)
            else:
               pass
    return newlist

#
def broadcast3(shape):
    newlist = shape.copy()
    len_s = len(shape)
    indexlist=[]
    if len(shape)<8:
        len_d = random.randint(len_s+1,8)
        for i in range(0,random.randint(0,len(newlist)-1)):
            axis=random.randint(0,len(newlist)-1)
            if np.sum(list(map(lambda  x:x>1,[abs(x-axis) for x in indexlist]))) <= 1:
               newlist[axis]=1
               indexlist.append(axis)
            else:
               pass
        for i in range(len_d-len_s):
            newlist.insert(0,1)
    return newlist
#��axis
def getAxis(shape):
    length = len(shape)
    axislist = []
    num = random.randint(1, length)
    for j in range(num):
        axis_loop = 1
        while axis_loop:
            axis = random.randint(-length, length-1)
            if axis not in axislist and  (axis + length) not in axislist and  (axis - length) not in axislist:
                axislist.append(axis)
                axis_loop = 0
    axislist = list(axislist)
    #print(type(axislist))
    return axislist



def dataCompare(dataCheck, dataExpect,  diffThd=0.01, pctThd=0.05):
    print("xxxxxxxxxxxxxx in dataCompare")
    # import pdb;
    # pdb.set_trace()
    npu_shape=dataCheck.shape
    expect_shape=dataExpect.shape
    print('npu_out_shape:',npu_shape)
    print('expect_out_shape:',expect_shape)
    dataCheck=dataCheck.flatten()
    dataExpect=dataExpect.flatten()
    start=0
    end=dataCheck.size - 1

    #for i in range(len(dataExpect)):
    #    if str(dataExpect[i]) =='nan' or str(dataExpect[i]) =='inf':
    #        print(i,dataExpect[i])
    # rdiff = list(map(lambda x,y: calRelativediff(x,y,diffThd), dataCheck.astype(np.float32), dataExpect.astype(np.float32)))
    rdiff = calRelativediffNumpy(dataCheck.astype(np.float32), dataExpect.astype(np.float32), diffThd)
    split_rdiff = rdiff[start:end+1]
    splitCount = int(end-start+1) if end != start else 1
    print('splitCount: ',  splitCount)
    print('float(splitCount): ',  float(splitCount))
    # ltNum = functools.reduce(lambda x, y: x+1 if y < diffThd else x, split_rdiff, 0)
    ltNum = rdiff[rdiff < diffThd].size
    j=0
    # for i in range(len(dataExpect)):
    #     if 'nan' in str(dataExpect[i])  or 'inf' in str(dataExpect[i]) :
    #         ltNum=ltNum+1
    #         if j<5:
    #            print(i,dataExpect[i],dataCheck[i])
    #            j=j+1
    ltNum = ltNum + dataExpect[np.isinf(dataExpect)].size + dataExpect[np.isnan(dataExpect)].size
    ltPct = float(ltNum)/float(splitCount) * 100.0
    displayOutput(dataCheck, dataExpect, start, end,diffThd)
    pctThd = (1-pctThd) * 100.0
    result = "Pass" if (ltPct>=pctThd) else "Failed"
    if npu_shape != expect_shape:
        result = "Failed"

    print ('\n--------------------------------------------------------------------')
    print ('DiffThd  \t PctThd   \t PctRlt   \t Result')
    print ('--------------------------------------------------------------------')
    print ('%.3f     \t %.2f%%   \t %.6f%%   \t %s   \n' % (diffThd,pctThd,ltPct,result))

    if npu_shape != expect_shape:
        print('============ out_shape is not equal expect!')
    else:
        print('============ out_shape is equal expect!')

    if result == "Failed":
        displayErrorOutput(dataCheck, dataExpect, rdiff ,start, end, diffThd)
    return result

#--------------------------------------------------
#Author:TangJianhua 00505743 @ 2020/01/12
#--------------------------------------------------
def calcdiff(Check,Expect,diffThd):
    #result = list(map(lambda x,y: calRelativediff(x,y,diffThd), Check.astype(np.float32), Expect.astype(np.float32)))
    #result = list(map(lambda x,y: calRelativediff(x,y,diffThd), Check, Expect))
    #return result
    return list(map(lambda x,y: calRelativediff(x,y,diffThd), Check, Expect))

def dataComparev2(dataCheck, dataExpect, start, end, diffThd=0.01, pctThd=0.05):
    print("xxxxxxxxxxxxxx in dataComparev2")
    manager = mp.Manager()
    return_list = manager.list()
    starttime=time.time()
    NUMPROC = min([16,int(mp.cpu_count()/2)])
    #for i in range(len(dataExpect)):
    #    if str(dataExpect[i]) =='nan' or str(dataExpect[i]) =='inf':
    #        print(i,dataExpect[i])
    splitstep = len(dataExpect) // NUMPROC  #3273600
    rdiff = []
    if splitstep > 100:
        runtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print('%s I: splitstep is %d,using multi-processing.' % (runtime,splitstep))
        pool = mp.Pool(processes=NUMPROC)
        e = start
        result = []
        while e <= end:
            s = e
            e = s + splitstep
            if e > end+1:
                e = end+1
            result.append(pool.apply_async(calcdiff,args=(dataCheck[s:e],dataExpect[s:e],diffThd)))
        pool.close()
        pool.join()
        for p in result:
            rdiff.extend(p.get())
    else:
        runtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print('%s I: splitstep is %d,smaller data-set,using single processing.' % (runtime,splitstep))
        rdiff = list(map(lambda x,y: calRelativediff(x,y,diffThd), dataCheck.astype(np.float32), dataExpect.astype(np.float32)))
    runtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('%s I: getting rdiff have compeled.Duration:%0.2f second.' % (runtime,time.time()-starttime))
    split_rdiff = rdiff[start:end+1]
    #print('split_rdiff',split_rdiff)
    splitCount = int(end-start+1) if end != start else 1
    print('splitCount:%s; float(splitCount):%s;' %(splitCount,float(splitCount)))
    ltNum = functools.reduce(lambda x, y: x+1 if y < diffThd else x, split_rdiff, 0)
    #print('=reduce=====Duration:',time.time()-starttime)
    ltPct = float(ltNum)/float(splitCount) * 100.0
    displayOutput(dataCheck, dataExpect, start, end,diffThd)
    pctThd = (1-pctThd) * 100.0
    result = "Pass" if (ltPct>=pctThd) else "Failed"
    print ('\n--------------------------------------------------------------------------------------------')
    print ('DiffThd  \t PctThd   \t PctRlt   \t Result')
    print ('--------------------------------------------------------------------------------------------')
    print ('%.3f     \t %.2f%%   \t %.6f%%   \t %s   \n' % (diffThd,pctThd,ltPct,result))
    if result == "Failed":
        displayErrorOutput(dataCheck, dataExpect, rdiff ,start, end, diffThd)
    #print('=display=====Duration:',time.time()-starttime)
    return result,ltPct

def dataCompare_new(NpuOut, ExpFP16Out, ExpFP32Out,  diffThd=0.01, pctThd=0.05, MaxdiffThd=0.1):
    print("xxxxxxxxxxxxxx in dataCompare_new")
    npu_shape=NpuOut.shape
    expect_shape=ExpFP16Out.shape
    print('npu_out_shape:',npu_shape)
    print('expect_out_shape:',expect_shape)

    dataCheck=NpuOut.flatten()
    dataCompe=ExpFP16Out.flatten()
    dataExpect=ExpFP32Out.flatten()

    errcnt=0
    start=0
    end=dataCheck.size - 1

    # j=0
    # for i in range(len(dataExpect)):
    #     if 'nan' in str(dataExpect[i])  or 'inf' in str(dataExpect[i]) :
    #         if j<5:
    #            print(i,dataExpect[i],dataCheck[i])
    #            j=j+1
    #         else:
    #            break



    #rdiff = list(map(lambda x,y: calRelativediff(x,y,diffThd), dataCheck.astype(np.float32), dataExpect.astype(np.float32)))
    #����������?
    # diff1 = list(map(lambda x,y: abs(x-y), dataCheck.astype(np.float32), dataExpect.astype(np.float32)))
    # diff2 = list(map(lambda x,y: abs(x-y), dataCompe.astype(np.float32), dataExpect.astype(np.float32)))

    diff1 = np.abs(np.subtract(dataCheck.astype(np.float32), dataExpect.astype(np.float32)))
    diff2 = np.abs(np.subtract(dataCompe.astype(np.float32), dataExpect.astype(np.float32)))

    #split_rdiff = rdiff[start:end+1]
    splitCount = int(end-start+1) if end != start else 1
    print('splitCount: ',  splitCount)
    print('float(splitCount): ',  float(splitCount))

    err_idx=[]
    # err_diff=[]

    diff = np.subtract(diff1, diff2)
    diff_index = np.where(diff > 0)

    rdiff = calRelativediffNumpy(dataCheck[diff_index], dataCompe[diff_index], diffThd)
    err_diff = rdiff[rdiff > diffThd]
    errcnt = err_diff.size

    # for i in range(splitCount):
    #     if diff1[i] > diff2[i]:
    #         rdiff=calRelativediff(dataCheck[i], dataCompe[i], diffThd)
    #         if rdiff > diffThd:
    #             errcnt=errcnt+1
    #             err_idx.append(i)
    #             err_diff.append(rdiff)
    ltNum= splitCount - errcnt


    #ltNum = functools.reduce(lambda x, y: x+1 if y < diffThd else x, split_rdiff, 0)
    ltPct = float(ltNum)/float(splitCount) * 100.0
    displayOutput_new(dataCheck, dataCompe, dataExpect, start, end,diffThd)
    pctThd = (1-pctThd) * 100.0
    result = "Pass" if (ltPct>=pctThd) else "Failed"
    if len(err_diff)>0:
        max_diff =  max(err_diff)
        if max(err_diff)>=MaxdiffThd:
            result = "Failed"
    if npu_shape != expect_shape:
        result = "Failed"

    print ('\n--------------------------------------------------------------------')
    print ('DiffThd  \t PctThd   \t PctRlt   \t Result')
    print ('--------------------------------------------------------------------')
    print ('%.3f     \t %.2f%%   \t %.6f%%   \t %s   \n' % (diffThd,pctThd,ltPct,result))

    if len(err_diff) > 0:
        print('=============maximum error is:', max_diff)

    if npu_shape != expect_shape:
        print('============ out_shape is not equal expect!')
    else:
        print('============ out_shape is equal expect!')

    if result == "Failed":
        displayErrorOutput_new(dataCheck, dataCompe,dataExpect, err_idx ,err_diff,start, end, diffThd)
    return result,ltPct



