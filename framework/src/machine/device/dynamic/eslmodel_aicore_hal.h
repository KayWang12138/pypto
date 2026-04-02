#include <cstdint>
#include <utility>
#include <dlfcn.h>

const int REG_SPR_DATA_MAIN_BASE_ADDR = 0x000000D0;
const int REG_SPR_COND_ADDR = 0x00005108;
const int SUBBLOCKDIM_NUM = 2;
const int BLOCKDIM = 32;
using CaReadReg64Func = uint32_t (*)(uint32_t coreId, uint32_t subcoreId, uint64_t addr, uint64_t *data);
using CaWriteReg64Func = uint32_t (*)(uint32_t coreId, uint32_t subcoreId, uint64_t addr, uint64_t *data);
using BusDirectReadFunc = uint32_t (*)(void *ptr, uint64_t size, uint64_t address, uint32_t devIdx);
using BusDirectWriteFunc = uint32_t (*)(uint64_t address, uint64_t size, void *ptr, uint32_t devIdx);


namespace npu::tile_fwk::dynamic {

class EslAicoreHal {
public:
    void Init() {
        void *eslDriverHandle = dlopen("libnpu_drv.so", RTLD_LAZY | RTLD_NOLOAD);
        caReadReg64_ = reinterpret_cast<CaReadReg64Func>(dlsym(eslDriverHandle, "ca_read_reg64"));
        caWriteReg64_ = reinterpret_cast<CaWriteReg64Func>(dlsym(eslDriverHandle, "ca_write_reg64"));
        busDirectRead_ = reinterpret_cast<BusDirectReadFunc>(dlsym(eslDriverHandle, "ca_read_ddr"));
        busDirectWrite_ = reinterpret_cast<BusDirectWriteFunc>(dlsym(eslDriverHandle, "ca_write_ddr"));
    }
    
    inline std::pair<int, int> GetSubCoreId(int coreIdx) {
        if (coreIdx < BLOCKDIM) {
            return { coreIdx, 0 };
        }
        int primaryCoreIdx = (coreIdx - BLOCKDIM) / SUBBLOCKDIM_NUM;
        int subCoreIdx = (coreIdx - BLOCKDIM - (primaryCoreIdx * 2)) % 2 + 1;
        return { primaryCoreIdx, subCoreIdx };
    }

    inline void WriteEslReg(int coreIdx, uint64_t *val) {
        auto coreInfo = GetSubCoreId(coreIdx);
        caWriteReg64_(coreInfo.first, coreInfo.second, REG_SPR_DATA_MAIN_BASE_ADDR, val);
    }

    inline uint64_t ReadEslReg(int coreIdx) {
        auto coreInfo = GetSubCoreId(coreIdx);
        uint64_t regVal;
        caReadReg64_(coreInfo.first, coreInfo.second, REG_SPR_COND_ADDR, &regVal);
        return regVal;
    }

    inline void WriteEslMem(uint64_t address, uint64_t size, uint64_t value) {
        uint64_t valToSend = value;
        busDirectWrite_(address, size, &valToSend, 0);
    }

    inline void SendDynFuncData(DynDeviceTask *dyntask, uint64_t stitchedSize, uint64_t dynFuncDataSize) {
        auto dyndata = &dyntask->dynFuncDataList->At(0);
        for (size_t funcIdx = 0; funcIdx < stitchedSize; ++funcIdx) {
            WriteEslMem(static_cast<int64_t>(PtrToValue(dyndata->rawTensorAddr)), dyndata->rawTensorAddrSize * sizeof(uint64_t), 
                                             static_cast<int64_t>(PtrToValue(dyndata->rawTensorAddr)));
            WriteEslMem(static_cast<int64_t>(PtrToValue(dyndata->exprTbl)), dyndata->exprNum * sizeof(uint64_t), static_cast<int64_t>(PtrToValue(dyndata->exprTbl)));
            dyndata++;
        }
        WriteEslMem(static_cast<int64_t>(PtrToValue(dyntask->GetDynFuncDataList())), dynFuncDataSize, static_cast<int64_t>(PtrToValue(dyntask->GetDynFuncDataList()))); 
    }
    
private:
    CaReadReg64Func caReadReg64_;
    CaWriteReg64Func caWriteReg64_;
    BusDirectReadFunc busDirectRead_;
    BusDirectWriteFunc busDirectWrite_;
};
}