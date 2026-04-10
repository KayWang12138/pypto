# 通信域管理
## 通信域创建
通过torch.dist.init_process_group接口,指定hccl后端初始化全局通信域,默认包含所有参与初始化的进程(rank0到world_size-1)
,通过torch.dist.new_group创建子通信域,需要显示传入rank列表,最后拿到通信域区分标识符group_name
```python
def init_hccl_comm(self, logical_rank_id: int) -> list[str]:
    physical_device_id = self.get_physical_device_id(logical_rank_id)
    torch_npu.npu.set_device(physical_device_id)
    dist.init_process_group(
        backend='hccl',
        rank=logical_rank_id,
        world_size=self.world_size,
        init_method=f'tcp://{self.master_ip}:{self.master_port}',
    )
    group_handle = dist.new_group(backend='hccl', ranks=self.logical_ranks)
    get_backend_method = getattr(group_handle, '_get_backend')
    backend = get_backend_method(torch.device('npu'))
    group_name = backend.get_hccl_comm_name(logical_rank_id)
    return [group_name]
```

## 通信context申请 
通过groupName可以获取对应handle,根据handle、stream和tilingStruct可以申请hccl资源,返回hcclContext结构体指针
```C++
HcclComm commHandle = nullptr;
ASSERT(HcomGetCommHandleByGroup(groupName.c_str(), &commHandle) == 0) << "Get hcom handle failed";
tilingStruct->MakeMc2TilingStruct(groupName);
auto ret = HcclAllocComResourceByTiling(commHandle, machine::GetRA()->GetStream(),
                tilingStruct->GetMc2CommConfig(), reinterpret_cast<void **>(&hcclContext[groupIndex]));
```
由于不同芯片架构hcclContext结构体不同,为了统一使用方式,对hcclContext结构体提取公共字段,定义一个归一的通信context结构体commContext,通过该commContext可以rankId、rankNum和winAddr,其中winAddr线性存储了共享内存起始地址,共享内存划分三个分区,winData用于不同卡之间数据传输,winStatus用于信号同步,winDebug可用于调试,winAddr内存排布为winData[0~rankNum-1], winStatus[0~rankNum-1], winDebug[0~rankNum-1],index分别指向了三个分区的起始索引,size表示共享内存区大小
```C++
struct CommContext {
    uint64_t rankId = 0; // 当前卡rankId
    uint64_t rankNum = 0;
    int64_t startIndex = 0; // 每个共享内存区起始Index
    int64_t statusIndex = -1; 
    int64_t debugIndex = -1; 
    uint64_t winDataSize = 0; // 每个共享内存区大小
    uint64_t winStatusSize = 0; 
    uint64_t winDebugSize = 0; 
    uint64_t totalWinNum = 0; 
    uint64_t winAddr[0]; // size大小rankNum*3,内存排布winData[0~rankNum-1], winStatus[0~rankNum-1], winDebug[0~rankNum-1]
};
```
目前在pyPTO框架中记录了前端使用的groupName及对应groupIndex,在hostmachine的device_launcher阶段会调用DeviceInitDistributedContext接口根据groupName申请context并作为DeviceKernelArgs的字段传到到kernel
```C++
DeviceInitDistributedContext(devMemoryUtilis, dynAttr->commGroupNames, kArgs);
```

## shmemTensor绑定通信域共享内存
```python
def create_shmem_tensor(
    group_name: str,
    n_pes: int,
    dtype: DataType,
    shape: list[int],
) -> ShmemTensor:
```
create_shmem_tensor会在指定通信域创建位于共享内存区的shmemTensor类型tensor,通过读写shmemTensor即可向同一通信域下的其他卡进行数据通信。接口底层会通过Opcode::OP_BIND_TENSOR将该shmemTensor映射一个编码了shmemType,offset以及groupIndex的虚拟地址,其中offset表示当前起始地址在共享内存中的偏移,memType表示使用共享内存分区类型。kernel运行时会从rawTensorAddr获取编码的虚拟地址并解析,根据kArgs中的commContext[groupIndex]访问对应通信域context,获取共享内存区地址

## kernel解析通信域共享内存地址
aicpu算子和aicore算子处理通信域地址逻辑相同
aicpu算子:
```C++
uint64_t ShmemWaitUntil::GetRawAddr(const uint64_t addr, const uint64_t dstRankId)
{
    uint64_t groupIndex = npu::tile_fwk::Distributed::GetVirtualAddrGroupIndex(addr);
    DEV_ASSERT(DistributedErrorCode::INVALID_GROUP_INDEX, groupIndex < commGroupNum_);
    uint64_t offset = npu::tile_fwk::Distributed::GetVirtualAddrOffset(addr);
    uint64_t memType = npu::tile_fwk::Distributed::GetVirtualAddrMemType(addr);
    auto hcclOpParam = reinterpret_cast<TileOp::CommContext*>(hcclContextAddr_[groupIndex]);
    auto winAddrOffset = (memType == 0) ? dstRankId : hcclOpParam->statusIndex + dstRankId;
    uint64_t rawAddr = hcclOpParam->winAddr[winAddrOffset] + offset;
    return rawAddr;
}
```

aicore算子:
```C++
template<typename T>
TILEOP __gm__ T* MapVirtualAddr(__gm__ int64_t *hcclContext, __gm__ T* vAddr, uint32_t dstRankId)
{
    auto groupIndex = GetVirtualAddrGroupIndex((uint64_t)vAddr);
    auto offset = GetVirtualAddrOffset((uint64_t)vAddr);
    auto memType = GetVirtualAddrMemType((uint64_t)vAddr);
    __gm__ TileOp::CommContext* commCtxParam = (__gm__ TileOp::CommContext*)hcclContext[groupIndex];
    if (memType == 0) {
        return (__gm__ T*)(commCtxParam->winAddr[dstRankId] + offset);
    } else {
        return (__gm__ T*)(commCtxParam->winAddr[commCtxParam->statusIndex + dstRankId] + offset);
    }
}
```

