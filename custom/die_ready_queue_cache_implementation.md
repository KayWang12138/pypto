/**
 * DieReadyQueue Cache 实现方案
 * 
 * 参考 ReadyQueueCache 的实现模式，为 dieReadyQue 适配 cache 实现
 * 
 * 关键信息：
 * - DIE_NUM = 2（core_func_data.h:117）
 * - DIE_READY_QUEUE_SIZE = 2（dev_encode_program_ctrlflow_cache.h:34）
 * - DieReadyQueueData 包含 2 个 AIV 队列 + 2 个 AIC 队列 = 4 个队列
 * - 队列存储在 devTask.dieReadyFunctionQue 中
 */

// 1. 数据结构定义（参考 ReadyQueueCache，38-47行）
// 注意：dieReadyQue 包含 DIE_NUM * 2 个队列（2个AIV + 2个AIC）
struct DieReadyQueueCache {
    uint32_t coreFunctionCnt;
    struct Queue {
        uint32_t head;
        uint32_t tail;
        uint32_t capacity;
        uint32_t *elem;
    } queueList[DIE_NUM * 2];  // 2个AIV + 2个AIC = 4个队列
    uint32_t readyTaskNum;
};

// 2. DynDeviceTaskBase 中添加成员（参考 readyQueueBackup，95行）
struct DynDeviceTaskBase {
    // ... 其他成员
    ReadyQueueCache *readyQueueBackup;
    DieReadyQueueCache *dieReadyQueueBackup;  // 新增
    // ... 其他成员
};

// 3. 备份函数 DieReadyQueueDataBackup（参考 ReadyQueueDataBackup，365-389行）
void DieReadyQueueDataBackup(DynDeviceTaskBase *base) {
    DieReadyQueueCache *dieReadyQueueBackup = reinterpret_cast<DieReadyQueueCache *>(AllocateCache(sizeof(DieReadyQueueCache)));
    if (dieReadyQueueBackup == nullptr) {
        return;
    }
    dieReadyQueueBackup->coreFunctionCnt = base->devTask.coreFunctionCnt;
    uint32_t readyTaskNum = 0;
    
    // 备份 2 个 AIV 队列
    for (size_t i = 0; i < DIE_NUM; i++) {
        ReadyCoreFunctionQueue *queue = reinterpret_cast<ReadyCoreFunctionQueue *>(base->devTask.dieReadyFunctionQue.readyDieAivCoreFunctionQue[i]);
        size_t backupSize = sizeof(uint32_t) * queue->capacity;
        uint32_t *dieReadyQueueBackupElem = reinterpret_cast<uint32_t *>(AllocateCache(backupSize));
        if (dieReadyQueueBackupElem == nullptr) {
            return;
        }

        dieReadyQueueBackup->queueList[i].head = queue->head;
        dieReadyQueueBackup->queueList[i].tail = queue->tail;
        dieReadyQueueBackup->queueList[i].capacity = queue->capacity;
        dieReadyQueueBackup->queueList[i].elem = dieReadyQueueBackupElem;
        memcpy_s(dieReadyQueueBackup->queueList[i].elem, backupSize, queue->elem, backupSize);

        readyTaskNum += queue->tail - queue->head;
    }
    
    // 备份 2 个 AIC 队列
    for (size_t i = 0; i < DIE_NUM; i++) {
        ReadyCoreFunctionQueue *queue = reinterpret_cast<ReadyCoreFunctionQueue *>(base->devTask.dieReadyFunctionQue.readyDieAicCoreFunctionQue[i]);
        size_t backupSize = {sizeof(uint32_t) * queue->capacity};
        uint32_t *dieReadyQueueBackupElem = reinterpret_cast<uint32_t *>(AllocateCachebacksSize));
        if (dieReadyQueueBackupElem == nullptr) {
            return;
        }

        dieReadyQueueBackup->queueList[DIE_NUM + i].head = queue->head;
        dieReadyQueueBackup->queueList[DIE_NUM + i].tail = queue->tail;
        dieReadyQueueBackup->queueList[DIE_NUM + i].capacity = queue->capacity;
        dieReadyQueueBackup->queueList[DIE_NUM + i].elem = dieReadyQueueBackupElem;
        memcpy_s(dieReadyQueueBackup->queueList[DIE_NUM + i].elem, backupSize, queue->elem, backupSize);

        readyTaskNum += queue->tail - queue->head;
    }
    
    dieReadyQueueBackup->readyTaskNum = readyTaskNum;
    base->dieReadyQueueBackup = dieReadyQueueBackup;
}

// 4. 恢复函数 DieReadyQueueDataRestore（参考 ReadyQueueDataRestore，391-401行）
void DieReadyQueueDataRestore(DynDeviceTaskBase *base) {
    DieReadyQueueCache *dieReadyQueueBackup = base->dieReadyQueueBackup;
    base->devTask.coreFunctionCnt = dieReadyQueueBackup->coreFunctionCnt;
    
    // 恢复 2 个 AIV 队列
    for (size_t i = 0; i < DIE_NUM; i++) {
        ReadyCoreFunctionQueue *queue = reinterpret_cast<ReadyCoreFunctionQueue *>(base->devTask.dieReadyFunctionQue.readyDieAivCoreFunctionQue[i]);
        size_t backupSize = sizeof(uint32_t) * queue->capacity;

        queue->head = dieReadyQueueBackup->queueList[i].head;
        queue->tail = dieReadyQueueBackup->queueList[i].tail;
        memcpy_s(queue->elem, backupSize, dieReadyQueueBackup->queueList[i].elem, backupSize);
    }
    
    // 恢复 2 个 AIC 队列
    for (size_t i = 0; i < DIE_NUM; i++) {
        ReadyCoreFunctionQueue *queue = reinterpret_cast<ReadyCoreFunctionQueue *>(base->devTask.dieReadyFunctionQue.readyDieAicCoreFunctionQue[i]);
        size_t backupSize = sizeof(uint32_t) * queue->capacity;

        queue->head = dieReadyQueueBackup->queueList[DIE_NUM + i].head;
        queue->tail = dieReadyQueueBackup->queueList[DIE_NUM + i].tail;
        memcpy_s(queue->elem, backupSize, dieReadyQueueBackup->queueList[DIE_NUM + i].elem, backupSize);
    }
}

// 5. 重定位函数 TaskAddrRelocProgramAndCtrlCache 中添加（参考 899-903行）
void TaskAddrRelocProgramAndCtrlCache(uint64_t srcProgram, uint64_t srcCtrlCache, uint64_t dstProgram, uint64_t dstCtrlCache) {
    RelocRange relocCtrlCache(srcCtrlCache, dstCtrlCache);
    RelocRange relocProgram(srcProgram, dstProgram);
    for (uint64_t deviceIndex = 0; deviceIndex < deviceTaskCount; deviceIndex++) {
        DynDeviceTaskBase *&dynTaskBaseRef = deviceTaskCacheList[deviceIndex].dynTaskBase;
        DynDeviceTaskBase *dynTaskBase = RelocControlFlowCachePointer(dynTaskBaseRef, relocCtrlCache);
        
        // ... 其他重定位逻辑
        
        // readyQueueBackup 重定位
        ReadyQueueCache *&readyQueueBackupRef = dynTaskBase->readyQueueBackup;
        ReadyQueueCache *readyQueueBackup = RelocControlFlowCachePointer(readyQueueBackupRef, relocCtrlCache);
        for (size_t i = 0; i < READY_QUEUE_SIZE; i++) {
            relocCtrlCache.Reloc(readyQueueBackup->queueList[i].elem);
        }

        // dieReadyQueueBackup 重定位（新增）
        DieReadyQueueCache *&dieReadyQueueBackupRef = dynTaskBase->dieReadyQueueBackup;
        DieReadyQueueCache *dieReadyQueueBackup = RelocControlFlowCachePointer(dieReadyQueueBackupRef, relocCtrlCache);
        for (size_t i = 0; i < DIE_NUM * 2; i++) {
            relocCtrlCache.Reloc(dieReadyQueueBackup->queueList[i].elem);
        }

        // ... 其他重定位逻辑
    }
}
