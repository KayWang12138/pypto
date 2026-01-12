#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <string>

namespace npu::tile_fwk{
#define TRACE_DATA_SIZE 64

enum class TracePhase {
    RUN_DEV_INIT = 0,
    RUN_DEV_SET_CAPTURE,
    RUN_DEV_ENV_READY,
    RUN_DEV_INIT_TILING_DATA,
    RUN_DEV_INIT_INOUT_TENSOR,
    RUN_DEV_REG_KERNEL_BIN,
    RUN_DEV_KERNEL_INIT,
    RUN_DEV_KERNEL_LAUNCH_AICPU_INIT,
    RUN_DEV_KERNEL_LAUNCH_AICPU_RUN,
    RUN_DEV_KERNEL_LAUNCH_AICORE,
    MAX_TRACE_PHASES
};

inline const std::string TraceName[] = {
    "RunDeviceInit",
    "RunDeviceSetCapture",
    "RunDevEnvReady",
    "RunDevInitTiling",
    "RunDevInitInOutTensor",
    "RunDevRegistKernelBin",
    "RunDevKernelInit",
    "RunDevKernelLaunchAicpuInit",
    "RunDevKernelLaunchAicpuRun",
    "RunDevKernelLaunchAIcore"
};

enum class EventPhase {
    COMPILE = 0,
    BUILD_CTRL_CACHE,
    RUN_DEV,
    MAX_EVENT_PHASES
};

inline const std::string EventName[] = {
    "Compile",
    "BuildCtrlFlowCache",
    "RunDevice",
};

struct PerfData {
    uint64_t totalTimeNs;
    uint64_t count;
    uint64_t maxTimeNs;
    uint64_t minTimeNs;
    std::string name;
    
    PerfData() : totalTimeNs(0), count(0), maxTimeNs(0), minTimeNs(UINT64_MAX) {}
    
    uint64_t AvgTimeNs() const {
        return count > 0 ? totalTimeNs / count : 0;
    }
    
    void AddStat(uint64_t durationNs) {
        totalTimeNs += durationNs;
        count++;
        if (durationNs > maxTimeNs) maxTimeNs = durationNs;
        if (durationNs < minTimeNs) minTimeNs = durationNs;
    }
};

#define HOST_PERF_TRACE(type) PerfAnalysis::Get().Trace(type)
#define HOST_PERF_EVT_BEGIN(type) PerfAnalysis::Get().EventBegin(type)
#define HOST_PERF_EVT_END(type) PerfAnalysis::Get().EventEnd(type)

class PerfAnalysis {
private:
    PerfAnalysis() {
        traceData_.resize(static_cast<size_t>(TracePhase::MAX_TRACE_PHASES));
        eventData_.resize(static_cast<size_t>(EventPhase::MAX_EVENT_PHASES));
        
        InitDataNames();

        eventStartTimes_.resize(static_cast<size_t>(EventPhase::MAX_EVENT_PHASES));
        for (auto& timePoint : eventStartTimes_) {
            timePoint = std::chrono::high_resolution_clock::time_point();
        }
        initTime_ = std::chrono::high_resolution_clock::now();
        
        isTraceInitialized_ = false;
    }
    
    PerfAnalysis(const PerfAnalysis&) = delete;
    PerfAnalysis& operator=(const PerfAnalysis&) = delete;
    
    std::vector<PerfData> traceData_;
    std::vector<PerfData> eventData_;
    std::vector<std::chrono::high_resolution_clock::time_point> eventStartTimes_;
    std::chrono::high_resolution_clock::time_point initTime_;
    std::chrono::high_resolution_clock::time_point lastTraceTime_;
    bool isTraceInitialized_;
    
    void InitDataNames() {
        for (int i = 0; i < static_cast<int>(TracePhase::MAX_TRACE_PHASES); i++) {
            traceData_[i].name = std::string(TraceName[i]);
        }

        for (int i = 0; i < static_cast<int>(EventPhase::MAX_EVENT_PHASES); i++) {
            eventData_[i].name = std::string(EventName[i]);
        }
    }
    
public:
    static PerfAnalysis& Get() {
        static PerfAnalysis instance;
        return instance;
    }
    
    void TraceInit() {
        lastTraceTime_ = std::chrono::high_resolution_clock::now();
        isTraceInitialized_ = true;
    }
    
    void Trace(TracePhase phase) {
        if (!isTraceInitialized_) {
            TraceInit();
            auto idx = static_cast<size_t>(phase);
            if (idx < traceData_.size()) {
                traceData_[idx].AddStat(0);
            }
            return;
        }
        
        auto now = std::chrono::high_resolution_clock::now();
        auto durationNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
            now - lastTraceTime_).count();
        
        auto idx = static_cast<size_t>(phase);
        if (idx < traceData_.size()) {
            traceData_[idx].AddStat(durationNs);
        }
        
        lastTraceTime_ = now;
    }
    
    void EventBegin(EventPhase phase) {
        auto idx = static_cast<size_t>(phase);
        
        if (eventStartTimes_[idx] != std::chrono::high_resolution_clock::time_point()) {
            EventEnd(phase);
        }
        
        eventStartTimes_[idx] = std::chrono::high_resolution_clock::now();
    }
    
    void EventEnd(EventPhase phase) {
        auto now = std::chrono::high_resolution_clock::now();
        auto idx = static_cast<size_t>(phase);
        
        if (eventStartTimes_[idx] != std::chrono::high_resolution_clock::time_point()) {
            auto durationNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                now - eventStartTimes_[idx]).count();
            
            if (idx < eventData_.size()) {
                eventData_[idx].AddStat(durationNs);
            }
            
            eventStartTimes_[idx] = std::chrono::high_resolution_clock::time_point();
        }
    }
    
    void ResetTrace() {
        for (auto& data : traceData_) {
            data.totalTimeNs = 0;
            data.count = 0;
            data.maxTimeNs = 0;
            data.minTimeNs = UINT64_MAX;
        }
        isTraceInitialized_ = false;
    }
    
    void ResetEvent() {
        for (auto& data : eventData_) {
            data.totalTimeNs = 0;
            data.count = 0;
            data.maxTimeNs = 0;
            data.minTimeNs = UINT64_MAX;
        }
        for (auto& timePoint : eventStartTimes_) {
            timePoint = std::chrono::high_resolution_clock::time_point();
        }
    }
    
    void Reset() {
        ResetTrace();
        ResetEvent();
        initTime_ = std::chrono::high_resolution_clock::now();
    }
    
    uint64_t GetTotalTimeUs() {
        auto now = std::chrono::high_resolution_clock::now();
        auto totalNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
            now - initTime_).count();
        return totalNs / 1000;
    }
    
    uint64_t GetTraceTotalTimeUs() {
        uint64_t totalNs = 0;
        for (const auto& data : traceData_) {
            totalNs += data.totalTimeNs;
        }
        return totalNs / 1000;
    }
    
    uint64_t GetEventTotalTimeUs() {
        uint64_t totalNs = 0;
        for (const auto& data : eventData_) {
            totalNs += data.totalTimeNs;
        }
        return totalNs / 1000;
    }
    
    uint64_t GetAllTotalTimeUs() {
        return GetTraceTotalTimeUs() + GetEventTotalTimeUs();
    }
    
    void Dump(bool toFile = false, const std::string& filename = "perf_stats.txt") {
        auto totalTimeUs = GetTotalTimeUs();
        auto traceTotalUs = GetTraceTotalTimeUs();
        auto eventTotalUs = GetEventTotalTimeUs();
        auto allTotalUs = GetAllTotalTimeUs();
        
        uint64_t totalTraceCount = 0;
        uint64_t totalEventCount = 0;
        
        std::ostream* output = &std::cout;
        std::ofstream fileStream;
        
        if (toFile) {
            fileStream.open(filename);
            if (fileStream.is_open()) {
                output = &fileStream;
            } else {
                std::cerr << "Failed to open file: " << filename << ", outputting to console instead." << std::endl;
            }
        }
        
        std::ostream& out = *output;
        
        out << "========== Performance Statistics ==========" << std::endl;
        out << "Total time since initialization: " 
                  << std::fixed << std::setprecision(3) << totalTimeUs / 1000.0 << " ms" 
                  << " (" << totalTimeUs << " us)" << std::endl;
        
        out << "\n--- Trace Statistics ---" << std::endl;
        for (int i = 0; i < static_cast<int>(TracePhase::MAX_TRACE_PHASES); i++) {
            const auto& data = traceData_[i];
            if (data.count > 0) {
                totalTraceCount += data.count;
                
                out << data.name << " (Phase " << i << "):" << std::endl;
                out << "  Count:      " << data.count << std::endl;
                out << "  Total Time: " << std::fixed << std::setprecision(3) 
                          << data.totalTimeNs / 1000.0 << " us" << std::endl;
                out << "  Avg Time:   " << std::fixed << std::setprecision(3) 
                          << data.AvgTimeNs() / 1000.0 << " us" << std::endl;
                out << "  Max Time:   " << std::fixed << std::setprecision(3) 
                          << data.maxTimeNs / 1000.0 << " us" << std::endl;
                out << "  Min Time:   " << std::fixed << std::setprecision(3) 
                          << data.minTimeNs / 1000.0 << " us" << std::endl;
                out << std::endl;
            }
        }
        
        out << "--- Event Statistics ---" << std::endl;
        for (int i = 0; i < static_cast<int>(EventPhase::MAX_EVENT_PHASES); i++) {
            const auto& data = eventData_[i];
            if (data.count > 0) {
                totalEventCount += data.count;
                
                out << data.name << " (Event " << i << "):" << std::endl;
                out << "  Count:      " << data.count << std::endl;
                out << "  Total Time: " << std::fixed << std::setprecision(3) 
                          << data.totalTimeNs / 1000.0 << " us" << std::endl;
                out << "  Avg Time:   " << std::fixed << std::setprecision(3) 
                          << data.AvgTimeNs() / 1000.0 << " us" << std::endl;
                out << "  Max Time:   " << std::fixed << std::setprecision(3) 
                          << data.maxTimeNs / 1000.0 << " us" << std::endl;
                out << "  Min Time:   " << std::fixed << std::setprecision(3) 
                          << data.minTimeNs / 1000.0 << " us" << std::endl;
                out << std::endl;
            }
        }
        
        out << "--- Summary ---" << std::endl;
        out << "Trace Statistics:" << std::endl;
        out << "  Total phases: " << totalTraceCount << std::endl;
        out << "  Total time:   " << std::fixed << std::setprecision(3) 
                  << traceTotalUs / 1000.0 << " ms" << std::endl;
        
        out << "\nEvent Statistics:" << std::endl;
        out << "  Total events: " << totalEventCount << std::endl;
        out << "  Total time:   " << std::fixed << std::setprecision(3) 
                  << eventTotalUs / 1000.0 << " ms" << std::endl;
        
        out << "\nCombined Total Time: " << std::fixed << std::setprecision(3) 
                  << allTotalUs / 1000.0 << " ms" 
                  << " (" << allTotalUs << " us)" << std::endl;
        
        if (allTotalUs > 0) {
            double percentage = (double)allTotalUs / totalTimeUs * 100.0;
            out << "Percentage of total time: " << std::fixed << std::setprecision(2) 
                      << percentage << "%" << std::endl;
        }
        
        out << "============================================" << std::endl;
        
        if (toFile && fileStream.is_open()) {
            fileStream.close();
            std::cout << "Statistics dumped to file: " << filename << std::endl;
        }
    }
    
    void DumpBrief() {
        std::cout << "===== Performance Statistics (Brief) =====" << std::endl;
        
        for (int i = 0; i < static_cast<int>(TracePhase::MAX_TRACE_PHASES); i++) {
            const auto& data = traceData_[i];
            if (data.count > 0) {
                std::cout << "Trace " << data.name << ": " 
                          << std::fixed << std::setprecision(3) << data.AvgTimeNs() / 1000.0 
                          << " us/op (" << data.count << " ops)" << std::endl;
            }
        }
        
        for (int i = 0; i < static_cast<int>(EventPhase::MAX_EVENT_PHASES); i++) {
            const auto& data = eventData_[i];
            if (data.count > 0) {
                std::cout << "Event " << data.name << ": " 
                          << std::fixed << std::setprecision(3) << data.AvgTimeNs() / 1000.0 
                          << " us/op (" << data.count << " ops)" << std::endl;
            }
        }
        
        std::cout << "==========================================" << std::endl;
    }
    
    void DumpToFile(const std::string& filename = "tmp/perf_stats.txt") {
        Dump(true, filename);
    }
};
}
