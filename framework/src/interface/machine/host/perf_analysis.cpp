
#include "machine/host/perf_analysis.h"

namespace npu::tile_fwk{

PerfAnalysis& PerfAnalysis::Get() {
    static PerfAnalysis instance;
    return instance;
}

}