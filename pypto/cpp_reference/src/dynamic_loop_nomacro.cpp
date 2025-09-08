#include <iostream>

#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_storage.h"
#include "interface/configs/config_manager.h"

using namespace npu::tile_fwk;

int main(){
    std::vector<int> shape = {128, 128};
    Tensor a(DT_FP32, shape, "a");
    Tensor b(DT_FP32, shape, "b");
    
    auto* recordFunc0 = new RecordFunc("main", FunctionType::DYNAMIC, {a}, {b});
    Program::GetInstance().GetTileShape().SetVecTileShapes({64, 64});
    auto record_loop = RecordLoopFunc("Dynamic", FunctionType::DYNAMIC_LOOP, "k", LoopRange(10));
    for (auto &k : record_loop) {
        b = Add(a, a);
        if (RecordIfBranch(k < 2, __FILE__, __LINE__)) {
            std::cout<< "(cond k<2)";
            b = Add(b, a);
        } else {
            std::cout<< "(cond k>=2)";
            b = Sub(b, a);
        }

        if (RecordIfBranch(k < 5, __FILE__, __LINE__)) {
            std::cout<< "(cond k<5)";
            b = Mul(b, a);
        } else {
            std::cout<< "(cond k>=5)";
            b = Div(b, a);
        }
        b = Sub(b, a);
        std::cout<<"(end loop)"<<std::endl;
    }
    delete recordFunc0;
    std::cout << "finished" << std::endl;
}
