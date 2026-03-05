#include <iostream>
#include "mock_inst.h"
#include "aicore_runtime.h"
#include "vector_dyn.h"
#include "cube_dyn.h"
#include "mte_dyn.h"
int main(int argc, char **argv) {
char charArray1[65536] = {0};
char* charArrayPtr1 = charArray1;
char charArray2[65536] = {0};
char* charArrayPtr2 = charArray2;
char charArray3[256] = {0};
char* charArrayPtr3 = charArray3;
char charArray4[256] = {0};
char* charArrayPtr4 = charArray4;
TileOp::DynTadd_<float, /*DS*/ 1, 4, 1, 64, /*S0*/ 1, 4, 1, 64, /*S1*/ 1, 4, 1, 64>((__ubuf__ float*)charArray2, (__ubuf__ float*)charArray1, (__ubuf__ float*)charArray3, 1, 1, 1, 1, 1, 1, 1, 1);
return 0;
}
