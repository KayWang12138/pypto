#ifndef PYPTO_IR_PIPE_H_
#define PYPTO_IR_PIPE_H_

namespace pypto {
namespace ir {

/**
 * @brief Pipeline type enumeration for hardware execution units
 */
enum PipeType : int {
  MTE1,  ///< Memory Transfer Engine 1 (L1 -> L0A/L0B)
  MTE2,  ///< Memory Transfer Engine 2 (DDR -> UB)
  MTE3,  ///< Memory Transfer Engine 3 (UB -> DDR)
  M,     ///< Matrix Unit
  V,     ///< Vector Unit
  S,     ///< Scalar Unit
  FIX,   ///< Fix Pipe (L0C -> UB)
  ALL    ///< All Pipes
};

/**
 * @brief Core type enumeration (numeric values must match runtime add_task expectation)
 */
enum CoreType : int {
  CUBE = 0,   ///< Cube Core (Alias for AIC)
  VECTOR = 1  ///< Vector Core (Alias for AIV)
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_PIPE_H_
