/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// clang-format off
#include <msgpack.hpp>
// clang-format on

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/serialization/type_registry.h"
#include "ir/stmt.h"
#include "ir/type.h"
#include "core/logging.h"

namespace pypto {
namespace ir {
namespace serialization {

// Use alias for cleaner code
using DeserializerContext = serialization::detail::DeserializerContext;

// Helper macros for deserializing fields
#define GET_FIELD(Type, name) ctx.GetField<Type>(fieldsObj, name)
#define GET_FIELD_OBJ(name) ctx.GetFieldObj(fieldsObj, name)

// Helper function to get optional field (returns nullopt if field doesn't exist or is null)
static std::optional<msgpack::object> GetOptionalFieldObj(const msgpack::object& fieldsObj,
                                                          const std::string& fieldName,
                                                          DeserializerContext& /*ctx*/) {
  if (fieldsObj.type != msgpack::type::MAP) {
    return std::nullopt;
  }
  msgpack::object_kv* p = fieldsObj.via.map.ptr;
  msgpack::object_kv* const pend = fieldsObj.via.map.ptr + fieldsObj.via.map.size;
  for (; p < pend; ++p) {
    std::string key;
    p->key.convert(key);
    if (key == fieldName) {
      auto obj = p->val;
      // Check if it's null or empty
      if (obj.type == msgpack::type::NIL) {
        return std::nullopt;
      }
      return obj;
    }
  }
  return std::nullopt;
}

DataType DeserializeDataType(const msgpack::object& fieldsObj, const std::string& fieldName) {
  msgpack::object_kv* mapP = fieldsObj.via.map.ptr;
  msgpack::object_kv* const mapPend = fieldsObj.via.map.ptr + fieldsObj.via.map.size;
  std::string typeName;
  bool isDtype = false;
  uint8_t dtypeCode = 0;

  for (; mapP < mapPend; ++mapP) {
    std::string keyName;
    mapP->key.convert(keyName);
    if (keyName == "type") {
      mapP->val.convert(typeName);
      isDtype = (typeName == "DataType");
    } else if (keyName == "code") {
      dtypeCode = mapP->val.as<uint8_t>();
    }
  }

  if (isDtype) {
    return DataType(dtypeCode);
  } else {
    throw TypeError("Invalid kwarg MAP type for key: " + fieldName);
  }
}

std::vector<std::pair<std::string, std::any>> DeserializeKwargs(const msgpack::object& kwargsObj,
                                                                const std::string& fieldName) {
  std::vector<std::pair<std::string, std::any>> kwargs;
  if (kwargsObj.type != msgpack::type::ARRAY) {
    throw TypeError("Invalid kwargs type for field: " + fieldName);
  }

  for (uint32_t i = 0; i < kwargsObj.via.array.size; ++i) {
    const msgpack::object& pairObj = kwargsObj.via.array.ptr[i];
    if (pairObj.type != msgpack::type::MAP) {
      throw TypeError("Invalid kwarg pair type for field: " + fieldName);
    }

    std::string key;
    msgpack::object valueObj;
    bool hasKey = false;
    bool hasValue = false;
    msgpack::object_kv* mapP = pairObj.via.map.ptr;
    msgpack::object_kv* const mapPend = pairObj.via.map.ptr + pairObj.via.map.size;
    for (; mapP < mapPend; ++mapP) {
      std::string mapKey;
      mapP->key.convert(mapKey);
      if (mapKey == "key") {
        mapP->val.convert(key);
        hasKey = true;
      } else if (mapKey == "value") {
        valueObj = mapP->val;
        hasValue = true;
      }
    }

    if (!hasKey || !hasValue) {
      throw TypeError("Invalid kwarg pair for field: " + fieldName);
    }

    // Deserialize value based on type
    if (valueObj.type == msgpack::type::BOOLEAN) {
      kwargs.emplace_back(key, valueObj.as<bool>());
    } else if (valueObj.type == msgpack::type::POSITIVE_INTEGER ||
               valueObj.type == msgpack::type::NEGATIVE_INTEGER) {
      kwargs.emplace_back(key, valueObj.as<int>());
    } else if (valueObj.type == msgpack::type::FLOAT32) {
      kwargs.emplace_back(key, valueObj.as<float>());
    } else if (valueObj.type == msgpack::type::FLOAT64) {
      kwargs.emplace_back(key, valueObj.as<double>());
    } else if (valueObj.type == msgpack::type::STR) {
      kwargs.emplace_back(key, valueObj.as<std::string>());
    } else if (valueObj.type == msgpack::type::MAP) {
      // Try to deserialize as DataType
      try {
        kwargs.emplace_back(key, DeserializeDataType(valueObj, key));
      } catch (const TypeError&) {
        throw TypeError("Invalid kwarg type for key: " + key);
      }
    } else {
      throw TypeError("Invalid kwarg type for key: " + key);
    }
  }

  return kwargs;
}

// Deserialize Var
static IRNodePtr DeserializeVar(const msgpack::object& fieldsObj, msgpack::zone& zone,
                                DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  auto type = ctx.DeserializeType(GET_FIELD_OBJ("type"), zone);
  std::string name = GET_FIELD(std::string, "name");
  return std::make_shared<Var>(name, type, span);
}

// Deserialize IterArg
static IRNodePtr DeserializeIterArg(const msgpack::object& fieldsObj, msgpack::zone& zone,
                                    DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  auto type = ctx.DeserializeType(GET_FIELD_OBJ("type"), zone);
  std::string name = GET_FIELD(std::string, "name");
  auto initValue =
      std::static_pointer_cast<const Expr>(ctx.DeserializeNode(GET_FIELD_OBJ("initValue"), zone));
  return std::make_shared<IterArg>(name, type, initValue, span);
}

// Deserialize MemRef
static IRNodePtr DeserializeMemRef(const msgpack::object& fieldsObj, msgpack::zone& zone,
                                   DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));

  // Deserialize memorySpace (stored as uint8_t)
  uint8_t memorySpaceCode = GET_FIELD(uint8_t, "memory_space");
  MemorySpace memorySpace = static_cast<MemorySpace>(memorySpaceCode);

  // Deserialize addr expression
  auto addr = std::static_pointer_cast<const Expr>(ctx.DeserializeNode(GET_FIELD_OBJ("addr"), zone));

  // Deserialize size and id
  uint64_t size = GET_FIELD(uint64_t, "size");
  uint64_t id = GET_FIELD(uint64_t, "id");

  return std::make_shared<MemRef>(memorySpace, addr, size, id, span);
}

// Deserialize ConstInt
static IRNodePtr DeserializeConstInt(const msgpack::object& fieldsObj, msgpack::zone& zone,
                                     DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  auto type = ctx.DeserializeType(GET_FIELD_OBJ("type"), zone);
  int64_t value = GET_FIELD(int64_t, "value");
  auto scalarType = As<ScalarType>(type);
  INTERNAL_CHECK(scalarType) << "ConstInt is expected to have ScalarType type, but got " + type->TypeName();
  return std::make_shared<ConstInt>(value, scalarType->dtype_, span);
}

// Deserialize ConstFloat
static IRNodePtr DeserializeConstFloat(const msgpack::object& fieldsObj, msgpack::zone& zone,
                                       DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  auto type = ctx.DeserializeType(GET_FIELD_OBJ("type"), zone);
  double value = GET_FIELD(double, "value");
  auto scalarType = As<ScalarType>(type);
  INTERNAL_CHECK(scalarType) << "ConstFloat is expected to have ScalarType type, but got " +
                                     type->TypeName();
  return std::make_shared<ConstFloat>(value, scalarType->dtype_, span);
}

// Deserialize ConstBool
static IRNodePtr DeserializeConstBool(const msgpack::object& fieldsObj, msgpack::zone& /*zone*/,
                                      DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  bool value = GET_FIELD(bool, "value");
  return std::make_shared<ConstBool>(value, span);
}

// Deserialize Call
static IRNodePtr DeserializeCall(const msgpack::object& fieldsObj, msgpack::zone& zone,
                                 DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  auto op = ctx.DeserializeOp(GET_FIELD_OBJ("op"));
  auto type = ctx.DeserializeType(GET_FIELD_OBJ("type"), zone);

  std::vector<ExprPtr> args;
  auto argsObj = GET_FIELD_OBJ("args");
  if (argsObj.type == msgpack::type::ARRAY) {
    for (uint32_t i = 0; i < argsObj.via.array.size; ++i) {
      args.push_back(
          std::static_pointer_cast<const Expr>(ctx.DeserializeNode(argsObj.via.array.ptr[i], zone)));
    }
  }

  // Deserialize kwargs (preserve order using vector)
  auto kwargsObj = GET_FIELD_OBJ("kwargs");
  std::vector<std::pair<std::string, std::any>> kwargs = DeserializeKwargs(kwargsObj, "kwargs");

  return std::make_shared<Call>(op, args, kwargs, type, span);
}

// Macro for binary expressions
#define DESERIALIZE_BINARY_EXPR(ClassName)                                                                \
  static IRNodePtr Deserialize##ClassName(const msgpack::object& fieldsObj, msgpack::zone& zone,         \
                                          DeserializerContext& ctx) {                                     \
    auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));                                               \
    auto type = ctx.DeserializeType(GET_FIELD_OBJ("type"), zone);                                         \
    auto scalarType = As<ScalarType>(type);                                                              \
    INTERNAL_CHECK(scalarType) << #ClassName " is expected to have ScalarType type, but got " +          \
                                       type->TypeName();                                                  \
    auto left = std::static_pointer_cast<const Expr>(ctx.DeserializeNode(GET_FIELD_OBJ("left"), zone));   \
    auto right = std::static_pointer_cast<const Expr>(ctx.DeserializeNode(GET_FIELD_OBJ("right"), zone)); \
    return std::make_shared<ClassName>(left, right, scalarType->dtype_, span);                           \
  }

DESERIALIZE_BINARY_EXPR(Add)
DESERIALIZE_BINARY_EXPR(Sub)
DESERIALIZE_BINARY_EXPR(Mul)
DESERIALIZE_BINARY_EXPR(FloorDiv)
DESERIALIZE_BINARY_EXPR(FloorMod)
DESERIALIZE_BINARY_EXPR(FloatDiv)
DESERIALIZE_BINARY_EXPR(Min)
DESERIALIZE_BINARY_EXPR(Max)
DESERIALIZE_BINARY_EXPR(Pow)
DESERIALIZE_BINARY_EXPR(Eq)
DESERIALIZE_BINARY_EXPR(Ne)
DESERIALIZE_BINARY_EXPR(Lt)
DESERIALIZE_BINARY_EXPR(Le)
DESERIALIZE_BINARY_EXPR(Gt)
DESERIALIZE_BINARY_EXPR(Ge)
DESERIALIZE_BINARY_EXPR(And)
DESERIALIZE_BINARY_EXPR(Or)
DESERIALIZE_BINARY_EXPR(Xor)
DESERIALIZE_BINARY_EXPR(BitAnd)
DESERIALIZE_BINARY_EXPR(BitOr)
DESERIALIZE_BINARY_EXPR(BitXor)
DESERIALIZE_BINARY_EXPR(BitShiftLeft)
DESERIALIZE_BINARY_EXPR(BitShiftRight)

// Macro for unary expressions
#define DESERIALIZE_UNARY_EXPR(ClassName)                                                          \
  static IRNodePtr Deserialize##ClassName(const msgpack::object& fieldsObj, msgpack::zone& zone,  \
                                          DeserializerContext& ctx) {                              \
    auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));                                        \
    auto type = ctx.DeserializeType(GET_FIELD_OBJ("type"), zone);                                  \
    auto scalarType = As<ScalarType>(type);                                                       \
    INTERNAL_CHECK(scalarType) << #ClassName " is expected to have ScalarType type, but got " +   \
                                       type->TypeName();                                           \
    auto operand =                                                                                 \
        std::static_pointer_cast<const Expr>(ctx.DeserializeNode(GET_FIELD_OBJ("operand"), zone)); \
    return std::make_shared<ClassName>(operand, scalarType->dtype_, span);                        \
  }

DESERIALIZE_UNARY_EXPR(Abs)
DESERIALIZE_UNARY_EXPR(Neg)
DESERIALIZE_UNARY_EXPR(Not)
DESERIALIZE_UNARY_EXPR(BitNot)
DESERIALIZE_UNARY_EXPR(Cast)

// Deserialize AssignStmt
static IRNodePtr DeserializeAssignStmt(const msgpack::object& fieldsObj, msgpack::zone& zone,
                                       DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  auto var = std::static_pointer_cast<const Var>(ctx.DeserializeNode(GET_FIELD_OBJ("var"), zone));
  auto value = std::static_pointer_cast<const Expr>(ctx.DeserializeNode(GET_FIELD_OBJ("value"), zone));
  return std::make_shared<AssignStmt>(var, value, span);
}

// Deserialize IfStmt
static IRNodePtr DeserializeIfStmt(const msgpack::object& fieldsObj, msgpack::zone& zone,
                                   DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  auto condition =
      std::static_pointer_cast<const Expr>(ctx.DeserializeNode(GET_FIELD_OBJ("condition"), zone));

  // Deserialize thenBody as single StmtPtr
  auto thenBody =
      std::static_pointer_cast<const Stmt>(ctx.DeserializeNode(GET_FIELD_OBJ("then_body"), zone));

  // Deserialize elseBody as optional StmtPtr
  std::optional<StmtPtr> elseBody;
  auto elseObjOpt = GetOptionalFieldObj(fieldsObj, "else_body", ctx);
  if (elseObjOpt.has_value()) {
    elseBody = std::static_pointer_cast<const Stmt>(ctx.DeserializeNode(*elseObjOpt, zone));
  }

  std::vector<VarPtr> returnVars;
  auto returnVarsObj = GET_FIELD_OBJ("return_vars");
  if (returnVarsObj.type == msgpack::type::ARRAY) {
    for (uint32_t i = 0; i < returnVarsObj.via.array.size; ++i) {
      returnVars.push_back(
          std::static_pointer_cast<const Var>(ctx.DeserializeNode(returnVarsObj.via.array.ptr[i], zone)));
    }
  }

  return std::make_shared<IfStmt>(condition, thenBody, elseBody, returnVars, span);
}

// Helper: deserialize a msgpack array field into a vector of ExprPtr
static std::vector<ExprPtr> DeserializeExprArray(const msgpack::object& arrayObj, msgpack::zone& zone,
                                                  DeserializerContext& ctx) {
  std::vector<ExprPtr> result;
  if (arrayObj.type == msgpack::type::ARRAY) {
    for (uint32_t i = 0; i < arrayObj.via.array.size; ++i) {
      result.push_back(
          std::static_pointer_cast<const Expr>(ctx.DeserializeNode(arrayObj.via.array.ptr[i], zone)));
    }
  }
  return result;
}

// Deserialize YieldStmt
static IRNodePtr DeserializeYieldStmt(const msgpack::object& fieldsObj, msgpack::zone& zone,
                                      DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  auto value = DeserializeExprArray(GET_FIELD_OBJ("value"), zone, ctx);
  return std::make_shared<YieldStmt>(value, span);
}

// Deserialize ReturnStmt
static IRNodePtr DeserializeReturnStmt(const msgpack::object& fieldsObj, msgpack::zone& zone,
                                       DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  auto value = DeserializeExprArray(GET_FIELD_OBJ("value"), zone, ctx);
  return std::make_shared<ReturnStmt>(value, span);
}

// Deserialize ForStmt
static IRNodePtr DeserializeForStmt(const msgpack::object& fieldsObj, msgpack::zone& zone,
                                    DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  auto loopVar = std::static_pointer_cast<const Var>(ctx.DeserializeNode(GET_FIELD_OBJ("loop_var"), zone));
  auto start = std::static_pointer_cast<const Expr>(ctx.DeserializeNode(GET_FIELD_OBJ("start"), zone));
  auto stop = std::static_pointer_cast<const Expr>(ctx.DeserializeNode(GET_FIELD_OBJ("stop"), zone));
  auto step = std::static_pointer_cast<const Expr>(ctx.DeserializeNode(GET_FIELD_OBJ("step"), zone));

  std::vector<IterArgPtr> iterArgs;
  auto iterArgsObj = GET_FIELD_OBJ("iter_args");
  if (iterArgsObj.type == msgpack::type::ARRAY) {
    for (uint32_t i = 0; i < iterArgsObj.via.array.size; ++i) {
      iterArgs.push_back(
          std::static_pointer_cast<const IterArg>(ctx.DeserializeNode(iterArgsObj.via.array.ptr[i], zone)));
    }
  }

  // Deserialize body as single StmtPtr
  auto body = std::static_pointer_cast<const Stmt>(ctx.DeserializeNode(GET_FIELD_OBJ("body"), zone));

  std::vector<VarPtr> returnVars;
  auto returnVarsObj = GET_FIELD_OBJ("return_vars");
  if (returnVarsObj.type == msgpack::type::ARRAY) {
    for (uint32_t i = 0; i < returnVarsObj.via.array.size; ++i) {
      returnVars.push_back(
          std::static_pointer_cast<const Var>(ctx.DeserializeNode(returnVarsObj.via.array.ptr[i], zone)));
    }
  }

  return std::make_shared<ForStmt>(loopVar, start, stop, step, iterArgs, body, returnVars, span);
}

// Helper: deserialize a msgpack array field into a vector of StmtPtr
static std::vector<StmtPtr> DeserializeStmtArray(const msgpack::object& arrayObj, msgpack::zone& zone,
                                                  DeserializerContext& ctx) {
  std::vector<StmtPtr> result;
  if (arrayObj.type == msgpack::type::ARRAY) {
    for (uint32_t i = 0; i < arrayObj.via.array.size; ++i) {
      result.push_back(
          std::static_pointer_cast<const Stmt>(ctx.DeserializeNode(arrayObj.via.array.ptr[i], zone)));
    }
  }
  return result;
}

// Deserialize SeqStmts
static IRNodePtr DeserializeSeqStmts(const msgpack::object& fieldsObj, msgpack::zone& zone,
                                     DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  auto stmts = DeserializeStmtArray(GET_FIELD_OBJ("stmts"), zone, ctx);
  return std::make_shared<SeqStmts>(stmts, span);
}

// Deserialize OpStmts
static IRNodePtr DeserializeOpStmts(const msgpack::object& fieldsObj, msgpack::zone& zone,
                                    DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  auto stmts = DeserializeStmtArray(GET_FIELD_OBJ("stmts"), zone, ctx);
  return std::make_shared<OpStmts>(stmts, span);
}

// Deserialize EvalStmt
static IRNodePtr DeserializeEvalStmt(const msgpack::object& fieldsObj, msgpack::zone& zone,
                                     DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  auto expr = std::static_pointer_cast<const Expr>(ctx.DeserializeNode(GET_FIELD_OBJ("expr"), zone));
  return std::make_shared<EvalStmt>(expr, span);
}

// Deserialize Function
static IRNodePtr DeserializeFunction(const msgpack::object& fieldsObj, msgpack::zone& zone,
                                     DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  std::string name = GET_FIELD(std::string, "name");

  // Deserialize funcType field (default to Opaque for backward compatibility)
  FunctionType funcType = FunctionType::OPAQUE;
  try {
    uint8_t typeCode = GET_FIELD(uint8_t, "func_type");
    funcType = static_cast<FunctionType>(typeCode);
  } catch (...) {
    // Field doesn't exist in old serialized data, use default
    funcType = FunctionType::OPAQUE;
  }

  std::vector<VarPtr> params;
  auto paramsObj = GET_FIELD_OBJ("params");
  if (paramsObj.type == msgpack::type::ARRAY) {
    for (uint32_t i = 0; i < paramsObj.via.array.size; ++i) {
      params.push_back(
          std::static_pointer_cast<const Var>(ctx.DeserializeNode(paramsObj.via.array.ptr[i], zone)));
    }
  }

  std::vector<TypePtr> returnTypes;
  auto returnTypesObj = GET_FIELD_OBJ("return_types");
  if (returnTypesObj.type == msgpack::type::ARRAY) {
    for (uint32_t i = 0; i < returnTypesObj.via.array.size; ++i) {
      returnTypes.push_back(ctx.DeserializeType(returnTypesObj.via.array.ptr[i], zone));
    }
  }

  auto body = std::static_pointer_cast<const Stmt>(ctx.DeserializeNode(GET_FIELD_OBJ("body"), zone));

  return std::make_shared<Function>(name, params, returnTypes, body, span, funcType);
}

// Deserialize Program
static IRNodePtr DeserializeProgram(const msgpack::object& fieldsObj, msgpack::zone& zone,
                                    DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  std::string name = GET_FIELD(std::string, "name");

  std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> functions;
  auto functionsObj = GET_FIELD_OBJ("functions");
  if (functionsObj.type == msgpack::type::ARRAY) {
    for (uint32_t i = 0; i < functionsObj.via.array.size; ++i) {
      auto entryObj = functionsObj.via.array.ptr[i];
      if (entryObj.type == msgpack::type::MAP) {
        msgpack::object keyObj, valueObj;
        bool hasKey = false, hasValue = false;

        msgpack::object_kv* p = entryObj.via.map.ptr;
        msgpack::object_kv* const pend = entryObj.via.map.ptr + entryObj.via.map.size;
        for (; p < pend; ++p) {
          std::string key;
          p->key.convert(key);
          if (key == "key") {
            keyObj = p->val;
            hasKey = true;
          } else if (key == "value") {
            valueObj = p->val;
            hasValue = true;
          }
        }

        if (hasKey && hasValue) {
          auto globalVar = std::static_pointer_cast<const GlobalVar>(ctx.DeserializeOp(keyObj));
          auto function = std::static_pointer_cast<const Function>(ctx.DeserializeNode(valueObj, zone));
          functions[globalVar] = function;
        }
      }
    }
  }

  return std::make_shared<Program>(functions, name, span);
}

// Deserialize MakeTuple
static IRNodePtr DeserializeMakeTuple(const msgpack::object& fieldsObj, msgpack::zone& zone,
                                      DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  auto elementsObj = GET_FIELD_OBJ("elements");
  auto elementsVec = elementsObj.as<std::vector<msgpack::object>>();
  std::vector<ExprPtr> elements;
  elements.reserve(elementsVec.size());
  for (const auto& elemObj : elementsVec) {
    elements.push_back(std::static_pointer_cast<const Expr>(ctx.DeserializeNode(elemObj, zone)));
  }
  return std::make_shared<MakeTuple>(std::move(elements), span);
}

// Deserialize TupleGetItemExpr
static IRNodePtr DeserializeTupleGetItemExpr(const msgpack::object& fieldsObj, msgpack::zone& zone,
                                             DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  auto tuple = std::static_pointer_cast<const Expr>(ctx.DeserializeNode(GET_FIELD_OBJ("tuple"), zone));
  int index = GET_FIELD(int, "index");
  return std::make_shared<TupleGetItemExpr>(tuple, index, span);
}

// Register all types with the registry
static TypeRegistrar _var_registrar("Var", DeserializeVar);
static TypeRegistrar _iter_arg_registrar("IterArg", DeserializeIterArg);
static TypeRegistrar _memref_registrar("MemRef", DeserializeMemRef);
static TypeRegistrar _const_int_registrar("ConstInt", DeserializeConstInt);
static TypeRegistrar _const_float_registrar("ConstFloat", DeserializeConstFloat);
static TypeRegistrar _const_bool_registrar("ConstBool", DeserializeConstBool);
static TypeRegistrar _call_registrar("Call", DeserializeCall);

static TypeRegistrar _add_registrar("Add", DeserializeAdd);
static TypeRegistrar _sub_registrar("Sub", DeserializeSub);
static TypeRegistrar _mul_registrar("Mul", DeserializeMul);
static TypeRegistrar _floor_div_registrar("FloorDiv", DeserializeFloorDiv);
static TypeRegistrar _floor_mod_registrar("FloorMod", DeserializeFloorMod);
static TypeRegistrar _float_div_registrar("FloatDiv", DeserializeFloatDiv);
static TypeRegistrar _min_registrar("Min", DeserializeMin);
static TypeRegistrar _max_registrar("Max", DeserializeMax);
static TypeRegistrar _pow_registrar("Pow", DeserializePow);
static TypeRegistrar _eq_registrar("Eq", DeserializeEq);
static TypeRegistrar _ne_registrar("Ne", DeserializeNe);
static TypeRegistrar _lt_registrar("Lt", DeserializeLt);
static TypeRegistrar _le_registrar("Le", DeserializeLe);
static TypeRegistrar _gt_registrar("Gt", DeserializeGt);
static TypeRegistrar _ge_registrar("Ge", DeserializeGe);
static TypeRegistrar _and_registrar("And", DeserializeAnd);
static TypeRegistrar _or_registrar("Or", DeserializeOr);
static TypeRegistrar _xor_registrar("Xor", DeserializeXor);
static TypeRegistrar _bit_and_registrar("BitAnd", DeserializeBitAnd);
static TypeRegistrar _bit_or_registrar("BitOr", DeserializeBitOr);
static TypeRegistrar _bit_xor_registrar("BitXor", DeserializeBitXor);
static TypeRegistrar _bit_shift_left_registrar("BitShiftLeft", DeserializeBitShiftLeft);
static TypeRegistrar _bit_shift_right_registrar("BitShiftRight", DeserializeBitShiftRight);

static TypeRegistrar _abs_registrar("Abs", DeserializeAbs);
static TypeRegistrar _neg_registrar("Neg", DeserializeNeg);
static TypeRegistrar _not_registrar("Not", DeserializeNot);
static TypeRegistrar _bit_not_registrar("BitNot", DeserializeBitNot);
static TypeRegistrar _cast_registrar("Cast", DeserializeCast);

static TypeRegistrar _assign_stmt_registrar("AssignStmt", DeserializeAssignStmt);
static TypeRegistrar _if_stmt_registrar("IfStmt", DeserializeIfStmt);
static TypeRegistrar _yield_stmt_registrar("YieldStmt", DeserializeYieldStmt);
static TypeRegistrar _return_stmt_registrar("ReturnStmt", DeserializeReturnStmt);
static TypeRegistrar _for_stmt_registrar("ForStmt", DeserializeForStmt);
static TypeRegistrar _seq_stmts_registrar("SeqStmts", DeserializeSeqStmts);
static TypeRegistrar _op_stmts_registrar("OpStmts", DeserializeOpStmts);
static TypeRegistrar _eval_stmt_registrar("EvalStmt", DeserializeEvalStmt);

static TypeRegistrar _function_registrar("Function", DeserializeFunction);
static TypeRegistrar _program_registrar("Program", DeserializeProgram);

static TypeRegistrar _make_tuple_registrar("MakeTuple", DeserializeMakeTuple);
static TypeRegistrar _tuple_get_item_expr_registrar("TupleGetItemExpr", DeserializeTupleGetItemExpr);

}  // namespace serialization
}  // namespace ir
}  // namespace pypto
