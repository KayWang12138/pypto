/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file attr_holder.h
 * \brief
 */

#pragma once
#include <iostream>
#include <map>
#include <string>
#include <any>

#include "interface/utils/any.h"
#include "interface/utils/common.h"
#include "interface/utils/log.h"
#include "element.h"

namespace npu::tile_fwk {
const std::string OP_ATTR_PREFIX = "op_attr_";

class AttrHolder {
private:
    std::map<std::string, npu::tile_fwk::Any> attributes;

public:
    std::map<std::string, npu::tile_fwk::Any> GetAttr() const {
        return attributes;
    }
    bool HasAttr(const std::string &key) const {
        if (key.empty()) {
            return false;
        }
        return attributes.find(key) != attributes.end();
    }

    // 设置属性值
    template <typename T>
    void SetAttr(const std::string &key, const T &value) {
        attributes[key] = value;
    }

    // 获取属性值
    template <typename T>
    const T *GetAttr(const std::string &key) const {
        auto it = attributes.find(key);
        if (it != attributes.end()) {
            try {
                return std::addressof(npu::tile_fwk::AnyCast<T &>(it->second));
            } catch (const std::bad_any_cast &) {
                return nullptr;
            }
        }
        return nullptr;
    }

    npu::tile_fwk::Any GetRawAttr(const std::string &key) const {
        auto it = attributes.find(key);
        if (it != attributes.end()) {
            return it->second;
        }
        return npu::tile_fwk::Any();
    }

    template <typename T>
    bool GetAttr(const std::string &key, T &value) const {
        auto it = attributes.find(key);
        if (it != attributes.end()) {
            if (it->second.Type() == typeid(T)) {
                value = npu::tile_fwk::AnyCast<T>(it->second);
            } else {
                return false;
            }
        } else {
            return false;
        }
        return true;
    }

    // 移除属性
    void RemoveAttr(const std::string &key) {
        auto it = attributes.find(key);
        if (it != attributes.end()) {
            attributes.erase(it);
        } else {
            throw std::out_of_range("Attribute not found: " + key);
        }
    }

    // 打印所有属性
    std::string DumpAttr(const std::string &key) const {
        std::string result;
        auto it = attributes.find(key);
        if (it != attributes.end()) {
            try {
                if (it->second.Type() == typeid(int)) {
                    result = std::to_string(npu::tile_fwk::AnyCast<int>(it->second));
                } else if (it->second.Type() == typeid(int64_t)) {
                    result = std::to_string(npu::tile_fwk::AnyCast<int64_t>(it->second));
                } else if (it->second.Type() == typeid(uint32_t)) {
                    result = std::to_string(npu::tile_fwk::AnyCast<uint32_t>(it->second));
                } else if (it->second.Type() == typeid(uint64_t)) {
                    result = std::to_string(npu::tile_fwk::AnyCast<uint64_t>(it->second));
                } else if (it->second.Type() == typeid(float)) {
                    result = std::to_string(npu::tile_fwk::AnyCast<float>(it->second));
                } else if (it->second.Type() == typeid(double)) {
                    result = std::to_string(npu::tile_fwk::AnyCast<double>(it->second));
                } else if (it->second.Type() == typeid(std::string)) {
                    result = npu::tile_fwk::AnyCast<std::string>(it->second);
                } else if (it->second.Type() == typeid(bool)) {
                    result = std::to_string(npu::tile_fwk::AnyCast<bool>(it->second));
                } else if (it->second.Type() == typeid(std::vector<int>)){
                    result = IntVecToStr(npu::tile_fwk::AnyCast<std::vector<int>>(it->second));
                } else if (it->second.Type() == typeid(Element)) {
                    auto tensorElement = npu::tile_fwk::AnyCast<Element>(it->second);
                    if (tensorElement.IsSigned()) {
                        result = std::to_string(tensorElement.GetSignedData());
                    } else if (tensorElement.IsUnsigned()) {
                        result = std::to_string(tensorElement.GetUnsignedData());
                    } else if (tensorElement.IsFloat()) {
                        result = std::to_string(tensorElement.GetFloatData());
                    }
                } else {
                    result += "unsupported type ";
                    result += it->second.Type().name();
                }
            } catch (const std::bad_any_cast &) {
                result = "Bad any cast";
            }
            return result;
        } else {
            result = "Invalid attribute key " + key;
            return result;
        }
    }

    void PrintAttributes() const {
        for (const auto &pair : attributes) {
            std::cout << pair.first << ": ";
            try {
                if (pair.second.Type() == typeid(int)) {
                    std::cout << npu::tile_fwk::AnyCast<int>(pair.second);
                } else if (pair.second.Type() == typeid(double)) {
                    std::cout << npu::tile_fwk::AnyCast<double>(pair.second);
                } else if (pair.second.Type() == typeid(std::string)) {
                    std::cout << (npu::tile_fwk::AnyCast<std::string>(pair.second));
                } else {
                    std::cout << "Unknown type";
                }
            } catch (const std::bad_any_cast &) {
                std::cout << "Bad any cast";
            }
            std::cout << std::endl;
        }
    }

    nlohmann::json DumpAttrJson() const {
        nlohmann::json attrJson;
        for (const auto &pair : attributes) {
            try {
                if (pair.second.Type() == typeid(int)) {
                    attrJson[pair.first] = npu::tile_fwk::AnyCast<int>(pair.second);
                } else if (pair.second.Type() == typeid(int64_t)) {
                    attrJson[pair.first] = npu::tile_fwk::AnyCast<int64_t>(pair.second);
                } else if (pair.second.Type() == typeid(uint64_t)) {
                    attrJson[pair.first] = npu::tile_fwk::AnyCast<uint64_t>(pair.second);
                } else if (pair.second.Type() == typeid(uint32_t)) {
                    attrJson[pair.first] = npu::tile_fwk::AnyCast<uint32_t>(pair.second);
                } else if (pair.second.Type() == typeid(std::vector<int>)) {
                    attrJson[pair.first] = npu::tile_fwk::AnyCast<std::vector<int>>(pair.second);
                } else if (pair.second.Type() == typeid(double)) {
                    attrJson[pair.first] = npu::tile_fwk::AnyCast<double>(pair.second);
                } else if (pair.second.Type() == typeid(float)) {
                    attrJson[pair.first] = npu::tile_fwk::AnyCast<float>(pair.second);
                } else if (pair.second.Type() == typeid(std::string)) {
                    attrJson[pair.first] = npu::tile_fwk::AnyCast<std::string>(pair.second);
                } else if (pair.second.Type() == typeid(bool)) {
                    attrJson[pair.first] = npu::tile_fwk::AnyCast<bool>(pair.second);
                } else if (pair.second.Type() == typeid(Element)) {
                    attrJson[pair.first] = ToJson(npu::tile_fwk::AnyCast<Element>(pair.second));
                } else {
                    attrJson[pair.first] = "Unsupported type";
                }
            } catch (const std::bad_any_cast &) {
                std::cout << "Bad any cast";
            }
        }
        return attrJson;
    }

    nlohmann::json DumpAttrJson(const std::string &key) const {
        auto iter = attributes.find(key);
        if (iter != attributes.end()) {
            auto &second = iter->second;
            try {
                if (second.Type() == typeid(int)) {
                    return nlohmann::json(npu::tile_fwk::AnyCast<int>(second));
                } else if (second.Type() == typeid(int64_t)) {
                    return nlohmann::json(npu::tile_fwk::AnyCast<int64_t>(second));
                } else if (second.Type() == typeid(uint64_t)) {
                    return nlohmann::json(npu::tile_fwk::AnyCast<uint64_t>(second));
                } else if (second.Type() == typeid(uint32_t)) {
                    return nlohmann::json(npu::tile_fwk::AnyCast<uint32_t>(second));
                } else if (second.Type() == typeid(std::vector<int>)) {
                    return nlohmann::json(npu::tile_fwk::AnyCast<std::vector<int>>(second));
                } else if (second.Type() == typeid(std::vector<float>)) {
                    return nlohmann::json(npu::tile_fwk::AnyCast<std::vector<float>>(second));
                } else if (second.Type() == typeid(std::vector<bool>)) {
                    return nlohmann::json(npu::tile_fwk::AnyCast<std::vector<bool>>(second));
                } else if (second.Type() == typeid(double)) {
                    return nlohmann::json(npu::tile_fwk::AnyCast<double>(second));
                } else if (second.Type() == typeid(float)) {
                    return nlohmann::json(npu::tile_fwk::AnyCast<float>(second));
                } else if (second.Type() == typeid(std::string)) {
                    return nlohmann::json(npu::tile_fwk::AnyCast<std::string>(second));
                } else if (second.Type() == typeid(bool)) {
                    return nlohmann::json(npu::tile_fwk::AnyCast<bool>(second));
                } else if (second.Type() == typeid(Element)) {
                    return ToJson(npu::tile_fwk::AnyCast<Element>(second));
                } else {
                    return nlohmann::json("Unsupported type");
                }
            } catch (const std::bad_any_cast &) {
                std::cout << "Bad any cast";
            }
        }
        return nlohmann::json();
    }

    void LoadVecAttr(const std::string &key, const std::vector<nlohmann::json> &vec) {
        if (vec[0].is_string()) {
            std::vector<std::string> strVec;
            for (const auto &j : vec) {
                strVec.emplace_back(j.get<std::string>());
            }
            SetAttr(key, strVec);
        } else if (vec[0].is_number()) {
            if (vec[0].is_number_integer()) {
                std::vector<int> intVec;
                for (const auto &j : vec) {
                    intVec.emplace_back(j.get<int>());
                }
                SetAttr(key, intVec);
            } else {
                std::vector<float> floatVec;
                for (const auto &j : vec) {
                    floatVec.emplace_back(j.get<float>());
                }
                SetAttr(key, floatVec);
            }
        } else if (vec[0].is_boolean()) {
            std::vector<bool> boolVec;
            for (const auto &j : vec) {
                boolVec.emplace_back(j.get<bool>());
            }
            SetAttr(key, boolVec);
        } else {
            return;
        }
    }

    void LoadAttrJson(const std::string &key, const nlohmann::json &attrJson) {
        try {
            if (attrJson.is_array()) {
                // 处理数组
                std::vector<nlohmann::json> vec;
                for (const auto &elem : attrJson) {
                    vec.push_back(elem);
                }
                if (!vec.empty()) {
                    LoadVecAttr(key, vec);
                }
            } else if (attrJson.is_object()) {
                SetAttr(key, parseElement(attrJson));
            } else if (attrJson.is_string()) {
                SetAttr(key, attrJson.get<std::string>());
            } else if (attrJson.is_number()) {
                if (attrJson.is_number_integer()) {
                    SetAttr(key, attrJson.get<int>());
                } else {
                    SetAttr(key, attrJson.get<float>());
                }
            } else if (attrJson.is_boolean()) {
                SetAttr(key, attrJson.get<bool>());
            } else if (attrJson.is_null()) {
                return;
            }
        } catch (...) {
            ALOG_ERROR_F("json parse error");
        }
    }
    [[nodiscard]] std::map<std::string, npu::tile_fwk::Any> GetAllAttr() const { return attributes; }
};
} // namespace npu::tile_fwk
