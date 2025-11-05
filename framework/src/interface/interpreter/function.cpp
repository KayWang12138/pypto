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
 * \file function.cpp
 * \brief
 */

#include "interface/interpreter/function.h"
#include "interface/interpreter/flow_verifier.h"
#include "interface/configs/config_manager.h"
#include "interface/utils/file_utils.h"

namespace npu::tile_fwk {
constexpr int MAX_IDENT_LEVEL = 20;

static std::string HtmlEscape(const std::string &src, bool escapeLineBreak = true) {
    std::string ret;
    for (auto &c : src) {
        switch (c) {
            case '<': ret += "&lt;"; break;
            case '>': ret += "&gt;"; break;
            case '&': ret += "&amp;"; break;
            case '\n':
                if (escapeLineBreak) {
                    ret += "<br/>";
                }
                ret.push_back(c);
                break;
            default: ret.push_back(c);
        }
    }
    return ret;
}

void FunctionInterpreter::DumpFunctionHead(Function *func) {
    if (execDumpLevel >= EXEC_DUMP_LEVEL_OPERATION) {
        int indent = GetFrameSize();
        auto head = func->DumpSSATitle();
        auto raw = func->DumpSSARawTensor(indent);
        auto incast = func->DumpSSAIncast(indent);
        auto outcast = func->DumpSSAOutcast(indent);
        auto attr = func->DumpSSAAttribute(indent);
        auto symbol = DumpSymbolDict();
        ALOG_INFO_F("%s Function %s\n", execDumpFuncKey.c_str(), head.c_str());
        ALOG_INFO_F("%s\n", raw.c_str());
        ALOG_INFO_F("%s\n", incast.c_str());
        ALOG_INFO_F("%s\n", outcast.c_str());
        ALOG_INFO_F("%s\n", attr.c_str());
        ALOG_INFO_F("%s\n", symbol.c_str());
        if (execDumpFile) {
            fprintf(execDumpFile, "<div class=\"function indent_%d\">%s</div>", indent, execDumpFuncKey.c_str());
            fprintf(execDumpFile, "<div class=\"function indent_%d\">frameIndex=%s</div>", indent, GetFrameCurrIndex().c_str());
            fprintf(execDumpFile, "<div class=\"function indent_%d\">%s</div>", indent, HtmlEscape(head).c_str());
            fprintf(execDumpFile, "<div class=\"function indent_%d\">%s</div>", indent, HtmlEscape(raw).c_str());
            fprintf(execDumpFile, "<div class=\"function indent_%d\">%s</div>", indent, HtmlEscape(incast).c_str());
            fprintf(execDumpFile, "<div class=\"function indent_%d\">%s</div>", indent, HtmlEscape(outcast).c_str());
            fprintf(execDumpFile, "<div class=\"function indent_%d\">%s</div>", indent, HtmlEscape(attr).c_str());
            fprintf(execDumpFile, "<div class=\"function indent_%d\">%s</div>", indent, HtmlEscape(symbol).c_str());
        }
    }
}

void FunctionInterpreter::DumpOperation(Operation *op) {
    if (execDumpLevel < EXEC_DUMP_LEVEL_OPERATION)
        return;
    int indent = GetFrameSize();
    auto dump = op->Dump();
    ALOG_INFO(execDumpFuncKey, " Operation: ", dump);
    if (execDumpFile) {
        std::string tensorId = GetDumpTensorId(GetFrameCurr(), op);
        std::string operationId = GetDumpOperationId(GetFrameCurr(), op);
        fprintf(execDumpFile,
            "<div class=\"indent_%d\" id=\"%s\">"
            "  <div class=\"operation\" id=\"%s\">%s</div>"
            "</div>\n",
            indent, tensorId.c_str(), operationId.c_str(), HtmlEscape(dump).c_str());
    }
}

static void DumpLine(FILE *f, const std::vector<std::string> &textList, const std::string &cssClass = "") {
    fprintf(f, "<tr>\n");
    for (size_t k = 0; k < textList.size(); k++) {
        std::string attr = " class=\"" + cssClass + "\"";
        fprintf(f, "  <td%s>%s</td>\n", attr.c_str(), textList[k].c_str());
    }
    fprintf(f, "</tr>\n");
}

static void DumpDataViewParallel(const std::shared_ptr<LogicalTensorData> &dataView,
    std::vector<ElementDump> &elementDumpList, util::ThreadPool *pool) {
    struct DumpTask {
        DumpTask(std::vector<ElementDump> *ret_, const LogicalTensorData *view_, int indexBegin_,
            int indexEnd_)
            : ret(ret_), view(view_), indexBegin(indexBegin_), indexEnd(indexEnd_) {}

        std::vector<ElementDump> *ret;
        const LogicalTensorData *view;
        int indexBegin;
        int indexEnd;

        static void Entry(void *c) {
            auto [ret, view, indexBegin, indexEnd] = *(DumpTask *)c;
            for (int i = indexBegin; i < indexEnd; i++) {
                view->DumpElement(i, &ret->at(i));
            }
        }
    };

    if (static_cast<int>(elementDumpList.size()) < dataView->GetSize()) {
        elementDumpList.resize(dataView->GetSize());
    }

    std::vector<DumpTask> dumpTaskList;
    int count = (dataView->GetSize() + pool->GetThreadCount() - 1) / pool->GetThreadCount();
    for (int i = 0; i < pool->GetThreadCount(); i++) {
        dumpTaskList.emplace_back(
            &elementDumpList, dataView.get(), count * i, std::min(count * (i + 1), dataView->GetSize()));
    }
    for (size_t i = 0; i < dumpTaskList.size(); i++) {
        pool->SubmitTask(&dumpTaskList[i], DumpTask::Entry);
    }
    pool->NotifyAll();
    pool->WaitForAll();
}

std::string FunctionInterpreter::DumpDataView(const std::shared_ptr<LogicalTensorData> &dataView) {
   DumpDataViewParallel(dataView, execDumpElementList, &operationInterpreter->GetPool());
   return dataView->Dump(&execDumpElementList);
}

std::string FunctionInterpreter::GetDumpFilePath(const std::string &lv0, const std::string &lv1, const std::string &filename) {
    std::string baseDirName = lv0 + "/" + lv1;
    if (!IsPathExist(baseDirName)) {
        CreateDir(baseDirName);
    }
    return baseDirName + "/" + filename;
}

void FunctionInterpreter::DumpTensorBinary(
        const std::shared_ptr<LogicalTensor> &tensor,
        const std::shared_ptr<LogicalTensorData> &dataView) {
    if (execDumpLevel < EXEC_DUMP_LEVEL_TENSOR || !execDumpFile)
        return;
    std::string dumpTensorDirName = GetDumpFrameDirName();
    std::string dumpTensorFileName = GetDumpTensorFileName(tensor);
    std::string dumpTensorFilePath = GetDumpFilePath(execDumpDir, dumpTensorDirName, dumpTensorFileName);
    dataView->Save(dumpTensorFilePath);
}

void FunctionInterpreter::DumpTensorList(const std::string &name, const std::vector<std::shared_ptr<LogicalTensor>> *tensorList,
    const std::vector<std::shared_ptr<LogicalTensorData>> *dataViewList) {
    if (execDumpLevel < EXEC_DUMP_LEVEL_TENSOR || !execDumpFile)
        return;

    std::string dumpTensorDirName = GetDumpFrameDirName();
    std::string dumpTensorFileName = GetDumpTensorListFileName(name);
    int indent = GetFrameSize();
    fprintf(execDumpFile, "<div class=\"detail indent_%d\"><a href=\"%s\">%s</a></div>", indent,
        (dumpTensorDirName + "/" + dumpTensorFileName).c_str(), dumpTensorFileName.c_str());

    std::string dumpTensorFilePath = GetDumpFilePath(execDumpDir, dumpTensorDirName, dumpTensorFileName);
    FILE *dumpTensorFile = fopen(dumpTensorFilePath.c_str(), "w");
    fprintf(dumpTensorFile, R"HTML(
<html>
    <head>
    <link rel="stylesheet" type="text/css" href="../../verifier.css">
    </head>
    <body>
)HTML");

    std::vector<std::string> textList(tensorList->size());
    constexpr int ONE_THOUSAND = 1000;
    fprintf(dumpTensorFile, "<table width=\"%dpx\">", static_cast<int>(tensorList->size() * ONE_THOUSAND));
    for (size_t k = 0; k < tensorList->size(); k++) {
        textList[k] = std::to_string(static_cast<int>(k));
        if (tensorList->at(k)) {
            textList[k] += " tileOpFormat:" + std::to_string(tensorList->at(k)->GetTileOpFormat());
        }
    }
    DumpLine(dumpTensorFile, textList, "table_head");

    for (size_t k = 0; k < tensorList->size(); k++) {
        if (tensorList->at(k) != nullptr) {
            textList[k] = HtmlEscape(tensorList->at(k)->Dump());
        } else {
            textList[k] = "";
        }
    }
    DumpLine(dumpTensorFile, textList);

    for (size_t k = 0; k < tensorList->size(); k++) {
        textList[k] = HtmlEscape(k < dataViewList->size() ? DumpDataView(dataViewList->at(k)) : "--");
    }
    DumpLine(dumpTensorFile, textList, "tensor_data");
    fprintf(dumpTensorFile, "</table>\n");

    fprintf(execDumpFile, R"HTML(
    </body>
</html>
)HTML");
    fclose(dumpTensorFile);
}

void FunctionInterpreter::DumpOperationTensor(Operation *op,
    const std::vector<std::shared_ptr<LogicalTensorData>> *ooperandDataViewList,
    const std::vector<std::shared_ptr<LogicalTensorData>> *ioperandDataViewList) {
    if (execDumpLevel < EXEC_DUMP_LEVEL_TENSOR || !execDumpFile)
        return;

    int indent = GetFrameSize();
    std::string dumpOperationDirName = GetDumpFrameDirName();
    std::string dumpOperationFileName = GetDumpOperationTensorFileName(op);
    fprintf(execDumpFile, "<div class=\"detail indent_%d\"><a href=\"%s\">%s</a></div>\n", indent,
        (dumpOperationDirName + "/" + dumpOperationFileName).c_str(), dumpOperationFileName.c_str());

    std::string dumpOperationFilePath = GetDumpFilePath(execDumpDir, dumpOperationDirName, dumpOperationFileName);
    FILE *dumpOperationFile = fopen(dumpOperationFilePath.c_str(), "w");
    fprintf(dumpOperationFile, R"HTML(
<html>
    <head>
    <link rel="stylesheet" type="text/css" href="../../verifier.css">
    </head>
    <body>
)HTML");
    auto dump = op->Dump();
    fprintf(dumpOperationFile, "<div class=\"operation\">%s</div>\n", dump.c_str());

    auto oopSize = op->GetOOperands().size();
    auto iopSize = op->GetIOperands().size();
    std::vector<std::string> textList(iopSize + oopSize);
    constexpr int ONE_THOUSAND = 1000;
    fprintf(dumpOperationFile, "<table width=\"%dpx\">", static_cast<int>((oopSize + iopSize) * ONE_THOUSAND));
    for (size_t k = 0; k < oopSize; k++) {
        textList[k] = "oop:" + std::to_string(static_cast<int>(k));
    }
    for (size_t k = 0; k < iopSize; k++) {
        textList[oopSize + k] = "iop:" + std::to_string(static_cast<int>(k));
    }
    DumpLine(dumpOperationFile, textList, "table_head");

    for (size_t k = 0; k < oopSize; k++) {
        auto oop = op->GetOOperands()[k];
        std::string tensorId = GetDumpTensorId(GetFrameCurr(), oop);
        textList[k] = "<a href=\"../entry.html#" + tensorId + "\">" + HtmlEscape(oop->Dump()) + "</a>";
    }
    for (size_t k = 0; k < iopSize; k++) {
        auto iop = op->GetIOperands()[k];
        std::string tensorId = GetDumpTensorId(GetFrameCurr(), iop);
        textList[oopSize + k] = "<a href=\"../entry.html#" + tensorId + "\">" + HtmlEscape(iop->Dump()) + "</a>";
    }
    DumpLine(dumpOperationFile, textList);

    size_t totalOOperandSize = 0;
    for (size_t k = 0; k < oopSize; k++) {
        if (k < ooperandDataViewList->size()) {
            auto dataView = ooperandDataViewList->at(k);
            totalOOperandSize += dataView->GetSize();
            textList[k] = HtmlEscape(DumpDataView(dataView));
            DumpTensorBinary(op->GetOOperands()[k], dataView);
        } else {
            textList[k] = "";
        }
    }
    size_t totalIOperandSize = 0;
    for (size_t k = 0; k < iopSize; k++) {
        if (k < ioperandDataViewList->size()) {
            auto dataView = ioperandDataViewList->at(k);
            if (totalIOperandSize + dataView->GetSize() < totalOOperandSize * 0x3) {
                totalIOperandSize += dataView->GetSize();
                textList[oopSize + k] = HtmlEscape(DumpDataView(dataView));
            } else {
                textList[oopSize + k] = "";
            }
        } else {
            textList[oopSize + k] = "";
        }
    }
    DumpLine(dumpOperationFile, textList, "tensor_data");
    fprintf(dumpOperationFile, "</table>\n");

    fprintf(dumpOperationFile, R"HTML(
    </body>
</html>
)HTML");
    fclose(dumpOperationFile);
}

void FunctionInterpreter::DumpPassTensorDiff(
        const std::shared_ptr<FunctionCaptureExecution> &captureExecution,
        const std::shared_ptr<FunctionCaptureExecution> &captureGolden) {
    if (execDumpLevel < EXEC_DUMP_LEVEL_TENSOR)
        return;
    if (captureExecution->GetFrameList().size() != captureGolden->GetFrameList().size())
        return;

    std::string dumpStyleFilePath = execDumpDir + "/" + "entry.css";
    execDumpStyleFile = fopen(dumpStyleFilePath.c_str(), "w");
    for (size_t idx = 0; idx < captureGolden->GetFrameList().size(); idx++) {
        auto frameExecution = captureExecution->GetFrameList()[idx];
        auto frameGolden = captureGolden->GetFrameList()[idx];
        const auto &tensorDictExecution = frameExecution->GetTensorDataViewDict();
        const auto &tensorDictGolden = frameGolden->GetTensorDataViewDict();
        std::vector<std::shared_ptr<LogicalTensor>> tensorList;
        for (auto &[tensor, dataViewExecution] : tensorDictExecution) {
            (void)dataViewExecution;
            if (tensorDictGolden.count(tensor)) {
                tensorList.push_back(tensor);
            }
        }
        for (size_t k = 0; k < tensorList.size(); k++) {
            auto tensor = tensorList[k];
            ALOG_INFO("Dump tensor diff: ", k, "/", tensorList.size(), " ", tensor->Dump(), " ",
                tensor->GetRawTensor()->Dump());
            auto dataViewExecution = tensorDictExecution.find(tensor)->second;
            auto dataViewGolden = tensorDictGolden.find(tensor)->second;
            auto compare = FlowVerifier::VerifyResult(dataViewGolden, dataViewExecution, static_cast<float>(1e-5));
            std::string tensorId = GetDumpTensorId(frameExecution, tensor);
            if (compare.Check()) {
                fprintf(execDumpStyleFile, "#%s { background-color: LightGreen; }\n", tensorId.c_str());
            } else {
                fprintf(execDumpStyleFile, "#%s { background-color: LightCoral; }\n", tensorId.c_str());
            }
        }
    }
    fclose(execDumpStyleFile);
}

void FunctionInterpreter::DumpBegin() {
    frameCount = 0;
    execDumpDir = config::LogTopFolder() + "/verify/" + execDumpFuncKey;
    CreateMultiLevelDir(execDumpDir);

    if (execDumpLevel < EXEC_DUMP_LEVEL_OPERATION)
        return;

    std::string styleFilePath = config::LogTopFolder() + "/verify/verifier.css";
    if (GetFileSize(styleFilePath) == 0) {
        FILE *fcss = fopen(styleFilePath.c_str(), "w");
        for (int i = 0; i < MAX_IDENT_LEVEL; i++) {
            fprintf(fcss, ".indent_%d { margin-left: %dpx; }\n", i, i * 50); // 50 is left margin
        }
        fprintf(fcss, ".table_head { width: 10%%; }\n");
        fprintf(fcss, ".tensor_data { vertical-align: top; }\n");
        fclose(fcss);
    }

    std::string dumpFilePath = execDumpDir + "/" + "entry.html";
    execDumpFile = fopen(dumpFilePath.c_str(), "w");
    fprintf(execDumpFile, R"HTML(
<html>
    <head>
    <link rel="stylesheet" type="text/css" href="../verifier.css">
    <link rel="stylesheet" type="text/css" href="entry.css">
    </head>
    <body>
)HTML");
}

void FunctionInterpreter::DumpEnd() {
    if (execDumpLevel < EXEC_DUMP_LEVEL_OPERATION)
        return;
    fprintf(execDumpFile, R"HTML(
    </body>
</html>
)HTML");
    fclose(execDumpFile);
}

} // namespace npu::tile_fwk
