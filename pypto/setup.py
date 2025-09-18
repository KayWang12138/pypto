# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
from setuptools import setup
from setuptools.command.install import install
import subprocess
import pathlib
import sysconfig


class InstallWithRpathFix(install):
    def run(self):
        # call normal install
        super().run()

        # get install dirs
        print("Install base:", self.install_base)
        print("Purelib:", self.install_lib)
        print("Platlib:", self.install_platlib)
        site = f"{self.install_lib}/pto"
        for so in pathlib.Path(site).rglob("*.so"):
            subprocess.run(["patchelf", "--set-rpath", "$ORIGIN", str(so)], check=True)


setup(
    name="pto",
    version="0.1.0",
    author="AscendC++ authors",
    packages=["pto"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    include_package_data=True,
    package_data={
        "pto": [
            "*.so",
            "*.json",
        ],  # include dynlibs and Json files from AscendC++ in the wheel
    },
    cmdclass={"install": InstallWithRpathFix},  # Work-around to fix RPATH in dylibs
    zip_safe=False,
)
