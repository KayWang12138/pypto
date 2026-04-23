# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

obj-m += pmu_user_access.o

KVER ?= $(shell uname -r)
KDIR ?= /lib/modules/$(KVER)/build
PWD  := $(shell pwd)
CC   ?= gcc

.PHONY: all clean install uninstall probe loader

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
	rm -f probe_pmu loader

install: all
	sudo insmod ./pmu_user_access.ko
	sudo dmesg | tail -n 3

uninstall:
	sudo rmmod pmu_user_access
	sudo dmesg | tail -n 2

probe: probe_pmu
	./probe_pmu

loader: loader.c
	$(CC) -O2 -Wall $< -o $@

probe_pmu: probe_pmu.c
	$(CC) -O2 -Wall $< -o $@
