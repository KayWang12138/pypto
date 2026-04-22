#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/driver/dsmi_common_interface.h"

int main() {
    DSMI_AICPU_INFO aicpu_info;
    memset(&aicpu_info, 0, sizeof(DSMI_AICPU_INFO));
    int ret = dsmi_get_aicpu_info(1, &aicpu_info);
    if (ret == 0) {
        printf("AICPU_NUM=%u\n", aicpu_info.aicpuNum);
        printf("AICPU_MAX_FREQ_MHZ=%u\n", aicpu_info.maxFreq);
        printf("AICPU_CUR_FREQ_MHZ=%u\n", aicpu_info.curFreq);
        return 0;
    }
    printf("ERROR=%d\n", ret);
    return 1;
}
