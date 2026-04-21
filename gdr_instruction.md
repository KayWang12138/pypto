通过python3 gdr_origin.py 运行，发现dq计算结果精度不正确，请进行代码二分，二分指导见/mnt/workspace/gitCode/cann/pypto/.agents/skills/pypto-precision-compare/reference/binary-search-guide.md
以及/mnt/workspace/gitCode/cann/pypto/.agents/skills/pypto-precision-compare/reference/binary-search.md

确认那行的计算存在精度问题，确认方法为：输入tensor精度正确，输出tensor精度错误
所谓精度就是kernel输出的tensor和golden进行比较，每次验证都copy gdr_origin.py代码到一个新的文件进行验证。