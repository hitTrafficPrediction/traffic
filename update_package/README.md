# 使用说明

## 0. 文件结构

- data 数据处理文件夹
  - process_data.py 处理数据文件执行脚本
  - road_names.py 路段名称
  - utils.py 辅助工具包
- train 训练数据文件夹
  - apis.py 辅助工具包
  - utils.py 辅助工具包
  - main.py 训练执行脚本

## 1. 数据处理

- 请执行data/process_data.py, 可根据实际情况修改输入文件, 输出文件路径, 详见函数入参。

- 使用的原始文件为数据库导出的带列名的csv文件, **且整个程序运行依赖于列名, 故使用时务必检查数据库及导出文件的列名, 并在必要时修改程序。**

- 本程序使用的数据库导出文件, 导出时间为2022/03/14, 具体文件为
  
- 链接：https://pan.baidu.com/s/12M9wDaENZc608lx1Xh84TA 
  提取码：fbrw 



## 2. 模型训练

- 请执行data/main.py, 具体参数请见文件107行, 可以通过外部传参指定训练路段等内容。

- 示例

```

python main.py --train_name='G1-1_up' --type='flow' --device='cpu'  --save_pth=Ture

```

