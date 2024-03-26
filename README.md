# train ldm
修改ldm代码，在单卡RTX4090上训练

文生图效果图

<p align="center">
<img src=assets/samples_gs-008529_e-000010_b-000003.png />
</p>



<p align="center">
<img src=assets/samples_gs-009562_e-000011_b-000003.png />
</p>

## code analysis

ldm代码解析见sd/sd.md文件

## train

### data

准备好数据放在datasets文件夹下



### weight

在

```
models\bert
models\ldm\text2img-large
```

下下载预训练权重



### run

运行main.py文件
