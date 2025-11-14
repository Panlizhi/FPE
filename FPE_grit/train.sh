
# 1 修改环境（指定路径）安装 Deformable Attention:
# mkdir -p /tmp/grit && cp -r /gemini/code/grit/models  /tmp/grit/models && cd /tmp/grit/models/ops/ && python setup.py build develop && cd /gemini/code/grit
# 离线训练切换deformer-detr路径为/tmp/grit，in /gemini/code/grit/models/detection/det_module.py

#2 修改评价指标  stanford-corenlp-3.4.1.jar 可读写的路径   in   /gemini/code/grit/datasets/caption/metrics/tokenizer.py
# cp /gemini/code/grit/datasets/caption/metrics/stanford-corenlp-3.4.1.jar  /tmp/grit/

#离线训练
mkdir -p /tmp/grit && cp -r /gemini/code/grit/models  /tmp/grit/models && cd /tmp/grit/models/ops/ && python setup.py build develop && cd /gemini/code/grit
cp /gemini/code/grit/datasets/caption/metrics/stanford-corenlp-3.4.1.jar  /tmp/grit/
cd /gemini/code/grit

# 监控GPU
# watch -n 1 nvidia-smi

# 在线训练
# 原项目(本路径)安装 Deformable Attention：
cd models/ops/ && python setup.py build develop
python test.py
pip install hydra-core==1.1.0


# spaCy 提供的语言模型是经过预训练的，能够处理文本数据并执行以下任务：分词，词性标注，命名实体识别，识别文本中实体。
# python -m spacy download en_core_web_sm
#or
# https://github.com/explosion/spacy-models/releases?q=en_core_web_sm&expanded=true
# pip install /gemini/code/grit/data/en_core_web_sm-3.7.1-py3-none-any.whl


# 参数已调试好等着跑整个离线实验，离线训练切换deformer-detr路径为/tmp/grit，in /gemini/code/grit/models/detection/det_module.py
# 离线训练切换 /gemini/code/grit/datasets/caption/metrics/tokenizer.py 的 path_to_jar_dirname 的路径为/tmp 可写


# Experiment 1
# train the entire model.
#【xe训练】并【sc微调】 backbone + detector (Swin + Project + Deformable DETR) 、grid_net 和 cap_generator，从图片加载并使用image aug。

# 两卡不可跑

cd models/ops/ && python setup.py build develop
cd /gemini/code/grit


export DATA_ROOT=${GEMINI_DATA_IN1}/COCO2014 
export DATA_ROOT2=${GEMINI_DATA_IN2}            
export OUTPUT_ROOT=${GEMINI_DATA_OUT}
export MODEL_ROOT1=${GEMINI_PRETRAIN}
export MODEL_ROOT2=${GEMINI_PRETRAIN2}
python train_caption.py exp.name=FPE_finetune_region_grid \
    model.detector.checkpoint=${MODEL_ROOT2}/detector_checkpoint_4ds.pth \
    optimizer.finetune_xe_epochs=10\
    optimizer.finetune_sc_epochs=10 \
    optimizer.batch_size=32 \
    optimizer.num_workers=2 \
    exp.ngpus_per_node=8 \
    exp.world_size=8 \
    model.cap_generator.decoder_name=sequential \
    dataset.overfit=False \
    model.freq_net.visual_type="CNN" \
    model.freq_net.gamma=1 \
    model.freq_net.perturbation_strength=1 \
    model.use_freq_feat=True
# GPU：8卡 每卡显存：80 GB  CPU：32核 内存：96 GB  临时存储：100 GB
#           从image加载  num_workers=2   batch_size=64     GPU 不够
#           从image加载  num_workers=2   batch_size=32     xe：68 min/epoch       SC：50 min/epoch    （batch_size 32  sc_batch_size 8）
#           从image加载  num_workers=2   batch_size=32     xe：68 min/epoch       SC：31  min/epoch   （batch_size 32  sc_batch_size 16）
#           从image加载  num_workers=2   batch_size=32     xe：68 min/epoch       SC：24  min/epoch   （batch_size 32  sc_batch_size 32）
#           从image加载  num_workers=2   batch_size=40     xe：67 min/epoch
#           从image加载  num_workers=2   batch_size=50     xe：66 min/epoch



# Experiment 2
# freeze the backbone and detector but the images are fed forward into the model every iteration you finetune the model
# 冻结backbone和detector，【xe训练】并【sc微调】 grid_net 和 cap_generator，从图片加载并使用image aug。
# cd models/ops/ && python setup.py build develop && cd /gemini/code/grit

# 两卡可跑cnn可跑，resnet15不可
export DATA_ROOT=${GEMINI_DATA_IN1}/COCO2014    # 数据集
export DATA_ROOT2=${GEMINI_DATA_IN2}            
export OUTPUT_ROOT=${GEMINI_DATA_OUT}           # 输出集路径
export MODEL_ROOT1=${GEMINI_PRETRAIN}           # 预训练模型
export MODEL_ROOT2=${GEMINI_PRETRAIN2}

python train_caption.py exp.name=FPE_no_reg_gamma1_stre1 \
    model.detector.checkpoint=${MODEL_ROOT1}/detector_checkpoint_4ds.pth \
    optimizer.freezing_xe_epochs=1 \
    optimizer.freezing_sc_epochs=1 \
    optimizer.finetune_xe_epochs=0 \
    optimizer.finetune_sc_epochs=0 \
    optimizer.freeze_backbone=True \
    optimizer.freeze_detector=True \
    optimizer.batch_size=32 \
    optimizer.num_workers=2\
    exp.ngpus_per_node=1 \
    exp.world_size=1 \
    model.cap_generator.decoder_name=sequential \
    dataset.overfit=True \
    model.freq_net.visual_type="CNN" \
    model.freq_net.gamma=1 \
    model.freq_net.perturbation_strength=0.1 \
    model.use_freq_feat=True \
    model.freq_net.visualization=False

# 参数已调试好等着跑整个离线实验，离线训练记得改deformer-detr路径
# dataset.overfit=True 表示小数据调试
# cap_generator.decoder_name: Parallel  # 'Concatenated_Sequential', 'Parallel', 'Sequential', 'Concatenated_Parallel'
# GPU：4卡 每卡显存：80 GB CPU：16核 内存：48 GB 临时存储：100 GB     
#           从image加载  num_workers=16  batch_size=64     cpu不够
#           从image加载  num_workers=10  batch_size=64     cpu不够 
#           从image加载  num_workers=6   batch_size=64     cpu不够 
#           从image加载  num_workers=2   batch_size=64     cpu 1600%            55 min/epoch       SC 
# GPU：4卡 每卡显存： 80 GB CPU：32核 内存：192 GB 临时存储：100 GB
#           从image加载  num_workers=20  batch_size=64     cpu 2000%            55 min/epoch       SC 
# GPU：8卡 每卡显存：80 GB  CPU：32核 内存：96 GB  临时存储：100 GB
#           从image加载  num_workers=20  batch_size=64     cpu 3200%，内存不够
#           从image加载  num_workers=10  batch_size=64     cpu 2000%  内存不够
#           从image加载  num_workers=2   batch_size=64     cpu 2800%  内存93.5   27 min/epoch       SC 20 min/epoch