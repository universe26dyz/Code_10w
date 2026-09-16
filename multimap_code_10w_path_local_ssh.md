【multimap svr——code_10w】

## codex运行环境为本地
### 本地设置
本地环境：knesvr_torch，含有torch,只有cpu,只能进行smoke和无需gpu的代码检查工作
本地代码路径：/home/universe/SVR/multimap_postprogramming/Code_10w/
本地原始dicom数据路径：/home/universe/SVR/multimap_postprogramming/subject_dicom/
本地实验结果保存路径：/home/universe/SVR/data/Code_10w_v1/ （或者v2\v3，以此类推）

### 服务器设置
服务器环境：cr_dreme
服务器代码路径：/data/dengyz/code/Code_10w
服务器原始dicom数据路径：/data/dengyz/dataset/mapping_subject_dicom
服务器实验结果保存路径：/data/dengyz/dataset/Code_10w_v1（或者v2\v3，以此类推）

### github仓库
git remote set-url origin git@github.com:universe26dyz/Code_10w.git

### 注意
每次在服务器上跑通流程生成结果时，需要同时保存这次实验使用的代码的commit sha，以便后续复现。
