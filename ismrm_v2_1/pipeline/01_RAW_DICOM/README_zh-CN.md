# 01 原始 DICOM

## 作用与顺序

这是第一个模块，只负责确认三种 stack 的原始输入和部署清单；不修改 DICOM。必须先于 MATLAB 预处理执行。

## 文件清单

`run_stage.sh`：生成单 subject manifest，并检查三种 stack 的 DICOM 数量。

## 输入与输出

- 输入：`sax/`、`2ch/`、`4ch/` 三个服务器 DICOM 目录；每个 group 必须恰好十个 weight。
- 输出：`configs/<SUBJECT>_deployment_manifest.json` 以及 DICOM 数量检查日志。
- 运行位置：服务器。原始 DICOM 不需要上传到 GitHub。

## 操作

```bash
PROJECT_ROOT=/path/to/ismrm_v2_1
SUBJECT=HHZ
RAW_ROOT=/data/.../mapping_subject_dicom
python "$PROJECT_ROOT/common/scripts/extract_subject_deployment_manifest.py" \
  --source "$PROJECT_ROOT/configs/subject_stack_manifest.example.json" \
  --subject "$SUBJECT" --output "$PROJECT_ROOT/subjects/$SUBJECT/manifest.json"
for s in sax 2ch 4ch; do
  find "$RAW_ROOT/$SUBJECT/$s" -maxdepth 1 -name '*.dcm' | wc -l
done
```

清单中的 `local_dicom_dir` 和 `server_dicom_dir` 必须解析为同一个服务器路径。不要用 NIfTI 代替 DICOM，因为 UID、采集时间和空间几何会在后续使用。

## 文件说明与故障排除

`extract_subject_deployment_manifest.py` 只抽取一个 subject；发现 0 个或多个匹配项时应停止。数量不是 10 的整数倍、路径不存在或 stack 缺失时不要进入 02。
