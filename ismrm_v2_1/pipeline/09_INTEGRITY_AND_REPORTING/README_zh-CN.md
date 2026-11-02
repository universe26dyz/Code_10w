# 09 完整性与汇总

最后一步只读检查并生成 summary。保存代码 commit、环境版本、protocol/model SHA、source-observation SHA、每个 subject/case 的完成状态、metrics manifest 和迁移 SHA。任何重跑必须写入新的 run root。

`run_stage.sh`：顺序运行项目结构、legacy path 和 runtime contract 三个检查。

```bash
python "$PROJECT_ROOT/scripts/verify_project_integrity.py" "$PROJECT_ROOT"
python "$PROJECT_ROOT/scripts/verify_runtime_contract.py" --root "$PROJECT_ROOT"
python "$PROJECT_ROOT/scripts/verify_no_legacy_code_path.py" "$PROJECT_ROOT"
```

不要把病人数据、模型或 result tree commit 到 GitHub。summary CSV/JSON 只引用 manifest 中的路径和 SHA。
