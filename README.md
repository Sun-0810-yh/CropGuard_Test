# CropGuard 农智云 · 自动化测试套件

CropGuard（27 类农业害虫检测系统）的**测试与评测套件**，从主仓库抽出、可独立克隆运行。

- **零第三方测试框架**：`tests/run_tests.py` 是纯标准库的迷你框架（现场机器装不上 pytest 也能跑），
  同时**兼容 pytest**（`pytest tests`）。
- **60 个用例，约 40 秒**；不需要 GPU、不需要摄像头、不需要模型权重。
- 相机用例用**合成图片序列**当视频源，没有硬件也能验证采集层的全部逻辑。
- 每个用例都是**真回归**：钉住的是实际踩过的坑（见下方"回归清单"）。

## 快速开始

```bash
git clone https://github.com/Sun-0810-yh/CropGuard_Test.git
cd CropGuard_Test
pip install numpy opencv-python requests PyYAML
python tests/run_tests.py
```

期望输出：

```
=== 60 passed, 0 failed ===
```

> 依赖 `ultralytics` / `PyQt5` 的用例（`test_gui_adapter_has_full_interface`，它要 import 主项目的
> `main.py`）在缺依赖时会显示 `SKIP` 而不是失败：
>
> ```
> === 59 passed, 1 skipped, 0 failed ===     # 退出码仍为 0
> ```
>
> 想吃满 60 项：`pip install ultralytics PyQt5`。

只跑部分模块（按文件名匹配）：

```bash
python tests/run_tests.py retriever guard     # 只跑名字含 retriever / guard 的模块
python tests/run_tests.py intent              # 只跑指令安全
```

## 用例分布

| 模块 | 用例 | 覆盖内容 |
| --- | --- | --- |
| `tests/test_intent.py` | 19 | 语音指令解析：片区/时长解析、时长安全钳制（100 秒 → 5 秒）、危险词拦截（"持续喷""加大剂量"）、未指定片区拒绝、中英文字母不误匹配 |
| `tests/test_feedback_and_camera.py` | 17 | 语音队列（连续多句都要播、`flush` 最新优先、去重窗口）+ 相机采集层（共享单例引用计数、`read()` 非阻塞、打不开时给出可读原因、多策略解析、首次失败后重试） |
| `tests/test_retriever.py` | 16 | TF-IDF 字符 n-gram 检索：库内召回、库外拒答（低于阈值不调模型）、别名映射、上下文过滤、问句改写 |
| `tests/test_guard.py` | 8 | 防幻觉后校验：编造药名/剂量被标注、离题回答拒绝、安全尾注幂等 |

## RAG / 防幻觉评测（答辩举证用）

```bash
python scripts/ask_eval.py            # 只测检索层，几秒出结果
python scripts/ask_eval.py --llm      # 额外真调本地 Ollama（需先 ollama serve）
```

输出四类指标：**库内召回率**、**库外拒答率**、**越权指令拦截率**、（`--llm`）**生成合规率**。
把这份输出打印出来，就是"防幻觉做过验证"最直接的证据。

## 回归清单：这些用例在钉什么

| 坑（真实踩过） | 钉它的用例 |
| --- | --- |
| 相机"日志说打开了、界面永远没画面" —— 适配器漏了 `wait_first_frame()`，`AttributeError` 被 `except` 吞掉 | `test_gui_adapter_has_full_interface` |
| 第一次打开失败（被别的程序占用）后，池里的"死"对象被一直复用 → 之后永远打不开 | `test_open_failure_then_retry_succeeds` |
| 多客户端（PC 页面 + 手机页面）抢同一个摄像头，第二个拿不到设备 | `test_camera_shared_singleton_refcount` |
| 相机设备不在时静默失败、看不出原因 | `test_camera_missing_device_reports_dead` / `test_open_failure_has_actionable_last_error` |
| 语音"只有第一句响" —— `pyttsx3` 的 COM 回调跨线程 + 锁没释放 | `test_tts_speaks_multiple_times` / `test_tts_latest_wins_when_flushed` |
| 大模型编造药名、剂量，或答非所问 | `test_guard.*`（配合召回闸门：低于阈值**不调模型**） |
| 危险语音指令（"一直喷不要停""加大剂量"）被当成正常指令下发硬件 | `test_intent.test_unsafe_words_rejected` |

## 目录

```
tests/
├─ run_tests.py                  ← 运行入口（迷你框架：用例登记 / 夹具注入 / SKIP / 失败回溯）
├─ _registry.py                  ← 用例与夹具注册表（框架内部；runner 是 __main__ 时的共享单例）
├─ conftest.py                   ← 装了 pytest 时也能直接 `pytest tests`
├─ test_intent.py / test_feedback_and_camera.py / test_retriever.py / test_guard.py
scripts/
└─ ask_eval.py                   ← RAG + 防幻觉评测报告
feedback/ llm/ tool/ config/ prompts/ main.py   ← 被测模块（见下）
requirements.txt                 ← 与主仓库一致的完整依赖清单
```

迷你框架的能力刻意做得很少，够用就停：

- `@test_case` / `@test_case('参数')` 登记用例，可叠加（等价 `pytest.mark.parametrize`）；
- `@fixture` / `@fixture(scope='module')` 按参数名注入，**支持生成器夹具做收尾**（等价 teardown）；
- `requires('ultralytics', 'PyQt5')` 缺可选依赖就 `SKIP`；
- `tmp_path` 内置（写在**工作区内**的短路径 —— `%TEMP%` 在受限环境里可能不可写，
  Windows 长路径还会让 OpenCV 的 `VideoCapture` 认不出来）。

## 与主仓库的关系

本仓库是主仓库 [CropGuard](https://github.com/Sun-0810-yh/CropGuard) 的**测试子集**：
保留测试本身 + 它们要 import 的模块（`feedback/ llm/ tool/ config/ prompts/ main.py`），
**不含**服务器前端 `serve/`、模型权重（`*.pt` / `*.onnx`）、数据集与训练产物 —— 因此体积只有几百 KB。

用例只跑真实存在的代码路径：`main.py`、`llm/ai_gateway.py`、`llm/retriever.py`、`llm/guard.py`、
`feedback/tts.py`、`feedback/feedback.py`、`tool/camera_source.py`，都是主仓库里正在用的同一份实现。
