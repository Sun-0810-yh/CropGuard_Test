import traceback
from datetime import datetime
import time
import os
import sys
import shutil
import threading
import queue

# CRITICAL: torch must be imported before PyQt5 to avoid DLL conflicts
import cv2
import numpy as np
import torch

from ultralytics import YOLO

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QComboBox, QMessageBox, QFileDialog
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtGui import QColor, QPixmap

from UI import Ui_MainWindow

from tool.parser import get_config
from tool.tools import draw_info, result_info_format, format_data, writexls, writecsv, resize_with_padding, assess_risk
from tool.camera_source import get_source, release_source
from prompts.core.prompt_manager import prompt_manager
from llm import local_llm, knowledge_base
from feedback.feedback import Feedback
from feedback.tts import TTSEngine

import winsound


# ==================== 从配置文件加载全局变量 ====================
# 注意：这些变量原来只在 `if __name__ == '__main__'` 里赋值，
# 导致「被别的模块 import 时」`MyMainWindow()` 一构造就 NameError
# （PyCharm 直接跑 main.py 恰好绕过）。现在统一在这里读，两条路径都能用。
config = get_config('./config/configs.yaml')
_cfg_ui = config.get('UI', {}) or {}

# UI配置
title = _cfg_ui.get('title', '农作物虫害检测识别系统')
label_title = _cfg_ui.get('label_title', '农作物虫害检测识别系统')
background = _cfg_ui.get('background', './icon/background.jpg')
zhutu2 = _cfg_ui.get('zhutu2', './icon/zhutu2.png')
background_img = _cfg_ui.get('background', background)     # 兼容旧变量名
padvalue = _cfg_ui.get('padvalue', [167, 200, 171])
column_widths = _cfg_ui.get('column_widths', [50, 220, 120, 200, 80, 80, 140])
column_color = _cfg_ui.get('column_color', [0, 0, 0])
label_control_color = _cfg_ui.get('label_control_color', '255,255,255, 80')
label_img_color = _cfg_ui.get('label_img_color', '255,255,255, 80')
_table_style = _cfg_ui.get('table_widget_info_styles', {}) or {}
header_background_color = _table_style.get('header_background_color', '48,77,49')
header_color = _table_style.get('header_color', '112,63,46')

# 摄像头配置
camera_num = config.get('CONFIG', {}).get('camera_num', 0)
# 中文名称映射
chinese_name = config.get('CONFIG', {}).get('chinese_name', {})

# ==================== AI配置 ====================
# AI模型显示名称映射
AI_MODEL_DISPLAY_NAMES = {
    'deepseek': 'DeepSeek',
    'qwen': '通义千问',
    'openai': 'OpenAI',
    'zhipu': '智谱AI',
    'qianfan': '百度千帆',
    'doubao': '豆包大模型',
    'custom': '自定义模型',
}

# 加载AI模型配置
AI_CONFIG = config.get('AI', {})
active_model = AI_CONFIG.get('active_model', 'openai')
ai_timeout = AI_CONFIG.get('timeout', 60)
ai_max_tokens = AI_CONFIG.get('max_tokens', 2000)
ai_temperature = AI_CONFIG.get('temperature', 0.7)

# 根据active_model获取当前模型配置
model_configs = AI_CONFIG.get('models', {})
current_model_config = model_configs.get(active_model, {})


# ==================== 摄像头 / 性能参数 ====================
# 摄像头采集与推理全部放到后台线程，Qt 主线程只负责显示，避免「跑一会就卡死」
_cam_cfg = config.get('CAMERA', {}) if config else {}
CAMERA_WIDTH = int(_cam_cfg.get('width', 1280) or 1280)    # 0=使用驱动默认
CAMERA_HEIGHT = int(_cam_cfg.get('height', 720) or 720)    # 0=使用驱动默认
# 强制 MJPG：很多 USB 摄像头默认走 YUY2，640x480 下只有 5~10fps，切 MJPG 才能上 30fps
CAMERA_FOURCC = _cam_cfg.get('fourcc', 'MJPG') or None     # 设为 None 则不强制
CAMERA_BACKEND = str(_cam_cfg.get('backend', 'auto') or 'auto')
CAMERA_STALE_SEC = float(_cam_cfg.get('stale_sec', 2.0) or 2.0)
DISPLAY_INTERVAL_MS = 30    # 画面刷新定时器间隔(ms)，越小越跟手
MAX_TABLE_ROWS = 500        # 表格最大行数，防止长时间运行内存/绘制开销无限增长
CAMERA_LOG_INTERVAL = 1.0   # 摄像头/视频模式每隔多少秒才往表格记一行
# 相机探测（打不开时自动找可用索引；放后台线程，绝不阻塞界面）
CAMERA_PROBE_INDICES = 3    # 扫描 0 ~ N-1 号
CAMERA_PROBE_TIMEOUT = 1.2  # 每个索引打开超时（秒）；试 3 个后端总共最多约 10 秒
CAMERA_BACKENDS_TO_TRY = ('dshow', 'msmf', 'default')
# 打开设备后等第一帧的最长时间（秒）。实测 USB 相机冷启动要 ~3 秒才出图，
# 所以这里给到 8 秒；等的是后台采集线程，界面不会卡（只是按钮文字变化稍晚）
CAMERA_FIRST_FRAME_WAIT = float(_cam_cfg.get('probe_timeout', 8.0) or 8.0)


# ==================== 模型 / 推理参数（模块级，import 时也可用） ====================
# `python main.py` 会在入口块里真正加载模型（见文件末尾），
# 这里先把参数与占位符定义好，避免"被 import 时" MyMainWindow 一构造就 NameError。
_cfg_model = config.get('MODEL', {}) or {}
weights_path = _cfg_model.get('WEIGHT', './runs/train/pest27_final4/weights/best.pt')
conf_thres = float(_cfg_model.get('CONF', 0.4) or 0.4)
imgsz = int(_cfg_model.get('IMGSIZE', 640) or 640)
device = _cfg_model.get('DEVICE', 'cuda:0')
_cfg_classes = _cfg_model.get('CLASSES', None)
try:
    classes = eval(_cfg_classes) if isinstance(_cfg_classes, str) else _cfg_classes
except Exception:
    classes = None
if str(device).startswith('cuda') and not torch.cuda.is_available():
    print('警告: CUDA 不可用，自动回退到 CPU 推理')
    device = 'cpu'
# 真正的模型对象在 __main__ 块里赋值（`global yolo`）；import 场景下保持 None
yolo = None


# ==================== 相机可用性探测（后台线程） ====================
def probe_available_cameras(max_index=CAMERA_PROBE_INDICES, timeout=CAMERA_PROBE_TIMEOUT):
    """扫一遍 0..max_index-1，返回"真的能出图"的索引列表。

    为什么要它：现场多数"打开摄像头就卡住"其实是**根本没有可用相机**
    （没插 / 被禁用 / 没驱动）。这里给出可执行结论，而不是让用户猜。
    每个索引都在**子进程**里探测并硬超时，驱动层阻塞也拖不垮界面。
    """
    import subprocess
    import sys as _sys
    code = (
        "import sys, cv2, time\n"
        "idx=int(sys.argv[1]); be=sys.argv[2]\n"
        "flag={'dshow':cv2.CAP_DSHOW,'msmf':cv2.CAP_MSMF,'default':None}[be]\n"
        "try:\n"
        "    cap=cv2.VideoCapture(idx, flag) if flag is not None else cv2.VideoCapture(idx)\n"
        "except Exception:\n"
        "    print('fail'); sys.exit(0)\n"
        "if not cap.isOpened():\n"
        "    print('fail'); sys.exit(0)\n"
        "ok=False\n"
        "for _ in range(5):\n"
        "    r,f=cap.read()\n"
        "    if r and f is not None:\n"
        "        ok=True; break\n"
        "    time.sleep(0.05)\n"
        "cap.release()\n"
        "print('ok' if ok else 'fail')\n"
    )
    found = []
    for idx in range(int(max_index)):
        for backend in CAMERA_BACKENDS_TO_TRY:
            try:
                proc = subprocess.run([_sys.executable, '-c', code, str(idx), backend],
                                      capture_output=True, text=True, timeout=timeout)
            except Exception:
                continue
            if 'ok' in (proc.stdout or ''):
                found.append(idx)
                break
    return found


def _camera_probe_worker(out_queue, max_index=CAMERA_PROBE_INDICES):
    """后台探测线程：结果丢进队列，由 Qt 定时器取（避免跨线程碰控件）。"""
    try:
        out_queue.put(probe_available_cameras(max_index))
    except Exception:
        out_queue.put([])


class LocalAdviceWorker(QThread):
    """本地大模型建议生成线程：优先本地 Ollama，失败静默由知识库兜底。"""
    success = pyqtSignal(str)

    def __init__(self, results, pest_text, risk_info, chinese_name):
        super().__init__()
        self.results = results
        self.pest_text = pest_text
        self.risk_info = risk_info
        self.chinese_name = chinese_name

    def run(self):
        try:
            system_prompt = prompt_manager.get_prompt('deepseek')
            user_prompt = (
                f"检测到的病虫害：{self.pest_text}{self.risk_info}\n\n"
                f"请生成简洁的诊断结论与综合防治要点（300字以内），"
                f"分点列出农业、物理、生物、化学防治措施。"
            )
            advice = local_llm.generate(system_prompt, user_prompt)
            self.success.emit(advice)
        except Exception:
            # 本地大模型不可用，静默保留知识库兜底版本
            pass


# ==================== 大模型问答线程（RAG + 防幻觉） ====================
class ChatWorker(QThread):
    """自由问答线程：调用 llm.ai_gateway（检索 → 接地提示词 → 本地 Ollama → 后校验）。

    与 LocalAdviceWorker 同一写法：网络/推理都在工作线程里跑，
    Qt 主线程只更新控件，避免 7B 生成时界面卡死。
    """
    done = pyqtSignal(dict)

    def __init__(self, question, history=None, context=None):
        super().__init__()
        self.question = question
        self.history = history or []
        self.context = context or None

    def run(self):
        try:
            from llm import ai_gateway
            res = ai_gateway.answer(self.question, history=self.history,
                                    context=self.context)
        except Exception as exc:
            res = {'success': False, 'source': 'error',
                   'answer': f'问答失败：{type(exc).__name__}: {str(exc)[:160]}',
                   'citations': [], 'guard': {}}
        self.done.emit(res)


# ==================== 摄像头采集 ====================
class CameraCapture:
    """摄像头采集适配器（GUI 侧）。

    真正的采集/看门狗/自愈已移到 tool/camera_source.py，
    这里只保留原来的接口（start/read/stop），避免 main.py 大面积改动。
    好处：GUI 与 api_server.py 共享同一套设备管理逻辑，不再两套实现各自卡死。
    """

    def __init__(self, index=0, width=0, height=0, fourcc=None, key=None):
        self.index = index
        # key 与 api_server 保持一致，同一相机全程只打开一次
        self.key = key if key is not None else ('cam', index)
        self._src = get_source(
            self.key, source=index,
            width=width or 0, height=height or 0, fourcc=fourcc,
            backend=CAMERA_BACKEND, stale_sec=CAMERA_STALE_SEC,
            name=f'相机{index}')

    def start(self):
        """确保设备已打开；返回是否"已打开"。

        语义与 CameraSource.start() 一致：**只保证设备打开成功**，
        首帧由后台采集线程继续等，调用方用 `wait_first_frame()` 显式等。

        注意：`get_source()` 在构造时已经尝试打开过一次，所以这里先看状态，
        已打开就直接返回 True（避免重复 open 抢设备、也避免重复起线程）。
        """
        if self._src is None:
            return False
        st = self._src.state()
        if st.get('opened_once') or st.get('opened'):
            return True
        return self._src.start()

    def wait_first_frame(self, timeout=None):
        """等首帧；返回是否拿到。实测 USB 相机冷启动要 ~3 秒，别用太小的超时。"""
        return self._src.wait_first_frame(timeout=timeout)

    def read(self):
        """非阻塞返回 (ok, 最新一帧副本)。"""
        return self._src.read()

    def peek(self):
        """零拷贝取最新帧引用（只读）。"""
        return self._src.peek()

    def state(self):
        return self._src.state()

    def stop(self):
        """释放本类引用；计数归零时才真正关设备。"""
        if self._src is not None:
            release_source(self.key)
            self._src = None


# ==================== 后台推理线程 ====================
class InferenceWorker(QThread):
    """在后台线程执行 YOLO 推理，Qt 主线程只刷新界面。

    单槽位 + 自动丢帧：推理跟不上采集时旧帧直接丢弃，永远只处理最新一帧，
    所以画面延迟不会随时间累积，界面也不会被推理阻塞。
    """
    result_ready = pyqtSignal(object, object, object, object, object)

    def __init__(self, infer_fn):
        super().__init__()
        self._infer = infer_fn
        self._cond = threading.Condition()
        self._pending = None
        self._running = True

    def submit(self, frame, img_name):
        with self._cond:
            self._pending = (frame, img_name)  # 只保留最新，旧的自动丢弃
            self._cond.notify()

    def clear(self):
        with self._cond:
            self._pending = None

    def run(self):
        while True:
            with self._cond:
                while self._running and self._pending is None:
                    self._cond.wait(0.1)
                if not self._running:
                    return
                frame, img_name = self._pending
                self._pending = None
            try:
                results, result_info, consum, input_time = self._infer(frame)
            except Exception:
                traceback.print_exc()
                continue
            self.result_ready.emit(results, result_info, consum, input_time, img_name)

    def stop(self):
        with self._cond:
            self._running = False
            self._pending = None
            self._cond.notify_all()
        self.wait(2000)


# ==================== 主窗口类 ====================
class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, cfg=None):
        super().__init__()
        self.frame_number = 0
        self.comboBox_index = None
        self.results = []
        self.result_img_name = None
        self.setupUi(self)

        # 根据config配置文件更新界面配置
        self.init_UI_config()
        self.start_type = None
        self.img = None
        self.img_path = None
        self.img_show = None
        self.img_name = None
        self.result_session_dir = None
        self.result_img_path = None
        self.result_txt = None
        self.output_dir = './output'

        # 默认选择为所有目标
        self.comboBox_value = '所有目标'
        self.comboBox_text = '所有目标'

        # 中文名映射
        self.chinese_name = chinese_name

        # 项目根路径
        self.ProjectPath = os.getcwd()

        self.number = 1
        self.RowLength = 0
        self.consum_time = 0
        self.input_time = 0

        # 打开图片
        self.pushButton_img.clicked.connect(self.open_img)

        # 异步建议线程w
        self.advice_thread = None
        # 打开文件夹
        self.pushButton_dir.clicked.connect(self.open_dir)
        # 打开视频
        self.pushButton_video.clicked.connect(self.open_video)
        # 打开摄像头
        self.pushButton_camera.clicked.connect(self.open_camera)
        # 拍照识别（手持人工确认：拍一张 → 立刻检测）
        if hasattr(self, 'pushButton_snap'):
            self.pushButton_snap.clicked.connect(self.snap_and_detect)
        # 关闭摄像头
        if hasattr(self, 'pushButton_cam_close'):
            self.pushButton_cam_close.clicked.connect(self.close_camera)
        # 绑定开始运行
        self.pushButton_start.clicked.connect(self.start)
        # 导出数据
        self.pushButton_export.clicked.connect(self.write_files)

        # 保存结果
        self.pushButton_save.clicked.connect(self.save_current_result)

        # 获取防治建议
        self.pushButton_advice.clicked.connect(self.get_advice)

        self.comboBox.activated.connect(self.onComboBoxActivated)
        self.comboBox.mousePressEvent = self.handle_mouse_press

        # 大模型自由问答（RAG + 防幻觉）：与网页/手机走同一个 llm.ai_gateway
        self.chat_thread = None
        self.chat_history = []
        if hasattr(self, 'pushButton_chat'):
            self.pushButton_chat.clicked.connect(self.ask_chat)
            self.lineEdit_chat.returnPressed.connect(self.ask_chat)

        # 表格点击事件绑定
        self.tableWidget_info.cellClicked.connect(self.cell_clicked)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)

        # 摄像头采集线程 / 后台推理线程
        self.camera = None
        self.worker = None
        self.awaiting_result = False
        self._last_camera_log = 0.0

        self.image_files = []
        self.current_index = 0
        self.is_running = False
        self.sign = False

        # 信号反馈：语音播报 + 可选 Arduino 声光报警（离线）
        # 语音引擎 = feedback/tts.py（专用线程 + 多后端兜底），
        # 解决了旧版"只有第一次检测有语音、后面全哑"的线程/COM 问题。
        _fb = config.get('FEEDBACK', {}) if config else {}
        _tts_cfg = _fb.get('tts', {}) or {}
        self.tts = TTSEngine(
            enabled=bool(_fb.get('enabled', True)) and str(_tts_cfg.get('engine', 'auto')) != 'none',
            engine=str(_tts_cfg.get('engine', 'auto') or 'auto'),
            rate=float(_tts_cfg.get('rate', 180) or 180),
            volume=float(_tts_cfg.get('volume', 1.0) or 1.0),
            dedup_sec=float(_tts_cfg.get('dedup_sec', 0.5) or 0),
        )
        self.feedback = Feedback(
            serial_port=_fb.get('serial_port') or None,
            baud=int(_fb.get('baud', 9600)),
            enabled=bool(_fb.get('enabled', True)),
            speak_interval=float(_fb.get('speak_interval', 3)),
            tts=self.tts,
            announce_no_risk=bool(_fb.get('announce_no_risk', False)),
        )
        _st = self.feedback.status()['tts']
        print(f"[INFO] TTS engine={_st.get('engine')} ready={_st.get('ready')} "
              f"errors={_st.get('init_errors') or []}")

        # 相机可用性探测（后台线程 + 队列，界面永不被它阻塞）
        self._cam_probe_q = queue.Queue()
        self._cam_probe_thread = None
        self._cam_probe_result = None
        self._cam_open_error = ''

    def init_UI_config(self):
        """
        根据config.yaml中的配置，更新界面
        """
        self.setWindowTitle(title)
        self.label_title.setText(label_title)
        # 背景图用配置里的 UI.background（原来这里引用了不存在的 background_img，
        # 一启动就 NameError 崩溃 —— 已修）。图片缺失时只跳过背景，不影响其他配置。
        try:
            if background and os.path.exists(background):
                self.setStyleSheet("#centralwidget {background-image: url('%s')}" % background)
        except Exception:
            traceback.print_exc()
        self.label_img.setPixmap(QtGui.QPixmap(zhutu2))
        self.label_control.setStyleSheet("background-color: rgba(%s); border-radius: 15px;" % label_control_color)
        self.label_img.setStyleSheet("background-color: rgba(%s); border-radius: 15px;" % label_img_color)
        # 设置 tableWidget 表头的样式
        header_style_sheet = """
                    QHeaderView::section {{
                        background-color: rgb({header_background_color});
                        color: {header_color};
                    }}
                    """.format(
            header_background_color=header_background_color,
            header_color=header_color
        )
        self.tableWidget_info.horizontalHeader().setStyleSheet(header_style_sheet)

    def cell_clicked(self, row, column):
        """
        列表 单元格点击事件
        """
        self.update_comboBox_default()

        result_info = {}
        if self.tableWidget_info.item(row, 1) is None:
            return

        self.img_path = self.tableWidget_info.item(row, 1).text()
        try:
            self.results = eval(self.tableWidget_info.item(row, 3).text())
        except Exception:
            self.results = []
        self.result_img_name = self.tableWidget_info.item(row, 6).text()

        try:
            if self.result_img_name and os.path.exists(self.result_img_name):
                self.img_show = cv2.imdecode(np.fromfile(self.result_img_name, dtype=np.uint8), -1)
            else:
                src_img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                self.img_show = draw_info(src_img, self.results)
        except Exception:
            self.img_show = None

        if len(self.results) > 0:
            box = self.results[0][2]
            score = self.results[0][1]
            cls_name = self.results[0][0]
            result_info = result_info_format({}, box, score, cls_name)
            self.show_info(result_info)

        if self.img_show is not None:
            self._display_image(self.img_show)

    def handle_mouse_press(self, event):
        """鼠标点击下拉列表时更新项目列表"""
        # 更新下拉列表项
        self.comboBox.clear()
        self.comboBox.addItems([self.comboBox_text])
        # 调用默认的mousePressEvent
        QtWidgets.QComboBox.mousePressEvent(self.comboBox, event)

    def onComboBoxActivated(self):
        """
        点击下拉列表
        """
        self.sign = True
        comboBox_text = self.comboBox.currentText()
        self.comboBox_index = self.comboBox.currentIndex()
        result_info = {}

        if len(self.results) == 0:
            print('图片中无目标！')
            QMessageBox.information(self, "信息", "图片中无目标", QMessageBox.Yes)
            return

        if comboBox_text == '所有目标':
            box = self.results[0][2]
            score = self.results[0][1]
            cls_name = self.results[0][0]
            lst_info = self.results
        else:
            select_result = self.results[self.comboBox_index - 1]
            box = select_result[2]
            cls_name = select_result[0]
            score = select_result[1]
            lst_info = [[cls_name, score, box]]

        result_info = result_info_format(result_info, box, score, cls_name)

        self.img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        self.img_show = draw_info(self.img, lst_info)
        self.show_all(self.img_show, result_info)

    def _display_image(self, img):
        """在label_img上显示图像"""
        if img is None:
            return
        # 调整图像大小以适应label
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qt_img = QtGui.QImage(img_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_img)
        # 缩放以适应label（实时画面用 FastTransformation，避免逐帧平滑缩放开销过大）
        scaled_pixmap = pixmap.scaled(
            self.label_img.width(), self.label_img.height(),
            QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation
        )
        self.label_img.setPixmap(scaled_pixmap)

    def show_frame(self, img):
        """显示一帧图像"""
        self._display_image(img)
        if img is not None:
            self.img_show = img

    def open_img(self):
        """打开单张图片"""
        try:
            self.update_comboBox_default()
            self.img_path, filetype = QFileDialog.getOpenFileName(
                None, "选择文件", self.ProjectPath,
                "JPEG Image (*.jpg);;PNG Image (*.png);;JFIF Image (*.jfif);;All Files (*)"
            )
            if self.img_path == "":
                self.start_type = None
                return

            self.img_name = os.path.basename(self.img_path)
            self.label_img_path.setText(" " + self.img_path)
            self.label_dir_path.setText(" 选择图片文件夹")
            self.label_video_path.setText(" 选择视频文件")
            self.label_camera_path.setText(" 打开摄像头")

            self.start_type = 'img'
            self.img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.show_frame(self.img)
        except Exception as e:
            traceback.print_exc()

    def open_dir(self):
        try:
            self.update_comboBox_default()
            self.img_path_dir = QFileDialog.getExistingDirectory(None, "选择文件夹")
            if not self.img_path_dir:
                self.start_type = None
                return

            self.start_type = 'dir'
            self.label_dir_path.setText(" " + self.img_path_dir)
            self.label_img_path.setText(" 选择图片文件")
            self.label_video_path.setText(" 选择视频文件")
            self.label_camera_path.setText(" 打开摄像头")

            self.image_files = [file for file in os.listdir(self.img_path_dir) if file.lower().endswith(
                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))]

            if not self.image_files:
                QMessageBox.information(self, "信息", "文件夹中没有符合条件的图片", QMessageBox.Yes)
                return

            self.current_index = 0
            self.img_path = os.path.join(self.img_path_dir, self.image_files[self.current_index])
            self.img_name = self.image_files[self.current_index]

            self.img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.show_frame(self.img)
        except Exception as e:
            traceback.print_exc()

    def open_video(self):
        try:
            self.update_comboBox_default()
            self.video_path, filetype = QFileDialog.getOpenFileName(None, "选择文件", self.ProjectPath,
                                                                     "mp4 Video (*.mp4);;avi Video (*.avi)")
            if not self.video_path:
                self.start_type = None
                return

            self.start_type = 'video'
            self.label_video_path.setText(" " + self.video_path)
            self.label_img_path.setText(" 选择图片文件")
            self.label_dir_path.setText(" 选择图片文件夹")
            self.label_camera_path.setText(" 打开摄像头")

            self.video_name = os.path.basename(self.video_path)
            self.video = cv2.VideoCapture(self.video_path)
            ret, self.img = self.video.read()
            if ret:
                self.img_name = f"{self.video_name}_frame.jpg"
                self.show_frame(self.img)
        except Exception as e:
            traceback.print_exc()

    def open_camera(self):
        """打开/关闭摄像头。

        ⚠️ 这里**绝不能弹阻塞式对话框、也不能在 UI 线程里长时间等待**：
        现场"打开摄像头就卡死"的经典原因就是——相机其实不可用，
        然后在 UI 线程弹了个模态框（窗口失去响应）或死等首帧。
        所以现在改成：状态写进界面提示 + 后台线程去探测哪个索引可用。
        """
        try:
            self.update_comboBox_default()
            text = self.label_camera_path.text()

            # ---------- 关闭 ----------
            if text == ' 摄像头已打开':
                self.pushButton_start.setText("开始运行 >")
                self.label_camera_path.setText(" 摄像头已关闭")
                self.start_type = None
                self.is_running = False
                self.awaiting_result = False
                self.timer.stop()
                if self.camera is not None:
                    self.camera.stop()
                    self.camera = None
                return

            # ---------- 打开 ----------
            self.start_type = 'camera'
            self.label_img_path.setText(" 选择图片文件")
            self.label_dir_path.setText(" 选择图片文件夹")
            self.label_video_path.setText(" 选择视频文件")
            self.video_name = camera_num
            self._cam_open_error = ''

            # 先释放上一个采集线程，避免设备被重复占用
            if self.camera is not None:
                self.camera.stop()
                self.camera = None

            # 先取缓存里的共享采集源（camera_source 的池），避免重复打开设备
            self.camera = CameraCapture(camera_num, CAMERA_WIDTH, CAMERA_HEIGHT,
                                        CAMERA_FOURCC)

            # start() 只负责"打开设备"；拿到首帧由后台采集线程继续等。
            # 这里等一会儿（默认 8 秒，见 config 的 CAMERA.probe_timeout），
            # 因为实测 USB 相机冷启动可能要 3 秒才吐第一帧。
            if not self.camera.start():
                st = self.camera.state()
                self._cam_open_error = st.get('last_error') or '未知原因'
                self.camera.stop()
                self.camera = None
                self.start_type = None
                self.label_camera_path.setText(" 打开摄像头")
                # 非阻塞提示：写在界面状态栏 + 控制台，不弹模态框
                self.label_class.setText('摄像头不可用')
                self.label_score.setText('--')
                self.label_risk.setText('--')
                hint = (f"⚠ 无法打开摄像头 {camera_num}（{self._cam_open_error}）"
                        f"；正在后台扫描可用索引…")
                self.label_camera_path.setText(" 摄像头不可用")
                print('[CAM] ' + hint, flush=True)
                self.textEdit_advice.setText(
                    hint + '\n\n控制台里能看到逐步日志（每个索引/后端/格式的尝试结果）。\n'
                    '常见原因：\n'
                    '1) 相机没插好 / 没装驱动 / 在设备管理器里被禁用；\n'
                    '2) 被别的程序占用（微信、QQ、浏览器、另一个 Python/PyCharm 窗口）；\n'
                    '3) 索引不对 —— 跑一次「7. 摄像头一键诊断 (camera_check)」看哪个能出图；\n'
                    '4) Windows 设置 → 隐私和安全性 → 相机：允许桌面应用访问相机。\n\n'
                    '（没有相机也能继续用：打开图片 / 文件夹 / 视频 都能检测）')
                self._start_camera_probe()
                return

            # 设备已打开：等首帧（后台线程在读，这里只等它把帧填上）
            got = self.camera.wait_first_frame(timeout=CAMERA_FIRST_FRAME_WAIT)
            if not got:
                st = self.camera.state()
                msg = (f"⚠ 摄像头 {camera_num} 已打开但 {CAMERA_FIRST_FRAME_WAIT:.0f}s 内没出画面"
                       f"（{st.get('last_error') or '无帧'}）；后台仍会继续尝试出帧。")
                print('[CAM] ' + msg, flush=True)
                self.textEdit_advice.setText(msg)
                self.label_camera_path.setText(" 摄像头无画面")
                self._start_camera_probe()
                return

            ok, frame = self.camera.read()
            if ok and frame is not None:
                self.img = frame
                self.img_name = "camera_frame.jpg"
                self.show_frame(frame)
                st = self.camera.state()
                print(f"[CAM] 画面就绪 size={st.get('size')} fps={st.get('fps')} "
                      f"（点「开始运行 >」进入连续实时检测）", flush=True)
            self.label_camera_path.setText(" 摄像头已打开")
        except Exception as exc:
            # 关键：这里以前只打 traceback，界面上什么都不显示 ——
            # 用户就会看到"点了没反应/没画面"。现在把错误写进界面，绝不静默。
            traceback.print_exc()
            msg = (f'\n⚠ 打开摄像头时发生异常：{type(exc).__name__}: {exc}\n'
                   f'（已把详情打到控制台，请把这行发我）')
            try:
                self.textEdit_advice.setText(msg)
                self.label_camera_path.setText(" 摄像头不可用")
            except Exception:
                pass

    # ---------- 拍照识别 / 关闭摄像头（手持人工确认场景） ----------
    def snap_and_detect(self):
        """拍照识别：抓当前一帧 → 立刻检测一次 → 显示带框结果。

        与"实时检测"的区别：实时是持续跑；拍照是**按需拍一张**，
        适合"农户手持、拍一下再决定要不要喷"的人工确认流程。
        没开摄像头时，用当前显示的图片也能跑一次检测（相当于"重检"）。
        """
        try:
            frame = None
            if self.camera is not None:
                ok, frame = self.camera.read()
                if not ok or frame is None:
                    st = self.camera.state()
                    msg = (f'⚠ 摄像头当前没有画面（{st.get("last_error") or "无帧"}）。'
                           f'请检查设备，或改用「打开图片」做检测。')
                    print('[CAM] ' + msg, flush=True)
                    self.textEdit_advice.setText(msg)
                    self._start_camera_probe()
                    return
                self.img = frame
                self.img_name = 'camera_snap.jpg'
            elif self.img is not None:
                frame = self.img
            else:
                self.textEdit_advice.setText(
                    '请先「打开摄像头」或「打开图片」，再点「📷 拍照识别」。')
                return

            self._ensure_worker()
            self.awaiting_result = True
            self.worker.submit(frame, self.img_name or 'camera_snap.jpg')
            self.input_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('[CAM] 已提交一帧做检测（拍照识别）', flush=True)
        except Exception:
            traceback.print_exc()

    def close_camera(self):
        """关闭摄像头：停定时器 + 释放采集源（引用计数归零才真关设备）。"""
        try:
            if self.camera is not None:
                self.camera.stop()
                self.camera = None
            self.timer.stop()
            self.is_running = False
            self.awaiting_result = False
            self.start_type = None
            self.pushButton_start.setText('开始运行 >')
            self.label_camera_path.setText(' 打开摄像头')
            print('[CAM] 摄像头已关闭（设备已释放）', flush=True)
        except Exception:
            traceback.print_exc()

    # ---------- 相机探测（后台） ----------
    def _start_camera_probe(self):
        """后台扫描可用索引；同一时间只跑一个探测线程。"""
        if self._cam_probe_thread is not None and self._cam_probe_thread.is_alive():
            return
        while not self._cam_probe_q.empty():      # 清掉上一次的旧结果
            try:
                self._cam_probe_q.get_nowait()
            except Exception:
                break
        self._cam_probe_result = None
        self._cam_probe_thread = threading.Thread(
            target=_camera_probe_worker, args=(self._cam_probe_q,),
            name='CamProbe', daemon=True)
        self._cam_probe_thread.start()

    def _poll_camera_probe(self):
        """在 UI 定时器里取探测结果（跨线程只通过队列传数据）。"""
        if self._cam_probe_result is not None or self._cam_probe_q.empty():
            return
        try:
            found = self._cam_probe_q.get_nowait()
        except Exception:
            return
        self._cam_probe_result = found or []
        if self._cam_probe_result:
            idx = self._cam_probe_result[0]
            extra = ('，另有：' + '、'.join(str(i) for i in self._cam_probe_result[1:])
                     if len(self._cam_probe_result) > 1 else '')
            msg = (f'🔍 后台扫描发现可用相机索引 {idx}{extra}。'
                   f'请把 config/configs.yaml 的 CONFIG.camera_num 改成 {idx} 后重开摄像头。')
        else:
            msg = ('🔍 后台扫描完成：0~'
                   f'{CAMERA_PROBE_INDICES - 1} 号索引都没有可用相机。'
                   '请检查相机是否插好/驱动是否正常（设备管理器 → 照相机），'
                   '或改用「打开图片/文件夹/视频」做检测。')
        print('[CAM] ' + msg, flush=True)
        self.label_camera_path.setText(" 摄像头不可用")
        self.textEdit_advice.append('\n' + msg)


    def show_all(self, img, info):
        '''
        展示所有的信息
        '''
        self.show_frame(img)
        self.show_info(info)

    def start(self):
        """开始运行检测"""
        self.update_comboBox_default()
        try:
            if self.start_type is None:
                QMessageBox.information(self, "信息", "请先选择输入类型！", QMessageBox.Yes)
                return

            # 如果正在运行，点击按钮 = 停止
            if self.is_running:
                self.stop()
                return

            # 惰性创建后台推理线程
            self._ensure_worker()

            if self.start_type == 'img':
                # 单张图片检测
                self.img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if self.img is None:
                    QMessageBox.information(self, "信息", "图片读取失败！", QMessageBox.Yes)
                    return
                self.is_running = True
                self.pushButton_start.setText('停止检测')
                self.awaiting_result = True
                self.worker.submit(self.img, self.img_name)
                return

            if self.start_type == 'camera' and self.camera is None:
                QMessageBox.information(self, "信息", "请先打开摄像头！", QMessageBox.Yes)
                return

            # 文件夹 / 视频 / 摄像头：启动定时器驱动
            self.is_running = True
            self.pushButton_start.setText('停止检测')
            self.timer.start(DISPLAY_INTERVAL_MS)

        except Exception as e:
            traceback.print_exc()

    def stop(self):
        """停止检测"""
        self.is_running = False
        self.awaiting_result = False
        self.timer.stop()
        self.pushButton_start.setText('开始运行 >')
        if self.worker is not None:
            self.worker.clear()
        # 只释放视频文件句柄；摄像头采集线程要保留，继续排空缓冲防止卡死
        if self.start_type == 'video' and hasattr(self, 'video') and self.video is not None:
            try:
                self.video.release()
            except Exception:
                pass
            self.video = None

    def update_frame(self):
        """定时器回调：摄像头模式刷新画面并投递推理；图片/视频模式顺序处理。"""
        try:
            # 相机探测结果（如果后台在扫）——无论什么模式都要取，才能及时给提示
            self._poll_camera_probe()

            # ---------- 摄像头：画面永远用最新原始帧渲染，推理在后台跑 ----------
            if self.start_type == 'camera':
                if self.camera is None:
                    return
                ok, frame = self.camera.read()
                if not ok or frame is None:
                    # 拿不到帧：不要静默什么都不做，否则用户看到的就是"画面卡住不动"
                    st = self.camera.state()
                    if st.get('state') != 'ok':
                        if not self._cam_open_error:
                            self._cam_open_error = st.get('last_error') or '无帧'
                        self.label_camera_path.setText(" 摄像头无画面")
                        self._start_camera_probe()
                    return

                # 投递最新一帧给推理线程（线程内部自动丢帧，不会积压）
                if self.is_running and self.worker is not None:
                    self.worker.submit(frame, 'camera_frame.jpg')

                # 用「最新原始帧 + 上一次检测框」渲染，预览流畅且不会闪
                display = frame.copy()
                if self.results:
                    draw_info(display, self.results)
                self.img = frame
                self.img_show = display
                self.img_name = 'camera_frame.jpg'
                self.show_frame(display)
                return

            # ---------- 文件夹 / 视频：等上一帧结果回来再读下一帧（背压） ----------
            if not self.is_running or self.awaiting_result:
                return

            if self.start_type == 'dir':
                if not self.image_files:
                    return
                if self.current_index >= len(self.image_files):
                    # 全部检测完毕，自动停止，不再循环
                    self.stop()
                    return

                self.img_name = self.image_files[self.current_index]
                self.img_path = os.path.join(self.img_path_dir, self.img_name)
                self.img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), -1)
                self.current_index += 1
                if self.img is None:
                    return
                self.awaiting_result = True
                self.worker.submit(self.img, self.img_name)
                return

            if self.start_type == 'video':
                ret, frame = self.video.read()
                if not ret:
                    # 视频播完，自动停止，不再循环
                    self.stop()
                    return
                self.img = frame
                self.frame_number += 1
                frame_number = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
                self.img_name = f"{self.video_name}_{frame_number}.jpg"
                self.img_path = self.video_path
                self.awaiting_result = True
                self.worker.submit(frame, self.img_name)
                return
        except Exception:
            traceback.print_exc()

    def _ensure_worker(self):
        """惰性创建后台推理线程（只创建一次）。"""
        if self.worker is None:
            self.worker = InferenceWorker(self.run_inference)
            self.worker.result_ready.connect(self.on_detect_result)
            self.worker.start()

    def run_inference(self, img):
        """纯推理：不接触任何 Qt 控件，可安全地在后台线程调用。"""
        t1 = time.time()

        # 模型识别
        raw = yolo.predict(img, imgsz=imgsz, conf=conf_thres, device=device,
                           classes=classes, verbose=False)
        results = format_data(raw)

        consum_time = str(round(time.time() - t1, 2)) + 's'
        input_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if len(results) > 0:
            box = results[0][2]
            score = results[0][1]
            cls_name = results[0][0]
        else:
            box = [0, 0, 0, 0]
            score = 0
            cls_name = '无目标'

        result_info = result_info_format({}, box, score, cls_name)

        # 病害风险等级评估
        risk_level, risk_color, risk_detail = assess_risk(results, img.shape[:2])
        result_info['risk_level'] = risk_level
        result_info['risk_color'] = risk_color
        result_info['risk_detail'] = risk_detail

        return results, result_info, consum_time, input_time

    def _allow_table_log(self):
        """摄像头/视频模式限制写入表格的频率，避免表格无限膨胀拖垮界面。"""
        if self.start_type in ('camera', 'video'):
            now = time.time()
            if now - self._last_camera_log < CAMERA_LOG_INTERVAL:
                return False
            self._last_camera_log = now
        return True

    def apply_results(self, results, result_info, consum_time, input_time, img_name):
        """在 Qt 主线程中根据推理结果刷新界面。"""
        self.results = results
        self.consum_time = consum_time
        self.input_time = input_time
        if img_name:
            self.img_name = img_name

        self.get_comboBox_value(self.results)
        self.show_info(result_info)

        if self._allow_table_log():
            self.show_table()

        # 触发信号反馈：语音播报 + 声光报警
        _names = []
        for _r in self.results:
            _n = _r[0]
            _names.append(str(self.chinese_name.get(_n, _n)) if isinstance(self.chinese_name, dict) else str(_n))
        if hasattr(self, 'feedback'):
            self.feedback.trigger(result_info.get('risk_level', '无风险'), _names,
                                  result_info.get('risk_detail', ''))

    def on_detect_result(self, results, result_info, consum_time, input_time, img_name):
        """后台推理线程回调（在 Qt 主线程执行）。"""
        self.awaiting_result = False
        self.apply_results(results, result_info, consum_time, input_time, img_name)

        # 图片 / 文件夹 / 视频模式：结果图直接显示
        if self.start_type in ('dir', 'video', 'img'):
            if self.img is not None:
                self.img_show = draw_info(self.img.copy(), results)
                self.show_frame(self.img_show)
            if self.start_type == 'img':
                self.is_running = False
                self.pushButton_start.setText('开始运行 >')

    def closeEvent(self, event):
        """窗口关闭时释放摄像头、推理线程与串口，避免设备被一直占用。"""
        try:
            self.timer.stop()
            if self.camera is not None:
                self.camera.stop()
                self.camera = None
            if self.worker is not None:
                self.worker.stop()
                self.worker = None
            if hasattr(self, 'video') and self.video is not None:
                self.video.release()
                self.video = None
            if hasattr(self, 'feedback'):
                self.feedback.close()
        except Exception:
            traceback.print_exc()
        super().closeEvent(event)

    def get_advice(self):
        """获取防治建议：知识库秒出兜底 + 本地大模型异步增强（离线）"""
        try:
            if not hasattr(self, 'results') or not self.results:
                self.textEdit_advice.setText("请先进行病虫害检测！")
                return

            # 构建检测结果文本
            pest_names = []
            for result in self.results:
                if len(result) >= 3:
                    pest_name = result[0]
                    confidence = result[1]
                    chinese_pest_name = self.chinese_name.get(pest_name, pest_name)
                    pest_names.append(f"{chinese_pest_name}(置信度:{confidence:.2f})")
            pest_text = "、".join(pest_names)

            # 获取风险等级
            risk_info = ""
            risk_level = '中风险'
            risk_detail = ''
            if self.results:
                risk_level, _, risk_detail = assess_risk(self.results)
                risk_info = f"\n病害风险等级：{risk_level}\n风险详情：{risk_detail}"

            # 1) 知识库立即秒出，保证离线稳定
            self.textEdit_advice.setText(
                knowledge_base.get_advice(self.results, risk_level, risk_detail)
            )

            # 2) 本地大模型异步增强，成功后覆盖知识库版本
            self.advice_thread = LocalAdviceWorker(
                self.results, pest_text, risk_info, self.chinese_name)
            self.advice_thread.success.connect(self.on_advice_success)
            self.advice_thread.start()

        except Exception as e:
            self.textEdit_advice.setText(f"获取防治建议出错：\n{str(e)}")

    def get_comboBox_value(self, results):
        '''
        获取当前所有的类别和ID，点击下拉列表时，使用
        '''
        lst = ["所有目标"]
        for bbox in results:
            cls_name = bbox[0]
            lst.append(str(cls_name))
        self.comboBox_value = lst

    def show_info(self, result):
        try:
            if len(result) == 0:
                print("未识别到目标")
                return
            cls_name = result['cls_name']
            if isinstance(self.chinese_name, dict) and cls_name in self.chinese_name:
                cls_name = self.chinese_name[cls_name]
            if len(str(cls_name)) > 10:
                lst_cls_name = str(cls_name).split('_')
                cls_name = lst_cls_name[0][:10] + '...'

            self.label_class.setText(str(cls_name))
            self.label_score.setText(str(result.get('score', '--')))

            # 显示风险等级
            risk_level = result.get('risk_level', '--')
            risk_color = result.get('risk_color', (0, 0, 0))
            self.label_risk.setText(str(risk_level))
            self.label_risk.setStyleSheet(
                f"color: rgb({risk_color[2]},{risk_color[1]},{risk_color[0]}); font-weight: bold;"
            )

            self.label_xmin_v.setText("xmin: " + str(result.get('label_xmin_v', '--')))
            self.label_ymin_v.setText("ymin: " + str(result.get('label_ymin_v', '--')))
            self.label_xmax_v.setText("xmax: " + str(result.get('label_xmax_v', '--')))
            self.label_ymax_v.setText("ymax: " + str(result.get('label_ymax_v', '--')))
        except Exception as e:
            traceback.print_exc()

    def update_comboBox_default(self):
        """
        将下拉列表更新为 所有目标 默认状态
        """
        self.comboBox.clear()
        self.comboBox.addItems([self.comboBox_text])

    def show_table(self):
        try:
            self.number += 1
            self.RowLength = self.RowLength + 1
            self.tableWidget_info.setRowCount(self.RowLength)
            row = self.RowLength - 1
            for column, content in enumerate(
                    [self.number, self.img_path, self.input_time, self.results,
                     len(self.results), self.consum_time, "待保存"]):
                item = QtWidgets.QTableWidgetItem(str(content))
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                item.setForeground(QColor.fromRgb(column_color[0], column_color[1], column_color[2]))
                self.tableWidget_info.setItem(row, column, item)
            # 限制最大行数：摄像头/视频长时间运行不会让表格无限膨胀拖垮界面
            if self.RowLength > MAX_TABLE_ROWS:
                self.tableWidget_info.removeRow(0)
                self.RowLength -= 1
            self.tableWidget_info.scrollToBottom()
        except Exception as e:
            traceback.print_exc()

    def write_files(self):
        """
        导出 excel、csv 数据
        """
        try:
            if self.RowLength == 0:
                QMessageBox.information(self, "信息", "没有数据可导出！", QMessageBox.Yes)
                return

            # 从表格收集所有数据
            all_data = []
            headers = ["编号", "图片路径", "输入时间", "识别结果", "目标数量", "耗时", "保存状态"]
            all_data.append(headers)

            for row in range(self.RowLength):
                row_data = []
                for col in range(7):
                    item = self.tableWidget_info.item(row, col)
                    row_data.append(item.text() if item else "")
                all_data.append(row_data)

            # 保存到output目录
            if not os.path.exists(self.output_dir):
                os.mkdir(self.output_dir)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 导出Excel
            xls_path = os.path.join(self.output_dir, f"detection_results_{timestamp}.xls")
            writexls(all_data, xls_path)

            # 导出CSV
            csv_path = os.path.join(self.output_dir, f"detection_results_{timestamp}.csv")
            writecsv(all_data, csv_path)

            QMessageBox.information(self, "成功",
                                    f"数据已导出到：\n{xls_path}\n{csv_path}",
                                    QMessageBox.Yes)
        except Exception as e:
            traceback.print_exc()
            QMessageBox.warning(self, "错误", f"导出失败：{str(e)}")

    def save_current_result(self):
        """保存：每次点击在 output/类别名/ 生成新的检测结果图与防治方案；无检测不保存"""
        try:
            if self.img_show is None or self.img_name is None or not self.results:
                QMessageBox.information(self, "信息", "未检测到目标，\n未保存。", QMessageBox.Yes)
                return

            if not os.path.exists(self.output_dir):
                os.mkdir(self.output_dir)

            self.result_session_dir = None
            self.result_txt = None
            self.result_img_path = None

            advice_txt = self.textEdit_advice.toPlainText().strip() if hasattr(self, 'textEdit_advice') else ''
            detected_names = []
            try:
                for r in (self.results or []):
                    name = r[0]
                    mapped = self.chinese_name.get(name, name) if hasattr(self, 'chinese_name') else name
                    detected_names.append(str(mapped))
            except Exception:
                pass

            if detected_names:
                seen = set()
                ordered = []
                for n in detected_names:
                    if n not in seen:
                        seen.add(n)
                        ordered.append(n)
                base_name = '+'.join(ordered)
            else:
                QMessageBox.information(self, "信息", "未检测到目标，\n未保存。", QMessageBox.Yes)
                return

            for ch in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
                base_name = base_name.replace(ch, '_')
            if len(base_name) > 60:
                base_name = base_name[:60]

            base_dir_name = base_name
            category_dir = os.path.join(self.output_dir, base_dir_name)
            suffix = 1
            while os.path.exists(category_dir):
                base_dir_name = f"{base_name}({suffix})"
                category_dir = os.path.join(self.output_dir, base_dir_name)
                suffix += 1

            os.mkdir(category_dir)

            # 保存检测结果图片
            img_path = os.path.join(category_dir, f"{base_dir_name}.jpg")
            cv2.imencode('.jpg', self.img_show)[1].tofile(img_path)
            self.result_img_path = img_path

            # 保存防治方案文本
            if advice_txt and '请先进行病虫害检测' not in advice_txt and '未配置' not in advice_txt:
                txt_path = os.path.join(category_dir, f"{base_dir_name}_防治方案.txt")
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(advice_txt)

            QMessageBox.information(self, "成功", f"已保存：\n{base_dir_name}\n检测结果图与防治方案。", QMessageBox.Yes)
        except Exception:
            traceback.print_exc()

    def on_advice_success(self, advice: str):
        """AI建议获取成功回调"""
        self.textEdit_advice.setText(advice)
        if hasattr(self, 'pushButton_advice'):
            self.pushButton_advice.setEnabled(True)

    def on_advice_error(self, msg: str):
        """AI建议获取失败回调"""
        current_model_display = AI_MODEL_DISPLAY_NAMES.get(active_model, active_model)

        error_text = f"""❌ 获取防治建议失败

错误信息：{msg}

💡 可能的解决方案：
1. 检查网络连接是否正常
2. 确认API Key是否有效
3. 尝试重新点击"获取防治建议"按钮
4. 如果问题持续，请检查配置文件中的API设置

🔧 技术支持：
- 当前模型：{current_model_display}
- 超时设置：{ai_timeout}秒
- 最大Token：{ai_max_tokens}"""

        self.textEdit_advice.setText(error_text)
        if hasattr(self, 'pushButton_advice'):
            self.pushButton_advice.setEnabled(True)

    # ==================== 大模型自由问答（RAG + 防幻觉） ====================
    def _current_context(self):
        """把当前检测结果整理成问答上下文（让大模型结合当前虫情回答）。"""
        try:
            if not self.results:
                return None
            dets = []
            for r in self.results:
                name = r[0]
                dets.append({
                    'class_name': name,
                    'chinese_name': self.chinese_name.get(name, name)
                    if isinstance(self.chinese_name, dict) else name,
                    'confidence': float(r[1]) if len(r) > 1 else 0.0,
                })
            risk_level, _, risk_detail = assess_risk(self.results)
            return {'detections': dets,
                    'risk': {'level': risk_level, 'detail': risk_detail}}
        except Exception:
            return None

    def ask_chat(self):
        """提交问题：工作线程里跑 RAG + 本地大模型，界面不卡。"""
        try:
            if not hasattr(self, 'lineEdit_chat'):
                return
            q = self.lineEdit_chat.text().strip()
            if not q:
                self.textEdit_advice.append('请输入问题后再提问。')
                return
            self.lineEdit_chat.clear()
            self.textEdit_advice.append(f'\n你：{q}\n正在检索知识库并调用本地大模型…')
            self._last_chat_question = q

            self.chat_thread = ChatWorker(q, history=self.chat_history,
                                          context=self._current_context())
            self.chat_thread.done.connect(self.on_chat_done)
            self.chat_thread.start()
        except Exception as exc:
            self.textEdit_advice.append(f'提问失败：{exc}')

    def on_chat_done(self, res: dict):
        """问答返回：显示回答 + 依据 + 防幻觉校验结果。"""
        try:
            res = res or {}
            answer = res.get('answer') or '（无回答）'
            src_txt = {'llm': '本地大模型（已接地）',
                       'knowledge_base': '知识库（大模型不可用，原文兜底）',
                       'refused': '拒答（知识库未涵盖）',
                       'error': '错误'}.get(res.get('source', ''), res.get('source', ''))

            guard = res.get('guard') or {}
            tips = []
            if guard.get('top_score') is not None:
                tips.append(f"召回分 {guard.get('top_score')}")
            if guard.get('unsupported'):
                tips.append('已标注可疑药名：' + '、'.join(guard['unsupported']))
            if guard.get('dosage_unsupported'):
                tips.append('已标注可疑剂量')
            if res.get('elapsed') is not None:
                tips.append(f"耗时 {res.get('elapsed')}s")

            self.textEdit_advice.append(f'AI（{src_txt}）：\n{answer}')
            cites = res.get('citations') or []
            if cites:
                self.textEdit_advice.append(
                    '依据：' + '、'.join(c.get('title', c.get('id', '')) for c in cites))
            if tips:
                self.textEdit_advice.append('［' + ' | '.join(tips) + '］')

            q = getattr(self, '_last_chat_question', '')
            if q:
                self.chat_history.append({'role': 'user', 'content': q})
            self.chat_history.append({'role': 'assistant', 'content': answer})
            self.chat_history = self.chat_history[-12:]     # 只留最近若干轮
        except Exception:
            traceback.print_exc()

    def show_model_info(self):
        """显示当前模型信息"""
        if not current_model_config:
            self.textEdit_advice.setText("❌ 未配置AI模型！")
            return

        current_model_display = AI_MODEL_DISPLAY_NAMES.get(active_model, active_model)

        info = f"""当前AI模型配置

模型名称: {current_model_display}
API地址: {current_model_config.get('api_base', 'N/A')}
模型: {current_model_config.get('model', 'N/A')}
SSL验证: {current_model_config.get('verify_ssl', True)}
超时时间: {ai_timeout}秒
最大Token: {ai_max_tokens}
温度: {ai_temperature}

要切换模型，请修改 config/configs.yaml 中的 AI 配置：
- active_model 可选值: deepseek, qwen, openai, zhipu, qianfan, doubao, custom
- 在 models 下配置相应的 api_key 和参数
- 重启程序即可生效"""
        self.textEdit_advice.setText(info)


# ==================== 程序入口 ====================
if __name__ == "__main__":
    # 配置已在文件顶部统一读入全局变量（含 CUDA 不可用时的 CPU 回退）。
    # 这里只做"依赖配置的运行时准备"：权重检查、模型加载预热。
    # 注意：别再把 UI/相机等配置在这里重读一遍 —— 那样 import 本模块时
    # MyMainWindow 会因为拿不到这些全局变量而 NameError（历史 bug，已修）；
    # 也别在这里写 `global yolo`，那会和模块级赋值冲突（SyntaxError）。
    if not os.path.exists(weights_path):
        print(f"警告: 权重文件 {weights_path} 不存在")
        # 兜底权重 yolov8s.pt（COCO 预训练）已在 2026-10-08 清理时删除。
        # 需要它就从 ultralytics 重新下载放到项目根目录；更推荐直接改
        # config/configs.yaml 的 MODEL.WEIGHT 指回 runs/train/pest27_final4/weights/best.pt
        print("      请修改 config/configs.yaml 的 MODEL.WEIGHT 指向现有权重")
    yolo = None
    try:
        yolo = YOLO(weights_path)
        # 模型预热
        yolo.predict(np.zeros((300, 300, 3), dtype='uint8'), device=device)
    except Exception:
        traceback.print_exc()
        QMessageBox.critical(None, '模型加载失败',
                             f'无法加载权重：{weights_path}\n请检查 config/configs.yaml 的 MODEL.WEIGHT。')
        sys.exit(1)

    # 创建QApplication实例
    app = QApplication([])
    # 创建自定义的主窗口对象
    window = MyMainWindow(config)
    # 显示窗口
    window.show()
    # 运行应用程序
    app.exec_()
