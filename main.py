import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, Text, ttk

import cv2
from PIL import Image, ImageTk
from deepface import DeepFace


# 1. 创建主窗口 - 暗黑极简风格
class FaceRecognitionApp:
    def __init__(self, master):
        self.master = master
        master.title("面部识别系统")
        master.geometry("1000x700")
        master.configure(bg="#121212")
        master.bind("<Escape>", self.stop_stream_analysis)

        # 设置应用图标
        try:
            icon_path = os.path.join(os.path.dirname(__file__), "face_icon.png")
            if os.path.exists(icon_path):
                master.iconphoto(False, tk.PhotoImage(file=icon_path))
        except:
            pass

        # 顶部标题栏
        title_frame = tk.Frame(master, bg="#1e1e1e", height=50)
        title_frame.pack(fill="x")

        title_label = tk.Label(title_frame, text="面部识别分析系统", font=("Helvetica", 18, "bold"), fg="#e0e0e0",
                               bg="#1e1e1e")
        title_label.pack(padx=20)

        # 主布局容器
        main_container = tk.Frame(master, bg="#121212")
        main_container.pack(fill="both", expand=True, padx=20, pady=10)

        # 侧边框 - 功能导航
        self.sidebar = tk.Frame(main_container, width=180, bg="#1e1e1e", bd=0, highlightthickness=0)
        self.sidebar.pack(side="left", fill="y", padx=(0, 10))

        # 添加图标占位符
        icon_placeholder = tk.Label(self.sidebar, text="⚡", font=("Arial", 24), bg="#1e1e1e", fg="#4fc3f7")
        icon_placeholder.pack(pady=20)

        # 功能按钮容器 - 使用Frame实现均匀分布
        button_container = tk.Frame(self.sidebar, bg="#1e1e1e", padx=10, pady=10)
        button_container.pack(fill="both", expand=True)

        # 按钮样式
        button_style = {"font": ("Segoe UI", 12, "bold"), "bg": "#4fc3f7", "fg": "#000000",
            "activebackground": "#29b6f6", "activeforeground": "#000000", "bd": 0, "highlightthickness": 0, "padx": 15,
            "pady": 12, "anchor": "center", "relief": "flat", "width": 15, "height": 2}

        # 功能按钮 - 均匀分布
        self.button_verify = tk.Button(button_container, text="人脸验证", command=self.show_verify_screen,
                                       **button_style)
        self.button_verify.pack(padx=20, pady=20, fill="x")

        self.button_find = tk.Button(button_container, text="人脸识别", command=self.show_find_screen, **button_style)
        self.button_find.pack(padx=20, pady=20, fill="x")

        self.button_analyze = tk.Button(button_container, text="面部属性分析", command=self.show_analyze_screen,
                                        **button_style)
        self.button_analyze.pack(padx=20, pady=20, fill="x")

        # 实时分析按钮 - 单独保存以便修改
        self.button_stream = tk.Button(button_container, text="实时分析", command=self.toggle_stream_analysis,
                                       **button_style)
        self.button_stream.pack(padx=20, pady=20, fill="x")

        # 添加分隔线
        separator = ttk.Separator(self.sidebar, orient="horizontal")
        separator.pack(fill="x", pady=10)

        # 状态指示器
        self.status_label = tk.Label(self.sidebar, text="就绪", font=("Segoe UI", 9), fg="#9e9e9e", bg="#1e1e1e")
        self.status_label.pack(side="bottom", fill="x", pady=10, padx=10)

        # 主要区域 - 输出和图像显示
        main_area_frame = tk.Frame(main_container, bg="#1e1e1e", bd=0, highlightthickness=0)
        main_area_frame.pack(side="right", fill="both", expand=True)

        # 创建标签页 - 按照要求互换位置
        self.notebook = ttk.Notebook(main_area_frame)
        self.notebook.pack(fill="both", expand=True, padx=0, pady=0)

        # 图像预览标签页 - 放在第一位
        self.image_tab = tk.Frame(self.notebook, bg="#1e1e1e")
        self.notebook.add(self.image_tab, text="图像预览")

        # 图像显示区域
        self.image_label = tk.Label(self.image_tab, bg="#1e1e1e")
        self.image_label.pack(fill="both", expand=True, padx=10, pady=10)

        # 输出标签页 - 放在第二位
        output_tab = tk.Frame(self.notebook, bg="#1e1e1e")
        self.notebook.add(output_tab, text="分析结果")

        # 输出文本框
        self.text_output = Text(output_tab, bg="#252525", fg="#e0e0e0", wrap="word", font=("Segoe UI", 11),
                                insertbackground="#e0e0e0", relief="flat")
        self.text_output.pack(fill="both", expand=True, padx=10, pady=10)

        # 设置按钮悬停效果
        buttons = [self.button_verify, self.button_find, self.button_analyze, self.button_stream]
        for btn in buttons:
            btn.bind("<Enter>", lambda e, b=btn: b.config(bg="#29b6f6"))
            btn.bind("<Leave>", lambda e, b=btn: b.config(bg="#4fc3f7"))

        # 实时分析状态
        self.stream_active = False
        self.stream_thread = None
        self.capture = None

    def update_status(self, message):
        """更新状态栏"""
        self.status_label.config(text=message)
        self.master.update_idletasks()

    # 2. 人脸验证
    def show_verify_screen(self):
        self.update_status("请选择两张图片进行人脸验证...")
        self.clear_output()

        img1_path = filedialog.askopenfilename(title="选择第一个图像", filetypes=[("图片文件", "*.jpg *.jpeg *.png")])

        # 如果用户取消选择，直接返回
        if not img1_path:
            self.update_status("取消选择图像1")
            return

        self.display_image(img1_path)
        self.update_status(f"已选择图像1: {os.path.basename(img1_path)}")

        img2_path = filedialog.askopenfilename(title="选择第二个图像", filetypes=[("图片文件", "*.jpg *.jpeg *.png")])

        # 如果用户取消选择，直接返回
        if not img2_path:
            self.update_status("取消选择图像2")
            return

        self.display_image(img2_path)
        self.update_status(f"已选择图像2: {os.path.basename(img2_path)}")

        if img1_path and img2_path:
            self.update_status("正在进行人脸验证...")
            try:
                result = DeepFace.verify(img1_path, img2_path, enforce_detection=False, detector_backend='retinaface')
                self.notebook.select(1)  # 切换到分析结果标签页
                verified = "匹配" if result['verified'] else "不匹配"
                distance = result['distance']
                threshold = result['threshold']

                self.text_output.insert(tk.END, "===== 人脸验证结果 =====\n", "title")
                self.text_output.insert(tk.END, f"验证结果: {verified}\n")
                self.text_output.insert(tk.END, f"相似度: {1 - distance:.4f}\n")
                self.text_output.insert(tk.END, f"距离值: {distance:.4f} (阈值: {threshold:.4f})\n")
                self.text_output.insert(tk.END, f"模型: {result['model']}\n")

                # 设置结果颜色
                if result['verified']:
                    self.text_output.tag_add("success", "2.0", "2.end")
                    self.text_output.tag_config("success", foreground="#4caf50")
                else:
                    self.text_output.tag_add("error", "2.0", "2.end")
                    self.text_output.tag_config("error", foreground="#f44336")

                self.update_status("人脸验证完成")
            except Exception as e:
                messagebox.showerror("验证错误", f"人脸验证失败: {str(e)}")
                self.text_output.insert(tk.END, f"错误详情: {str(e)}\n")
                self.update_status("验证出错")

    # 3. 人脸识别
    def show_find_screen(self):
        self.update_status("请选择待识别人脸图像和数据库文件夹...")
        self.clear_output()

        img_path = filedialog.askopenfilename(title="选择待识别人脸图像",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png")])

        # 如果用户取消选择，直接返回
        if not img_path:
            self.update_status("取消选择图像")
            return

        self.display_image(img_path)
        self.update_status(f"已选择图像: {os.path.basename(img_path)}")

        db_path = filedialog.askdirectory(title="选择数据库文件夹")

        # 如果用户取消选择，直接返回
        if not db_path:
            self.update_status("取消选择数据库")
            return

        if img_path and db_path:
            self.update_status(f"正在识别人脸，数据库: {db_path}...")
            try:
                dfs = DeepFace.find(img_path=img_path, db_path=db_path, enforce_detection=False,
                                    detector_backend='retinaface')
                self.notebook.select(1)  # 切换到分析结果标签页
                self.text_output.insert(tk.END, "===== 人脸识别结果 =====\n", "title")

                if not dfs or dfs[0].empty:
                    self.text_output.insert(tk.END, "未在数据库中找到匹配的人脸\n")
                    self.update_status("未找到匹配")
                    return

                for df in dfs:
                    # 只显示前5个匹配结果
                    top_results = df.head(5)

                    for idx, row in top_results.iterrows():
                        identity = os.path.basename(os.path.dirname(row['identity']))
                        distance = row['distance']
                        similarity = (1 - distance) * 100

                        self.text_output.insert(tk.END, f"\n身份: {identity}\n")
                        self.text_output.insert(tk.END, f"相似度: {similarity:.2f}%\n")
                        self.text_output.insert(tk.END, f"文件路径: {row['identity']}\n")
                        self.text_output.insert(tk.END, "-" * 50 + "\n")

                self.update_status(f"找到 {len(dfs[0])} 个匹配结果")
            except Exception as e:
                messagebox.showerror("识别错误", f"人脸识别失败: {str(e)}")
                self.text_output.insert(tk.END, f"错误详情: {str(e)}\n")
                self.update_status("识别出错")

    # 4. 面部属性分析
    def show_analyze_screen(self):
        self.update_status("请选择待分析图像...")
        self.clear_output()

        img_path = filedialog.askopenfilename(title="选择待分析图像", filetypes=[("图片文件", "*.jpg *.jpeg *.png")])

        # 如果用户取消选择，直接返回
        if not img_path:
            self.update_status("取消选择图像")
            return

        self.display_image(img_path)
        self.update_status(f"正在分析: {os.path.basename(img_path)}...")
        try:
            objs = DeepFace.analyze(img_path, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False,
                                    detector_backend='retinaface')
            self.notebook.select(1)  # 切换到分析结果标签页
            for obj in objs:
                # 格式化性别输出
                gender_data = obj['gender']
                man_prob = gender_data.get('Man', 0)
                woman_prob = gender_data.get('Woman', 0)

                if man_prob > 90 or woman_prob > 90:
                    gender_str = "男" if man_prob > woman_prob else "女"
                    gender_color = "#4caf50"  # 绿色表示高置信度
                elif man_prob > 60 or woman_prob > 60:
                    dominant = "男" if man_prob > woman_prob else "女"
                    gender_str = f"大概率{dominant} (男:{man_prob:.1f}%, 女:{woman_prob:.1f}%)"
                    gender_color = "#ff9800"  # 橙色表示中等置信度
                else:
                    gender_str = f"无法判断 (男:{man_prob:.1f}%, 女:{woman_prob:.1f}%)"
                    gender_color = "#f44336"  # 红色表示低置信度

                # 格式化种族输出
                race_data = obj['race']
                # 获取概率最高的种族
                dominant_race = max(race_data, key=race_data.get)

                # 种族名称映射到中文
                race_mapping = {'asian': '亚洲人', 'indian': '印度人', 'black': '黑人', 'white': '白人',
                    'middle eastern': '中东人', 'latino hispanic': '拉丁裔'}

                race_str = race_mapping.get(dominant_race, dominant_race)

                # 格式化情感输出
                emotion_mapping = {'angry': '生气', 'disgust': '厌恶', 'fear': '恐惧', 'happy': '开心', 'sad': '伤心',
                    'surprise': '惊讶', 'neutral': '中性'}

                emotion_str = emotion_mapping.get(obj['dominant_emotion'], obj['dominant_emotion'])

                # 输出格式化结果
                self.text_output.insert(tk.END, "===== 面部分析结果 =====\n", "title")
                self.text_output.insert(tk.END, f"年龄: {obj['age']}\n")
                self.text_output.insert(tk.END, f"性别: {gender_str}\n")
                self.text_output.tag_add("gender", "3.0", "3.end")
                self.text_output.tag_config("gender", foreground=gender_color)

                self.text_output.insert(tk.END, f"种族: {race_str}\n")
                self.text_output.insert(tk.END, f"情感: {emotion_str}\n\n")

                # 添加详细数据
                self.text_output.insert(tk.END, "详细分析数据:\n", "subtitle")
                self.text_output.insert(tk.END, f"  - 性别概率: 男 {man_prob:.1f}%, 女 {woman_prob:.1f}%\n")

                # 种族概率 - 按概率降序排列
                self.text_output.insert(tk.END, "  - 种族概率 (降序):\n")
                # 按概率值降序排序
                sorted_race = sorted(race_data.items(), key=lambda x: x[1], reverse=True)
                for race, prob in sorted_race:
                    chinese_race = race_mapping.get(race, race)
                    self.text_output.insert(tk.END, f"      {chinese_race}: {prob:.1f}%\n")

                # 情感概率 - 按概率降序排列
                self.text_output.insert(tk.END, "  - 情感概率 (降序):\n")
                emotion_data = obj['emotion']
                # 按概率值降序排序
                sorted_emotion = sorted(emotion_data.items(), key=lambda x: x[1], reverse=True)
                for emotion, prob in sorted_emotion:
                    chinese_emotion = emotion_mapping.get(emotion, emotion)
                    self.text_output.insert(tk.END, f"      {chinese_emotion}: {prob:.1f}%\n")

            self.update_status("面部分析完成")
        except Exception as e:
            messagebox.showerror("分析错误", f"面部分析失败: {str(e)}")
            self.text_output.insert(tk.END, f"错误详情: {str(e)}\n")
            self.update_status("分析出错")

    def toggle_stream_analysis(self):
        """切换实时分析状态"""
        if not self.stream_active:
            self.start_stream_analysis()
        else:
            self.stop_stream_analysis()

    def start_stream_analysis(self):
        """开始实时分析"""
        self.update_status("请选择数据库文件夹...")
        self.clear_output()
        self.notebook.select(1)  # 切换到分析结果标签页

        db_path = filedialog.askdirectory(title="选择数据库文件夹")

        # 如果用户取消选择，直接返回
        if not db_path:
            self.update_status("取消选择数据库")
            return

        self.text_output.insert(tk.END, "===== 实时分析 =====\n", "title")
        self.text_output.insert(tk.END, f"数据库路径: {db_path}\n")
        self.text_output.insert(tk.END, "启动实时分析中...\n")
        self.text_output.insert(tk.END, "请查看弹出的摄像头窗口\n\n")
        self.text_output.insert(tk.END, "按ESC键可退出实时分析\n")

        # 更新按钮状态和样式
        self.button_stream.config(text="停止分析", bg="#ff4081",  # 鲜艳的玫瑰红色
            activebackground="#e91e63",  # 更深的玫瑰红
            fg="#ffffff"  # 白色文字更醒目
        )
        # 更新悬停效果
        self.button_stream.unbind("<Enter>")
        self.button_stream.unbind("<Leave>")
        self.button_stream.bind("<Enter>", lambda e: self.button_stream.config(bg="#e91e63"))
        self.button_stream.bind("<Leave>", lambda e: self.button_stream.config(bg="#ff4081"))

        self.update_status("启动实时分析...")
        self.stream_active = True
        self.stream_thread = threading.Thread(target=self.stream_analysis, args=(db_path,), daemon=True)
        self.stream_thread.start()

    def stream_analysis(self, db_path):
        try:
            # 创建VideoCapture对象并保存引用
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                self.update_status("无法打开摄像头")
                self.text_output.insert(tk.END, "错误: 无法访问摄像头\n")
                self.stop_stream_analysis()
                return

            # 设置窗口名称
            cv2.namedWindow('Real-time Face Recognition', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Real-time Face Recognition', 800, 600)

            frame_counter = 0

            while self.stream_active:
                # 检查ESC键
                if cv2.waitKey(1) == 27:  # 27是ESC键的ASCII码
                    self.stop_stream_analysis()
                    break

                # 读取帧
                ret, frame = self.capture.read()
                frame_counter += 1
                if not ret:
                    break
                if frame_counter % 2 == 0:
                    continue
                # 进行人脸识别
                try:
                    results = DeepFace.find(img_path=frame, db_path=db_path, enforce_detection=False, silent=True)
                    if results and not results[0].empty:
                        if len(results) > 1:
                            print(results[0])
                            print(results[1])
                        result = results[0].iloc[0]
                        identity = os.path.basename(os.path.dirname(result['identity']))
                        distance = result['distance']
                        similarity = (1 - distance) * 100
                        # 在图像上显示身份和相似度
                        text = f"{identity}: {similarity:.2f}%"
                        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                except Exception:
                    pass
                cv2.imshow('Real-time Face Recognition', frame)
            self.cleanup_stream()
            self.update_status("实时分析已停止")
        except Exception as e:
            self.update_status(f"实时分析出错: {str(e)}")
            self.text_output.insert(tk.END, f"错误详情: {str(e)}\n")
            self.cleanup_stream()
        finally:
            self.stream_active = False
            self.master.after(0, self.restore_stream_button)

    def cleanup_stream(self):
        """清理实时分析资源"""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        cv2.destroyAllWindows()

    def stop_stream_analysis(self, event=None):
        """停止实时分析"""
        if self.stream_active:
            self.stream_active = False
            self.update_status("正在停止实时分析...")
            self.text_output.insert(tk.END, "实时分析已停止\n")
            self.cleanup_stream()
            self.restore_stream_button()

    def restore_stream_button(self):
        """恢复实时分析按钮的原始状态"""
        self.button_stream.config(text="实时分析", bg="#4fc3f7", activebackground="#29b6f6", fg="#000000")
        # 恢复悬停效果
        self.button_stream.unbind("<Enter>")
        self.button_stream.unbind("<Leave>")
        self.button_stream.bind("<Enter>", lambda e: self.button_stream.config(bg="#29b6f6"))
        self.button_stream.bind("<Leave>", lambda e: self.button_stream.config(bg="#4fc3f7"))

    def display_image(self, img_path):
        """显示选中的图片"""
        try:
            self.notebook.select(0)
            img = Image.open(img_path)
            tab_width = self.image_tab.winfo_width() - 20
            tab_height = self.image_tab.winfo_height() - 20
            # 调整图片大小以适应窗口
            if tab_width > 0 and tab_height > 0:
                img.thumbnail((tab_width, tab_height), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk  # 保持引用
            self.update_status(f"显示图像: {os.path.basename(img_path)}")
        except Exception as e:
            self.update_status(f"无法加载图像: {str(e)}")

    def clear_output(self):
        """清空输出区域"""
        self.text_output.delete(1.0, tk.END)
        # 配置文本样式
        self.text_output.tag_config("title", font=("Segoe UI", 12, "bold"), foreground="#4fc3f7")
        self.text_output.tag_config("subtitle", font=("Segoe UI", 10, "bold"), foreground="#bb86fc")

        # 清空图像预览
        self.image_label.config(image=None)
        self.image_label.image = None


# 6. 启动应用
if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use('clam')
    # 配置暗黑主题颜色
    style.configure('.', background='#121212', foreground='#e0e0e0')
    style.configure('TNotebook', background='#1e1e1e', borderwidth=0)
    style.configure('TNotebook.Tab', background='#2d2d2d', foreground='#e0e0e0', padding=[10, 5], font=('Segoe UI', 10))
    style.map('TNotebook.Tab', background=[('selected', '#1e1e1e')])
    app = FaceRecognitionApp(root)
    root.mainloop()
