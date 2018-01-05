import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import numpy as np
import tkinter as tk

from math import ceil
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler
from PIL import Image, ImageTk
from random import randrange
from tkinter import filedialog, messagebox

from image import *


class InfoErea(tk.Frame):
    def __init__(self, master, *args, **kwargs):
        super(InfoErea, self).__init__(master, *args, **kwargs)

        tk.Label(self, text="图像尺寸:", font=("PingFang", 12)).grid(row=0, column=0, sticky=tk.W)
        self.img_size = tk.Label(self, text="", font=("PingFang", 12))
        self.img_size.grid(row=1, column=0, sticky=tk.W)
        tk.Label(self, text="像素坐标:", font=("PingFang", 12)).grid(row=2, column=0, sticky=tk.W)
        self.pixel_cord = tk.Label(self, text="", font=("PingFang", 12))
        self.pixel_cord.grid(row=3, column=0, sticky=tk.W)
        tk.Label(self, text="像素值:", font=("PingFang", 12)).grid(row=4, column=0, sticky=tk.W)
        self.pixel_val = tk.Label(self, text="", font=("PingFang", 12))
        self.pixel_val.grid(row=5, column=0, sticky=tk.W)

    def set_img_size(self, info):
        self.img_size.config(text=info)

    def set_pixel_cord(self, info):
        self.pixel_cord.config(text=info)

    def set_pixel_val(self, info):
        self.pixel_val.config(text=info)


class SliderEntry(tk.Frame):
    def __init__(self, master, command, from_=0, to=100, resolution=1.0, init=0):
        super(SliderEntry, self).__init__(master)

        def sync_from_slider(*args):
            self.num.set(str(self.slider.get()))
            command()

        def sync_from_entry(*args):
            try:
                val = float(self.num.get())
            except Exception as e:
                print(e)
                return

            if val > to:
                val = to
            if val < from_:
                val = from_

            self.slider.set(val)
            command()

        self.slider = tk.Scale(self, from_=from_, to=to, orient=tk.HORIZONTAL, showvalue=0, resolution=resolution,
                               command=sync_from_slider)
        self.slider.set(init)
        self.slider.pack(side=tk.LEFT)

        self.num = tk.StringVar()
        self.num.trace("w", sync_from_entry)
        tk.Entry(self, textvariable=self.num, width=4).pack(side=tk.LEFT)

    def get(self):
        return self.slider.get()


class ButtonGroup(tk.Frame):
    def __init__(self, master):
        super(ButtonGroup, self).__init__(master)

    def set_cansel(self, cancel):
        tk.Button(self, text="取消", command=cancel).grid(row=0, column=0)

    def set_preview(self, preview):
        tk.Button(self, text="预览", command=preview).grid(row=0, column=1)

    def set_confirm(self, confirm):
        tk.Button(self, text="确定", command=confirm).grid(row=0, column=2)


class KernelInput(tk.Frame):
    def __init__(self, master):
        super(KernelInput, self).__init__(master)

        tk.Label(self, text="请用逗号(\",\")分隔单行元素\n用换行(\"\\n\")分隔各行\n每行每列元素应为奇数个").pack()
        self.kernel = tk.Text(self, width=30, height=6)
        self.kernel.pack()

    def get(self):
        try:
            kernel = self.kernel.get(1.0, tk.END)
            kernel = [[float(x) for x in line.split(",")] for line in kernel.splitlines()]
        except Exception as e:
            print(e)

        if len(set([len(line) for line in kernel])) > 1:  # line length differs
            return None

        if len(kernel) % 2 != 1 or len(kernel[0]) % 2 != 1:  # not odd
            return None

        return np.array(kernel)


class ImageInput(tk.Frame):
    def __init__(self, master, msg=None):
        super(ImageInput, self).__init__(master)

        def select_img():
            temp = AppData(filedialog.askopenfilename())
            if temp.img_ok():
                temp.to_grey()
                self.target = temp.img

                max_size = 100, 100
                img = ImageTk.PhotoImage(Image.fromarray(temp.get_img(max_size), "RGB"))
                self.temp_img.config(image=img)
                self.temp_img.image = img
            else:
                self.target = None

        self.temp_img = tk.Label(self)
        self.temp_img.pack()

        self.target = None
        if msg is None:
            msg = "选择运算图"
        tk.Button(self, text=msg, command=select_img).pack()

    def get(self):
        return self.target


class PopupBase():
    def __init__(self, master, clean_up):
        self.top = tk.Toplevel(master)
        self.container = tk.Frame(self.top, padx=20, pady=30)
        self.container.pack()

        def finish():
            clean_up()
            self.top.destroy()

        self.top.protocol("WM_DELETE_WINDOW", finish)


class PopupCommon(PopupBase):
    def __init__(self, master, clean_up, title):
        super().__init__(master, clean_up)
        self.clean_up = clean_up

        def cancel():
            clean_up()
            self.top.destroy()

        self.top.title(title)

        self.body = tk.Frame(self.container)
        self.body.pack()

        self.buttons = ButtonGroup(self.container)
        self.buttons.pack()
        self.buttons.set_cansel(cancel)

    def set_preview(self, preview):
        def confirm():
            preview()
            self.clean_up(True)  # need to exit preview first
            self.top.destroy()

        self.buttons.set_confirm(confirm)
        self.buttons.set_preview(preview)


class PopupChannel(PopupCommon):
    def __init__(self, master, update, clean_up):
        super().__init__(master, clean_up, "选择通道")

        def preview():
            update(self.v.get())

        self.set_preview(preview)

        self.v = tk.StringVar()
        self.v.set("r")
        tk.Radiobutton(self.body, text="R (Red)", variable=self.v, value="r", command=preview).pack(anchor=tk.W)
        tk.Radiobutton(self.body, text="G (Green)", variable=self.v, value="g", command=preview).pack(anchor=tk.W)
        tk.Radiobutton(self.body, text="B (Blue)", variable=self.v, value="b", command=preview).pack(anchor=tk.W)


class PopupRotate(PopupCommon):
    def __init__(self, master, update, clean_up):
        super().__init__(master, clean_up, "旋转图像")

        def preview():
            try:
                degree = int(self.degree_entry.get()) * self.direction.get()
                bi_lin = self.bi_lin == 1
            except Exception as e:
                print(e)
                return

            update(degree, bi_lin)

        self.set_preview(preview)

        tk.Label(self.body, text="角度:").grid(row=0, column=0)
        self.degree_entry = tk.Entry(self.body, width=6)
        self.degree_entry.grid(row=0, column=1)

        directions = tk.Frame(self.body)
        directions.grid(row=0, column=2)
        self.direction = tk.IntVar()
        tk.Radiobutton(directions, text="顺时针", variable=self.direction, value=1).pack()
        tk.Radiobutton(directions, text="逆时针", variable=self.direction, value=-1).pack()

        self.bi_lin = tk.IntVar()
        tk.Checkbutton(self.body, text="双线性插值", variable=self.bi_lin).grid(row=1, column=0, columnspan=3)


class PopupCrop(PopupCommon):
    def __init__(self, master, update, clean_up):
        super().__init__(master, clean_up, "剪裁")

        def preview():
            try:
                x1, y1, x2, y2 = int(self.x1.get()), int(self.y1.get()), int(self.x2.get()), int(self.y2.get())
            except Exception as e:
                print(e)
                return

            update((x1, y1), (x2, y2))

        self.set_preview(preview)

        tk.Label(self.body, text="x1:").grid(row=0, column=0)
        self.x1 = tk.Entry(self.body, width=4)
        self.x1.grid(row=0, column=1)
        tk.Label(self.body, text="y1:").grid(row=0, column=2)
        self.y1 = tk.Entry(self.body, width=4)
        self.y1.grid(row=0, column=3)
        tk.Label(self.body, text="x2:").grid(row=1, column=0)
        self.x2 = tk.Entry(self.body, width=4)
        self.x2.grid(row=1, column=1)
        tk.Label(self.body, text="y2:").grid(row=1, column=2)
        self.y2 = tk.Entry(self.body, width=4)
        self.y2.grid(row=1, column=3)


class PopupResize(PopupCommon):
    def __init__(self, master, update, clean_up):
        super().__init__(master, clean_up, "调整尺寸")

        def preview():
            try:
                w, h = int(self.w.get()), int(self.h.get())
                bi_lin = self.bi_lin == 1
            except Exception as e:
                print(e)
                return

            update(w, h, bi_lin)

        self.set_preview(preview)

        tk.Label(self.body, text="宽:").grid(row=0, column=0)
        self.w = tk.Entry(self.body, width=7)
        self.w.grid(row=0, column=1)
        tk.Label(self.body, text="高:").grid(row=0, column=2)
        self.h = tk.Entry(self.body, width=7)
        self.h.grid(row=0, column=3)

        self.bi_lin = tk.IntVar()
        tk.Checkbutton(self.body, text="双线性插值", variable=self.bi_lin).grid(row=1, column=0, columnspan=4)


class PopupHue(PopupBase):
    def __init__(self, master, update_hue, update_sat, update_light, clean_up):
        super().__init__(master, clean_up)

        def do_update_hue():
            update_hue(self.hue.get())

        def do_update_sat():
            update_sat(self.sat.get())

        def do_update_light():
            update_light(self.light.get())

        def confirm():
            clean_up(True)  # need to exit preview first
            self.top.destroy()

        def cancel():
            clean_up()
            self.top.destroy()

        self.top.title("色相/饱和度/亮度")

        paras = tk.Frame(self.container)
        paras.pack()

        tk.Label(paras, text="色相:").grid(row=0, column=0, sticky=tk.E)
        self.hue = SliderEntry(paras, command=do_update_hue, from_=-180, to=180, resolution=1, init=0)
        self.hue.grid(row=0, column=1, sticky=tk.W)
        tk.Label(paras, text="饱和度:").grid(row=1, column=0, sticky=tk.E)
        self.sat = SliderEntry(paras, command=do_update_sat, from_=0, to=2, resolution=0.01, init=1)
        self.sat.grid(row=1, column=1, sticky=tk.W)
        tk.Label(paras, text="亮度:").grid(row=2, column=0, sticky=tk.E)
        self.light = SliderEntry(paras, command=do_update_light, from_=0, to=2, resolution=0.01, init=1)
        self.light.grid(row=2, column=1, sticky=tk.W)

        buttons = tk.Frame(self.container)
        buttons.pack()
        tk.Button(buttons, text="确定", command=confirm).pack(side=tk.LEFT)
        tk.Button(buttons, text="取消", command=cancel).pack(side=tk.LEFT)


class PopupLinMap(PopupCommon):
    def __init__(self, master, update, clean_up):
        super().__init__(master, clean_up, "线性变换")

        def preview():
            try:
                from_ = [float(x) / 255 for x in self.from_.get().split(",")]
                to = [float(x) / 255 for x in self.to.get().split(",")]
                if len(from_) != len(to):
                    return
                for data in [*from_, *to]:
                    if data < 0 or data > 1:
                        return
            except Exception as e:
                print(e)
                return

            anchors = [pair for pair in zip(from_, to)]
            update(anchors)

        self.set_preview(preview)

        tk.Label(self.body, text="From:").grid(row=0, column=0, sticky=tk.E)
        self.from_ = tk.Entry(self.body, width=20)
        self.from_.grid(row=0, column=1, sticky=tk.W)
        tk.Label(self.body, text="To:").grid(row=1, column=0, sticky=tk.E)
        self.to = tk.Entry(self.body, width=20)
        self.to.grid(row=1, column=1, sticky=tk.W)


class PopupLogMap(PopupCommon):
    def __init__(self, master, update, clean_up):
        super().__init__(master, clean_up, "对数变换")

        def preview():
            try:
                c = float(self.c.get())
            except Exception as e:
                print(e)
                return

            update(c)

        self.set_preview(preview)

        tk.Label(self.body, text="s = c * log(1 + r)").grid(row=0, column=0, columnspan=2)
        tk.Label(self.body, text="c:").grid(row=1, column=0, sticky=tk.E)
        self.c = tk.Entry(self.body, width=7)
        self.c.grid(row=1, column=1, sticky=tk.W)


class PopupPowMap(PopupCommon):
    def __init__(self, master, update, clean_up):
        super().__init__(master, clean_up, "指数变换")

        def preview():
            try:
                c, g = float(self.c.get()), float(self.g.get())
            except Exception as e:
                print(e)
                return

            update(c, g)

        self.set_preview(preview)

        tk.Label(self.body, text="s = c * r ^ g").grid(row=0, column=0, columnspan=2)
        tk.Label(self.body, text="c:").grid(row=1, column=0, sticky=tk.E)
        self.c = tk.Entry(self.body, width=7)
        self.c.grid(row=1, column=1, sticky=tk.W)
        tk.Label(self.body, text="g:").grid(row=2, column=0, sticky=tk.E)
        self.g = tk.Entry(self.body, width=7)
        self.g.grid(row=2, column=1, sticky=tk.W)


class PopupThresh(PopupCommon):
    def __init__(self, master, update, clean_up):
        super().__init__(master, clean_up, "手动阈值")

        def preview():
            double_thresh = self.double_thresh.get() == 1
            if double_thresh:
                th1, th2 = self.thresh1.get() / 255, self.thresh2.get() / 255
                update(th1, th2)
            else:
                th1 = self.thresh1.get() / 255
                update(th1)

        self.set_preview(preview)

        tk.Label(self.body, text="阈值1:").grid(row=0, column=0, sticky=tk.E)
        self.thresh1 = SliderEntry(self.body, preview, 0, 255, 1, 127)
        self.thresh1.grid(row=0, column=1, sticky=tk.W)

        tk.Label(self.body, text="阈值2:").grid(row=1, column=0, sticky=tk.E)
        self.thresh2 = SliderEntry(self.body, preview, 0, 255, 1, 200)
        self.thresh2.grid(row=1, column=1, sticky=tk.W)

        self.double_thresh = tk.IntVar()
        tk.Checkbutton(self.body, text="双阈值", variable=self.double_thresh).grid(row=2, column=0, columnspan=2)


class PopupGaus(PopupCommon):
    def __init__(self, master, update, clean_up):
        super().__init__(master, clean_up, "高斯滤波")

        def preview():
            try:
                w, h = int(self.w.get()), int(self.h.get())
            except Exception as e:
                print(e)
                return

            update(h, w)

        self.set_preview(preview)

        tk.Label(self.body, text="卷积核宽度:").grid(row=0, column=0, sticky=tk.E)
        self.w = tk.Entry(self.body, width=7)
        self.w.grid(row=0, column=1, sticky=tk.W)

        tk.Label(self.body, text="卷积核高度:").grid(row=1, column=0, sticky=tk.E)
        self.h = tk.Entry(self.body, width=7)
        self.h.grid(row=1, column=1, sticky=tk.W)


class PopupMean(PopupCommon):
    def __init__(self, master, update, clean_up):
        super().__init__(master, clean_up, "均值滤波")

        def preview():
            try:
                w, h = int(self.w.get()), int(self.h.get())
            except Exception as e:
                print(e)
                return

            update(h, w)

        self.set_preview(preview)

        tk.Label(self.body, text="卷积核宽度:").grid(row=0, column=0, sticky=tk.E)
        self.w = tk.Entry(self.body, width=7)
        self.w.grid(row=0, column=1, sticky=tk.W)

        tk.Label(self.body, text="卷积核高度:").grid(row=1, column=0, sticky=tk.E)
        self.h = tk.Entry(self.body, width=7)
        self.h.grid(row=1, column=1, sticky=tk.W)


class PopupMed(PopupCommon):
    def __init__(self, master, update, clean_up):
        super().__init__(master, clean_up, "中值滤波")

        def preview():
            try:
                w, h = int(self.w.get()), int(self.h.get())
            except Exception as e:
                print(e)
                return

            update(h, w)

        self.set_preview(preview)

        tk.Label(self.body, text="卷积核宽度:").grid(row=0, column=0, sticky=tk.E)
        self.w = tk.Entry(self.body, width=7)
        self.w.grid(row=0, column=1, sticky=tk.W)

        tk.Label(self.body, text="卷积核高度:").grid(row=1, column=0, sticky=tk.E)
        self.h = tk.Entry(self.body, width=7)
        self.h.grid(row=1, column=1, sticky=tk.W)


class PopupKernel(PopupCommon):
    def __init__(self, master, update, clean_up):
        super().__init__(master, clean_up, "自定义卷积核")

        def preview():
            kernel = self.kernel.get()
            if kernel is not None:
                update(kernel, self.norm_k.get() == 1, self.norm_r.get() == 1)

        self.set_preview(preview)

        self.kernel = KernelInput(self.body)
        self.kernel.pack()

        self.norm_k = tk.IntVar()
        tk.Checkbutton(self.body, text="归一化卷积核", variable=self.norm_k).pack()

        self.norm_r = tk.IntVar()
        tk.Checkbutton(self.body, text="归一化卷积结果", variable=self.norm_r).pack()


class PopupCanny(PopupCommon):
    def __init__(self, master, update, clean_up):
        super().__init__(master, clean_up, "Canny边缘检测")

        def preview():
            try:
                lo, hi = float(self.lo.get()) / 255, float(self.hi.get()) / 255
            except Exception as e:
                print(e)
                return

            if lo >= hi:
                return

            update(lo, hi)

        self.set_preview(preview)

        tk.Label(self.body, text="低阈值:").grid(row=0, column=0, sticky=tk.E)
        self.lo = tk.Entry(self.body, width=7)
        self.lo.grid(row=0, column=1, sticky=tk.W)
        tk.Label(self.body, text="高阈值:").grid(row=1, column=0, sticky=tk.E)
        self.hi = tk.Entry(self.body, width=7)
        self.hi.grid(row=1, column=1, sticky=tk.W)


class PopupAlg(PopupCommon):
    def __init__(self, master, update, clean_up):
        super().__init__(master, clean_up, "代数运算")

        def preview():
            target = self.target.get()
            if target is not None:
                update(target, self.method.get())

        self.set_preview(preview)

        self.method = tk.StringVar()
        self.method.set("plus")
        tk.Radiobutton(self.body, text="加法", variable=self.method, value="plus").grid(row=0, column=0)
        tk.Radiobutton(self.body, text="减法", variable=self.method, value="minus").grid(row=0, column=1)
        tk.Radiobutton(self.body, text="乘法", variable=self.method, value="times").grid(row=0, column=2)

        self.target = ImageInput(self.body)
        self.target.grid(row=1, column=0, columnspan=3)


class PopupMorph(PopupCommon):
    def __init__(self, master, update, clean_up):
        super().__init__(master, clean_up, "基本形态学操作")

        def preview():
            kernel = self.kernel.get()
            if kernel is None:
                return

            update(kernel, self.method.get(), self.flat_k.get() == 1)

        self.set_preview(preview)

        self.kernel = KernelInput(self.body)
        self.kernel.pack()

        self.flat_k = tk.IntVar()
        self.flat_k.set(1)
        tk.Checkbutton(self.body, text="平坦卷积核", variable=self.flat_k).pack()

        methods = tk.Frame(self.body)
        methods.pack()
        self.method = tk.StringVar()
        self.method.set("erode")
        tk.Radiobutton(methods, text="腐蚀", variable=self.method, value="erode").grid(row=0, column=0)
        tk.Radiobutton(methods, text="膨胀", variable=self.method, value="dilate").grid(row=0, column=1)
        tk.Radiobutton(methods, text="开", variable=self.method, value="open").grid(row=0, column=2)
        tk.Radiobutton(methods, text="闭", variable=self.method, value="close").grid(row=0, column=3)


class PopupMorphRe(PopupCommon):
    def __init__(self, master, update, clean_up):
        super().__init__(master, clean_up, "形态学重建")

        def preview():
            kernel = self.kernel.get()
            if kernel is None:
                return

            ref = self.ref.get()
            if ref is None:
                return

            update(ref, kernel, self.method.get())

        self.set_preview(preview)

        self.kernel = KernelInput(self.body)
        self.kernel.pack()

        self.ref = ImageInput(self.body, msg="选择参照图")
        self.ref.pack()

        methods = tk.Frame(self.body)
        methods.pack()
        self.method = tk.StringVar()
        self.method.set("erode")
        tk.Radiobutton(methods, text="腐蚀", variable=self.method, value="erode").grid(row=0, column=0)
        tk.Radiobutton(methods, text="膨胀", variable=self.method, value="dilate").grid(row=0, column=1)


class PopupMorphBi(PopupCommon):
    def __init__(self, master, update, clean_up):
        super().__init__(master, clean_up, "基本形态学操作")

        def preview():
            kernel = self.kernel.get()
            if kernel is None:
                return

            update(kernel, self.method.get())

        self.set_preview(preview)

        self.kernel = KernelInput(self.body)
        self.kernel.pack()

        methods = tk.Frame(self.body)
        methods.pack()
        self.method = tk.StringVar()
        self.method.set("erode")
        tk.Radiobutton(methods, text="腐蚀", variable=self.method, value="erode").grid(row=0, column=0)
        tk.Radiobutton(methods, text="膨胀", variable=self.method, value="dilate").grid(row=0, column=1)
        tk.Radiobutton(methods, text="开", variable=self.method, value="open").grid(row=0, column=2)
        tk.Radiobutton(methods, text="闭", variable=self.method, value="close").grid(row=0, column=3)


class PopupThin(PopupCommon):
    def __init__(self, master, update, clean_up):
        super().__init__(master, clean_up, "细化/粗化")

        def preview():
            try:
                iter_n = int(self.iter_n.get())
            except Exception as e:
                print(e)
                return

            kernel = self.kernel.get()
            if kernel is None:
                return

            if iter_n <= 0:
                iter_n = None

            update(self.method.get(), kernel, iter_n)

        self.set_preview(preview)

        self.kernel = KernelInput(self.body)
        self.kernel.grid(row=0, column=0, columnspan=2)

        tk.Label(self.body, text="重复次数:").grid(row=1, column=0, sticky=tk.E)
        self.iter_n = tk.Entry(self.body, width=6)
        self.iter_n.grid(row=1, column=1, sticky=tk.W)

        self.method = tk.StringVar()
        self.method.set("thin")
        tk.Radiobutton(self.body, text="细化", variable=self.method, value="thin").grid(row=2, column=0, sticky=tk.E)
        tk.Radiobutton(self.body, text="粗化", variable=self.method, value="thick").grid(row=2, column=1, sticky=tk.W)


class PopupSkel(PopupCommon):
    def __init__(self, master, update, clean_up):
        super().__init__(master, clean_up, "骨架及重建")

        def preview():
            kernel = self.kernel.get()
            if kernel is None:
                return

            update(self.method.get(), kernel)

        self.set_preview(preview)

        self.kernel = KernelInput(self.body)
        self.kernel.grid(row=0, column=0, columnspan=2)

        self.method = tk.StringVar()
        self.method.set("xixi")
        tk.Radiobutton(self.body, text="骨架", variable=self.method, value="xixi").grid(row=2, column=0, sticky=tk.E)
        tk.Radiobutton(self.body, text="骨架重建", variable=self.method, value="rexixi").grid(row=2, column=1, sticky=tk.W)


class ImageMode(Enum):
    BGR = 0
    GREY = 1
    HSL = 2
    # BIN = 4
    # DIST = 5


class AppData:
    def __init__(self, filename=None):
        if filename is None:
            filename = "Lenna.png"

        self.img = read(filename)
        self.in_preview = False
        self.img_preview = None
        self.mode_preview = None
        self.history = []
        self.future = []
        if self.img_ok():
            self.mode = ImageMode.GREY if is_greyscale(self.img) else ImageMode.BGR
            self.log()  # history should include the original point

        self.recent_img_src = None
        self.recent_img_rgb = None

        self.sks = []

    def img_ok(self):
        return self.img is not None

    def get_img(self, max_size=None):
        img = self.img if not self.in_preview else self.img_preview
        mode = self.mode if not self.in_preview else self.mode_preview

        # REVIEW: this should be error-prone
        if np.all(self.recent_img_src == img):
            return self.recent_img_rgb
        else:
            self.recent_img_src = img

        if max_size is not None:
            h, w = max_size
            if h / w > img.shape[0] / img.shape[1]:
                if img.shape[1] > w:
                    img = resize(img, w, ceil(img.shape[0] / img.shape[1] * w), InterpolationMethod.Linear)
            else:
                if img.shape[0] > h:
                    img = resize(img, ceil(img.shape[1] / img.shape[0] * h), h, InterpolationMethod.Linear)

        if mode == ImageMode.BGR:
            self.recent_img_rgb = to8bit(bgr_to_rgb(img))
        elif mode == ImageMode.GREY:
            self.recent_img_rgb = to8bit(grey_to_rgb(img))
        elif mode == ImageMode.HSL:
            img[:, :, 0] /= 360
            self.recent_img_rgb = hsl2rgb(img)
        else:
            logger.error("get_img: wrong image mode")
            self.recent_img_rgb = np.zeros((100, 100, 3))

        return self.recent_img_rgb

    def get_hist(self):

        def get_fig(img):
            levels = 256
            n = get_histogram_int(img, levels)
            bins = np.linspace(0, 1, levels + 1)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.axis("off")

            left = np.array(bins[:-1])
            right = np.array(bins[1:])
            bottom = np.zeros(len(left))
            top = bottom + n
            nrects = len(left)

            nverts = nrects * (1 + 3 + 1)
            verts = np.zeros((nverts, 2))
            codes = np.ones(nverts, int) * path.Path.LINETO
            codes[0::5] = path.Path.MOVETO
            codes[4::5] = path.Path.CLOSEPOLY
            verts[0::5, 0] = left
            verts[0::5, 1] = bottom
            verts[1::5, 0] = left
            verts[1::5, 1] = top
            verts[2::5, 0] = right
            verts[2::5, 1] = top
            verts[3::5, 0] = right
            verts[3::5, 1] = bottom

            barpath = path.Path(verts, codes)
            patch = patches.PathPatch(barpath, facecolor='green', edgecolor='yellow', alpha=0.5)
            ax.add_patch(patch)

            ax.set_xlim(left[0], right[-1])
            ax.set_ylim(bottom.min(), top.max())

            fig.tight_layout()
            return fig

        if self.mode == ImageMode.GREY:
            return get_fig(self.img)
        else:
            return get_fig(bgr_to_grey(self.img))

    def save(self, filename):
        write(filename, self.img)

    def undo(self):
        if self.in_preview:
            logger.warn("undo: cannot undo when in preview")
            return

        if len(self.history) > 1:  # history include the first one, which will never pop out
            self.future.append(self.history.pop())  # put current state to future
            self.img, self.mode = self.history[-1]
            self.img = self.img.copy()

    def redo(self):
        if self.in_preview:
            logger.warn("redo: cannot redo when in preview")
            return

        if len(self.future):
            self.history.append(self.future.pop())
            self.img, self.mode = self.history[-1]  # pop the last one
            self.img = self.img.copy()

    def enter_preview(self):
        if self.in_preview:
            logger.warn("enter_preview: in preview but try to enter")
            return

        self.in_preview = True
        self.img_preview = self.img.copy()
        self.mode_preview = self.mode

    def exit_preview(self, save_preview=False):
        if not self.in_preview:
            logger.warn("exit_preview: not in preview but try to exit")
            return

        self.in_preview = False
        if save_preview:
            self.img = self.img_preview.copy()
            self.mode = self.mode_preview
            self.log()

    def clear_future(self):
        self.future.clear()

    def log(self):
        if not self.in_preview:  # don't log during preview
            self.clear_future()
            self.history.append((self.img.copy(), self.mode))

    def do_process(self, func, *args, **kwargs):
        if self.in_preview:
            if "mode" in kwargs:
                self.mode_preview = kwargs.pop("mode")
            self.img_preview = func(*args, **kwargs)
        else:
            if "mode" in kwargs:
                self.mode = kwargs.pop("mode")
            self.img = func(*args, **kwargs)
            self.log()

    def to_grey(self):
        if self.mode == ImageMode.BGR:
            self.do_process(bgr_to_grey, self.img, mode=ImageMode.GREY)

    def extract_channel(self, channel):
        if channel == "r":
            channel = Channel.Red
        elif channel == "g":
            channel = Channel.Green
        elif channel == "b":
            channel = Channel.Blue
        else:
            logger.error("extract_channel: unknown channel {}".format(channel))
            return

        if self.mode != ImageMode.BGR:
            logger.error("extract_channel: image is not colored")
            return
        self.do_process(get_channel, self.img, channel, mode=ImageMode.GREY)

    def rotate(self, degree, bi_lin):
        method = InterpolationMethod.Bilinear if bi_lin else InterpolationMethod.Linear
        self.do_process(rotate, self.img, degree % 360, method)

    def crop(self, point0, point1):
        if point0[0] <= 0 or point0[1] <= 0 or point1[0] <= 0 or point1[1] <= 0 or \
                        point0[0] >= self.img.shape[0] or point0[1] >= self.img.shape[1] or \
                        point1[0] >= self.img.shape[0] or point1[1] >= self.img.shape[1]:
            logger.error("crop: invalid point(s): ({}, {}), ({}, {}), shape is ({}, {})".format(*point0, *point1,
                                                                                                *self.img.shape))
            return

        self.do_process(crop, self.img, point0, point1)

    def resize(self, w, h, bi_lin):
        if w <= 0 or h <= 0:
            logger.error("resize: invalid w/h: {}, {}".format(w, h))
            return

        method = InterpolationMethod.Bilinear if bi_lin else InterpolationMethod.Linear
        self.do_process(resize, self.img, w, h, method)

    def brg_to_hsl(self):
        if self.mode == ImageMode.BGR:
            self.do_process(bgr_to_hsl, self.img, mode=ImageMode.HSL)

    def hsl_to_brg(self):
        if self.mode == ImageMode.HSL:
            self.do_process(hsl_to_bgr, self.img, mode=ImageMode.BGR)

    def hue(self, off):
        if self.mode == ImageMode.HSL:
            self.do_process(adjust_hue, self.img, off)

    def sat(self, fac):
        if self.mode == ImageMode.HSL:
            self.do_process(adjust_saturation, self.img, fac)

    def light(self, fac):
        if self.mode == ImageMode.HSL:
            self.do_process(adjust_lightness, self.img, fac)

    def lin_map(self, anchors):
        if self.mode == ImageMode.GREY:
            self.do_process(linear_mapping, self.img, anchors)

    def log_map(self, c):
        if self.mode == ImageMode.GREY:
            self.do_process(log_mapping, self.img, c)

    def pow_map(self, c, g):
        if self.mode == ImageMode.GREY:
            self.do_process(power_mapping, self.img, c, g)

    def hist_eq(self):
        if self.mode == ImageMode.GREY:
            self.do_process(histogram_qualization, self.img)

    def otus(self):
        if self.mode == ImageMode.GREY:
            self.do_process(otus, self.img, 256)

    def thresh(self, thresh1, thresh2=None):
        if self.mode == ImageMode.GREY:
            if thresh2 is None:
                self.do_process(threshold, self.img, thresh1)
            else:
                if 0 <= thresh1 <= thresh2 <= 1:
                    self.do_process(threshold2, self.img, thresh1, thresh2)

    def gaus(self, h, w):
        self.do_process(gaussian_filter, self.img, h, w)

    def med(self, h, w):
        self.do_process(median_filter, self.img, h, w)

    def mean(self, h, w):
        self.do_process(mean_filter, self.img, h, w)

    def kernel(self, kernel, normalize_kernal, normalize_result):
        def conv_n_trunc():
            img = convolve(self.img, kernel, normalize_kernel=normalize_kernal, normalize_result=normalize_result)
            if is_greyscale(img):
                img[img[:, :] > 1] = 1.0
            else:
                img[img[:, :, :] > 1] = 1.0
            return img

        self.do_process(conv_n_trunc)

    def sobel(self):
        if self.mode == ImageMode.GREY:
            self.do_process(lambda x: normalize(sobel(x)), self.img)

    def lapl(self):
        if self.mode == ImageMode.GREY:
            self.do_process(lambda x: normalize(laplacian_filter(x)), self.img)

    def canny(self, thresh_lo, thresh_hi):
        if self.mode == ImageMode.GREY:
            self.do_process(canny, self.img, thresh_lo, thresh_hi)

    def alg(self, target, method):
        if not is_greyscale(target):
            target = bgr_to_grey(target)

        if self.mode == ImageMode.GREY and target.shape == self.img.shape:
            if method.startswith("p"):
                method = plus
            elif method.startswith("m"):
                method = minus
            elif method.startswith("t"):
                method = times
            else:
                logger.error("alg: unexpected algebra operation {}".format(method))
                return

            self.do_process(method, self.img, target)

    def morph_grey(self, kernel, method, flat_k):
        if self.mode == ImageMode.GREY:
            if method == "erode":
                method = MorphMethod.Erode
            elif method == "dilate":
                method = MorphMethod.Dilate
            elif method == "open":
                method = MorphMethod.Open
            elif method == "close":
                method = MorphMethod.Close
            else:
                logger.error("morph: unknown method {}".format(method))
                return

            self.do_process(morph_greyscale, self.img, kernel, method, flat_k)

    def morph_re_grey(self, reference, kernel, method):
        if not is_greyscale(reference):
            reference = bgr_to_grey(reference)

        if self.mode == ImageMode.GREY and self.img.shape == reference.shape:
            if method == "erode":
                method = MorphMethod.Erode
            elif method == "dilate":
                method = MorphMethod.Dilate
            else:
                logger.error("morph: unknown method {}".format(method))
                return

            self.do_process(morphological_reconstruct_greyscale, self.img, reference, kernel, method)

    def watershed_grey(self):
        if self.mode == ImageMode.GREY:
            self.do_process(watershed, self.img)

    def morph_bi(self, kernel, method):
        if self.mode == ImageMode.GREY:
            if method == "erode":
                method = MorphMethod.Erode
            elif method == "dilate":
                method = MorphMethod.Dilate
            elif method == "open":
                method = MorphMethod.Open
            elif method == "close":
                method = MorphMethod.Close
            else:
                logger.error("morph_bi: unknown method {}".format(method))
                return

            self.do_process(morph_binary, self.img, kernel, method)

    def morph_re_bi(self, reference, kernel, method):
        if self.mode == ImageMode.GREY and is_greyscale(reference) and self.img.shape == reference.shape:
            if method == "erode":
                method = MorphMethod.Erode
            elif method == "dilate":
                method = MorphMethod.Dilate
            else:
                logger.error("morph: unknown method {}".format(method))
                return

            self.do_process(morphological_reconstruct_binary, self.img, reference, kernel, method)

    def thin_thick(self, method, kernel, iter_n=None):
        if self.mode == ImageMode.GREY:
            if method.startswith("thin"):
                self.do_process(thinning, self.img, kernel, iter_n)
            elif method.startswith("thick"):
                self.do_process(thickening, self.img, kernel, iter_n)
            else:
                logger.error("thin_thick: unknown method {}".format(method))

    def thin_for_skel(self, iter_n=None):
        if self.mode == ImageMode.GREY:
            self.do_process(thinning_skeleton, self.img, iter_n)

    def skel(self, method, kernel):
        if self.mode == ImageMode.GREY:
            if method.startswith("re"):
                self.do_process(skeleton_reconstruct, self.sks, kernel)
            else:
                if self.in_preview:
                    self.img_preview, self.sks = skeleton(self.img, kernel)
                else:
                    self.img, self.sks = skeleton(self.img, kernel)
                    self.log()

    def dist(self):
        if self.mode == ImageMode.GREY:
            self.do_process(lambda x: normalize(distance_transform(x)), self.img)


class App:
    def __init__(self):
        self.data = AppData()
        self.init_gui()

    @staticmethod
    def random_color():
        # return "#%X" % randrange(0x808080, 0xFFFFFF)
        return "#FFFFFF"

    def init_gui(self):
        rhs_w = 220
        hist_h = 200
        info_h = 400

        self.root = tk.Tk()
        self.root.title("RPS - a Retarded PhotoShop")

        # Define layout here
        container = tk.Frame(self.root, padx=35, pady=25, bg=self.random_color())
        container.pack(fill=tk.BOTH, expand=tk.YES)

        # Operation bar
        op_bar_frame = tk.Frame(container, bg=self.random_color())
        op_bar_frame.pack()

        tk.Frame(container, height=15, bg=self.random_color()).pack()

        # Body
        body_frame = tk.Frame(container, bg=self.random_color())
        body_frame.pack(fill=tk.BOTH, expand=tk.YES)

        # Body - Left hand side
        self.lhs_frame = tk.Frame(body_frame, bg=self.random_color())
        self.lhs_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)

        # Body - Left hand side - Image display
        # display_frame = tk.Frame(self.lhs_frame, bg=self.random_color())
        # display_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        tk.Frame(body_frame, width=27, bg=self.random_color()).pack(side=tk.LEFT)

        # Body - Right hand side
        rhs_frame = tk.Frame(body_frame, width=rhs_w, bg=self.random_color())
        rhs_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Body - Right hand side - Histogram
        self.hist_frame = tk.Frame(rhs_frame, width=rhs_w, height=hist_h, bg=self.random_color())
        self.hist_frame.pack_propagate(False)
        self.hist_frame.pack()

        tk.Frame(rhs_frame, height=20, bg=self.random_color()).pack()

        # Body - Right hand side - Detail info
        self.info_erea = InfoErea(rhs_frame, bg=self.random_color())
        self.info_erea.pack(fill=tk.BOTH, expand=tk.YES)

        # Main menu
        main_menu = tk.Menu(self.root)
        self.root.config(menu=main_menu)

        # Image menu
        image_menu = tk.Menu(main_menu)
        main_menu.add_cascade(label="图像", menu=image_menu)

        # Image - Basic
        basic_menu = tk.Menu(image_menu)
        image_menu.add_cascade(label="入门", menu=basic_menu)
        basic_menu.add_command(label="色相/饱和度/亮度", command=self.hsl)
        basic_menu.add_separator()

        # Image - Basic - Contrast
        contrast_menu = tk.Menu(basic_menu)
        basic_menu.add_cascade(label="对比度调整", menu=contrast_menu)
        contrast_menu.add_command(label="线性变换", command=self.lin_map)
        contrast_menu.add_command(label="对数变换", command=self.log_map)
        contrast_menu.add_command(label="指数变换", command=self.pow_map)
        contrast_menu.add_command(label="直方图均衡化", command=self.hist_eq)

        # Image - Binary
        bin_menu = tk.Menu(image_menu)
        image_menu.add_cascade(label="二值化", menu=bin_menu)
        bin_menu.add_command(label="Otus", command=self.otus)
        bin_menu.add_command(label="手动阈值", command=self.thresh)

        # Image - Filter
        smooth_menu = tk.Menu(image_menu)
        image_menu.add_cascade(label="滤波", menu=smooth_menu)
        smooth_menu.add_command(label="高斯滤波", command=self.gaus)
        smooth_menu.add_command(label="均值滤波", command=self.mean)
        smooth_menu.add_command(label="中值滤波", command=self.med)
        smooth_menu.add_command(label="自定义...", command=self.kernel)

        # Image - Edge detection
        edge_menu = tk.Menu(image_menu)
        image_menu.add_cascade(label="边缘检测", menu=edge_menu)
        edge_menu.add_command(label="Sobel算子", command=self.sobel)
        edge_menu.add_command(label="Laplace算子", command=self.lapl)
        edge_menu.add_command(label="Canny边缘检测", command=self.canny)

        # Edit
        edit_menu = tk.Menu(main_menu)
        main_menu.add_cascade(label="编辑", menu=edit_menu)

        # Edit - Algebra
        edit_menu.add_command(label="代数操作", command=self.alg)

        # Edit - Greyscale Morphology
        grey_morph_menu = tk.Menu(edit_menu)
        edit_menu.add_cascade(label="灰度形态学", menu=grey_morph_menu)
        grey_morph_menu.add_command(label="膨胀/腐蚀/开/闭", command=self.morph_grey)
        grey_morph_menu.add_command(label="形态学重构", command=self.morph_re_grey)
        grey_morph_menu.add_command(label="分水岭算法", command=self.watershed_grey)

        # Edit - Binary Morphology
        bin_morph_menu = tk.Menu(edit_menu)
        edit_menu.add_cascade(label="二值形态学", menu=bin_morph_menu)
        bin_morph_menu.add_command(label="膨胀/腐蚀/开/闭", command=self.morph_bi)
        bin_morph_menu.add_command(label="细化/粗化", command=self.thin_thick)
        bin_morph_menu.add_command(label="细化得骨架", command=self.thin_for_skel)
        bin_morph_menu.add_command(label="骨架/骨架重构", command=self.skel)
        bin_morph_menu.add_command(label="形态学重构", command=self.morph_re_bi)
        bin_morph_menu.add_command(label="距离变换", command=self.dist)

        tk.Button(op_bar_frame, text="打开", command=self.open_img, bg=self.random_color()).pack(side=tk.LEFT)
        tk.Button(op_bar_frame, text="保存", command=self.save_img, bg=self.random_color()).pack(side=tk.LEFT)
        tk.Frame(op_bar_frame, width=15).pack(side=tk.LEFT, fill=tk.Y)
        tk.Button(op_bar_frame, text="撤销", command=self.undo).pack(side=tk.LEFT)
        tk.Button(op_bar_frame, text="重做", command=self.redo).pack(side=tk.LEFT)
        tk.Frame(op_bar_frame, width=15).pack(side=tk.LEFT, fill=tk.Y)
        tk.Button(op_bar_frame, text="黑白", command=self.to_grey).pack(side=tk.LEFT)
        tk.Button(op_bar_frame, text="通道分离...", command=self.channel).pack(side=tk.LEFT)
        tk.Frame(op_bar_frame, width=15).pack(side=tk.LEFT, fill=tk.Y)
        tk.Button(op_bar_frame, text="旋转90度", command=self.rotate90).pack(side=tk.LEFT)
        tk.Button(op_bar_frame, text="旋转...", command=self.rotate).pack(side=tk.LEFT)
        tk.Frame(op_bar_frame, width=15).pack(side=tk.LEFT, fill=tk.Y)
        tk.Button(op_bar_frame, text="剪裁...", command=self.crop).pack(side=tk.LEFT)
        tk.Button(op_bar_frame, text="调整尺寸...", command=self.resize).pack(side=tk.LEFT)

        img = ImageTk.PhotoImage(Image.fromarray(self.data.get_img(), "RGB"))
        self.img_label = tk.Label(self.lhs_frame, bd=0, image=img)
        self.img_label.image = img  # keep a reference from GC
        # self.img_label.pack(fill=tk.BOTH)
        self.img_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.hist_canvas = FigureCanvasTkAgg(self.data.get_hist(), master=self.hist_frame)
        self.hist_canvas.show()
        self.hist_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.hist_canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # track display size change
        self.lhs_frame.bind("<Configure>", self.update_image)

        # track mouse movement
        self.img_label.bind('<Motion>', self.update_mouse)

        # set window size
        self.root.update()
        self.root.minsize(max(800, self.root.winfo_width()), max(600, self.root.winfo_height()))

    def update_mouse(self, event):
        col_i, row_i = event.x, event.y
        if len(self.data.img.shape) == 3:
            real_row_n, real_col_n, _ = self.data.img.shape
        else:
            real_row_n, real_col_n = self.data.img.shape
        display_row_n, display_col_n = self.img_label.winfo_height(), self.img_label.winfo_width()

        mapped_row_i, mapped_col_i = round(real_row_n / display_row_n * row_i), \
                                     round(real_col_n / display_col_n * col_i)
        if mapped_row_i < 0 or mapped_row_i >= real_row_n or mapped_col_i < 0 or mapped_col_i >= real_col_n:
            return  # out of scope

        self.info_erea.set_pixel_cord("({}, {})".format(mapped_row_i, mapped_col_i))

        pixel = self.data.img[mapped_row_i, mapped_col_i]
        if len(pixel.shape) == 0:  # greyscale
            self.info_erea.set_pixel_val("灰度 {}".format(round(pixel * 255)))
        else:
            self.info_erea.set_pixel_val(
                "RGB ({}, {}, {})".format(int(pixel[0] * 255), int(pixel[1] * 255), int(pixel[2] * 255)))

    # TODO: show 3 hist for color image
    def update_image(self, event=None):
        # main image
        max_size = self.lhs_frame.winfo_height(), self.lhs_frame.winfo_width()
        img = ImageTk.PhotoImage(Image.fromarray(self.data.get_img(max_size), "RGB"))
        self.img_label.config(image=img)
        self.img_label.image = img

        self.info_erea.set_img_size("{}px x {}px".format(*self.data.img.shape))

        # histogram
        if self.hist_canvas is not None:
            plt.clf()
            self.hist_canvas.get_tk_widget().pack_forget()

        self.hist_canvas = FigureCanvasTkAgg(self.data.get_hist(), master=self.hist_frame)
        self.hist_canvas.show()
        self.hist_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.hist_canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def open_img(self):
        filename = filedialog.askopenfilename()
        self.data = AppData(filename)
        if not self.data.img_ok():
            messagebox.showerror("打开错误", "不能读取该图像文件，请重试。")
            return

        self.update_image()

    def save_img(self):
        filename = filedialog.asksaveasfilename()
        self.data.save(filename)
        if not self.data.img_ok():
            messagebox.showerror("保存错误", "不能保存该图像文件，请重试。")
            return

    def undo(self):
        self.data.undo()
        self.update_image()

    def redo(self):
        self.data.redo()
        self.update_image()

    def common_clean_up(self):
        def clean_up(save_op=False):
            self.data.exit_preview(save_op)
            self.update_image()

        return clean_up

    def update_template(self, func):
        def update(*args, **kwargs):
            func(*args, **kwargs)
            self.update_image()

        return update

    def popup_callback_template(self, data_api, popup_class):
        def callback():
            self.data.enter_preview()
            popup_class(self.root, self.update_template(data_api), self.common_clean_up())

        return callback

    def direct_callback_template(self, data_api):
        def callback():
            data_api()
            self.update_image()

        return callback

    def lin_map(self):
        self.data.enter_preview()
        PopupLinMap(self.root, self.update_template(self.data.lin_map), self.common_clean_up())

    def log_map(self):
        self.data.enter_preview()
        PopupLogMap(self.root, self.update_template(self.data.log_map), self.common_clean_up())

    def pow_map(self):
        self.data.enter_preview()
        PopupPowMap(self.root, self.update_template(self.data.pow_map), self.common_clean_up())

    def hist_eq(self):
        self.data.hist_eq()
        self.update_image()

    def otus(self):
        self.data.otus()
        self.update_image()

    def thresh(self):
        self.data.enter_preview()
        PopupThresh(self.root, self.update_template(self.data.thresh), self.common_clean_up())

    def gaus(self):
        self.data.enter_preview()
        PopupGaus(self.root, self.update_template(self.data.gaus), self.common_clean_up())

    def mean(self):
        self.data.enter_preview()
        PopupMean(self.root, self.update_template(self.data.mean), self.common_clean_up())

    def med(self):
        self.data.enter_preview()
        PopupMed(self.root, self.update_template(self.data.med), self.common_clean_up())

    def kernel(self):
        self.data.enter_preview()
        PopupKernel(self.root, self.update_template(self.data.kernel), self.common_clean_up())

    def sobel(self):
        self.data.sobel()
        self.update_image()

    def lapl(self):
        self.data.lapl()
        self.update_image()

    def canny(self):
        self.data.enter_preview()
        PopupCanny(self.root, self.update_template(self.data.canny), self.common_clean_up())

    def alg(self):
        self.data.enter_preview()
        PopupAlg(self.root, self.update_template(self.data.alg), self.common_clean_up())

    def morph_grey(self):
        self.data.enter_preview()
        PopupMorph(self.root, self.update_template(self.data.morph_grey), self.common_clean_up())

    def morph_re_grey(self):
        self.data.enter_preview()
        PopupMorphRe(self.root, self.update_template(self.data.morph_re_grey), self.common_clean_up())

    def watershed_grey(self):
        self.data.watershed_grey()
        self.update_image()

    def morph_bi(self):
        self.data.enter_preview()
        PopupMorphBi(self.root, self.update_template(self.data.morph_bi), self.common_clean_up())

    def thin_thick(self):
        self.data.enter_preview()
        PopupThin(self.root, self.update_template(self.data.thin_thick), self.common_clean_up())

    def thin_for_skel(self):
        self.data.thin_for_skel()
        self.update_image()

    def skel(self):
        self.data.enter_preview()
        PopupSkel(self.root, self.update_template(self.data.skel), self.common_clean_up())

    def morph_re_bi(self):
        self.data.enter_preview()
        PopupMorphRe(self.root, self.update_template(self.data.morph_re_bi), self.common_clean_up())

    def dist(self):
        self.data.dist()
        self.update_image()

    def to_grey(self):
        self.data.to_grey()
        self.update_image()

    def channel(self):
        self.data.enter_preview()
        PopupChannel(self.root, self.update_template(self.data.extract_channel), self.common_clean_up())

    def rotate90(self):
        self.data.rotate(90, True)
        self.update_image()

    def rotate(self):
        self.data.enter_preview()
        PopupRotate(self.root, self.update_template(self.data.rotate), self.common_clean_up())

    def crop(self):
        self.data.enter_preview()
        PopupCrop(self.root, self.update_template(self.data.crop), self.common_clean_up())

    def resize(self):
        self.data.enter_preview()
        PopupResize(self.root, self.update_template(self.data.resize), self.common_clean_up())

    # TODO: use canvas to show realtime result

    def hsl(self):
        def clean_up(save_op=False):  # customed
            self.data.exit_preview(save_op)
            self.data.hsl_to_brg()
            self.update_image()

        self.data.brg_to_hsl()
        self.data.enter_preview()
        PopupHue(self.root, self.update_template(self.data.hue), self.update_template(self.data.sat),
                 self.update_template(self.data.light), clean_up)

    # TODO: use canvas to draw lines

    def run(self):
        self.root.mainloop()
