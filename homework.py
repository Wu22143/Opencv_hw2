from fileinput import filename
from glob import glob
from unittest import result
from PIL import ImageTk, Image
from cProfile import label
import tkinter as tk 
from cv2 import cvtColor, getHardwareFeatureName, threshold
from matplotlib import pyplot as plt
from tkinter import COMMAND, Y, Button, Entry, Label, Menu, PhotoImage, Toplevel, filedialog
import tkinter
import HW_Functions


class App():
    def __init__(self,window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry('1000x600')
        self.functions = HW_Functions.MyFunctions()
        # Add a menubar
        self.main_menu = tk.Menu(window)
        # Add file submenu
        self.file_menu = tk.Menu(self.main_menu, tearoff=0)
        self.menu_setting = tk.Menu(self.main_menu,tearoff=0)
        self.menu_imageprocess = tk.Menu(self.main_menu,tearoff=0)
        self.menu_filter = tk.Menu(self.main_menu,tearoff=0)
        self.menu_cvt = tk.Menu(self.main_menu,tearoff=0)
        self.menu_affine = tk.Menu(self.main_menu,tearoff=0)
        #
        self.menu_cvt.add_command(label='to RGB' , command=self.functions.cvttoRGB)
        self.menu_cvt.add_command(label='to HSV' , command=self.functions.cvttoHSV)
        self.menu_cvt.add_command(label='to Gray', command=self.functions.cvttoGray)

        self.file_menu.add_command(label='開啟檔案' , command=self.functions.open_files)
        self.file_menu.add_command(label='儲存檔案' , command=self.functions.save_files)
        self.file_menu.add_separator()
        self.file_menu.add_command(label='離開程式', command=window.quit)

        self.menu_setting.add_command(label='設定ROI',command=self.functions.ROI)
        self.menu_setting.add_command(label='顯示影像直方圖',command=self.functions.calHist)
        self.menu_setting.add_cascade(label='顯示或改變色彩空間',menu=self.menu_cvt)

        self.menu_imageprocess.add_command(label='影像二值化(Thresholding)',command=self.functions.fun_Threshoulding)
        self.menu_imageprocess.add_command(label='直方圖等化(Histogram Equalization)',command=self.functions.fun_EqualizeHist)
        self.menu_imageprocess.add_cascade(label='濾波器(Filter)',menu=self.menu_filter)
        self.menu_imageprocess.add_cascade(label='仿射轉換(Affine Transform',menu=self.menu_affine)
        self.menu_imageprocess.add_command(label='透視變換',command=self.functions.Perspective_Transformation)
        self.menu_imageprocess.add_command(label='Canny Detector',command=self.functions.canny_detector)
        self.menu_imageprocess.add_command(label='霍夫轉換',command=self.functions.hough_transform)
        self.menu_imageprocess.add_command(label='Corner_Harris',command=self.functions.conrer_Harris)

        self.menu_filter.add_command(label='中值濾波',command=self.functions.Median_filter)
        self.menu_filter.add_command(label='均值濾波',command=self.functions.averaging_filter)
        self.menu_filter.add_command(label='高斯濾波',command=self.functions.Gaussian_filter)
        self.menu_filter.add_command(label='索伯算子',command=self.functions.sobel_filter)
        self.menu_filter.add_command(label='拉普拉斯',command=self.functions.laplacian_filter)
        self.menu_affine.add_command(label='平移',command=self.functions.Affine_Transform)
        self.menu_affine.add_command(label='旋轉',command=self.functions.Affine_Transform_Rotate)
        
        # Add mainmenu
        self.main_menu.add_cascade(label='檔案(File)', menu = self.file_menu)
        self.main_menu.add_cascade(label='設定(Setting)' , menu = self.menu_setting)
        self.main_menu.add_cascade(label='影像處理(Image Processing)' , menu = self.menu_imageprocess)
        
        self.window.config(menu=self.main_menu)
        self.window.mainloop()

#開始
App(tk.Tk(),'4A8G0023_HW')
