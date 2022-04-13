from ast import IsNot
from fileinput import filename
from glob import glob
from tkinter.tix import Tree
from unittest import result
from PIL import ImageTk, Image
from cProfile import label
import tkinter as tk , numpy as np
from cv2 import cvtColor, getHardwareFeatureName, threshold
from matplotlib import pyplot as plt
import cv2 as cv
from tkinter import filedialog , messagebox
import tkinter

def trackChaned(x):
    pass

point_matrix = np.zeros((4,2),np.int)
counter = 0

#滑鼠事件
def mousePoints(event,x,y,flags,params):
    global counter
    if event == cv.EVENT_LBUTTONDOWN:
        point_matrix[counter] = x,y
        counter = counter + 1


class MyFunctions():
    def __init__(self):
        self.picture = None #要顯示在tk的
        self.file_path = None  #讀檔路徑
        self.img = None   #讀取的照片
        self.tkimg = None #要顯示在tk的照片
        self.original_img = None
        self.save_img = None
        self.size = None

        self.dt = True
    #載入新照片    
    def reload(self,pic):
        im = Image.fromarray(pic)
        im = ImageTk.PhotoImage(im)
        if self.picture !=None:
            self.picture.destroy()
        self.picture = tkinter.Label(image=im)
        self.picture.pack()
        cv.destroyAllWindows()
        super().mainmenu.mainloop()
    
    #開檔
    def open_files(self):
        filetypes = (
            ('jpg files', '*.jpg'),
            ('png files', '*.png'),
            ('All files', '*.*')
        )
        
        self.file_path = filedialog.askopenfilename(filetypes=filetypes)
        if not self.file_path:
            tkinter,messagebox.showerror(title = '錯誤!', message='請選擇檔案!')
            return
        
        if self.picture != None:
            self.picture.destroy()
                
        self.img = cv.imread(self.file_path)
        self.original_img = self.img
        self.save_img = self.img
        self.tkimg = cv.cvtColor(self.img,cv.COLOR_BGR2RGB)
        self.size = self.img.shape
        self.reload(self.tkimg)
    #寫檔
    def save_files(self):
        file_path = filedialog.asksaveasfilename(defaultextension='.jpg',filetypes=[("jpg" , '*.jpg')])
        if not file_path:
            return
        cv.imwrite(file_path,self.save_img)

    
#ROI    
    def ROI(self):
        while True:
            r = cv.selectROI('Select Roi',self.original_img, False,False)
            if (max(r)==0):
                cv.destroyAllWindows()
                return
            roi_img = self.tkimg[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
            self.save_img = cv.cvtColor(roi_img,cv.COLOR_BGR2RGB)
            if (max(r)!=0):
                break
        self.reload(roi_img)

#轉換色彩空間
    def cvttoRGB(self):
        cvt = cv.cvtColor(self.original_img,cv.COLOR_BGR2RGB)
        self.save_img = self.original_img
        self.reload(cvt)
        
    def cvttoHSV(self):
        self.save_img = cv.cvtColor(self.original_img,cv.COLOR_BGR2HSV)
        cvt = cv.cvtColor(self.save_img,cv.COLOR_BGR2RGB)
        self.reload(cvt)
        
    def cvttoGray(self):
        cvt = cv.cvtColor(self.original_img,cv.COLOR_BGR2GRAY)
        self.save_img = cvt
        self.reload(cvt)


    #影像二值化
    def fun_Threshoulding(self):
        cv.namedWindow('Threshold Track Bar',)
        cv.resizeWindow('Threshold Track Bar',200,245)
        cv.createTrackbar("Thresh", "Threshold Track Bar",0,255, trackChaned)
        cv.createTrackbar("Min", "Threshold Track Bar",0,255, trackChaned)
        tmp = self.img.copy()
        tmp = cv.cvtColor(tmp,cv.COLOR_BGR2GRAY)
        while (True):
            hul=cv.getTrackbarPos("Thresh", "Threshold Track Bar")
            huh=cv.getTrackbarPos("Min", "Threshold Track Bar")
            ret,thresh1 = cv.threshold(tmp,hul,huh,cv.THRESH_BINARY)
            cv.imshow("Threshold Track Bar",thresh1)
            if cv.waitKey(1) == ord("c"):
                threshold, thresh = cv.threshold(tmp, hul, huh,cv.THRESH_BINARY)
                self.save_img = thresh
                thresh = cv.cvtColor(thresh,cv.COLOR_BGR2RGB)
                cv.destroyAllWindows()
                break
        self.reload(thresh)
        
#直方圖等化    
    def fun_EqualizeHist(self):
        (b,g,r) = cv.split(self.img)
        bH = cv.equalizeHist(b)
        gH = cv.equalizeHist(g)
        rH = cv.equalizeHist(r)
        result = cv.merge((bH,gH,rH))
        self.save_img = result
        result = cv.cvtColor(result,cv.COLOR_BGR2RGB)
        self.reload(result)

#直方圖
    def calHist(self):
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist = cv.calcHist([self.img], [i], None, [256], [0,256])
            plt.title("Image Size:" + str(self.size))
            plt.plot(hist, color=col)
            plt.xlim([0, 256])
        plt.show()

#中值濾波
    def Median_filter(self):
        blur = cv.medianBlur(self.img,11)
        self.save_img = blur
        blur = cv.cvtColor(blur,cv.COLOR_BGR2RGB)
        self.reload(blur)

#均值濾波
    def averaging_filter(self):
        blur = cv.blur(self.img,(11,11))
        self.save_img = blur
        blur = cv.cvtColor(blur,cv.COLOR_BGR2RGB)
        self.reload(blur)

#高斯濾波
    def Gaussian_filter(self):
        blur = cv.GaussianBlur(self.img,(11,11),-1)
        self.save_img = blur
        blur = cv.cvtColor(blur,cv.COLOR_BGR2RGB)
        self.reload(blur)
        
#索伯算子
    def sobel_filter(self):
        # Sobel filter
        x = cv.Sobel(self.img, cv.CV_16S, 1, 0)
        y = cv.Sobel(self.img, cv.CV_16S, 0, 1)
        abs_x = cv.convertScaleAbs(x) 
        abs_y = cv.convertScaleAbs(y)
        img_sobel = cv.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
        self.save_img = img_sobel
        img_sobel = cv.cvtColor(img_sobel,cv.COLOR_BGR2RGB)
        self.reload(img_sobel)

#拉普拉斯
    def laplacian_filter(self):
        gray_lap = cv.Laplacian(self.img, cv.CV_16S, ksize=3)
        img_laplacian = cv.convertScaleAbs(gray_lap)
        self.save_img = img_laplacian
        img_laplacian = cv.cvtColor(img_laplacian,cv.COLOR_BGR2RGB)
        self.reload(img_laplacian)
            
            
#仿射轉換-平移
    def Affine_Transform(self):
        H = np.float32([[1,0,100],[0,1,50]])
        rows,cols = self.img.shape[:2]
        res = cv.warpAffine(self.img,H,(rows,cols))
        self.save_img = res
        res = cv.cvtColor(res,cv.COLOR_BGR2RGB)
        self.reload(res)
        
#仿射轉換-旋轉    
    def Affine_Transform_Rotate(self):
        rows,cols = self.img.shape[:2]
        M = cv.getRotationMatrix2D((cols/2,rows/2),45,1)
        res = cv.warpAffine(self.img,M,(rows,cols))
        self.save_img = res
        res = cv.cvtColor(res,cv.COLOR_BGR2RGB)
        self.reload(res)
        
#透視變換
    def Perspective_Transformation(self):
        global counter , point_matrix
        tmp_img = self.img.copy()
        counter = 0
        point_matrix = np.zeros((4,2),np.int)
        while True:
            for x in range (0,4):
                cv.circle(tmp_img,(point_matrix[x][0],point_matrix[x][1]),3,(0,255,0),cv.FILLED)
 
            if counter == 4:
                left_up_x = point_matrix[0][0]
                left_up_y = point_matrix[0][1]
                left_down_x = point_matrix[1][0]
                left_down_y = point_matrix[1][1]
                right_up_x = point_matrix[2][0]
                right_up_y = point_matrix[2][1]
                right_down_x = point_matrix[3][0]
                right_down_y = point_matrix[3][1]
                
            cv.imshow("Perspective", tmp_img)
            cv.setMouseCallback("Perspective",mousePoints)
            
            if cv.waitKey(1) == ord('c'):
                cv.destroyAllWindows()
                break
            
        pts_o = np.float32([[left_up_x, left_up_y], [left_down_x, left_down_y], [right_up_x, right_up_y], [right_down_x, right_down_y]])
        pts_d = np.float32([[0, 0], [600, 0], [0, 600], [600, 600]])
        M = cv.getPerspectiveTransform(pts_o,pts_d)
        dst = cv.warpPerspective(self.img,M,(600,600))
        self.save_img = dst
        dst = cv.cvtColor(dst,cv.COLOR_BGR2RGB)
        self.reload(dst)

        