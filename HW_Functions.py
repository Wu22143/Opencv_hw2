from fileinput import filename
from glob import glob
from time import sleep
from PIL import ImageTk, Image
from cProfile import label
import tkinter as tk , numpy as np
from cv2 import cvtColor, getHardwareFeatureName, threshold
from matplotlib import pyplot as plt
import cv2 as cv
from tkinter import filedialog , messagebox
import tkinter
import math ,random

from scipy.fft import dst

def trackChaned(x):
    pass

point_matrix = np.zeros((4,2),np.int32)
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
        point_matrix = np.zeros((4,2),np.int32)
        while True:
            for x in range (0,4):
                cv.circle(tmp_img,(point_matrix[x][0],point_matrix[x][1]),3,(0,255,0),cv.FILLED)
 
            if counter == 4:
                left_up_x = point_matrix[0][0]
                left_up_y = point_matrix[0][1]
                right_up_x = point_matrix[1][0]
                right_up_y = point_matrix[1][1]
                right_down_x = point_matrix[2][0]
                right_down_y = point_matrix[2][1]
                left_down_x = point_matrix[3][0]
                left_down_y = point_matrix[3][1]
                
            cv.imshow("Perspective", tmp_img)
            cv.setMouseCallback("Perspective",mousePoints)
            
            if cv.waitKey(1) == ord('c'):
                cv.destroyAllWindows()
                break
            
        pts_o = np.float32([[left_up_x, left_up_y],[right_up_x, right_up_y],[left_down_x, left_down_y],[right_down_x, right_down_y]])
        pts_d = np.float32([[0, 0], [600, 0], [0, 600], [600, 600]])
        M = cv.getPerspectiveTransform(pts_o,pts_d)
        dst = cv.warpPerspective(self.img,M,(600,600))
        self.save_img = dst
        dst = cv.cvtColor(dst,cv.COLOR_BGR2RGB)
        self.reload(dst)

#canny detector Week8
    def canny_detector(self):
        ratio = 3
        kernel_size = 3
        # the callback function for trackbar
        cv.namedWindow('Edge Map',)
        cv.createTrackbar('Min Threshold', 'Edge Map', 0, 100, trackChaned)
        src = self.img.copy()
        src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        while True:
            low_threshold = cv.getTrackbarPos("Min Threshold", "Edge Map")
            print(low_threshold)
            img_blur = cv.blur(src_gray, (3, 3))
            detected_edges = cv.Canny(img_blur, low_threshold, low_threshold * ratio, kernel_size)
            mask = detected_edges != 0
            dst = src * (mask[:, :, None].astype(src.dtype))
            cv.imshow('Edge Map', dst)
            if cv.waitKey(1) == ord("c"):
                self.save_img = dst
                dst = cv.cvtColor(dst,cv.COLOR_BGR2RGB)
                self.reload(dst)
                break

#霍夫轉換
    def hough_transform(self):
        src = self.img.copy()
        dst = cv.Canny(src,50,200,None,3)

        cdstP = np.copy(self.img)

        lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv.line(src, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

        linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

        cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", src)
        cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

    #coner harris week11

    def cornerHarris_event_handler(self,val):
        print(1)
        thresh = val
        blockSize = 2
        apertureSize = 3
        k = 0.04
        src = cv.cvtColor(self.img,cv.COLOR_BGR2GRAY)

        dst = cv.cornerHarris(src, blockSize, apertureSize, k)

        dst_norm = np.empty(dst.shape, dtype=np.float32)
        cv.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        dst_norm_scaled = cv.convertScaleAbs(dst_norm)

        for i in range(dst_norm.shape[0]):
            for j in range(dst_norm.shape[1]):
                if int(dst_norm[i, j]) > thresh:
                    cv.circle(dst_norm_scaled, (j, i), 5, (0), 2)

        cv.namedWindow('Source image')
        cv.imshow('Corners detected', dst_norm_scaled)

    def conrer_Harris(self):
        src = cv.cvtColor(self.img,cv.COLOR_BGR2GRAY)
        source_window = 'Source image'
        max_thresh = 255
        thresh = 200
        cv.namedWindow(source_window)
        cv.createTrackbar('Threshold: ',source_window,200,max_thresh,self.cornerHarris_event_handler)
        cv.imshow(source_window,src)
        self.cornerHarris_event_handler(thresh)
        cv.waitKey()

#week12 add simple_contour , find_contour , bounding_box
    def simple_contour(self):
        src = self.img.copy()
        src_gray = cv.cvtColor(src,cv.COLOR_BGR2GRAY)
        
        ret, src_thresh = cv.threshold(src_gray,127,255,cv.THRESH_BINARY)
        cv.imshow('Threshold image',src_thresh)

        contours, hierarchy = cv.findContours(src_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contour_all = cv.drawContours(image=src, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
        cv.imshow('Contours', contour_all)
        cv.waitKey()


    def find_contour(self):
        def contour_threshold_callback(val):
            threshold = val
            canny_output = cv.Canny(src_gray, threshold, threshold * 2)
            contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            # Draw contours
            drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
            for i in range(len(contours)):
                color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
                cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
            cv.imshow('Contours', drawing)

        src = self.img.copy()
        src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        src_gray = cv.blur(src_gray, (3, 3))

        source_window = 'Source image'
        cv.namedWindow(source_window)
        cv.imshow(source_window, src)
        max_thresh = 255
        thresh = 100
        cv.createTrackbar('Threshold: ', source_window, thresh, max_thresh, contour_threshold_callback)
        contour_threshold_callback(thresh)
        cv.waitKey()

    def bounding_box(self):
        def bounding_box_callback(val):
            threshold = val
            contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            contours_poly = [None] * len(contours)
            boundRect = [None] * len(contours)
            centers = [None] * len(contours)
            radius = [None] * len(contours)
            for i, c in enumerate(contours):
                contours_poly[i] = cv.approxPolyDP(c, 3, True)
                boundRect[i] = cv.boundingRect(contours_poly[i])
                centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])

            drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

            for i in range(len(contours)):
                color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
                cv.drawContours(drawing, contours_poly, i, color)
                cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])),
                             (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
            cv.imshow('Contours', drawing)

        src = self.img.copy()
        src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        src_gray = cv.blur(src_gray, (3, 3))

        max_thresh = 255
        thresh = 100
        source_window = 'Source image'
        cv.namedWindow(source_window)
        canny_output = cv.Canny(src_gray, thresh, thresh * 2)
        cv.imshow(source_window, src)
        cv.createTrackbar('Threshold: ', source_window, thresh, max_thresh, bounding_box_callback)
        bounding_box_callback(thresh)
        cv.waitKey()


    def convex_hull(self):
        def convex_hull_callback(val):
            threshold = val
            canny_output = cv.Canny(src_gray, threshold, threshold * 2)

            contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            hull_list = []
            for i in range(len(contours)):
                hull = cv.convexHull(contours[i])
                hull_list.append(hull)

            drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
            for i in range(len(contours)):
                color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
                cv.drawContours(drawing, contours, i, color)
                cv.drawContours(drawing, hull_list, i, color)
            cv.imshow('Contours', drawing)

        src = self.img.copy()
        src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        src_gray = cv.blur(src_gray, (3, 3))

        source_window = 'Source image'
        cv.namedWindow(source_window)
        cv.imshow(source_window, src)
        max_thresh = 255
        thresh = 100
        cv.createTrackbar('Threshold: ', source_window, thresh, max_thresh, convex_hull_callback)
        convex_hull_callback(thresh)
        cv.waitKey()