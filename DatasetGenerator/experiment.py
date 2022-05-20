# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

import os
import sys
import random
import csv
import pandas as pd
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.label_cur = 0
        
        
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1080, 640)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(100, 140, 201, 201))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.PicPos_P0 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.PicPos_P0.setObjectName("PicPos_P0")
        self.verticalLayout.addWidget(self.PicPos_P0)
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(770, 140, 201, 201))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.PicPos_P1 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.PicPos_P1.setObjectName("PicPos_P1")
        self.verticalLayout_2.addWidget(self.PicPos_P1)
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(120, 370, 160, 51))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.LeftButton = QtWidgets.QPushButton(self.verticalLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.LeftButton.sizePolicy().hasHeightForWidth())
        self.LeftButton.setSizePolicy(sizePolicy)
        self.LeftButton.setObjectName("LeftButton")
        self.LeftButton.clicked.connect(self.click_leftButton)
        self.LeftButton.setEnabled(False)
        
        self.verticalLayout_3.addWidget(self.LeftButton)
        self.verticalLayoutWidget_4 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_4.setGeometry(QtCore.QRect(790, 370, 160, 51))
        self.verticalLayoutWidget_4.setObjectName("verticalLayoutWidget_4")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_4)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.RightButton = QtWidgets.QPushButton(self.verticalLayoutWidget_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.RightButton.sizePolicy().hasHeightForWidth())
        self.RightButton.setSizePolicy(sizePolicy)
        self.RightButton.setObjectName("RightButton")
        self.RightButton.clicked.connect(self.click_rightButton)
        self.RightButton.setEnabled(False)
        
        self.verticalLayout_4.addWidget(self.RightButton)
        self.verticalLayoutWidget_5 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_5.setGeometry(QtCore.QRect(460, 480, 160, 51))
        self.verticalLayoutWidget_5.setObjectName("verticalLayoutWidget_5")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_5)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.DownButton = QtWidgets.QPushButton(self.verticalLayoutWidget_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.DownButton.sizePolicy().hasHeightForWidth())
        self.DownButton.setSizePolicy(sizePolicy)
        self.DownButton.setObjectName("DownButton")
        self.DownButton.clicked.connect(self.start_experiment)
        
        
        self.verticalLayout_5.addWidget(self.DownButton)
        self.verticalLayoutWidget_6 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_6.setGeometry(QtCore.QRect(440, 140, 201, 201))
        self.verticalLayoutWidget_6.setObjectName("verticalLayoutWidget_6")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_6)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.PicPos_O = QtWidgets.QLabel(self.verticalLayoutWidget_6)
        self.PicPos_O.setObjectName("PicPos_O")
        self.verticalLayout_6.addWidget(self.PicPos_O)
        self.verticalLayoutWidget_7 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_7.setGeometry(QtCore.QRect(900, 530, 160, 51))
        self.verticalLayoutWidget_7.setObjectName("verticalLayoutWidget_7")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_7)
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.BackButton = QtWidgets.QPushButton(self.verticalLayoutWidget_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.BackButton.sizePolicy().hasHeightForWidth())
        self.BackButton.setSizePolicy(sizePolicy)
        self.BackButton.setObjectName("BackButton")
        self.BackButton.clicked.connect(self.click_backButton)
        self.BackButton.setEnabled(False)
        
        self.verticalLayout_7.addWidget(self.BackButton)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1080, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Dataset_Experiment"))
        self.PicPos_P0.setText(_translate("MainWindow", "Welcome"))
        self.PicPos_P1.setText(_translate("MainWindow", "Welcome"))
        self.LeftButton.setText(_translate("MainWindow", "PushButton"))
        self.RightButton.setText(_translate("MainWindow", "PushButton"))
        self.DownButton.setText(_translate("MainWindow", "Start Next Part"))
        self.PicPos_O.setText(_translate("MainWindow", "Welcome"))
        self.BackButton.setText(_translate("MainWindow", "I WANNA GO BACK"))

    def changePic(self, num_in_Piclist=0):
        
        o_path = 'Test_Pics/'+self.label[self.label_cur]+'/o'
        p0_path = 'Test_Pics/'+self.label[self.label_cur]+'/p0'
        p1_path = 'Test_Pics/'+self.label[self.label_cur]+'/p1'
        
        o_img_path = os.path.join(o_path, str(num_in_Piclist).zfill(5)+'.png')
        p0_img_path = os.path.join(p0_path, str(num_in_Piclist).zfill(5)+'.png')
        p1_img_path = os.path.join(p1_path, str(num_in_Piclist).zfill(5)+'.png')
        
        o_img = QtGui.QPixmap(o_img_path).scaled(self.PicPos_O.width(), self.PicPos_O.height())
        p0_img = QtGui.QPixmap(p0_img_path).scaled(self.PicPos_P0.width(), self.PicPos_P0.height())
        p1_img = QtGui.QPixmap(p1_img_path).scaled(self.PicPos_P1.width(), self.PicPos_P1.height())
        
        self.PicPos_O.setPixmap(o_img)
        self.PicPos_P0.setPixmap(p0_img)
        self.PicPos_P1.setPixmap(p1_img)

    def start_experiment(self):
        self.label = ['albedo', 'lit', 'normal']
        
        self.test_total_num = 20
        self.Pic_list = random.sample(range(0,1000),self.test_total_num)
        f_groundtruth = pd.read_csv(f'Test_Pics/{self.label[self.label_cur]}/GroundTruth.csv', header=None)
        self.groundtruth = []
        for i in self.Pic_list:
            self.groundtruth.append(f_groundtruth.iloc[i,0])
        
        self.test_cur_num=0
        self.ans = []
        self.changePic(self.Pic_list[self.test_cur_num])
        # print(self.Pic_list[self.test_cur_num])
        self.DownButton.setEnabled(False)
        self.LeftButton.setEnabled(True)
        self.RightButton.setEnabled(True)
        self.BackButton.setEnabled(True)
        # pass
    
    def click_leftButton(self):
        self.ans.append(0)
        print(self.test_cur_num)
        self.test_cur_num = self.test_cur_num+1
        if self.test_cur_num<self.test_total_num:
            self.changePic(self.Pic_list[self.test_cur_num])
        else :
            
            self.finish_experiment()
        
        
    def click_rightButton(self):
        self.ans.append(1)
        self.test_cur_num = self.test_cur_num+1
        if self.test_cur_num<self.test_total_num:
            self.changePic(self.Pic_list[self.test_cur_num])
        else :
            self.finish_experiment()
        # print(self.test_cur_num)
        
    def click_backButton(self):
        if len(self.ans)!=0:
            self.ans.pop()
            self.test_cur_num = self.test_cur_num-1
            self.changePic(self.Pic_list[self.test_cur_num])
    
    def restart_experiment(self):
        self.Pic_list = random.sample(range(0,1000),self.test_total_num)
        f_groundtruth = pd.read_csv(f'Test_Pics/{self.label[self.label_cur]}/GroundTruth.csv', header=None)
        self.groundtruth = []
        for i in self.Pic_list:
            self.groundtruth.append(f_groundtruth.iloc[i,0])
        
        self.test_cur_num=0
        self.ans = []
        self.changePic(self.Pic_list[self.test_cur_num])
        # print(self.Pic_list[self.test_cur_num])
        self.DownButton.setEnabled(False)
        self.LeftButton.setEnabled(True)
        self.RightButton.setEnabled(True)
        self.BackButton.setEnabled(True)
    
    def finish_experiment(self):
        _translate = QtCore.QCoreApplication.translate
        all_results_folder = 'results'
        if self.label_cur==0:            
            self.csv_num = len(os.listdir(all_results_folder))
        csv_folder = os.path.join(all_results_folder, str(self.csv_num).zfill(5))
        os.makedirs(csv_folder, exist_ok=True)
        csv_path = os.path.join(csv_folder, f'{self.label[self.label_cur]}results.csv')

            
        if not os.path.exists(csv_path):
            print("Create New CSV File!")
            with open(csv_path, "w") as csvfile:
                file = csv.writer(csvfile)
                file.writerow(self.groundtruth)
                file.writerow(self.ans)
        else:
            print("Open Exist CSV File!")
            with open(csv_path, "a") as csvfile:
                file = csv.writer(csvfile)
                file.writerow(self.groundtruth)
                file.writerow(self.ans)
        self.LeftButton.setEnabled(False)
        self.RightButton.setEnabled(False)
        self.BackButton.setEnabled(False)
        print(self.label[self.label_cur])
        self.label_cur = self.label_cur+1
        if self.label_cur<3:
            self.PicPos_O.setText(_translate("MainWindow", "Part Finished"))
            self.PicPos_P0.setText(_translate("MainWindow", "Part Finished"))
            self.PicPos_P1.setText(_translate("MainWindow", "Part Finished"))
            self.DownButton.setEnabled(True)
            self.DownButton.clicked.connect(self.restart_experiment)
        else:
            self.PicPos_O.setText(_translate("MainWindow", "All Finished"))
            self.PicPos_P0.setText(_translate("MainWindow", "All Finished"))
            self.PicPos_P1.setText(_translate("MainWindow", "All Finished"))
        
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


