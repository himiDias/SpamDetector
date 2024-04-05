import email
from re import L
import numpy as num
import pandas as pd
import sys
from PySide2.QtCore import QSize, Qt
from PySide2.QtWidgets import QApplication,QMainWindow,QPushButton,QLineEdit,QWidget,QLabel,QVBoxLayout,QGridLayout

 

class MainWin(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Spam Filter")
        self.setMinimumSize(QSize(700,400))
        self.setMaximumSize(QSize(900,600))

        self.email = QLineEdit()
        self.email.setPlaceholderText("Email Address")
        

        self.subject = QLineEdit()
        self.subject.setPlaceholderText("Subject")
        

        self.content = QLineEdit()
        self.content.setPlaceholderText("Content")
        
        enter = QPushButton("Enter")
        enter.setFixedSize(QSize(100,25))
        enter.setCheckable(True)
        enter.clicked.connect(self.data_entered)
        
 

        layout = QGridLayout()
        layout.addWidget(self.email,0,0)
        layout.addWidget(self.subject,0,1) 
        layout.addWidget(self.content,1,0)
        layout.addWidget(enter)


        email = QWidget()
        email.setLayout(layout)



        #self.setCentralWidget(enter)
        self.setCentralWidget(email)
    
    def data_entered(self):
        email_address = self.email.text()
        e_subject = self.subject.text()
        e_content = self.content.text()
        print("email : ",email_address,"::::::::::::: subject : ",e_subject, ":::::::::::::: content : ",e_content)
        list_of_words = e_content.split()
        print(list_of_words)



myApp = QApplication(sys.argv)
window = MainWin()
window.show()

myApp.exec_()

 