# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'info_dialog.ui'
#
# Created: Tue May 15 23:28:45 2012
#      by: PyQt4 UI code generator 4.9.1
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_NSInfoDialog(object):
    def setupUi(self, NSInfoDialog):
        NSInfoDialog.setObjectName(_fromUtf8("NSInfoDialog"))
        NSInfoDialog.resize(610, 485)
        self.verticalLayout = QtGui.QVBoxLayout(NSInfoDialog)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.info_text = QtGui.QTextBrowser(NSInfoDialog)
        self.info_text.setObjectName(_fromUtf8("info_text"))
        self.verticalLayout.addWidget(self.info_text)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.okay_button = QtGui.QPushButton(NSInfoDialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.okay_button.sizePolicy().hasHeightForWidth())
        self.okay_button.setSizePolicy(sizePolicy)
        self.okay_button.setObjectName(_fromUtf8("okay_button"))
        self.horizontalLayout.addWidget(self.okay_button)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(NSInfoDialog)
        QtCore.QMetaObject.connectSlotsByName(NSInfoDialog)

    def retranslateUi(self, NSInfoDialog):
        NSInfoDialog.setWindowTitle(QtGui.QApplication.translate("NSInfoDialog", "Info", None, QtGui.QApplication.UnicodeUTF8))
        self.okay_button.setText(QtGui.QApplication.translate("NSInfoDialog", "Okay", None, QtGui.QApplication.UnicodeUTF8))

