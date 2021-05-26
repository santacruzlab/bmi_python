# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'nsbrowser_about.ui'
#
# Created: Tue May 15 20:34:28 2012
#      by: PyQt4 UI code generator 4.9.1
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_NSAboutDialog(object):
    def setupUi(self, NSAboutDialog):
        NSAboutDialog.setObjectName(_fromUtf8("NSAboutDialog"))
        NSAboutDialog.resize(551, 334)
        self.label = QtGui.QLabel(NSAboutDialog)
        self.label.setGeometry(QtCore.QRect(160, 10, 381, 281))
        self.label.setWordWrap(True)
        self.label.setObjectName(_fromUtf8("label"))
        self.okay_button = QtGui.QPushButton(NSAboutDialog)
        self.okay_button.setGeometry(QtCore.QRect(430, 290, 108, 27))
        self.okay_button.setObjectName(_fromUtf8("okay_button"))

        self.retranslateUi(NSAboutDialog)
        QtCore.QMetaObject.connectSlotsByName(NSAboutDialog)

    def retranslateUi(self, NSAboutDialog):
        NSAboutDialog.setWindowTitle(QtGui.QApplication.translate("NSAboutDialog", "About Neuroshare File Browser", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("NSAboutDialog", "<html><head/><body><p><span style=\" font-size:14pt; font-weight:600;\">Neuroshare File Browser</span></p>\n"
"<p>Neuroshare File Browser provides a simple interface to browse through NEV and NSx files (versions 2.1 and 2.2) using the pyns version of the Neuroshare API.</p>\n"
"<p>The purpose of this program is to provide a simple but useful program to look at data but be easily extendable by users.</p><p>This program makes use of Qt4 and PyQt binding.</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.okay_button.setText(QtGui.QApplication.translate("NSAboutDialog", "Okay", None, QtGui.QApplication.UnicodeUTF8))

