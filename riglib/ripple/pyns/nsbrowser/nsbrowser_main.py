# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ns_browser_frame_layout_v2.ui'
#
# Created: Tue May 15 20:23:16 2012
#      by: PyQt4 UI code generator 4.9.1
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_NSBrowserMainWindow(object):
    def setupUi(self, NSBrowserMainWindow):
        NSBrowserMainWindow.setObjectName(_fromUtf8("NSBrowserMainWindow"))
        NSBrowserMainWindow.resize(856, 524)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(NSBrowserMainWindow.sizePolicy().hasHeightForWidth())
        NSBrowserMainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtGui.QWidget(NSBrowserMainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.splitter = QtGui.QSplitter(self.centralwidget)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName(_fromUtf8("splitter"))
        self.layoutWidget = QtGui.QWidget(self.splitter)
        self.layoutWidget.setObjectName(_fromUtf8("layoutWidget"))
        self.gridLayout = QtGui.QGridLayout(self.layoutWidget)
        self.gridLayout.setMargin(0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.segment_label = QtGui.QLabel(self.layoutWidget)
        self.segment_label.setObjectName(_fromUtf8("segment_label"))
        self.gridLayout.addWidget(self.segment_label, 0, 0, 1, 1)
        self.item_label = QtGui.QLabel(self.layoutWidget)
        self.item_label.setObjectName(_fromUtf8("item_label"))
        self.gridLayout.addWidget(self.item_label, 0, 1, 1, 1)
        self.segment_list = QtGui.QListWidget(self.layoutWidget)
        self.segment_list.setObjectName(_fromUtf8("segment_list"))
        self.gridLayout.addWidget(self.segment_list, 1, 0, 1, 1)
        self.item_list = QtGui.QListWidget(self.layoutWidget)
        self.item_list.setObjectName(_fromUtf8("item_list"))
        self.gridLayout.addWidget(self.item_list, 1, 1, 1, 1)
        self.analog_label = QtGui.QLabel(self.layoutWidget)
        self.analog_label.setObjectName(_fromUtf8("analog_label"))
        self.gridLayout.addWidget(self.analog_label, 2, 0, 1, 1)
        self.event_label = QtGui.QLabel(self.layoutWidget)
        self.event_label.setObjectName(_fromUtf8("event_label"))
        self.gridLayout.addWidget(self.event_label, 2, 1, 1, 1)
        self.analog_list = QtGui.QListWidget(self.layoutWidget)
        self.analog_list.setObjectName(_fromUtf8("analog_list"))
        self.gridLayout.addWidget(self.analog_list, 3, 0, 1, 1)
        self.event_list = QtGui.QListWidget(self.layoutWidget)
        self.event_list.setObjectName(_fromUtf8("event_list"))
        self.gridLayout.addWidget(self.event_list, 3, 1, 1, 1)
        self.file_info_button = QtGui.QPushButton(self.layoutWidget)
        self.file_info_button.setObjectName(_fromUtf8("file_info_button"))
        self.gridLayout.addWidget(self.file_info_button, 4, 0, 1, 1)
        self.entity_info_button = QtGui.QPushButton(self.layoutWidget)
        self.entity_info_button.setObjectName(_fromUtf8("entity_info_button"))
        self.gridLayout.addWidget(self.entity_info_button, 4, 1, 1, 1)
        self.mpl = MplWidget(self.splitter)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mpl.sizePolicy().hasHeightForWidth())
        self.mpl.setSizePolicy(sizePolicy)
        self.mpl.setObjectName(_fromUtf8("mpl"))
        self.horizontalLayout.addWidget(self.splitter)
        NSBrowserMainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(NSBrowserMainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 856, 27))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuMy_program = QtGui.QMenu(self.menubar)
        self.menuMy_program.setObjectName(_fromUtf8("menuMy_program"))
        self.menuHelp = QtGui.QMenu(self.menubar)
        self.menuHelp.setObjectName(_fromUtf8("menuHelp"))
        NSBrowserMainWindow.setMenuBar(self.menubar)
        self.action_open = QtGui.QAction(NSBrowserMainWindow)
        self.action_open.setObjectName(_fromUtf8("action_open"))
        self.action_exit = QtGui.QAction(NSBrowserMainWindow)
        self.action_exit.setObjectName(_fromUtf8("action_exit"))
        self.action_about = QtGui.QAction(NSBrowserMainWindow)
        self.action_about.setObjectName(_fromUtf8("action_about"))
        self.menuMy_program.addAction(self.action_open)
        self.menuMy_program.addSeparator()
        self.menuMy_program.addAction(self.action_exit)
        self.menuHelp.addAction(self.action_about)
        self.menubar.addAction(self.menuMy_program.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(NSBrowserMainWindow)
        QtCore.QMetaObject.connectSlotsByName(NSBrowserMainWindow)

    def retranslateUi(self, NSBrowserMainWindow):
        NSBrowserMainWindow.setWindowTitle(QtGui.QApplication.translate("NSBrowserMainWindow", "Neuroshare File Browser", None, QtGui.QApplication.UnicodeUTF8))
        self.segment_label.setText(QtGui.QApplication.translate("NSBrowserMainWindow", "Segments", None, QtGui.QApplication.UnicodeUTF8))
        self.item_label.setText(QtGui.QApplication.translate("NSBrowserMainWindow", "Segment Items", None, QtGui.QApplication.UnicodeUTF8))
        self.segment_list.setWhatsThis(QtGui.QApplication.translate("NSBrowserMainWindow", "<html><head/><body><p>Hello</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.analog_label.setText(QtGui.QApplication.translate("NSBrowserMainWindow", "Analog", None, QtGui.QApplication.UnicodeUTF8))
        self.event_label.setText(QtGui.QApplication.translate("NSBrowserMainWindow", "Digital Events", None, QtGui.QApplication.UnicodeUTF8))
        self.file_info_button.setText(QtGui.QApplication.translate("NSBrowserMainWindow", "Get File Info", None, QtGui.QApplication.UnicodeUTF8))
        self.entity_info_button.setText(QtGui.QApplication.translate("NSBrowserMainWindow", "Get Entity Info", None, QtGui.QApplication.UnicodeUTF8))
        self.menuMy_program.setTitle(QtGui.QApplication.translate("NSBrowserMainWindow", "File", None, QtGui.QApplication.UnicodeUTF8))
        self.menuHelp.setTitle(QtGui.QApplication.translate("NSBrowserMainWindow", "Help", None, QtGui.QApplication.UnicodeUTF8))
        self.action_open.setText(QtGui.QApplication.translate("NSBrowserMainWindow", "Open...", None, QtGui.QApplication.UnicodeUTF8))
        self.action_exit.setText(QtGui.QApplication.translate("NSBrowserMainWindow", "Exit", None, QtGui.QApplication.UnicodeUTF8))
        self.action_about.setText(QtGui.QApplication.translate("NSBrowserMainWindow", "About", None, QtGui.QApplication.UnicodeUTF8))

from mplwidget import MplWidget
