#   Copyright 2018 by Paolo Inglese, National Phenome Centre, Imperial College
#   London
#   All rights reserved.
#   This file is part of DESI-MSI recalibration, and is released under the
#   "MIT License Agreement".
#   Please see the LICENSE file that should have been included as part of this
#   package.

import os
import sys
import time
from enum import Enum
from typing import Union

import cv2
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PyQt5.Qt import *
from PyQt5.QtGui import *
from scipy.stats import pearsonr
from skimage import measure
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale, scale
from sklearn.svm import SVC

from tools.binner import MSBinner
from tools.gui import ImageWithColorbar
from tools.msi import MSI
from tools import qrc_resources


def transform_data(intmat: np.ndarray) -> np.ndarray:
    zero_rows = (np.sum(intmat, axis=1) == 0)
    intmat[intmat == 0] = np.nan
    sf = np.nanmedian(intmat[~zero_rows, :], axis=1)
    intmat[~zero_rows, :] /= sf[:, None]
    intmat[~zero_rows, :] *= np.nanmean(sf)
    intmat[np.isnan(intmat)] = 0
    intmat = np.log(intmat + 1.0)
    # Remove const vars
    intmat = intmat[:, np.var(intmat, axis=0) != 0]
    return intmat


def pca_scores(intmat: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=3)
    const_vars = (np.var(intmat, axis=0) == 0)
    scores = pca.fit_transform(scale(intmat[:, ~const_vars]))
    if pearsonr(scores[:, 0], np.mean(intmat, axis=1))[0] < 0:
        scores = -1 * scores
    return scores


def rem_small_obj_roi(roi_mat: np.ndarray, min_size: int) -> np.ndarray:
    bw_labels = measure.label(roi_mat != 0)
    unique_labels = np.unique(bw_labels[bw_labels != 0])
    areas = np.empty(0)
    for lx in unique_labels:
        areas = np.append(areas, np.sum(bw_labels == lx))
    small_objs = [unique_labels[i] for i in range(len(areas)) if
                  areas[i] < min_size]
    for lx in small_objs:
        roi_mat[bw_labels == lx] = False

    return roi_mat


class LabelSignal(Enum):
    BACKGROUND = 0
    SAMPLE = 1


# noinspection PyUnresolvedReferences
class PredictThread(QThread):
    curr_operation = pyqtSignal(str)
    finished = pyqtSignal()
    mask_: np.ndarray
    intmat_: np.ndarray

    def __init__(self, mask_, yimat, msi_dim_xy, parent=None):
        QThread.__init__(self, parent)
        self.threadactive = True
        self.mask_ = mask_
        self.X = yimat
        self.msi_dim_xy = msi_dim_xy

    def run(self):
        if self.threadactive:
            self.curr_operation.emit('Predicting ...')
            mask_all = cv2.resize(
                self.mask_, (self.msi_dim_xy[1], self.msi_dim_xy[2]),
                interpolation=cv2.INTER_NEAREST)
            lbl = np.asarray(mask_all, dtype=int).ravel()
            lbl = np.digitize(lbl, bins=[0, 1, 2], right=True) - 1
            svm = SVC(kernel='linear')
            svm.fit(self.X[lbl != 0, :], lbl[lbl != 0])
            lbl[lbl == 0] = svm.predict(self.X[lbl == 0, :])
            lbl = lbl.reshape((self.msi_dim_xy[2], self.msi_dim_xy[1]))
            self.mask_ = lbl

        self.curr_operation.emit('Done!')
        self.threadactive = False
        self.finished.emit()

    def stop(self):
        self.threadactive = False
        self.quit()
        self.wait()


# noinspection PyUnresolvedReferences
class SaveThread(QThread):
    curr_operation = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()
    threadactive: bool

    def __init__(self, save_mask: np.ndarray, save_dir: str, parent) -> None:
        super(SaveThread, self).__init__(parent=parent)
        self.threadactive = True
        self.mask_ = save_mask
        self.save_dir = save_dir

    # noinspection PyTypeChecker
    def run(self) -> None:
        if self.threadactive:
            self.curr_operation.emit('Saving ROI ...')

            output_filename = os.path.join(self.save_dir, 'roi.csv')
            np.savetxt(fname=output_filename, X=self.mask_, delimiter=',',
                       fmt='%d')

            self.curr_operation.emit('ROI saved')
            print('ROI saved')
            time.sleep(2)

            self.threadactive = False
            self.finished.emit()

    def stop(self) -> None:
        self.threadactive = False
        self.quit()
        self.wait()


# noinspection PyUnresolvedReferences
class LoadPipeline(QThread):
    currsig = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal(list)
    threadactive: bool
    datapath: str

    def __init__(self, datapath: str, parent=None):
        super(LoadPipeline, self).__init__(parent=parent)
        self.binner = MSBinner(decimals=0)
        self.datapath = datapath
        self.threadactive = True

    def run(self) -> None:
        scores = None
        intmat = None
        if self.threadactive:
            self.currsig.emit('Loading MS data ...')
            # Ion mode and analyzer are irrelevant
            meta = {
                'ion_mode': 'ES-',
                'analyzer': 'TOF'
            }
            msi = MSI(imzml=self.datapath, meta=meta)
            self.currsig.emit('Uniform binning ...')
            intmat = self.binner.bin(msi)
            self.currsig.emit('Transforming data ...')
            intmat = transform_data(intmat)
            self.currsig.emit('PCA ...')
            scores = pca_scores(intmat)
            scores = np.reshape(scores, (msi.dim_xy[1], msi.dim_xy[0], 3))
        self.finished.emit([intmat, scores])
        self.threadactive = False

    def stop(self):
        self.threadactive = False
        self.quit()
        self.wait()


class BusySpinner(QWidget):
    def __init__(self, parent):
        super(BusySpinner, self).__init__(parent=parent)
        self.setFixedSize(200, 200)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.CustomizeWindowHint)

        self.label_operation = QLabel('Please wait ...')
        self.label_animation = QLabel()
        self.movie = QMovie(':busy.gif')
        self.movie.setScaledSize(QSize(24, 24))
        self.label_animation.setMovie(self.movie)
        self.label_animation.adjustSize()

        layout = QHBoxLayout(self)
        layout.addWidget(self.label_operation)
        layout.addWidget(self.label_animation)
        self.startAnimation()
        self.show()

    def startAnimation(self):
        self.movie.start()

    def stopAnimation(self):
        self.movie.stop()
        self.close()


# noinspection PyUnresolvedReferences
class LabelsBox(QWidget):
    signal_lbl = pyqtSignal(LabelSignal)

    def __init__(self, parent):
        super(LabelsBox, self).__init__(parent=parent)

        self.btn_bg = QRadioButton('Background')
        self.btn_bg.setChecked(True)
        self.btn_bg.toggled.connect(self.emit_value)
        self.btn_sm = QRadioButton('Sample')
        self.btn_sm.toggled.connect(self.emit_value)

        internal_widget = QGroupBox('Labels')
        layout_labels = QVBoxLayout(internal_widget)
        layout_labels.addWidget(self.btn_bg)
        # layout_labels.addWidget(self.btn_bg_col, 0, 1)
        layout_labels.addWidget(self.btn_sm)
        # layout_labels.addWidget(self.btn_sm_col, 1, 1)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(internal_widget)

    def emit_value(self):
        if self.btn_bg.isChecked():
            self.signal_lbl.emit(LabelSignal.BACKGROUND)
        else:
            self.signal_lbl.emit(LabelSignal.SAMPLE)


# noinspection PyUnresolvedReferences
class MainWindow(QMainWindow):
    busyWidget: BusySpinner
    comboLabel: Union[None, QComboBox]
    curr_label: int
    filename: Union[None, str]
    imageWidget: Union[None, ImageWithColorbar]
    interfaceWidget: Union[None, QWidget]
    intmat: Union[None, np.ndarray]
    load_thread: Union[None, LoadPipeline]
    mainWidget: Union[None, QWidget]
    mask_: Union[None, np.ndarray]
    qpixmap: Union[None, QPixmap]
    rgb_im: Union[None, np.ndarray]
    save_dir: Union[None, str]
    thread: Union[None, PredictThread]

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Actions --------------------------------------------------------------
        self.predictAction = QAction(
            QIcon(':neural.svg'), 'Predict regions', self)
        self.deleteAction = QAction(
            QIcon(':delete.svg'), 'Delete regions', self)
        self.eraseAction = QAction(QIcon(':eraser.svg'), 'Erase contour', self)
        self.exitAction = QAction(QIcon(':exit.svg'), 'Exit', self)
        self.loadAction = QAction(
            QIcon(':file-open.svg'), "Open raw peaks ...", self)

        # Buttons --------------------------------------------------------------
        self.btnSave = QPushButton('Save...')
        self.btnProcess = QPushButton('Process...')
        self.btnReset = QPushButton('Reset')
        self.btnDel = QPushButton('Del selection')
        self.btnAdd = QPushButton('Add selection')
        self.btnZoomIn = QPushButton('Zoom in')
        self.btnZoomOut = QPushButton('Zoom out')

        # Others ---------------------------------------------------------------
        self.boxMinRoi = QSpinBox()
        self.labels_widget = LabelsBox(self)
        self.pbar_ = QProgressBar(self)
        self.status_bar = QStatusBar()
        self.tools_toolbar = QToolBar('Tools')
        self.file_toolbar = QToolBar('File')

        # Main -----------------------------------------------------------------
        self.__height = 768
        self.__title = 'DESI-MSI: select ROI tool'
        self.__width = 1024
        self.__img_height = 500
        self.__img_width = 500
        self.busy_spinner = Union[None, BusySpinner]
        self.comboLabel = None
        self.curr_label = 0
        self.filename = None
        self.imageWidget = None
        self.interfaceWidget = None
        self.intmat = None
        self.load_thread = None
        self.mainWidget = None
        self.rgb_im = None
        self.msi_dim_xy = None
        self.save_dir = None
        self.thread = None

        self.init_actions()

        self.initMenu()
        self.initToolbar()
        self.initUI()

    # def closeEvent(self, event) -> None:
    #     if self.thread is not None:
    #         if self.thread.threadactive:
    #             self.thread.stop()
    #     self.deleteLater()

    def init_actions(self):
        self.loadAction.setShortcut("CTRL+O")
        self.loadAction.setStatusTip("Load h5 file containing raw peaks")
        self.loadAction.triggered.connect(self.open_file_dialog)

        self.exitAction.setShortcut('Ctrl+Q')
        self.exitAction.setStatusTip('Exit application')
        self.exitAction.triggered.connect(qApp.quit)

        self.eraseAction.setShortcut('Ctrl+E')
        self.eraseAction.setStatusTip('Erase current region contour')

        self.deleteAction.setShortcut('Ctrl+R')
        self.deleteAction.setStatusTip('Delete all regions')

        self.deleteAction.setShortcut('Ctrl+P')
        self.deleteAction.setStatusTip('Predict labels from selected regions')

    def initMenu(self):
        mainMenu = self.menuBar()
        mainMenu.setNativeMenuBar(False)

        fileMenu = mainMenu.addMenu('File')
        fileMenu.addAction(self.loadAction)
        fileMenu.addSeparator()
        fileMenu.addAction(self.exitAction)

        toolsMenu = mainMenu.addMenu('Tools')
        toolsMenu.addAction(self.eraseAction)
        toolsMenu.addAction(self.deleteAction)
        toolsMenu.addSeparator()
        toolsMenu.addAction(self.predictAction)

    def initToolbar(self):

        btn_draw = QToolButton()
        btn_draw.setIcon(QIcon(':edit.svg'))
        btn_draw.setCheckable(True)

        self.file_toolbar.setIconSize(QSize(16, 16))
        self.file_toolbar.addAction(self.loadAction)
        self.file_toolbar.addAction(self.exitAction)

        self.tools_toolbar.setIconSize(QSize(16, 16))
        self.tools_toolbar.addAction(self.eraseAction)
        self.tools_toolbar.addAction(self.deleteAction)
        self.tools_toolbar.addAction(self.predictAction)

        self.addToolBar(self.file_toolbar)
        self.addToolBar(self.tools_toolbar)

    def initUI(self):
        self.setGeometry(100, 100, self.__width, self.__height)
        self.move(100, 100)
        self.setWindowTitle(self.__title)

        self.setStatusBar(self.status_bar)

        self.mainWidget = QWidget()

        self.imageWidget = ImageWithColorbar(
            parent=self, title='Reference Image',
            colorbar_title=['PC1', 'PC2', 'PC3'])
        self.imageWidget.setDisabled(True)

        self.interfaceWidget = QWidget()

        self.labels_widget.signal_lbl.connect(self.update_sel_label)
        self.labels_widget.setEnabled(False)

        smallRoiWidget = QWidget()
        self.boxMinRoi.setMinimum(0)
        self.boxMinRoi.setValue(50)
        layout_small_roi = QFormLayout(smallRoiWidget)
        layout_small_roi.addRow(QLabel('Smallest ROI size:'), self.boxMinRoi)

        self.btnAdd.clicked.connect(self.add_selection)
        self.btnDel.clicked.connect(self.del_selection)
        self.btnReset.clicked.connect(self.reset_selection)
        self.btnProcess.clicked.connect(self.process_data)
        self.btnSave.clicked.connect(self.save_mask)
        self.btnZoomIn.clicked.connect(self.zoomin)
        self.btnZoomOut.clicked.connect(self.zoomout)

        self.resetInterface(enable=False)

        vspacer = QSpacerItem(QSizePolicy.Minimum, QSizePolicy.Expanding)

        layout_panel = QVBoxLayout(self.interfaceWidget)
        layout_panel.addWidget(self.btnZoomIn)
        layout_panel.addWidget(self.btnZoomOut)
        layout_panel.addItem(vspacer)
        layout_panel.addWidget(self.labels_widget)
        layout_panel.addWidget(self.btnAdd)
        layout_panel.addItem(vspacer)
        layout_panel.addWidget(self.btnDel)
        layout_panel.addWidget(self.btnReset)
        layout_panel.addItem(vspacer)
        layout_panel.addWidget(smallRoiWidget)
        layout_panel.addWidget(self.btnProcess)
        layout_panel.addWidget(self.btnSave)
        layout_panel.addStretch()

        layout = QGridLayout(self.mainWidget)
        layout.addWidget(self.imageWidget, 0, 0)
        layout.addWidget(self.interfaceWidget, 0, 1)
        layout.setColumnStretch(0, 2)

        self.setCentralWidget(self.mainWidget)

    def resetInterface(self, enable: bool):
        self.labels_widget.btn_bg.setChecked(True)
        self.labels_widget.setEnabled(enable)
        self.imageWidget.setEnabled(enable)
        self.btnAdd.setEnabled(enable)
        self.btnDel.setEnabled(enable)
        self.btnReset.setEnabled(enable)
        self.btnProcess.setEnabled(enable)
        self.btnSave.setEnabled(enable)
        self.boxMinRoi.setEnabled(enable)
        self.btnZoomIn.setEnabled(enable)
        self.btnZoomOut.setEnabled(enable)

    def change_spinner_text(self, msg):
        self.busy_spinner.label_operation.setText(msg)
        self.busy_spinner.startAnimation()

    def end_loading(self, out_loading):
        # Retrieve loaded data
        self.intmat = out_loading[0]
        self.rgb_im = out_loading[1]
        self.msi_dim_xy = out_loading[1].shape[::-1]
        # Reshape to fit the screen
        h, w, ch = self.rgb_im.shape
        if w > h:
            sc_fact = self.__img_width / w
        else:
            sc_fact = self.__img_height / h
        scaled_im = cv2.resize(
            self.rgb_im,
            (int(self.rgb_im.shape[1] * sc_fact),
             int(self.rgb_im.shape[0] * sc_fact)),
            interpolation=cv2.INTER_NEAREST)
        self.imageWidget.plot_data(scaled_im)
        self.imageWidget.canvas.reset_mask()
        self.imageWidget.canvas.reset_temp()
        # Close the spinner
        self.busy_spinner.stopAnimation()
        self.busy_spinner.close()
        # Re-enable the interface
        self.mainWidget.setEnabled(True)
        self.resetInterface(enable=True)

    def load_data(self, filename: str):
        self.filename = filename
        if self.filename is None or self.filename == [] \
                or len(self.filename) == 0:
            return

        self.save_dir = os.path.dirname(self.filename)
        self.busy_spinner = BusySpinner(parent=self)
        # Place the spinner at the centre of the window
        p = self.window().rect().center() - self.busy_spinner.rect().center()
        self.busy_spinner.move(p)
        # Disable the main window
        self.mainWidget.setEnabled(False)
        self.load_thread = LoadPipeline(datapath=filename, parent=self)
        self.load_thread.currsig.connect(self.change_spinner_text)
        self.load_thread.finished.connect(self.end_loading)
        self.load_thread.start()

    def open_file_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(
            self, "QFileDialog.getOpenFileName()", "", "imzML Files (*.imzML)",
            options=options)
        self.load_data(filename)

    @pyqtSlot(LabelSignal)
    def update_sel_label(self, sigval: LabelSignal):
        self.imageWidget.canvas.draw_value = sigval.value
        self.imageWidget.canvas.reset_temp()
        self.imageWidget.canvas.reset_plot(add_mask=True)
        if sigval == LabelSignal.BACKGROUND:
            self.imageWidget.canvas.draw_color = QColor(Qt.red)
        else:
            self.imageWidget.canvas.draw_color = QColor(Qt.blue)

    @pyqtSlot()
    def add_selection(self) -> None:
        self.imageWidget.canvas.add_selection()

    @pyqtSlot()
    def del_selection(self) -> None:
        self.imageWidget.canvas.reset_temp()
        self.imageWidget.canvas.reset_plot(add_mask=True)

    @pyqtSlot()
    def reset_selection(self):
        self.imageWidget.canvas.reset_mask()
        self.imageWidget.canvas.reset_plot()

    def zoomin(self):
        self.imageWidget.canvas.zoom(2)

    def zoomout(self):
        self.imageWidget.canvas.zoom(0.5)

    def get_mask(self):
        return self.imageWidget.canvas.roi

    def process_end(self):
        s = self.imageWidget.canvas.image.size()
        pred_mask_ = cv2.resize(
            self.thread.mask_, (s.width(), s.height()),
            interpolation=cv2.INTER_NEAREST)
        pred_mask_ = np.digitize(pred_mask_, bins=[0, 1, 2], right=True) - 1
        self.imageWidget.canvas.roi = pred_mask_
        self.imageWidget.canvas.plot_image_and_mask_overlay()
        self.busy_spinner.stopAnimation()
        self.busy_spinner.close()
        if self.thread.threadactive:
            self.thread.stop()

    def process_data(self):
        if np.all(self.imageWidget == 0):
            self.showDialog('No ROI selected')
        elif np.all(self.imageWidget.canvas.roi != 0):
            self.showDialog('All pixels already assigned.')
        elif len(np.unique(self.imageWidget.canvas.roi[
                               self.imageWidget.canvas.roi != 0])) == 1:
            self.showDialog('Only one class annotated.')
        else:
            # Do SVM
            self.busy_spinner = BusySpinner(parent=self)
            # Place the spinner at the centre of the window
            p = self.window().rect().center() \
                - self.busy_spinner.rect().center()
            self.busy_spinner.move(p)
            self.thread = PredictThread(
                yimat=self.intmat, mask_=self.imageWidget.canvas.roi,
                msi_dim_xy=self.msi_dim_xy, parent=self)
            self.thread.finished.connect(self.process_end)
            self.thread.curr_operation.connect(self.change_spinner_text)
            self.thread.start()

    def end_save(self):
        self.busy_spinner.stopAnimation()
        self.busy_spinner.label_operation.setText('Done!')
        time.sleep(1)
        self.busy_spinner.close()

    def save_mask(self):
        if np.any(self.imageWidget.canvas.roi == 0):
            self.showDialog('Still unassigned pixels.')
            return
        else:
            self.busy_spinner = BusySpinner(parent=self)
            # Place the spinner at the centre of the window
            p = self.window().rect().center() \
                - self.busy_spinner.rect().center()
            self.busy_spinner.move(p)

            # Label the separated ROIs
            bin_roi = np.asarray(self.imageWidget.canvas.roi != 1, dtype=int)

            # Save the final ROI
            h, w, ch = self.rgb_im.shape
            save_mask = cv2.resize(bin_roi, (w, h),
                                   interpolation=cv2.INTER_NEAREST)
            save_mask = np.ceil(save_mask).reshape(h, w)

            print('Removing objects smaller than {} px ...'.format(
                int(self.boxMinRoi.value())))
            sample_mask = rem_small_obj_roi(save_mask, min_size=int(
                self.boxMinRoi.value()))

            rgb_im = self.rgb_im
            for ch in range(rgb_im.shape[2]):
                rgb_im[:, :, ch] = minmax_scale(rgb_im[:, :, ch])
                rgb_im[:, :, ch] = np.clip(rgb_im[:, :, ch], 0, 1)
            plt.figure(dpi=150)
            plt.imshow(rgb_im, interpolation='none')
            mask_im = (sample_mask != 0).astype(float)
            # mask_im[mask_im == 0] = np.nan
            mask_im = np.clip(mask_im, 0, 1)
            plt.imshow(mask_im.reshape(self.rgb_im.shape[:2]),
                       cmap='gray', alpha=0.5, interpolation='none')
            plt.title('ROI overlap')
            plt.savefig(os.path.join(self.save_dir, 'roi_overlap.png'))
            plt.close()

            n_ticks = len(np.unique(sample_mask))
            cmap = matplotlib.cm.get_cmap('Set1', n_ticks)
            plt.figure(dpi=150)
            im = plt.imshow(sample_mask.astype(int), cmap=cmap)
            cbar = plt.colorbar(im)
            tick_locs = (np.arange(n_ticks) + 0.5) * (n_ticks - 1) / n_ticks
            cbar.set_ticks(tick_locs)
            cbar.set_ticklabels(np.arange(n_ticks))
            cbar.set_label('Label')
            plt.title('ROI')
            plt.savefig(os.path.join(self.save_dir, 'roi.png'))
            plt.close()

            self.thread = SaveThread(
                parent=self, save_mask=sample_mask, save_dir=self.save_dir)
            self.thread.curr_operation.connect(self.change_spinner_text)
            self.thread.finished.connect(self.end_save)
            self.thread.start()

    def showDialog(self, msg: str):
        QMessageBox.warning(self, 'Cannot process', msg)


qapp = None

if __name__ == '__main__':
    qapp = QApplication(sys.argv)
    m = MainWindow()
    m.show()
    sys.exit(qapp.exec_())
