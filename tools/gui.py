#   Copyright 2018 by Paolo Inglese, National Phenome Centre, Imperial College
#   London
#   All rights reserved.
#   This file is part of DESI-MSI recalibration, and is released under the
#   "MIT License Agreement".
#   Please see the LICENSE file that should have been included as part of this
#   package.


from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PIL import Image, ImageDraw

from typing import Union, List
import numpy as np
from sklearn.preprocessing import minmax_scale

from tools.dbl_slider import QRangeSlider
from tools._colpicker import MplColorHelper


def data_to_qpixmap(img, set_alpha: bool = True):

    def _scale(values):
        if np.var(values) == 0:
            values = np.zeros(values.shape, dtype=np.uint8)
        else:
            values = (minmax_scale(values, axis=0) * 255).astype(np.uint8)
        return values

    if set_alpha:
        try:
            h, w, ch = img.shape
            _rgb = True
        except ValueError:
            h, w = img.shape
            _rgb = False

        if _rgb:
            img = np.reshape(img, (-1, 3))
            img = _scale(img)
            is_black = (np.sum(img, axis=1) == 0)
            img = np.c_[
                img, 255 * np.ones(img.shape[0], dtype=np.uint8)]
        else:
            img = np.reshape(img, (-1,))
            img = _scale(img)
            is_black = (img == 0)
            img = minmax_scale(img.astype(np.float))
            colpick = MplColorHelper('viridis', start_val=0, stop_val=1)
            img = (colpick.get_rgb(img) * 255).astype(np.uint8)
        img[is_black, 3] = 0
        img = np.reshape(img, (h, w, 4))
    height, width, channel = img.shape
    bytesPerLine = 4 * width
    qimg = QImage(
        img.data, width, height, bytesPerLine, QImage.Format_RGBA8888)
    return QPixmap(qimg)


class SelectableImage(QWidget):
    __MASK_NO_LBL = 0
    __title: str
    imageWidget: QLabel
    draw_value: int
    draw_color: QColor
    drawing: bool
    image: Union[None, QPixmap]
    last_point: Union[None, QPoint]
    roi: Union[None, np.ndarray]
    temp_roi: Union[None, np.ndarray]
    temp_contour: list

    def __init__(self, title: str, parent):
        super(SelectableImage, self).__init__(parent)

        self.setCursor(QCursor(Qt.CrossCursor))

        self.__title = title
        self.imageWidget = QLabel()
        self.draw_value = 0
        self.draw_color = QColor(Qt.black)
        self.drawing = False
        self.image = None
        self.last_point = None
        self.roi = None
        self.temp_roi = None
        self.temp_contour = []

        self.initUI()

    def __draw_line(self, pos):
        painter = QPainter(self.imageWidget.pixmap())
        painter.setPen(QPen(self.draw_color, 1, Qt.SolidLine))
        painter.drawLine(pos, self.last_point)
        painter.end()
        self.update()

    def __empty_mask(self):
        s = self.image.size()
        return np.zeros((s.height(), s.width()), dtype=np.int)

    def __gen_polygon(self):
        if len(self.temp_contour) == 0:
            return
        else:
            s = self.imageWidget.size()
            points = np.array(self.temp_contour)
            points = points.ravel().tolist()
            poly_img = Image.new(
                "L", [s.width(), s.height()], self.__MASK_NO_LBL)
            # Fill the mask with the current label value (+1)
            ImageDraw.Draw(poly_img).polygon(
                points, outline=self.draw_value + 1,
                fill=self.draw_value + 1)
            self.temp_roi = np.array(poly_img)

    # TODO: use hex color
    def __mask_to_rgb(self):
        # Map the colors of the labels
        h, w = self.roi.shape
        mask_img = np.zeros((h * w, 3), dtype=np.uint8)
        mask_img[self.roi.flatten() == 1, 0] = 255
        mask_img[self.roi.flatten() == 2, 1] = 255
        mask_img = np.reshape(mask_img, (h, w, 3))
        return mask_img

    def add_selection(self):
        self.__gen_polygon()
        if self.temp_roi is None:
            return
        else:
            self.roi[self.temp_roi != self.__MASK_NO_LBL] \
                = self.temp_roi[self.temp_roi != self.__MASK_NO_LBL]
            # Update image
            self.plot_image_and_mask_overlay()
            # Remove the selected pixels
            self.reset_temp()

    def initUI(self):
        self.imageWidget = QLabel(self)
        self.imageWidget.setStyleSheet('border:1px solid rgb(0, 0, 0);')

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(self.__title), 10)
        layout.addWidget(self.imageWidget, 90)
        layout.addStretch()

    def mouseMoveEvent(self, e) -> None:
        if self.last_point is None:
            self.last_point = self.imageWidget.mapFromParent(e.pos())

        if self.drawing:
            map_pos = self.imageWidget.mapFromParent(e.pos())
            self.__draw_line(map_pos)
            self.last_point = map_pos
            self.temp_contour.append([map_pos.x(), map_pos.y()])

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = self.imageWidget.mapFromParent(e.pos())

    def mouseReleaseEvent(self, event):
        if event.button == Qt.LeftButton:
            self.drawing = False
            self.last_point = None

    def plot_image(self) -> None:
        self.imageWidget.setPixmap(self.image)
        self.imageWidget.setFixedSize(self.image.size())
        self.imageWidget.update()

    def plot_image_and_mask_overlay(self):
        if self.roi is not None:
            mask_img = self.__mask_to_rgb()
            mask_qpixmap = data_to_qpixmap(mask_img)
            mode = QPainter.CompositionMode_SourceOver
            self.imageWidget.clear()

            pixmap = QPixmap(self.image.size())
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.drawPixmap(0, 0, self.image)
            painter.setCompositionMode(mode)
            painter.setOpacity(0.6)
            painter.drawPixmap(0, 0, mask_qpixmap)
            painter.end()
            self.imageWidget.setPixmap(pixmap)
        else:
            self.plot_image()

    def reset_mask(self):
        if self.image is None:
            self.roi = None
        else:
            self.roi = self.__empty_mask()

    def reset_plot(self, add_mask: bool = False):
        if add_mask:
            self.plot_image_and_mask_overlay()
        else:
            self.plot_image()
        self.update()

    def reset_temp(self):
        self.temp_contour = []
        self.temp_roi = None

    def set_draw_color(self, color):
        self.draw_color = QColor(color)


class RangeColorBar(QWidget):
    data: Union[None, np.ndarray]
    wig_image: QLabel
    __nch: int
    __rgb: bool
    pixmap: QPixmap
    colorbar: List[QRangeSlider]
    drawing: bool
    last_point: QPoint
    mouse_pos = pyqtSignal(list)

    def __init__(self, bar_title: List[str], parent=None):
        super(RangeColorBar, self).__init__(parent=parent)

        self.data = None
        self.__nch = 3
        self.__rgb = False

        self.colorbar = []
        for i in range(3):
            w = QRangeSlider()
            w.setMin(0)
            w.setMax(1)
            w.setRange(0, 1)
            w.setSpanStyle('background: #999999ff;')
            w.setEnabled(False)
            w.setMinimumWidth(150)
            w.setMaximumWidth(200)
            self.colorbar.append(w)

        lay_central = QFormLayout(self)
        lay_central.setFieldGrowthPolicy(1)
        lay_central.setFormAlignment(Qt.AlignLeft)
        lay_central.setLabelAlignment(Qt.AlignLeft)
        for ch in range(self.__nch):
            lay_central.addRow(QLabel(bar_title[ch]), self.colorbar[ch])

    def update_minmax(self, minmax):
        if len(minmax) == 3:
            self.__rgb = True
            self.__nch = 3
        else:
            self.__rgb = False
            self.__nch = 1

        cols = ['#ff0000', '#00ff00', '#0000ff']
        for ch in range(self.__nch):
            m_ = minmax[ch][0]
            M_ = minmax[ch][1]
            self.colorbar[ch].setMin(m_)
            self.colorbar[ch].setMax(M_)
            self.colorbar[ch].setRange(m_, M_)
            if self.__rgb:
                self.colorbar[ch].setSpanStyle(
                    'background: qlineargradient(x1:0, y1:0, x2:1, y2:0, '
                    'stop:0 #000000, stop:1 {});'.format(cols[ch]))
            else:
                self.colorbar[ch].setSpanStyle(
                    'background: qlineargradient(x1:0, y1:0, x2:1, y2:0, '
                    'stop:0 #042333, stop:1 #e8fa5b);')
            self.colorbar[ch].setEnabled(True)


class ImageWithColorbar(QWidget):
    rangebar: RangeColorBar
    data: np.ndarray
    __rgb: bool

    def __init__(self, title: str, colorbar_title: List[str], parent=None):
        super(ImageWithColorbar, self).__init__(parent=parent)

        self.canvas = SelectableImage(title=title, parent=self)
        self.rangebar = RangeColorBar(parent=self, bar_title=colorbar_title)

        lay = QVBoxLayout(self)
        lay.addWidget(self.canvas, 75)
        lay.addWidget(self.rangebar, 25)
        lay.addStretch()

    def plot_data(self, data):
        self.data = data
        self.canvas.image = QPixmap(data_to_qpixmap(self.data))
        minmax = []
        try:
            h, w, ch = data.shape
            for ch in range(ch):
                minmax.append(
                    [np.min(data[:, :, ch]), np.max(data[:, :, ch])])
            self.__rgb = True
        except ValueError:
            for i in range(1):
                minmax.append([np.min(data), np.max(data)])
            self.__rgb = False

        self.rangebar.update_minmax(minmax=minmax)
        for ch in range(len(self.rangebar.colorbar)):
            self.rangebar.colorbar[ch].startValueChanged.connect(
                self.update_colorscale)
            self.rangebar.colorbar[ch].endValueChanged.connect(
                self.update_colorscale)
        self.canvas.plot_image()

    def update_colorscale(self):
        data_ = self.data.copy()
        if self.__rgb:
            for ch in range(data_.shape[2]):
                s_ = self.rangebar.colorbar[ch].start()
                e_ = self.rangebar.colorbar[ch].end()
                d_ = data_[:, :, ch]
                d_[d_ < s_] = s_
                d_[d_ > e_] = e_
                data_[:, :, ch] = d_
        else:
            s_ = self.rangebar.colorbar[0].start()
            e_ = self.rangebar.colorbar[0].end()
            data_[data_ < s_] = s_
            data_[data_ > e_] = e_

        self.canvas.image = data_to_qpixmap(data_)
        self.canvas.plot_image_and_mask_overlay()
