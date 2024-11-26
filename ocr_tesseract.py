import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QTextEdit
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import cv2
import pytesseract
import numpy as np

class OCRDesktopApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("OCR Desktop Application")
        self.setGeometry(200, 200, 1000, 600)

        # Label untuk menampilkan gambar input
        self.input_image_label = QLabel("Input Image", self)
        self.input_image_label.setAlignment(Qt.AlignCenter)
        self.input_image_label.setFixedSize(400, 400)

        # Area teks untuk menampilkan hasil OCR
        self.output_text_area = QTextEdit(self)
        self.output_text_area.setReadOnly(True)
        self.output_text_area.setPlaceholderText("OCR result will appear here...")

        # Tombol untuk mengunggah gambar
        self.upload_button = QPushButton("Upload Image", self)
        self.upload_button.clicked.connect(self.upload_image)

        # Tombol untuk menjalankan OCR
        self.ocr_button = QPushButton("Run OCR", self)
        self.ocr_button.clicked.connect(self.run_ocr)
        self.ocr_button.setEnabled(False)  # Tombol tidak aktif sebelum gambar diunggah

        # Layout untuk input dan output
        io_layout = QHBoxLayout()
        io_layout.addWidget(self.input_image_label)
        io_layout.addWidget(self.output_text_area)

        # Layout utama
        main_layout = QVBoxLayout()
        main_layout.addLayout(io_layout)
        main_layout.addWidget(self.upload_button)
        main_layout.addWidget(self.ocr_button)

        # Atur widget utama
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def upload_image(self):
        # Dialog untuk memilih file gambar
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
        
        if file_path:
            self.image_path = file_path

            # Tampilkan gambar di QLabel
            pixmap = QPixmap(file_path)
            self.input_image_label.setPixmap(pixmap.scaled(self.input_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            # Aktifkan tombol OCR
            self.ocr_button.setEnabled(True)

    def run_ocr(self):
        # Baca gambar menggunakan OpenCV
        img = cv2.imread(self.image_path)

        # Konversi ke grayscale untuk meningkatkan akurasi OCR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Normalisasi gambar
        norm_img = np.zeros((gray.shape[0], gray.shape[1]))
        img_normalized = cv2.normalize(gray, norm_img, 0, 255, cv2.NORM_MINMAX)

        # thresholding untuk mempermudah OCR
        _, thresholded = cv2.threshold(img_normalized, 100, 255, cv2.THRESH_BINARY)

        # GaussianBlur untuk mengurangi noise
        blurred = cv2.GaussianBlur(thresholded, (1, 1), 0)

        # Ekstraksi teks menggunakan Tesseract dengan konfigurasi khusus
        custom_config = r'--oem 3 --psm 6'  # Mode OCR dan segmen halaman
        extracted_text = pytesseract.image_to_string(blurred, config=custom_config)

        # Tampilkan hasil OCR di QTextEdit
        self.output_text_area.setText(extracted_text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = OCRDesktopApp()
    mainWindow.show()
    sys.exit(app.exec_())
