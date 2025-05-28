import os

from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from config.settings import Settings
from models.fit_file import FitFile
from utils.file_handling import get_results_dir  # ← add this import


class FileSelector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = Settings()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Virtual Elevation Analyzer")
        self.setGeometry(100, 100, 600, 200)

        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # File selection
        file_layout = QHBoxLayout()
        self.file_label = QLabel("FIT File:")
        self.file_path = QLineEdit()
        if self.settings.last_file and os.path.exists(self.settings.last_file):
            self.file_path.setText(self.settings.last_file)
        self.file_button = QPushButton("Browse")
        self.file_button.clicked.connect(self.select_file)

        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_path)
        file_layout.addWidget(self.file_button)

        # Result directory selection
        dir_layout = QHBoxLayout()
        self.dir_label = QLabel("Result Directory:")
        self.dir_path = QLineEdit()

        self.dir_path.setText(self.settings.result_dir or str(get_results_dir()))

        self.dir_button = QPushButton("Select Directory")
        self.dir_button.clicked.connect(self.select_directory)

        dir_layout.addWidget(self.dir_label)
        dir_layout.addWidget(self.dir_path)
        dir_layout.addWidget(self.dir_button)

        # DEM File selection
        dem_file_layout = QHBoxLayout()
        self.dem_file_label = QLabel("Correct Elevation:")
        self.dem_file_path = QLineEdit()
        self.dem_file_path.setPlaceholderText("OPTIONAL: Select DEM file to correct GPS elevation data...")
        if self.settings.last_dem_file and os.path.exists(self.settings.last_dem_file):
            self.dem_file_path.setText(self.settings.last_dem_file)
        self.dem_file_button = QPushButton("Browse")
        self.dem_file_button.clicked.connect(self.select_dem_file)
        self.dem_file_path.editingFinished.connect(self.dem_file_path_edited)

        dem_file_layout.addWidget(self.dem_file_label)
        dem_file_layout.addWidget(self.dem_file_path)
        dem_file_layout.addWidget(self.dem_file_button)

        # Analyze and Close buttons
        button_layout = QHBoxLayout()
        self.analyze_button = QPushButton("Analyze File")
        self.analyze_button.clicked.connect(self.analyze_file)
        self.analyze_button.setStyleSheet(f"background-color: #4363d8; color: white;")

        self.close_button = QPushButton("Close App")
        self.close_button.clicked.connect(self.close)

        button_layout.addWidget(self.close_button)
        button_layout.addStretch()
        button_layout.addWidget(self.analyze_button)

        # Add layouts to main layout
        main_layout.addLayout(file_layout)
        main_layout.addLayout(dir_layout)
        main_layout.addLayout(dem_file_layout)
        main_layout.addStretch()
        main_layout.addLayout(button_layout)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select FIT File", "", "FIT Files (*.fit)"
        )

        if file_path:
            self.file_path.setText(file_path)
            self.settings.last_file = file_path
            self.settings.save_settings()

    def select_dem_file(self):
        dem_file_path, _ = QFileDialog.getOpenFileName(
            self, "Select DEM File (*.vrt *.tif)", "",
            "DEM Files (*.vrt *.tif *.tiff);;VRT Files (*.vrt);;GeoTIFF Files (*.tif *.tiff)"
        )

        if dem_file_path:
            self.dem_file_path.setText(dem_file_path)
            self.settings.last_dem_file = dem_file_path
            self.settings.save_settings()

    def dem_file_path_edited(self):
        dem_file_path = self.dem_file_path.text()
        self.settings.last_dem_file = dem_file_path
        self.settings.save_settings()

    def analyze_file(self):
        file_path = self.file_path.text()
        dem_file_path = self.dem_file_path.text()

        if not file_path or not os.path.exists(file_path):
            QMessageBox.warning(self, "Invalid File", "Please select a valid FIT file.")
            return

        if dem_file_path and not os.path.exists(dem_file_path):
            QMessageBox.warning(self, "Invalid DEM File", "Not correcting elevation")
            dem_file_path = None

        # Create results directory if it doesn't exist
        result_dir = self.dir_path.text()
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
            self.settings.result_dir = result_dir
            self.settings.save_settings()

        try:
            # Load the fit file and open analysis window
            fit_file = FitFile(file_path, dem_file_path)

            elevation_error_rate = fit_file.get_elevation_error_rate()
            if (elevation_error_rate != 0):
                elevation_error_rate = int(elevation_error_rate * 100)
                QMessageBox.warning(self, "Invalid DEM File",
                                    f"Could not correct {elevation_error_rate}% of altitude points")

            # Import here to avoid circular import
            from ui.analysis_window import AnalysisWindow

            self.analysis_window = AnalysisWindow(fit_file, self.settings)
            self.analysis_window.show()
            self.hide()
        except Exception as e:
            # Handle invalid fit file
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setText("Error loading FIT file")
            error_dialog.setInformativeText(str(e))
            error_dialog.setWindowTitle("Error")
            error_dialog.exec()

    def select_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Result Directory")

        if dir_path:
            self.dir_path.setText(dir_path)
            self.settings.result_dir = dir_path
            self.settings.save_settings()
