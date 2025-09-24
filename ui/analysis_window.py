import os
from datetime import timedelta

from PySide6.QtCore import Qt
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ui.locale_utils import parse_float
from ui.map_widget import MapMode, MapWidget


class AnalysisWindow(QMainWindow):
    def __init__(self, fit_file, settings):
        super().__init__()
        self.fit_file = fit_file
        self.settings = settings
        self.file_settings = settings.get_file_settings(fit_file.filename)
        self.selected_laps = []
        self.initUI()
        QApplication.instance().aboutToQuit.connect(self.join_threads)

    def initUI(self):
        self.setWindowTitle(
            f"Virtual Elevation Analyzer - {os.path.basename(self.fit_file.filename)}"
        )
        self.setGeometry(50, 50, 1200, 800)

        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # Create horizontal layout for map and controls
        content_layout = QHBoxLayout()

        # Left side - Map and lap table
        left_layout = QVBoxLayout()

        # Add map widget if GPS data is available
        self.map_widget = MapWidget(
            MapMode.LAPS, self.fit_file.resampled_df, self.file_settings
        )
        if self.map_widget.has_gps():
            self.map_widget.update()
            left_layout.addWidget(self.map_widget, 2)
        else:
            no_gps_label = QLabel("No GPS data available in this FIT file")
            no_gps_label.setAlignment(Qt.AlignCenter)
            left_layout.addWidget(no_gps_label, 2)

        # Lap table
        lap_groupbox = QGroupBox("Laps")
        lap_layout = QVBoxLayout()

        # Lap table
        self.lap_table = QTableWidget()
        self.populate_lap_table()
        lap_layout.addWidget(self.lap_table)

        # Select/deselect all button
        select_button_layout = QHBoxLayout()
        self.select_all_button = QPushButton("Select / Deselect All Laps")
        self.select_all_button.clicked.connect(self.toggle_lap_selection)
        select_button_layout.addWidget(self.select_all_button)
        select_button_layout.addStretch()
        lap_layout.addLayout(select_button_layout)

        lap_groupbox.setLayout(lap_layout)
        left_layout.addWidget(lap_groupbox, 1)

        # Right side - Parameters
        right_layout = QVBoxLayout()

        # Parameters group
        self.param_groupbox = QGroupBox("Analysis Parameters")
        param_layout = QFormLayout()

        # System mass
        self.system_mass = QLineEdit(str(self.file_settings["system_mass"]))
        self.system_mass.setValidator(QDoubleValidator(0, 200, 1))
        param_layout.addRow("System Mass (kg):", self.system_mass)

        # Air density
        self.rho = QLineEdit(str(self.file_settings["rho"]))
        self.rho.setValidator(QDoubleValidator(0, 2, 4))
        param_layout.addRow("Rho (kg/m³):", self.rho)

        # CdA
        self.cda = QLineEdit()
        if self.file_settings["cda"] is not None:
            self.cda.setText(str(self.file_settings["cda"]))
        self.cda.setValidator(QDoubleValidator(0, 1, 4))
        self.cda.setPlaceholderText("Empty for optimization")
        param_layout.addRow("Fixed CdA:", self.cda)

        # Crr
        self.crr = QLineEdit()
        if self.file_settings["crr"] is not None:
            self.crr.setText(str(self.file_settings["crr"]))
        self.crr.setValidator(QDoubleValidator(0, 0.1, 6))
        self.crr.setPlaceholderText("Empty for optimization")
        param_layout.addRow("Fixed Crr:", self.crr)

        # CdA bounds
        cda_bounds_layout = QHBoxLayout()
        self.cda_min = QLineEdit(str(self.file_settings["cda_min"]))
        self.cda_min.setValidator(QDoubleValidator(0, 1, 4))
        self.cda_max = QLineEdit(str(self.file_settings["cda_max"]))
        self.cda_max.setValidator(QDoubleValidator(0, 1, 4))
        cda_bounds_layout.addWidget(self.cda_min)
        cda_bounds_layout.addWidget(QLabel("to"))
        cda_bounds_layout.addWidget(self.cda_max)
        param_layout.addRow("CdA Bounds:", cda_bounds_layout)

        # Crr bounds
        crr_bounds_layout = QHBoxLayout()
        self.crr_min = QLineEdit(str(self.file_settings["crr_min"]))
        self.crr_min.setValidator(QDoubleValidator(0, 0.1, 6))
        self.crr_max = QLineEdit(str(self.file_settings["crr_max"]))
        self.crr_max.setValidator(QDoubleValidator(0, 0.1, 6))
        crr_bounds_layout.addWidget(self.crr_min)
        crr_bounds_layout.addWidget(QLabel("to"))
        crr_bounds_layout.addWidget(self.crr_max)
        param_layout.addRow("Crr Bounds:", crr_bounds_layout)

        # Drivetrain efficiency
        self.eta = QLineEdit(str(self.file_settings["eta"]))
        self.eta.setValidator(QDoubleValidator(0, 1, 4))
        param_layout.addRow("Eta (efficiency):", self.eta)

        # Velodrome - we'll use this to zero altitude.
        velodrome_layout = QHBoxLayout()
        self.velodrome = QCheckBox("Sets Altitude to Zero")

        # Set its checked state based on the bool in file_settings
        if "velodrome" in self.file_settings and self.file_settings["velodrome"] is not None:
            self.velodrome.setChecked(bool(self.file_settings["velodrome"]))
        else:
            self.velodrome.setChecked(False)

        velodrome_layout.addWidget(self.velodrome)
        param_layout.addRow("Velodrome:", velodrome_layout)

        self.wind_speed = QLineEdit()
        if self.file_settings["wind_speed"] is not None:
            self.wind_speed.setText(str(self.file_settings["wind_speed"]))
        self.wind_speed.setValidator(QDoubleValidator(-30, 30, 2))
        self.wind_speed.setPlaceholderText("Optional")
        self.wind_speed.textChanged.connect(self.update_wind_settings)
        param_layout.addRow("Wind Speed (m/s):", self.wind_speed)

        # Wind direction
        self.wind_direction = QLineEdit()
        if self.file_settings["wind_direction"] is not None:
            self.wind_direction.setText(str(self.file_settings["wind_direction"]))
        self.wind_direction.setValidator(QDoubleValidator(0, 360, 1))
        self.wind_direction.setPlaceholderText("Optional")
        self.wind_direction.textChanged.connect(self.update_wind_settings)
        param_layout.addRow("Wind Direction (°):", self.wind_direction)

        # Auto lap detection
        self.auto_lap_detection = QComboBox()
        self.auto_lap_detection.addItems(
            [
                "None",
                "GPS based lap splitting",
                "GPS based out and back",
                "GPS gate one way",
            ]
        )
        self.auto_lap_detection.setCurrentText(self.file_settings["auto_lap_detection"])
        param_layout.addRow("Auto Lap Detection:", self.auto_lap_detection)
        self.param_groupbox.setLayout(param_layout)
        right_layout.addWidget(self.param_groupbox)

        # Analysis button
        self.analyze_button = QPushButton("Analyse Laps")
        self.analyze_button.clicked.connect(self.analyze_laps)
        self.analyze_button.setEnabled(False)  # Disabled until laps are selected
        self.analyze_button.setStyleSheet(f"background-color: #4363d8; color: white;")
        right_layout.addWidget(self.analyze_button)

        # Navigation buttons
        nav_layout = QHBoxLayout()
        self.back_button = QPushButton("Back to File Selection")
        self.back_button.clicked.connect(self.back_to_file_selection)

        self.close_button = QPushButton("Close App")
        self.close_button.clicked.connect(self.close)

        nav_layout.addWidget(self.back_button)
        nav_layout.addWidget(self.close_button)
        right_layout.addLayout(nav_layout)

        # Add stretch to push everything up
        right_layout.addStretch()

        # Add layouts to content layout
        content_layout.addLayout(left_layout, 2)
        content_layout.addLayout(right_layout, 1)

        # Add content layout to main layout
        main_layout.addLayout(content_layout)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Maximize the window
        self.showMaximized()

    def populate_lap_table(self):
        """Populate the lap table with data from FIT file"""
        lap_data = self.fit_file.get_lap_data()

        # Configure table
        self.lap_table.setColumnCount(6)
        self.lap_table.setRowCount(len(lap_data))
        self.lap_table.setHorizontalHeaderLabels(
            ["Select", "Lap", "Duration", "Distance", "Avg Power", "Avg Speed"]
        )

        # Set column widths
        header = self.lap_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for i in range(1, 6):
            header.setSectionResizeMode(i, QHeaderView.Stretch)

        # Populate table
        for row, lap in enumerate(lap_data):
            # Checkbox for selection
            checkbox = QCheckBox()
            checkbox.stateChanged.connect(self.update_selected_laps)
            self.lap_table.setCellWidget(row, 0, checkbox)

            # Lap number
            self.lap_table.setItem(row, 1, QTableWidgetItem(str(lap["lap_number"])))

            # Duration in MM:SS format
            duration = timedelta(seconds=lap["duration"])
            duration_str = f"{int(duration.total_seconds() // 60):02d}:{int(duration.total_seconds() % 60):02d}"
            self.lap_table.setItem(row, 2, QTableWidgetItem(duration_str))

            # Distance in km
            self.lap_table.setItem(
                row, 3, QTableWidgetItem(f"{lap['distance']:.2f} km")
            )

            # Average power
            self.lap_table.setItem(
                row, 4, QTableWidgetItem(f"{int(lap['avg_power'])} W")
            )

            # Average speed
            self.lap_table.setItem(
                row, 5, QTableWidgetItem(f"{lap['avg_speed']:.1f} km/h")
            )

            # Make cells read-only
            for col in range(1, 6):
                item = self.lap_table.item(row, col)
                if item:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)

    def update_selected_laps(self):
        """Update the list of selected laps when checkboxes change"""
        self.selected_laps = []

        for row in range(self.lap_table.rowCount()):
            checkbox = self.lap_table.cellWidget(row, 0)
            if checkbox and checkbox.isChecked():
                lap_item = self.lap_table.item(row, 1)
                if lap_item:
                    self.selected_laps.append(int(lap_item.text()))

        # Sort laps
        self.selected_laps.sort()

        # Enable analyze button if laps are selected
        self.analyze_button.setEnabled(len(self.selected_laps) > 0)

        # Update the map to show selected laps
        if self.fit_file.has_gps:
            self.map_widget.set_selected_laps(
                self.fit_file.get_lap_data(), self.selected_laps
            )
            self.map_widget.update()

    def toggle_lap_selection(self):
        """Select or deselect all laps"""
        # Determine if any are already selected
        any_selected = False
        for row in range(self.lap_table.rowCount()):
            checkbox = self.lap_table.cellWidget(row, 0)
            if checkbox and checkbox.isChecked():
                any_selected = True
                break

        # Toggle all checkboxes
        for row in range(self.lap_table.rowCount()):
            checkbox = self.lap_table.cellWidget(row, 0)
            if checkbox:
                checkbox.setChecked(not any_selected)

        self.update_selected_laps()

    def analyze_laps(self):
        """Handle the Analyze Laps button click"""
        # Check if selected laps are consecutive
        consecutive = True
        for i in range(len(self.selected_laps) - 1):
            if self.selected_laps[i + 1] - self.selected_laps[i] != 1:
                consecutive = False
                break

        # Check if multiple non-consecutive laps are selected
        if not consecutive and len(self.selected_laps) > 1:
            QMessageBox.warning(
                self,
                "Non-consecutive laps",
                "Analysis of non-consecutive laps will be implemented later. "
                "Please select consecutive laps only.",
            )
            return

        # Get the current file settings
        current_settings = self.settings.get_file_settings(self.fit_file.filename)

        # Save the existing trim_settings before updating other parameters
        existing_trim_settings = current_settings.get("trim_settings", {})

        # Prepare the new settings to save
        new_settings = {
            "system_mass": parse_float(self.system_mass.text()),
            "rho": parse_float(self.rho.text()),
            "cda": parse_float(self.cda.text()),
            "crr": parse_float(self.crr.text()),
            "cda_min": parse_float(self.cda_min.text()),
            "cda_max": parse_float(self.cda_max.text()),
            "crr_min": parse_float(self.crr_min.text()),
            "crr_max": parse_float(self.crr_max.text()),
            "eta": parse_float(self.eta.text()),
            "wind_speed": parse_float(self.wind_speed.text()),
            "wind_direction": parse_float(self.wind_direction.text()),
            "auto_lap_detection": self.auto_lap_detection.currentText(),
            "velodrome": self.velodrome.isChecked(),
            # Preserve the existing trim settings
            "trim_settings": existing_trim_settings,
        }

        # Save the combined settings
        self.settings.save_file_settings(self.fit_file.filename, new_settings)

        auto_lap_detection = self.auto_lap_detection.currentText()

        # Collect parameters
        params = {
            "system_mass": parse_float(self.system_mass.text()),
            "rho": parse_float(self.rho.text()),
            "cda": parse_float(self.cda.text()),
            "crr": parse_float(self.crr.text()),
            "cda_min": parse_float(self.cda_min.text()),
            "cda_max": parse_float(self.cda_max.text()),
            "crr_min": parse_float(self.crr_min.text()),
            "crr_max": parse_float(self.crr_max.text()),
            "eta": parse_float(self.eta.text()),
            "wind_speed": parse_float(self.wind_speed.text()),
            "wind_direction": parse_float(self.wind_direction.text()),
            "auto_lap_detection": auto_lap_detection,
            "velodrome": self.velodrome.isChecked(),
        }

        # Launch appropriate analysis window based on auto lap detection
        try:
            if auto_lap_detection == "None":
                # Standard analysis without auto lap detection
                from ui.analysis_result import AnalysisResult

                self.analysis_result_window = AnalysisResult(
                    self, self.fit_file, self.settings, self.selected_laps, params
                )
                self.analysis_result_window.show()
                self.hide()
            elif auto_lap_detection == "GPS based lap splitting":
                # GPS based lap splitting
                from ui.gps_lap_result import GPSLapResult

                self.gps_lap_result_window = GPSLapResult(
                    self, self.fit_file, self.settings, self.selected_laps, params
                )
                self.gps_lap_result_window.show()
                self.hide()
            elif auto_lap_detection == "GPS based out and back":
                # GPS based out and back analysis
                from ui.out_and_back_result import OutAndBackResult

                self.out_and_back_result_window = OutAndBackResult(
                    self, self.fit_file, self.settings, self.selected_laps, params
                )
                self.out_and_back_result_window.show()
                self.hide()
            elif auto_lap_detection == "GPS gate one way":
                # GPS gate one way analysis
                from ui.gps_gate_result import GPSGateResult

                self.gps_gate_result_window = GPSGateResult(
                    self, self.fit_file, self.settings, self.selected_laps, params
                )
                self.gps_gate_result_window.show()
                self.hide()
            else:
                # Other auto lap detection methods not yet implemented
                QMessageBox.information(
                    self,
                    "Feature Coming Soon",
                    f"Analysis with auto lap detection '{auto_lap_detection}' "
                    "will be implemented in a future update.",
                )
        except Exception as e:
            QMessageBox.critical(
                self, "Analysis Error", f"Error performing analysis: {str(e)}"
            )

    def save_parameters(self):
        """Save the current parameters to settings"""
        settings = {
            "system_mass": parse_float(self.system_mass.text()),
            "rho": parse_float(self.rho.text()),
            "cda": parse_float(self.cda.text()),
            "crr": parse_float(self.crr.text()),
            "cda_min": parse_float(self.cda_min.text()),
            "cda_max": parse_float(self.cda_max.text()),
            "crr_min": parse_float(self.crr_min.text()),
            "crr_max": parse_float(self.crr_max.text()),
            "eta": parse_float(self.eta.text()),
            "wind_speed": parse_float(self.wind_speed.text()),
            "wind_direction": parse_float(self.wind_direction.text()),
            "auto_lap_detection": self.auto_lap_detection.currentText(),
            "velodrome": self.velodrome.isChecked(),
        }

        self.settings.save_file_settings(self.fit_file.filename, settings)

    def back_to_file_selection(self):
        """Return to file selection window"""
        from ui.file_selector import FileSelector

        self.file_selector = FileSelector()
        self.file_selector.show()
        self.close()

    def update_wind_settings(self):
        """Update the wind settings and refresh the map"""
        # Parse wind speed and direction from UI
        wind_speed = parse_float(self.wind_speed.text())
        wind_direction = parse_float(self.wind_direction.text())

        # Update the file settings
        self.file_settings["wind_speed"] = wind_speed
        self.file_settings["wind_direction"] = wind_direction

        # Save settings to disk
        self.settings.save_file_settings(self.fit_file.filename, self.file_settings)

        # Refresh the map if it exists
        if self.fit_file.has_gps:
            self.map_widget.set_wind(wind_speed, wind_direction)
            self.map_widget.update()

    def join_threads(self):
        if self.map_widget:
            self.map_widget.close()

    def closeEvent(self, event):
        self.join_threads()
        event.accept()
