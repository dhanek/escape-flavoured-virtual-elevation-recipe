import io

import folium
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QVBoxLayout, QWidget


class MapWidget(QWidget):
    def __init__(self, records, params):
        super().__init__()
        self.records = records
        if ("position_lat" in records and "position_long" in records):
            self.has_gps = True
            # Filter out records with no position data
            records = records.dropna(subset=["position_lat", "position_long"])
            self.route_points = list(
                zip(records["position_lat"], records["position_long"])
            )
            self.route_timestamps = records["timestamp"].tolist()
            self.center_lat = records["position_lat"].mean()
            self.center_lon = records["position_long"].mean()
            if not self.route_points:
                self.has_gps = False
        else:
            self.has_gps = False

        self.lap_data = []
        self.selected_laps = []
        self.layout = QVBoxLayout(self)
        self.webview = QWebEngineView()
        self.layout.addWidget(self.webview)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.wind_speed = params.get("wind_speed", None)
        self.wind_direction = params.get("wind_direction", None)

    def set_selected_laps(self, lap_data, selected_laps):
        """Update the selected laps and redraw the map"""
        self.lap_data = lap_data
        self.selected_laps = selected_laps

    def set_wind(self, wind_speed, wind_direction):
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction

    def draw_selected_laps(self, m):
        selected_lap_points = []
        # We have selected laps - need to identify selected vs non-selected portions

        # Create a mask to mark which points belong to selected laps
        selected_mask = [False] * len(self.route_points)

        # Mark all points that belong to selected laps
        for lap_number in self.selected_laps:
            # Find the lap in self.lap_data
            lap_info = None
            for lap in self.lap_data:
                if lap["lap_number"] == lap_number:
                    lap_info = lap
                    break

            if lap_info and "start_time" in lap_info and "end_time" in lap_info:
                # Mark all points in this lap as selected
                for i, timestamp in enumerate(self.route_timestamps):
                    if lap_info["start_time"] <= timestamp <= lap_info["end_time"]:
                        selected_mask[i] = True
                        selected_lap_points.append(self.route_points[i])

        # Now draw non-selected parts (dashed blue with reduced opacity)
        non_selected_segments = []
        current_segment = []

        for i, selected in enumerate(selected_mask):
            if not selected:
                # Add to current non-selected segment
                current_segment.append(self.route_points[i])
            else:
                # End of non-selected segment
                if current_segment:
                    non_selected_segments.append(current_segment)
                    current_segment = []

        # Add the last segment if it exists
        if current_segment:
            non_selected_segments.append(current_segment)

        # Draw all non-selected segments as dashed blue
        for segment in non_selected_segments:
            if len(segment) > 1:
                folium.PolyLine(
                    segment,
                    color="#4363d8",  # Blue color
                    weight=3,
                    opacity=0.5,
                    dash_array="5,10",  # Dashed line
                ).add_to(m)

        # Draw selected parts (solid blue with full opacity)
        selected_segments = []
        current_segment = []

        for i, selected in enumerate(selected_mask):
            if selected:
                # Add to current selected segment
                current_segment.append(self.route_points[i])
            else:
                # End of selected segment
                if current_segment:
                    selected_segments.append(current_segment)
                    current_segment = []

        # Add the last segment if it exists
        if current_segment:
            selected_segments.append(current_segment)

        # Draw all selected segments as solid blue
        for segment in selected_segments:
            if len(segment) > 1:
                folium.PolyLine(
                    segment,
                    color="#4363d8",  # Blue color
                    weight=5,  # Slightly thicker
                    opacity=1.0,  # Full opacity
                ).add_to(m)

        # Add markers for the start and end of each selected lap
        for lap_number in self.selected_laps:
            # Find the lap in self.lap_data
            lap_info = None
            for lap in self.lap_data:
                if lap["lap_number"] == lap_number:
                    lap_info = lap
                    break

            if lap_info:
                # Find the first and last point in this lap
                start_point = None
                end_point = None

                for i, timestamp in enumerate(self.route_timestamps):
                    if lap_info["start_time"] <= timestamp <= lap_info["end_time"]:
                        if start_point is None:
                            start_point = self.route_points[i]
                        end_point = self.route_points[i]

                if start_point and end_point:
                    # Start marker
                    folium.Marker(
                        location=start_point,
                        popup=f"Lap {lap_number} Start",
                        icon=folium.Icon(color="green", icon="play", prefix="fa"),
                    ).add_to(m)

                    # End marker
                    folium.Marker(
                        location=end_point,
                        popup=f"Lap {lap_number} End",
                        icon=folium.Icon(color="red", icon="stop", prefix="fa"),
                    ).add_to(m)
        return selected_lap_points

    def draw_default(self, m):
        folium.PolyLine(
            self.route_points, color="#4363d8", weight=4, opacity=1.0
        ).add_to(m)
        return self.route_points

    def update(self):
        """Create and display a map from FIT file GPS data"""
        # Create a folium map

        if not self.has_gps:
            return

        # Create map centered on activity
        m = folium.Map(location=[self.center_lat, self.center_lon], zoom_start=12)

        # If no laps are selected, draw the full track as solid blue
        if self.selected_laps:
            points_to_zoom = self.draw_selected_laps(m)
        else:
            points_to_zoom = self.draw_default(m)

        # Calculate bounds for automatic zoom
        try:
            if points_to_zoom:
                lats = [p[0] for p in points_to_zoom]
                lons = [p[1] for p in points_to_zoom]
                min_lat, max_lat = min(lats), max(lats)
                min_lon, max_lon = min(lons), max(lons)

                # Add some padding (5%)
                lat_padding = (max_lat - min_lat) * 0.05
                lon_padding = (max_lon - min_lon) * 0.05
                bounds = [
                    [min_lat - lat_padding, min_lon - lon_padding],
                    [max_lat + lat_padding, max_lon + lon_padding],
                ]
                m.fit_bounds(bounds)
        except Exception as e:
            print(f"Error fitting map bounds: {e}")

        if self.wind_speed not in [None, 0] and self.wind_direction:
            # Create a custom HTML element for the wind arrow
            wind_html = f"""
            <div id="wind-arrow" style="position: absolute; top: 10px; right: 10px; 
                    background-color: rgba(255, 255, 255, 0.9); padding: 10px; 
                    border-radius: 5px; border: 1px solid #4363d8; z-index: 1000;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
                <div style="font-weight: bold; text-align: center; margin-bottom: 5px; color: #4363d8;">Wind</div>
                <div style="text-align: center;">
                    <div style="font-size: 28px; transform: rotate({self.wind_direction + 180}deg); color: #4363d8;">↑</div>
                    <div style="font-size: 12px; margin-top: 5px;">{self.wind_speed} m/s</div>
                    <div style="font-size: 12px;">{self.wind_direction}°</div>
                </div>
            </div>
            """

            # Add the HTML element to the map
            m.get_root().html.add_child(folium.Element(wind_html))

        # Save map to HTML
        data = io.BytesIO()
        m.save(data, close_file=False)

        # Load the HTML into the QWebEngineView
        html_content = data.getvalue().decode()
        self.webview.setHtml(html_content)
