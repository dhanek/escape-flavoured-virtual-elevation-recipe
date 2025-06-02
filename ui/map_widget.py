import io

import folium
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QVBoxLayout, QWidget
from enum import Enum, auto

class MapMode(Enum):
    LAPS = auto()
    TRIM = auto()
    MARKER = auto()
    MARKER_AB = auto()
    MARKER_GATE_SETS = auto()

class MapWidget(QWidget):
    def __init__(self, mode: MapMode, records, params):
        super().__init__()
        self.records = records
        if ("position_lat" in records and "position_long" in records):
            self.has_gps = True
            self.merged_data = records
            # Filter out records with no position data
            records = self.merged_data.dropna(subset=["position_lat", "position_long"])
            self.route_points = list(
                zip(records["position_lat"], records["position_long"])
            )
            self.route_timestamps = records["timestamp"].tolist()
            self.center_lat = records["position_lat"].mean()
            self.center_lon = records["position_long"].mean()
            if not self.route_points:
                self.has_gps = False
            self.trim_start = 0
            self.trim_end = len(self.merged_data) - 1
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
        self.marker_pos = 0
        self.marker_b_pos = 0
        self.detected_sections = []
        self.gate_sets = []
        self.mode = mode

    def set_selected_laps(self, lap_data, selected_laps):
        """Update the selected laps and redraw the map"""
        self.lap_data = lap_data
        self.selected_laps = selected_laps

    def set_wind(self, wind_speed, wind_direction):
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction

    def set_trim_start(self, start):
        self.trim_start = start

    def set_trim_end(self, end):
        self.trim_end = end

    def set_marker_pos(self, value):
        self.marker_pos = value

    def set_marker_a_pos(self, value):
        self.set_marker_pos(value)

    def set_marker_b_pos(self, value):
        self.marker_b_pos = value

    def set_gate_sets(self, gate_sets, detected_sections):
        self.gate_sets = gate_sets
        self.detected_sections = detected_sections

    def map_time_to_route_index(self, time_index):
        """
        Map a time index from the full dataset to the corresponding index in route_points

        Parameters:
        -----------
        time_index : int
            Index in the merged_data dataframe

        Returns:
        --------
        int
            Corresponding index in the route_points list, or None if not mappable
        """

        if time_index < 0 or time_index >= len(self.merged_data):
            return None

        # Get the timestamp at this index
        target_timestamp = self.merged_data["timestamp"].iloc[time_index]

        # Find the closest timestamp in route_timestamps
        if target_timestamp in self.route_timestamps:
            return self.route_timestamps.index(target_timestamp)

        # If not found directly, find the closest one
        for i, ts in enumerate(self.route_timestamps):
            if ts >= target_timestamp:
                return i

        # If we get here, target_timestamp is after all route_timestamps
        return len(self.route_timestamps) - 1

    def draw_marker_gate_sets(self, m, route_trim_start, trimmed_route,
                              valid_gate_positions):
        # Add gate markers
        gate_colors = [
            "blue",
            "purple",
            "orange",
            "cadetblue",
            "darkred",
            "darkgreen",
        ]

        for i, (gate_a_pos, gate_b_pos) in enumerate(valid_gate_positions):
            color = gate_colors[i % len(gate_colors)]
            gate_set_idx = i

            # Gate A marker
            if 0 <= gate_a_pos - route_trim_start < len(trimmed_route):
                gate_a_loc = trimmed_route[gate_a_pos - route_trim_start]

                # Use CircleMarker for Gate A
                folium.CircleMarker(
                    location=gate_a_loc,
                    radius=8,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.8,
                    popup=f"Gate {gate_set_idx+1}A",
                    weight=2,
                ).add_to(m)

                # Add label
                folium.map.Marker(
                    gate_a_loc,
                    icon=folium.DivIcon(
                        icon_size=(20, 20),
                        icon_anchor=(10, 10),
                        html=f'<div style="font-size: 12pt; font-weight: bold; color: white; background-color: {color}; border-radius: 50%; width: 20px; height: 20px; line-height: 20px; text-align: center;">{gate_set_idx+1}A</div>',
                    ),
                ).add_to(m)

                # Add detection radius
                folium.Circle(
                    location=gate_a_loc,
                    radius=20,  # 20 meters
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.2,
                    popup=f"Gate {gate_set_idx+1}A Detection Zone",
                ).add_to(m)

            # Gate B marker
            if 0 <= gate_b_pos - route_trim_start < len(trimmed_route):
                gate_b_loc = trimmed_route[gate_b_pos - route_trim_start]

                # Use CircleMarker for Gate B
                folium.CircleMarker(
                    location=gate_b_loc,
                    radius=8,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.8,
                    popup=f"Gate {gate_set_idx+1}B",
                    weight=2,
                ).add_to(m)

                # Add label
                folium.map.Marker(
                    gate_b_loc,
                    icon=folium.DivIcon(
                        icon_size=(20, 20),
                        icon_anchor=(10, 10),
                        html=f'<div style="font-size: 12pt; font-weight: bold; color: white; background-color: {color}; border-radius: 50%; width: 20px; height: 20px; line-height: 20px; text-align: center;">{gate_set_idx+1}B</div>',
                    ),
                ).add_to(m)

                # Add detection radius
                folium.Circle(
                    location=gate_b_loc,
                    radius=20,  # 20 meters
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.2,
                    popup=f"Gate {gate_set_idx+1}B Detection Zone",
                ).add_to(m)

        # Highlight detected sections if available
        section_colors = {
            0: "#ff7f0e",  # Orange
            1: "#2ca02c",  # Green
            2: "#d62728",  # Red
            3: "#9467bd",  # Purple
            4: "#8c564b",  # Brown
            5: "#e377c2",  # Pink
        }

        for i, section in enumerate(self.detected_sections):
            gate_idx = section.get("gate_set", 0)
            section_color = section_colors.get(
                gate_idx, "#1f77b4"
            )  # Default blue

            # Section: A to B
            if "start_idx" in section and "end_idx" in section:
                section_start = self.map_time_to_route_index(
                    section["start_idx"]
                )
                section_end = self.map_time_to_route_index(section["end_idx"])

                if section_start is not None and section_end is not None:
                    section_start = max(
                        0, min(section_start, len(self.route_points) - 1)
                    )
                    section_end = max(
                        section_start,
                        min(section_end, len(self.route_points) - 1),
                    )

                    section_route = self.route_points[
                        section_start : section_end + 1
                    ]
                    if len(section_route) > 1:
                        folium.PolyLine(
                            section_route,
                            color=section_color,
                            weight=5,
                            opacity=0.7,
                            popup=f"Section {section.get('section_id', i+1)}: Gate {gate_idx+1}A → {gate_idx+1}B",
                        ).add_to(m)


    def draw_marker_ab(self, m, route_trim_start, trimmed_route, route_marker_a_pos,
                       route_marker_b_pos):
        if 0 <= route_marker_a_pos - route_trim_start < len(trimmed_route):
            marker_a_location = trimmed_route[route_marker_a_pos - route_trim_start]

            # Use a CircleMarker instead of standard marker
            folium.CircleMarker(
                location=marker_a_location,
                radius=8,
                color="#3186cc",
                fill=True,
                fill_color="#3186cc",
                fill_opacity=0.8,
                popup="Point A",
                weight=2,
            ).add_to(m)

            # Add a letter label using DivIcon
            folium.map.Marker(
                marker_a_location,
                icon=folium.DivIcon(
                    icon_size=(20, 20),
                    icon_anchor=(10, 10),
                    html='<div style="font-size: 12pt; font-weight: bold; color: white; background-color: #3186cc; border-radius: 50%; width: 20px; height: 20px; line-height: 20px; text-align: center;">A</div>',
                ),
            ).add_to(m)

            # Add detection radius
            folium.Circle(
                location=marker_a_location,
                radius=20,  # 20 meters
                color="#3186cc",
                fill=True,
                fill_color="#3186cc",
                fill_opacity=0.2,
                popup="Point A Detection Zone",
            ).add_to(m)

        # Replace marker B implementation
        if 0 <= route_marker_b_pos - route_trim_start < len(trimmed_route):
            marker_b_location = trimmed_route[route_marker_b_pos - route_trim_start]

            # Use a CircleMarker instead of standard marker
            folium.CircleMarker(
                location=marker_b_location,
                radius=8,
                color="#9c27b0",
                fill=True,
                fill_color="#9c27b0",
                fill_opacity=0.8,
                popup="Point B",
                weight=2,
            ).add_to(m)

            # Add a letter label using DivIcon
            folium.map.Marker(
                marker_b_location,
                icon=folium.DivIcon(
                    icon_size=(20, 20),
                    icon_anchor=(10, 10),
                    html='<div style="font-size: 12pt; font-weight: bold; color: white; background-color: #9c27b0; border-radius: 50%; width: 20px; height: 20px; line-height: 20px; text-align: center;">B</div>',
                ),
            ).add_to(m)

            # Add detection radius
            folium.Circle(
                location=marker_b_location,
                radius=20,  # 20 meters
                color="#9c27b0",
                fill=True,
                fill_color="#9c27b0",
                fill_opacity=0.2,
                popup="Point B Detection Zone",
            ).add_to(m)

            # 4. Highlight detected sections if available
            if hasattr(self, "detected_sections") and self.detected_sections:
                for i, section in enumerate(self.detected_sections):
                    # Use different colors for outbound and inbound parts
                    outbound_color = "#ff7f0e"  # Orange
                    inbound_color = "#2ca02c"  # Green

                    # Outbound: A to B
                    if (
                        "outbound_start_idx" in section
                        and "outbound_end_idx" in section
                    ):
                        outbound_start = self.map_time_to_route_index(
                            section["outbound_start_idx"]
                        )
                        outbound_end = self.map_time_to_route_index(
                            section["outbound_end_idx"]
                        )

                        if outbound_start is not None and outbound_end is not None:
                            outbound_start = max(
                                0, min(outbound_start, len(self.route_points) - 1)
                            )
                            outbound_end = max(
                                outbound_start,
                                min(outbound_end, len(self.route_points) - 1),
                            )

                            outbound_route = self.route_points[
                                outbound_start : outbound_end + 1
                            ]
                            if len(outbound_route) > 1:
                                folium.PolyLine(
                                    outbound_route,
                                    color=outbound_color,
                                    weight=5,
                                    opacity=0.7,
                                    popup=f"Section {i+1} Outbound (A→B)",
                                ).add_to(m)

                    # Inbound: B to A
                    if (
                        "inbound_start_idx" in section
                        and "inbound_end_idx" in section
                    ):
                        inbound_start = self.map_time_to_route_index(
                            section["inbound_start_idx"]
                        )
                        inbound_end = self.map_time_to_route_index(
                            section["inbound_end_idx"]
                        )

                        if inbound_start is not None and inbound_end is not None:
                            inbound_start = max(
                                0, min(inbound_start, len(self.route_points) - 1)
                            )
                            inbound_end = max(
                                inbound_start,
                                min(inbound_end, len(self.route_points) - 1),
                            )

                            inbound_route = self.route_points[
                                inbound_start : inbound_end + 1
                            ]
                            if len(inbound_route) > 1:
                                folium.PolyLine(
                                    inbound_route,
                                    color=inbound_color,
                                    weight=5,
                                    opacity=0.7,
                                    popup=f"Section {i+1} Inbound (B→A)",
                                ).add_to(m)



    def draw_marker(self, m, route_trim_start, trimmed_route, route_marker_pos):
        if 0 <= route_marker_pos - route_trim_start < len(trimmed_route):
            marker_location = trimmed_route[route_marker_pos - route_trim_start]
            folium.Marker(
                location=marker_location,
                popup="GPS Lap Marker",
                icon=folium.Icon(color="orange", icon="flag", prefix="fa"),
            ).add_to(m)

            # Add a circle around the GPS marker to show the detection radius
            folium.Circle(
                location=marker_location,
                radius=20,  # 20 meters radius for detection
                color="#FFA500",
                fill=True,
                fill_color="#FFA500",
                fill_opacity=0.2,
                popup="GPS Marker Detection Zone",
            ).add_to(m)

    def draw_trim(self, m):
        try:
            # Map time indices to route indices
            if not self.route_timestamps:
                # Fall back to simple index mapping if no timestamps available
                total_points = len(self.route_points)
                total_records = len(self.merged_data)

                if total_points > 0 and total_records > 0:
                    route_trim_start = min(
                        int(self.trim_start * total_points / total_records),
                        total_points - 1,
                    )
                    route_trim_end = min(
                        int(self.trim_end * total_points / total_records),
                        total_points - 1,
                    )
                    route_marker_a_pos = route_marker_pos = min(
                        int(self.marker_pos * total_points / total_records),
                        total_points - 1,
                    )
                    route_marker_b_pos = min(
                        int(self.marker_b_pos * total_points / total_records),
                        total_points - 1,
                    )

                    # Map gate positions
                    route_gate_positions = []
                    for gate_set in self.gate_sets:
                        gate_a = min(
                            int(gate_set["gate_a_pos"] * total_points / total_records),
                            total_points - 1,
                        )
                        gate_b = min(
                            int(gate_set["gate_b_pos"] * total_points / total_records),
                            total_points - 1,
                        )
                        route_gate_positions.append((gate_a, gate_b))
                else:
                    route_trim_start = 0
                    route_trim_end = len(self.route_points) - 1
                    route_marker_pos = int((route_trim_start + route_trim_end) / 2)
                    route_marker_a_pos = int((route_trim_start + route_trim_end) * 0.25)
                    route_marker_b_pos = int((route_trim_start + route_trim_end) * 0.75)
                    route_gate_positions = []
            else:
                # Map using timestamps
                route_trim_start = self.map_time_to_route_index(self.trim_start)
                route_trim_end = self.map_time_to_route_index(self.trim_end)
                route_marker_a_pos = route_marker_pos = self.map_time_to_route_index(self.marker_pos)
                route_marker_b_pos = self.map_time_to_route_index(self.marker_b_pos)

                # Map gate positions
                route_gate_positions = []
                for gate_set in self.gate_sets:
                    gate_a = self.map_time_to_route_index(gate_set["gate_a_pos"])
                    gate_b = self.map_time_to_route_index(gate_set["gate_b_pos"])
                    route_gate_positions.append((gate_a, gate_b))

                # Fall back if mapping fails
                if (route_trim_start is None or route_trim_end is None
                    or route_marker_pos is None
                ):
                    total_points = len(self.route_points)
                    total_records = len(self.merged_data)

                    if total_points > 0 and total_records > 0:
                        route_trim_start = min(
                            int(self.trim_start * total_points / total_records),
                            total_points - 1,
                        )
                        route_trim_end = min(
                            int(self.trim_end * total_points / total_records),
                            total_points - 1,
                        )
                        route_marker_a_pos = route_marker_pos = min(
                            int(self.marker_pos * total_points / total_records),
                            total_points - 1,
                        )
                        route_marker_b_pos = min(
                            int(self.marker_b_pos * total_points / total_records),
                            total_points - 1,
                        )
                    else:
                        route_trim_start = 0
                        route_trim_end = len(self.route_points) - 1
                        route_marker_pos = int((route_trim_start + route_trim_end) / 2)
                        route_marker_a_pos = int(
                            (route_trim_start + route_trim_end) * 0.25
                        )
                        route_marker_b_pos = int(
                            (route_trim_start + route_trim_end) * 0.75
                        )

            # Make sure indices are valid
            route_trim_start = max(0, min(route_trim_start, len(self.route_points) - 1))
            route_trim_end = max(
                route_trim_start, min(route_trim_end, len(self.route_points) - 1)
            )
            route_marker_pos = max(
                route_trim_start, min(route_marker_pos, route_trim_end)
            )
            route_marker_a_pos = max(
                route_trim_start, min(route_marker_a_pos, route_trim_end)
            )
            route_marker_b_pos = max(
                route_trim_start, min(route_marker_b_pos, route_trim_end)
            )

            # Clean up gate positions
            valid_gate_positions = []
            for gate_a, gate_b in route_gate_positions:
                if gate_a is not None and gate_b is not None:
                    gate_a = max(0, min(gate_a, len(self.route_points) - 1))
                    gate_b = max(gate_a, min(gate_b, len(self.route_points) - 1))
                    valid_gate_positions.append((gate_a, gate_b))

            # 1. Draw the parts before trim_start with dashed blue line (if exists)
            if route_trim_start > 0:
                pre_trim_route = self.route_points[: route_trim_start + 1]
                folium.PolyLine(
                    pre_trim_route,
                    color="#4363d8",  # Blue color
                    weight=3,
                    opacity=0.5,
                    dash_array="5,10",  # Dashed line
                    popup="Pre-selected portion",
                ).add_to(m)

            # 2. Draw the parts after trim_end with dashed blue line (if exists)
            if route_trim_end < len(self.route_points) - 1:
                post_trim_route = self.route_points[route_trim_end:]
                folium.PolyLine(
                    post_trim_route,
                    color="#4363d8",  # Blue color
                    weight=3,
                    opacity=0.5,
                    dash_array="5,10",  # Dashed line
                    popup="Post-selected portion",
                ).add_to(m)

            # 3. Draw the selected portion with solid blue line
            trimmed_route = self.route_points[route_trim_start : route_trim_end + 1]
            if len(trimmed_route) > 1:
                folium.PolyLine(
                    trimmed_route,
                    color="#4363d8",  # Blue color
                    weight=5,  # Slightly thicker
                    opacity=1.0,  # Full opacity
                    popup="Selected portion",
                ).add_to(m)

                # Add trim markers
                folium.Marker(
                    location=trimmed_route[0],
                    popup="Trim Start",
                    icon=folium.Icon(color="green", icon="play", prefix="fa"),
                ).add_to(m)

                folium.Marker(
                    location=trimmed_route[-1],
                    popup="Trim End",
                    icon=folium.Icon(color="red", icon="stop", prefix="fa"),
                ).add_to(m)

                # Add GPS marker
                if self.mode == MapMode.MARKER:
                    self.draw_marker(m, route_trim_start, trimmed_route,
                                     route_marker_pos)
                elif self.mode == MapMode.MARKER_AB:
                    self.draw_marker_ab(m, route_trim_start, trimmed_route,
                                        route_marker_a_pos, route_marker_b_pos)
                elif self.mode == MapMode.MARKER_GATE_SETS:
                    self.draw_marker_gate_sets(m, route_trim_start, trimmed_route,
                                               valid_gate_positions)

        except Exception as e:
            print(f"Error highlighting trimmed route: {e}")

        return self.route_points

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
        if (self.mode == MapMode.TRIM
            or self.mode == MapMode.MARKER
            or self.mode == MapMode.MARKER_AB
            or self.mode == MapMode.MARKER_GATE_SETS):
            points_to_zoom = self.draw_trim(m)
        elif self.mode == MapMode.LAPS:
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
