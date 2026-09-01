# gui.py
"""
APT GUI
"""

import csv
import os
import tkinter as tk
from datetime import datetime, timedelta
from tkinter import filedialog, messagebox, ttk
import webbrowser
from config import get_base_url


class PrecipGUI:
    def __init__(self, root, dispatcher, data_dir="data"):
        self.root = root
        self.dispatcher = dispatcher
        self.data_dir = data_dir

        self.root.title("Antecedent Precipitation Tool")
        self.root.geometry("+0+0")
        self.root.minsize(480, 400)
        self.root.resizable(True, True)

        # Set window icon
        self._set_window_icon()

        # State
        self.date_mode = "unique"  # "unique" | "range" | "csv"
        self.date_entries = []  # list of (frame, entry) for unique mode
        self.custom_polygon_path = None

        self._build_ui()
        self._set_date_mode("unique")

    # ------------------------------------------------------------------
    # Set window icon
    # ------------------------------------------------------------------
    def _set_window_icon(self):
        """Set the application icon from data/Graph.ico"""
        icon_path = os.path.join(self.data_dir, "Graph.ico")
        try:
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
            else:
                # Fallback: try relative path from current working directory
                alt_path = os.path.join("data", "Graph.ico")
                if os.path.exists(alt_path):
                    self.root.iconbitmap(alt_path)
                else:
                    print(f"Warning: Icon not found at {icon_path} or {alt_path}")
        except Exception as e:
            print(f"Warning: Could not load icon: {e}")

    # ------------------------------------------------------------------
    # UI construction – mirrors official APT layout
    # ------------------------------------------------------------------
    def _build_ui(self):
        # Main container
        main = ttk.Frame(self.root, padding=6)
        main.pack(fill="both", expand=True)

        # ===== TOP ROW: Gridded + Streamflow + Help =====
        top = ttk.Frame(main)
        top.pack(fill="x", pady=(0, 4))

        self.gridded_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            top,
            text="Use Gridded Precipitation?",
            variable=self.gridded_var,
        ).pack(side="left", padx=(0, 12))

        self.streamflow_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            top,
            text="Calculate Local Streamflow Normal?",
            variable=self.streamflow_var,
            command=self._on_streamflow_toggle,
        ).pack(side="left", padx=(0, 12))

        ttk.Button(
            top, text="Help / More Info", command=self._open_help, width=16
        ).pack(side="right")

        ttk.Separator(main, orient="horizontal").pack(fill="x", pady=4)

        # ===== LAT / LON / SCOPE =====
        loc = ttk.Frame(main)
        loc.pack(fill="x", pady=2)

        ttk.Label(loc, text="Latitude (DD):").grid(row=0, column=0, sticky="w", padx=2)
        ttk.Label(loc, text="Longitude (-DD):").grid(
            row=0, column=1, sticky="w", padx=2
        )
        ttk.Label(loc, text="Scope").grid(row=0, column=2, sticky="w", padx=2)

        self.lat_entry = ttk.Entry(loc, width=14)
        self.lat_entry.grid(row=1, column=0, sticky="w", padx=2, pady=2)

        self.lon_entry = ttk.Entry(loc, width=14)
        self.lon_entry.grid(row=1, column=1, sticky="w", padx=2, pady=2)

        self.scope_var = tk.StringVar(value="Single Point")
        scope_opts = ["Single Point", "HUC12", "HUC10", "HUC8", "Custom Polygon"]
        self.scope_menu = ttk.OptionMenu(
            loc,
            self.scope_var,
            "Single Point",
            *scope_opts,
            command=self._on_scope_change,
        )
        self.scope_menu.grid(row=1, column=2, sticky="w", padx=2)

        # Custom polygon controls (hidden by default)
        self.custom_frame = ttk.Frame(main)
        ttk.Label(self.custom_frame, text="Custom Watershed Name:").grid(
            row=0, column=0, sticky="w", padx=2
        )
        self.custom_name_entry = ttk.Entry(self.custom_frame, width=30)
        self.custom_name_entry.grid(row=0, column=1, sticky="ew", padx=2)

        ttk.Label(self.custom_frame, text="Custom Watershed Shapefile:").grid(
            row=1, column=0, sticky="w", padx=2, pady=2
        )
        self.custom_path_entry = ttk.Entry(self.custom_frame, width=30)
        self.custom_path_entry.grid(row=1, column=1, sticky="ew", padx=2)
        ttk.Button(
            self.custom_frame, text="Browse…", command=self._browse_shapefile
        ).grid(row=1, column=2, padx=4)

        ttk.Separator(main, orient="horizontal").pack(fill="x", pady=6)

        # ===== DATES FRAME (content changes with mode) =====
        self.dates_outer = ttk.LabelFrame(main, text="Observation Date(s)", padding=6)
        self.dates_outer.pack(fill="both", expand=True, pady=2)

        self.dates_content = ttk.Frame(self.dates_outer)
        self.dates_content.pack(fill="both", expand=True)

        # ===== BOTTOM BUTTONS =====
        ttk.Separator(main, orient="horizontal").pack(fill="x", pady=6)

        bottom = ttk.Frame(main)
        bottom.pack(fill="x")

        self.calc_btn = ttk.Button(
            bottom, text="Calculate", command=self.run_btn_clicked, width=14
        )
        self.calc_btn.pack(side="left", padx=(0, 8))

        self.mode_btn_var = tk.StringVar(value="Switch to Date Range")
        self.mode_btn = ttk.Button(
            bottom,
            textvariable=self.mode_btn_var,
            command=self._cycle_date_mode,
            width=20,
        )
        self.mode_btn.pack(side="left", padx=(0, 8))

        ttk.Button(bottom, text="Quit", command=self.root.destroy, width=10).pack(
            side="right"
        )

        # Status
        self.status_var = tk.StringVar(value="Ready for Input")
        ttk.Label(main, textvariable=self.status_var, relief="sunken", anchor="w").pack(
            fill="x", side="bottom", pady=(6, 0)
        )

        # Internal analysis toggles (driven by the two top checkboxes)
        self.usgs_var = tk.BooleanVar(value=False)
        self.nwm_var = tk.BooleanVar(value=False)
        self.wimp_var = tk.BooleanVar(value=True)
        self.pdsi_var = tk.BooleanVar(value=True)

    # ------------------------------------------------------------------
    # Date mode switching (Unique → Range → CSV → Unique …)
    # ------------------------------------------------------------------
    def _cycle_date_mode(self):
        order = ["unique", "range", "csv"]
        idx = order.index(self.date_mode)
        next_mode = order[(idx + 1) % 3]
        self._set_date_mode(next_mode)

    def _set_date_mode(self, mode: str):
        self.date_mode = mode
        # Clear current content
        for w in self.dates_content.winfo_children():
            w.destroy()
        self.date_entries.clear()

        if mode == "unique":
            self.mode_btn_var.set("Switch to Date Range")
            self._build_unique_dates()
        elif mode == "range":
            self.mode_btn_var.set("Switch to CSV Input")
            self._build_date_range()
        else:
            self.mode_btn_var.set("Switch to Unique Dates")
            self._build_csv_input()

    def _build_unique_dates(self):
        ttk.Label(
            self.dates_content,
            text='Run a single date or click "+" to add more',
        ).pack(anchor="w", pady=(0, 4))

        ttk.Separator(self.dates_content, orient="horizontal").pack(fill="x", pady=2)

        header = ttk.Frame(self.dates_content)
        header.pack(fill="x")
        ttk.Label(header, text="#", width=4).pack(side="left")
        ttk.Label(header, text="YYYY-MM-DD").pack(side="left", padx=8)

        self.unique_list_frame = ttk.Frame(self.dates_content)
        self.unique_list_frame.pack(fill="both", expand=True, pady=4)

        btn_row = ttk.Frame(self.dates_content)
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="+", width=3, command=self._add_unique_date).pack(
            side="left"
        )
        ttk.Button(btn_row, text="–", width=3, command=self._remove_unique_date).pack(
            side="left", padx=4
        )

        # Start with one date
        self._add_unique_date()

    def _add_unique_date(self):
        row = ttk.Frame(self.unique_list_frame)
        row.pack(fill="x", pady=1)
        num = len(self.date_entries) + 1
        ttk.Label(row, text=str(num), width=4).pack(side="left")
        entry = ttk.Entry(row, width=14)
        entry.pack(side="left", padx=8)
        self.date_entries.append((row, entry))

    def _remove_unique_date(self):
        if len(self.date_entries) <= 1:
            return
        row, _ = self.date_entries.pop()
        row.destroy()
        # renumber
        for i, (r, _) in enumerate(self.date_entries, 1):
            for child in r.winfo_children():
                if isinstance(child, ttk.Label):
                    child.config(text=str(i))
                    break

    def _build_date_range(self):
        ttk.Label(
            self.dates_content,
            text="Get daily results between a Start Date and End Date",
        ).pack(anchor="w", pady=(0, 6))

        ttk.Separator(self.dates_content, orient="horizontal").pack(fill="x", pady=2)

        row1 = ttk.Frame(self.dates_content)
        row1.pack(fill="x", pady=4)
        ttk.Label(row1, text="Start Date (YYYY-MM-DD):", width=22).pack(side="left")
        self.start_entry = ttk.Entry(row1, width=14)
        self.start_entry.pack(side="left")

        row2 = ttk.Frame(self.dates_content)
        row2.pack(fill="x", pady=4)
        ttk.Label(row2, text="End Date (YYYY-MM-DD):", width=22).pack(side="left")
        self.end_entry = ttk.Entry(row2, width=14)
        self.end_entry.pack(side="left")

    def _build_csv_input(self):
        ttk.Label(
            self.dates_content,
            text="Use a CSV file to run many dates at once",
        ).pack(anchor="w", pady=(0, 6))

        ttk.Separator(self.dates_content, orient="horizontal").pack(fill="x", pady=2)

        ttk.Label(self.dates_content, text="CSV File Path:").pack(
            anchor="w", pady=(4, 0)
        )
        row = ttk.Frame(self.dates_content)
        row.pack(fill="x", pady=2)
        self.csv_entry = ttk.Entry(row)
        self.csv_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(row, text="Browse…", command=self._browse_csv).pack(side="left")

    # ------------------------------------------------------------------
    # Scope / Streamflow helpers
    # ------------------------------------------------------------------
    def _on_scope_change(self, *_):
        scope = self.scope_var.get()
        if scope == "Custom Polygon":
            self.custom_frame.pack(
                fill="x",
                pady=4,
                after=self.root.nametowidget(str(self.scope_menu.master)),
            )
        else:
            self.custom_frame.pack_forget()

    def _on_streamflow_toggle(self):
        on = self.streamflow_var.get()
        self.usgs_var.set(on)
        self.nwm_var.set(on)

    def _browse_shapefile(self):
        path = filedialog.askopenfilename(
            title="Select Watershed Shapefile",
            filetypes=[
                ("Shapefile", "*.shp"),
                ("GeoJSON", "*.geojson"),
                ("All", "*.*"),
            ],
        )
        if path:
            self.custom_path_entry.delete(0, tk.END)
            self.custom_path_entry.insert(0, path)
            self.custom_polygon_path = path

    def _browse_csv(self):
        path = filedialog.askopenfilename(
            title="Select CSV of dates",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
        )
        if path:
            self.csv_entry.delete(0, tk.END)
            self.csv_entry.insert(0, path)

    def _open_help(self):
        help_url = get_base_url("apt_help")
        if help_url:
            webbrowser.open(help_url)

    # ------------------------------------------------------------------
    # Collect dates
    # ------------------------------------------------------------------
    def _get_dates(self):
        dates = []
        if self.date_mode == "unique":
            for _, entry in self.date_entries:
                txt = entry.get().strip()
                if not txt:
                    continue
                dates.append(datetime.strptime(txt, "%Y-%m-%d"))
        elif self.date_mode == "range":
            start = datetime.strptime(self.start_entry.get().strip(), "%Y-%m-%d")
            end = datetime.strptime(self.end_entry.get().strip(), "%Y-%m-%d")
            if end < start:
                raise ValueError("End date must be on or after start date")
            cur = start
            while cur <= end:
                dates.append(cur)
                cur += timedelta(days=1)
        else:  # csv
            path = self.csv_entry.get().strip()
            if not path or not os.path.isfile(path):
                raise ValueError("Valid CSV file path required")
            with open(path, newline="", encoding="utf-8-sig") as fh:
                reader = csv.reader(fh)
                for row in reader:
                    for cell in row:
                        cell = cell.strip()
                        try:
                            dates.append(datetime.strptime(cell, "%Y-%m-%d"))
                            break
                        except ValueError:
                            continue
            dates = sorted(set(dates))

        if not dates:
            raise ValueError("No valid dates provided")
        return dates

    def _selected_analysis_types(self):
        types = ["precip"]  # always run precip
        if self.usgs_var.get():
            types.append("usgs")
        if self.nwm_var.get():
            types.append("nwm")
        if self.wimp_var.get():
            types.append("wimp")
        if self.pdsi_var.get():
            types.append("pdsi")
        return types

    # ------------------------------------------------------------------
    # Calculate
    # ------------------------------------------------------------------
    def run_btn_clicked(self):
        try:
            lat = float(self.lat_entry.get().strip())
            lon = float(self.lon_entry.get().strip())
            dates = self._get_dates()
            analysis_types = self._selected_analysis_types()
            scope = self.scope_var.get()

            self.calc_btn.config(state="disabled")
            self.status_var.set("Processing")
            self.root.update_idletasks()

            is_huc = scope in ("HUC12", "HUC10", "HUC8")
            is_custom = scope == "Custom Polygon"

            if is_huc or is_custom:
                # HUC / custom currently single-date in the engine
                if len(dates) > 1:
                    messagebox.showinfo(
                        "Note",
                        "Watershed modes currently use the first date only.\n"
                        "For multi-date watershed runs, use Single Point + date list/range.",
                    )
                analysis_date = dates[0]
                huc_level = {"HUC12": 12, "HUC10": 10, "HUC8": 8}.get(scope, 8)

                msg = {
                    "message_type": "huc_analysis",
                    "lat": lat,
                    "lon": lon,
                    "analysis_date": analysis_date,
                    "output_dir": "output",
                    "data_dir": self.data_dir,
                    "huc_level": huc_level,
                    "analysis_types": analysis_types,
                }
                if self.gridded_var.get():
                    msg["gridded"] = True
                if is_custom and self.custom_polygon_path:
                    msg["custom_polygon"] = self.custom_polygon_path
                    msg["custom_name"] = (
                        self.custom_name_entry.get().strip() or "CUSTOM"
                    )
                self.dispatcher.notify(msg)
            else:
                # Single Point – full multi-date support
                for current_date in dates:
                    bulk = {
                        "lat": lat,
                        "lon": lon,
                        "analysis_date": current_date,
                        "output_dir": "output",
                        "data_dir": self.data_dir,
                    }
                    if "precip" in analysis_types:
                        msg = {"message_type": "precip_analysis", **bulk}
                        if self.gridded_var.get():
                            msg["gridded"] = True
                        self.dispatcher.notify(msg)
                    if "usgs" in analysis_types:
                        self.dispatcher.notify(
                            {"message_type": "usgs_analysis", **bulk}
                        )
                    if "nwm" in analysis_types:
                        self.dispatcher.notify({"message_type": "nwm_analysis", **bulk})
                    if "wimp" in analysis_types:
                        self.dispatcher.notify(
                            {"message_type": "wimp_analysis", **bulk}
                        )
                    if "pdsi" in analysis_types:
                        self.dispatcher.notify(
                            {"message_type": "pdsi_analysis", **bulk}
                        )
                    self.dispatcher.notify(
                        {
                            "message_type": "generate_pdf",
                            "analysis_types": analysis_types,
                            **bulk,
                        }
                    )

                if len(dates) > 1:
                    self.dispatcher.notify(
                        {
                            "message_type": "merge_pdfs",
                            "lat": lat,
                            "lon": lon,
                            "start_date": dates[0],
                            "end_date": dates[-1],
                            "output_dir": "output",
                        }
                    )

        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Error")
        finally:
            self.status_var.set("Ready for Input")
            self.calc_btn.config(state="normal")
