# gui.py
"""
APT-style GUI – drop-in replacement.
Preserves the EventDispatcher contract used by the rest of the codebase.
"""

import csv
import tkinter as tk
from datetime import datetime, timedelta
from tkinter import filedialog, messagebox, ttk
from urllib.request import urlopen
import webbrowser


class PrecipGUI:
    def __init__(self, root, dispatcher):
        self.root = root
        self.dispatcher = dispatcher
        self.root.title("Antecedent Precipitation Tool (APT)")
        self.root.geometry("520x640")
        self.root.minsize(480, 580)

        # ----- state -----
        self.date_list = []  # list of datetime objects for multi-date mode
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        pad = {"padx": 8, "pady": 3}
        f = ttk.Frame(self.root, padding=10)
        f.pack(fill="both", expand=True)

        # ---- Location ----
        loc = ttk.LabelFrame(f, text="Location", padding=6)
        loc.pack(fill="x", **pad)

        ttk.Label(loc, text="Latitude:").grid(row=0, column=0, sticky="e", padx=4)
        self.lat_entry = ttk.Entry(loc, width=12)
        self.lat_entry.grid(row=0, column=1, sticky="w", padx=4)
        self.lat_entry.insert(0, "30.0")

        ttk.Label(loc, text="Longitude:").grid(row=0, column=2, sticky="e", padx=4)
        self.lon_entry = ttk.Entry(loc, width=12)
        self.lon_entry.grid(row=0, column=3, sticky="w", padx=4)
        self.lon_entry.insert(0, "-90.0")

        # ---- Dates ----
        date_fr = ttk.LabelFrame(f, text="Observation Date(s)", padding=6)
        date_fr.pack(fill="x", **pad)

        ttk.Label(date_fr, text="Date (YYYY-MM-DD):").grid(row=0, column=0, sticky="e")
        self.date_entry = ttk.Entry(date_fr, width=12)
        self.date_entry.grid(row=0, column=1, sticky="w", padx=4)
        self.date_entry.insert(0, "2023-11-11")

        btn_row = ttk.Frame(date_fr)
        btn_row.grid(row=0, column=2, columnspan=2, sticky="w")
        ttk.Button(btn_row, text="Add →", width=8, command=self._add_date).pack(
            side="left", padx=2
        )
        ttk.Button(btn_row, text="Clear", width=6, command=self._clear_dates).pack(
            side="left", padx=2
        )
        ttk.Button(btn_row, text="CSV…", width=6, command=self._load_csv).pack(
            side="left", padx=2
        )

        # list of selected dates
        self.date_listbox = tk.Listbox(
            date_fr, height=4, width=50, exportselection=False
        )
        self.date_listbox.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(4, 0))
        date_fr.columnconfigure(3, weight=1)

        # batch / range mode
        self.batch_mode_var = tk.BooleanVar(value=False)
        self.chk_batch = ttk.Checkbutton(
            date_fr,
            text="Date Range (start → end)",
            variable=self.batch_mode_var,
            command=self._toggle_batch_mode,
        )
        self.chk_batch.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

        self.end_date_label = ttk.Label(date_fr, text="End Date:")
        self.end_date_entry = ttk.Entry(date_fr, width=12)

        # ---- Analysis toggles ----
        an = ttk.LabelFrame(f, text="Analyses to run", padding=6)
        an.pack(fill="x", **pad)

        # Precipitation source
        ttk.Label(an, text="Precipitation:").grid(row=0, column=0, sticky="w")
        self.precip_source = tk.StringVar(value="ghcn")
        ttk.Radiobutton(
            an, text="GHCN Station", variable=self.precip_source, value="ghcn"
        ).grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(
            an, text="Gridded (nClimGrid)", variable=self.precip_source, value="gridded"
        ).grid(row=0, column=2, sticky="w")

        # Streamflow master
        self.streamflow_var = tk.BooleanVar(value=True)
        self.chk_stream = ttk.Checkbutton(
            an,
            text="Calculate Local Streamflow Normal (USGS + NWM)",
            variable=self.streamflow_var,
            command=self._toggle_streamflow,
        )
        self.chk_stream.grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))

        # individual streamflow (kept for flexibility, driven by master)
        self.usgs_var = tk.BooleanVar(value=True)
        self.nwm_var = tk.BooleanVar(value=True)
        self.chk_usgs = ttk.Checkbutton(
            an, text="USGS Streamflow", variable=self.usgs_var
        )
        self.chk_usgs.grid(row=2, column=1, sticky="w")
        self.chk_nwm = ttk.Checkbutton(an, text="NWM Streamflow", variable=self.nwm_var)
        self.chk_nwm.grid(row=2, column=2, sticky="w")

        # other indices
        self.wimp_var = tk.BooleanVar(value=True)
        self.pdsi_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(an, text="WIMP Season", variable=self.wimp_var).grid(
            row=3, column=1, sticky="w"
        )
        ttk.Checkbutton(an, text="PDSI Analysis", variable=self.pdsi_var).grid(
            row=3, column=2, sticky="w"
        )

        # ---- AOI / Scope ----
        aoi = ttk.LabelFrame(f, text="Area of Interest", padding=6)
        aoi.pack(fill="x", **pad)

        self.aoi_mode = tk.StringVar(value="point")
        ttk.Radiobutton(
            aoi,
            text="Single Point",
            variable=self.aoi_mode,
            value="point",
            command=self._toggle_aoi,
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            aoi,
            text="HUC Watershed",
            variable=self.aoi_mode,
            value="huc",
            command=self._toggle_aoi,
        ).grid(row=0, column=1, sticky="w")

        self.huc_level_label = ttk.Label(aoi, text="HUC Level (2–12):")
        self.huc_level_entry = ttk.Entry(aoi, width=6)
        self.huc_level_entry.insert(0, "8")

        # ---- Run / status ----
        btn_fr = ttk.Frame(f)
        btn_fr.pack(fill="x", pady=12)

        self.run_btn = ttk.Button(
            btn_fr, text="Run Analysis", command=self.run_btn_clicked
        )
        self.run_btn.pack(side="left", padx=(0, 8))

        ttk.Button(btn_fr, text="Help", command=self._open_help).pack(side="left")

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(f, textvariable=self.status_var, relief="sunken", anchor="w").pack(
            fill="x", side="bottom", pady=(4, 0)
        )

        # initial state
        self._toggle_batch_mode()
        self._toggle_aoi()
        self._toggle_streamflow()

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def _toggle_batch_mode(self):
        if self.batch_mode_var.get():
            self.end_date_label.grid(row=2, column=2, sticky="e", padx=4)
            self.end_date_entry.grid(row=2, column=3, sticky="w")
            if not self.end_date_entry.get():
                self.end_date_entry.insert(0, "2023-11-20")
            # range mode clears explicit list for clarity
            self._clear_dates()
            if self.aoi_mode.get() == "huc":
                self.aoi_mode.set("point")
                self._toggle_aoi()
        else:
            self.end_date_label.grid_forget()
            self.end_date_entry.grid_forget()

    def _toggle_aoi(self):
        if self.aoi_mode.get() == "huc":
            self.huc_level_label.grid(row=1, column=0, sticky="e", padx=4, pady=4)
            self.huc_level_entry.grid(row=1, column=1, sticky="w", pady=4)
            # HUC currently single-date only in the engine path
            self.batch_mode_var.set(False)
            self._toggle_batch_mode()
            self._clear_dates()
        else:
            self.huc_level_label.grid_forget()
            self.huc_level_entry.grid_forget()

    def _toggle_streamflow(self):
        on = self.streamflow_var.get()
        self.usgs_var.set(on)
        self.nwm_var.set(on)
        state = "normal" if on else "disabled"
        self.chk_usgs.configure(state=state)
        self.chk_nwm.configure(state=state)

    def _add_date(self):
        try:
            d = datetime.strptime(self.date_entry.get().strip(), "%Y-%m-%d")
        except ValueError:
            messagebox.showerror("Invalid Date", "Use YYYY-MM-DD format.")
            return
        if d not in self.date_list:
            self.date_list.append(d)
            self.date_list.sort()
            self._refresh_listbox()
        self.batch_mode_var.set(False)
        self._toggle_batch_mode()

    def _clear_dates(self):
        self.date_list.clear()
        self._refresh_listbox()

    def _refresh_listbox(self):
        self.date_listbox.delete(0, tk.END)
        for d in self.date_list:
            self.date_listbox.insert(tk.END, d.strftime("%Y-%m-%d"))

    def _load_csv(self):
        path = filedialog.askopenfilename(
            title="Select CSV of dates",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        added = 0
        try:
            with open(path, newline="", encoding="utf-8-sig") as fh:
                reader = csv.reader(fh)
                for row in reader:
                    if not row:
                        continue
                    # accept first column that looks like a date
                    for cell in row:
                        cell = cell.strip()
                        try:
                            d = datetime.strptime(cell, "%Y-%m-%d")
                            if d not in self.date_list:
                                self.date_list.append(d)
                                added += 1
                            break
                        except ValueError:
                            continue
            self.date_list.sort()
            self._refresh_listbox()
            self.batch_mode_var.set(False)
            self._toggle_batch_mode()
            self.status_var.set(f"Loaded {added} date(s) from CSV")
        except Exception as e:
            messagebox.showerror("CSV Error", str(e))

    def _open_help(self):
        # Official technical / user guide
        webbrowser.open("http://dx.doi.org/10.21079/11681/49835")

    # ------------------------------------------------------------------
    # Analysis selection helpers
    # ------------------------------------------------------------------
    def _selected_analysis_types(self):
        types = []
        # precip is always implied if either source is chosen
        if self.precip_source.get() in ("ghcn", "gridded"):
            types.append("precip")
        if self.usgs_var.get():
            types.append("usgs")
        if self.nwm_var.get():
            types.append("nwm")
        if self.wimp_var.get():
            types.append("wimp")
        if self.pdsi_var.get():
            types.append("pdsi")
        return types

    def _get_dates(self):
        """Return sorted list of datetime objects to process."""
        if self.batch_mode_var.get():
            start = datetime.strptime(self.date_entry.get().strip(), "%Y-%m-%d")
            end = datetime.strptime(self.end_date_entry.get().strip(), "%Y-%m-%d")
            if end < start:
                raise ValueError("End date must be on or after start date.")
            dates = []
            cur = start
            while cur <= end:
                dates.append(cur)
                cur += timedelta(days=1)
            return dates

        if self.date_list:
            return list(self.date_list)

        # single date from the entry box
        return [datetime.strptime(self.date_entry.get().strip(), "%Y-%m-%d")]

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run_btn_clicked(self):
        try:
            lat = float(self.lat_entry.get().strip())
            lon = float(self.lon_entry.get().strip())
            analysis_types = self._selected_analysis_types()
            if not analysis_types:
                messagebox.showwarning(
                    "No Analysis Selected",
                    "Please select at least one analysis type to run.",
                )
                return

            is_huc = self.aoi_mode.get() == "huc"
            dates = self._get_dates()

            self.run_btn.config(state="disabled")
            self.status_var.set("Queuing analyses…")
            self.root.update_idletasks()

            if is_huc:
                # HUC path – single date only for now (matches current engine)
                if len(dates) > 1:
                    messagebox.showwarning(
                        "HUC Mode",
                        "HUC watershed analysis currently uses the first date only.\n"
                        "Clear the date list or turn off Date Range for multi-date HUC runs.",
                    )
                analysis_date = dates[0]
                try:
                    huc_level = int(self.huc_level_entry.get().strip() or "8")
                except ValueError:
                    messagebox.showerror("Error", "HUC level must be an integer 2–12.")
                    return

                msg = {
                    "message_type": "huc_analysis",
                    "lat": lat,
                    "lon": lon,
                    "analysis_date": analysis_date,
                    "output_dir": "output",
                    "data_dir": "data",
                    "huc_level": huc_level,
                    "analysis_types": analysis_types,
                }
                if self.precip_source.get() == "gridded":
                    msg["gridded"] = True
                self.dispatcher.notify(msg)
                self.status_var.set(f"HUC analysis queued (level {huc_level})")
                return

            # Point / multi-date path
            for current_date in dates:
                bulk = {
                    "lat": lat,
                    "lon": lon,
                    "analysis_date": current_date,
                    "output_dir": "output",
                    "data_dir": "data",
                }

                if "precip" in analysis_types:
                    msg = {"message_type": "precip_analysis", **bulk}
                    if self.precip_source.get() == "gridded":
                        msg["gridded"] = True
                    self.dispatcher.notify(msg)

                if "usgs" in analysis_types:
                    self.dispatcher.notify({"message_type": "usgs_analysis", **bulk})

                if "nwm" in analysis_types:
                    self.dispatcher.notify({"message_type": "nwm_analysis", **bulk})

                if "wimp" in analysis_types:
                    self.dispatcher.notify({"message_type": "wimp_analysis", **bulk})

                if "pdsi" in analysis_types:
                    self.dispatcher.notify({"message_type": "pdsi_analysis", **bulk})

                self.dispatcher.notify({"message_type": "generate_pdf", **bulk})

            # merge when more than one date
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

            n = len(dates)
            self.status_var.set(f"Queued {n} date(s) – see console / logs for progress")

        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            self.status_var.set("Ready")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Error")
        finally:
            self.run_btn.config(state="normal")
