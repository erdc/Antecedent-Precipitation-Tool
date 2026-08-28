import tkinter as tk
from datetime import datetime, timedelta
from tkinter import messagebox


class PrecipGUI:
    def __init__(self, root, dispatcher):
        self.root = root
        self.dispatcher = dispatcher
        self.root.title("Precipitation Analysis Tool")
        self.root.geometry("440x520")

        # ----- Location / Date -----
        tk.Label(root, text="Latitude:").grid(
            row=0, column=0, padx=10, pady=5, sticky="e"
        )
        self.lat_entry = tk.Entry(root)
        self.lat_entry.grid(row=0, column=1, padx=10, pady=5)
        self.lat_entry.insert(0, "30.0")

        tk.Label(root, text="Longitude:").grid(
            row=1, column=0, padx=10, pady=5, sticky="e"
        )
        self.lon_entry = tk.Entry(root)
        self.lon_entry.grid(row=1, column=1, padx=10, pady=5)
        self.lon_entry.insert(0, "-90.0")

        tk.Label(root, text="Start Date (YYYY-MM-DD):").grid(
            row=2, column=0, padx=10, pady=5, sticky="e"
        )
        self.date_entry = tk.Entry(root)
        self.date_entry.grid(row=2, column=1, padx=10, pady=5)
        self.date_entry.insert(0, "2023-11-11")

        self.end_date_label = tk.Label(root, text="End Date (YYYY-MM-DD):")
        self.end_date_entry = tk.Entry(root)

        # ----- Analysis toggles -----
        tk.Label(root, text="Analyses to run:", font=("", 9, "bold")).grid(
            row=4, column=0, columnspan=2, padx=10, pady=(12, 2), sticky="w"
        )
        self.gridded_var = tk.BooleanVar(value=False)
        self.chk_gridded = tk.Checkbutton(
            root, text="Gridded (nClimGrid)", variable=self.gridded_var
        )
        self.chk_gridded.grid(row=5, column=0, padx=10, pady=2, sticky="w")

        self.ghcn_var = tk.BooleanVar(value=True)
        self.chk_ghcn = tk.Checkbutton(
            root, text="GHCN Station", variable=self.ghcn_var
        )
        self.chk_ghcn.grid(row=5, column=1, padx=10, pady=2, sticky="w")

        self.usgs_var = tk.BooleanVar(value=True)
        self.chk_usgs = tk.Checkbutton(
            root, text="USGS Streamflow", variable=self.usgs_var
        )
        self.chk_usgs.grid(row=6, column=0, padx=10, pady=2, sticky="w")

        self.nwm_var = tk.BooleanVar(value=False)
        self.chk_nwm = tk.Checkbutton(
            root, text="NWM Streamflow", variable=self.nwm_var
        )
        self.chk_nwm.grid(row=6, column=1, padx=10, pady=2, sticky="w")

        self.wimp_var = tk.BooleanVar(value=True)
        self.chk_wimp = tk.Checkbutton(root, text="WIMP Season", variable=self.wimp_var)
        self.chk_wimp.grid(row=7, column=0, padx=10, pady=2, sticky="w")

        self.pdsi_var = tk.BooleanVar(value=True)
        self.chk_pdsi = tk.Checkbutton(
            root, text="PDSI Analysis", variable=self.pdsi_var
        )
        self.chk_pdsi.grid(row=7, column=1, padx=10, pady=2, sticky="w")

        # ----- Batch mode -----
        self.batch_mode_var = tk.BooleanVar(value=False)
        self.chk_batch = tk.Checkbutton(
            root,
            text="Batch Analysis Mode",
            variable=self.batch_mode_var,
            command=self.toggle_batch_mode,
        )
        self.chk_batch.grid(row=8, column=0, columnspan=2, padx=10, pady=(15, 5))

        # ----- HUC mode -----
        self.huc_mode_var = tk.BooleanVar(value=False)
        self.chk_huc = tk.Checkbutton(
            root,
            text="HUC Watershed Analysis (sample points in HUC)",
            variable=self.huc_mode_var,
            command=self.toggle_huc_mode,
        )
        self.chk_huc.grid(row=9, column=0, columnspan=2, padx=10, pady=5)

        self.huc_level_label = tk.Label(root, text="HUC Level (2–12):")
        self.huc_level_entry = tk.Entry(root, width=6)
        self.huc_level_entry.insert(0, "8")

        # ----- Run button -----
        self.run_btn = tk.Button(
            root, text="Run Analysis", command=self.run_btn_clicked
        )
        self.run_btn.grid(row=11, column=0, columnspan=2, pady=15)

    def toggle_batch_mode(self):
        if self.batch_mode_var.get():
            self.end_date_label.grid(row=3, column=0, padx=10, pady=5, sticky="e")
            self.end_date_entry.grid(row=3, column=1, padx=10, pady=5)
            if not self.end_date_entry.get():
                self.end_date_entry.insert(0, "2023-11-20")
            # Batch and HUC are mutually exclusive for Phase 1
            if self.huc_mode_var.get():
                self.huc_mode_var.set(False)
                self.toggle_huc_mode()
        else:
            self.end_date_label.grid_forget()
            self.end_date_entry.grid_forget()

    def toggle_huc_mode(self):
        if self.huc_mode_var.get():
            self.huc_level_label.grid(row=10, column=0, padx=10, pady=2, sticky="e")
            self.huc_level_entry.grid(row=10, column=1, padx=10, pady=2, sticky="w")
            if self.batch_mode_var.get():
                self.batch_mode_var.set(False)
                self.toggle_batch_mode()
        else:
            self.huc_level_label.grid_forget()
            self.huc_level_entry.grid_forget()

    def _selected_analysis_types(self):
        types = []
        if self.ghcn_var.get() or self.gridded_var.get():
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

    def run_btn_clicked(self):
        try:
            lat = float(self.lat_entry.get())
            lon = float(self.lon_entry.get())
            start_date = datetime.strptime(self.date_entry.get(), "%Y-%m-%d")
            is_batch = self.batch_mode_var.get()
            is_huc = self.huc_mode_var.get()
            end_date = (
                datetime.strptime(self.end_date_entry.get(), "%Y-%m-%d")
                if is_batch
                else start_date
            )

            analysis_types = self._selected_analysis_types()
            if not analysis_types:
                messagebox.showwarning(
                    "No Analysis Selected",
                    "Please select at least one analysis type to run.",
                )
                return

            self.run_btn.config(state=tk.DISABLED)
            self.root.update()

            if is_huc:
                try:
                    huc_level = int(self.huc_level_entry.get().strip() or "8")
                except ValueError:
                    messagebox.showerror("Error", "HUC level must be an integer.")
                    return

                msg = {
                    "message_type": "huc_analysis",
                    "lat": lat,
                    "lon": lon,
                    "analysis_date": start_date,
                    "output_dir": "output",
                    "data_dir": "data",
                    "huc_level": huc_level,
                    "analysis_types": analysis_types,
                }
                if self.gridded_var.get():
                    msg["gridded"] = True
                self.dispatcher.notify(msg)
                return

            current_date = start_date
            while current_date <= end_date:
                bulk = {
                    "lat": lat,
                    "lon": lon,
                    "analysis_date": current_date,
                    "output_dir": "output",
                    "data_dir": "data",
                }

                if "precip" in analysis_types:
                    msg = {"message_type": "precip_analysis", **bulk}
                    if self.gridded_var.get():
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
                current_date += timedelta(days=1)

            if is_batch:
                self.dispatcher.notify(
                    {
                        "message_type": "merge_pdfs",
                        "lat": lat,
                        "lon": lon,
                        "start_date": start_date,
                        "end_date": end_date,
                        "output_dir": "output",
                    }
                )

        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.run_btn.config(state=tk.NORMAL)
