import tkinter as tk
from datetime import datetime, timedelta
from tkinter import messagebox


class PrecipGUI:
    def __init__(self, root, dispatcher):
        self.root = root
        self.dispatcher = dispatcher
        self.root.title("Precipitation Analysis Tool")
        self.root.geometry("380x400")

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

        self.gridded_var = tk.BooleanVar(value=False)
        self.chk_gridded = tk.Checkbutton(
            root, text="Gridded", variable=self.gridded_var
        )
        self.chk_gridded.grid(row=4, column=0, padx=10, pady=5, sticky="w")

        self.ghcn_var = tk.BooleanVar(value=True)
        self.chk_ghcn = tk.Checkbutton(
            root, text="GHCN Station", variable=self.ghcn_var
        )
        self.chk_ghcn.grid(row=4, column=1, padx=10, pady=5, sticky="w")

        self.batch_mode_var = tk.BooleanVar(value=False)
        self.chk_batch = tk.Checkbutton(
            root,
            text="Batch Analysis Mode",
            variable=self.batch_mode_var,
            command=self.toggle_batch_mode,
        )
        self.chk_batch.grid(row=6, column=0, columnspan=2, padx=10, pady=5)

        self.run_btn = tk.Button(
            root, text="Run Analysis", command=self.run_btn_clicked
        )
        self.run_btn.grid(row=7, column=0, columnspan=2, pady=15)

    def toggle_batch_mode(self):
        if self.batch_mode_var.get():
            self.end_date_label.grid(row=3, column=0, padx=10, pady=5, sticky="e")
            self.end_date_entry.grid(row=3, column=1, padx=10, pady=5)
            if not self.end_date_entry.get():
                self.end_date_entry.insert(0, "2023-11-20")
        else:
            self.end_date_label.grid_forget()
            self.end_date_entry.grid_forget()

    def run_btn_clicked(self):
        try:
            lat = float(self.lat_entry.get())
            lon = float(self.lon_entry.get())
            start_date = datetime.strptime(self.date_entry.get(), "%Y-%m-%d")
            is_batch = self.batch_mode_var.get()
            end_date = (
                datetime.strptime(self.end_date_entry.get(), "%Y-%m-%d")
                if is_batch
                else start_date
            )

            self.run_btn.config(state=tk.DISABLED)
            self.root.update()

            if is_batch:
                self.dispatcher.notify(
                    {
                        "message_type": "batch_prefetch",
                        "lat": lat,
                        "lon": lon,
                        "start_date": start_date,
                        "end_date": end_date,
                        "ghcn": self.ghcn_var.get(),
                        "gridded": self.gridded_var.get(),
                        "data_dir": "data",  # will be resolved properly
                        "output_dir": "output",
                    }
                )

            current_date = start_date
            while current_date <= end_date:
                bulk = {
                    "lat": lat,
                    "lon": lon,
                    "analysis_date": current_date,
                    "output_dir": "output",
                    "data_dir": "data",
                }

                if self.ghcn_var.get() or self.gridded_var.get():
                    msg = {"message_type": "precip_analysis", **bulk}
                    if self.gridded_var.get():
                        msg["gridded"] = True
                    self.dispatcher.notify(msg)

                # Other analyses (USGS, NWM, PDSI) will be added later
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
