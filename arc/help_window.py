#  This software was developed by United States Army Corps of Engineers (USACE)
#  employees in the course of their official duties.  USACE used copyrighted,
#  open source code to develop this software, as such this software
#  (per 17 USC § 101) is considered "joint work."  Pursuant to 17 USC § 105,
#  portions of the software developed by USACE employees in the course of their
#  official duties are not subject to copyright protection and are in the public
#  domain.
#
#  USACE assumes no responsibility whatsoever for the use of this software by
#  other parties, and makes no guarantees, expressed or implied, about its
#  quality, reliability, or any other characteristic.
#
#  The software is provided "as is," without warranty of any kind, express or
#  implied, including but not limited to the warranties of merchantability,
#  fitness for a particular purpose, and noninfringement.  In no event shall the
#  authors or U.S. Government be liable for any claim, damages or other
#  liability, whether in an action of contract, tort or otherwise, arising from,
#  out of or in connection with the software or the use or other dealings in the
#  software.
#
#  Public domain portions of this software can be redistributed and/or modified
#  freely, provided that any derivative works bear some notice that they are
#  derived from it, and any modified versions bear some notice that they have
#  been modified.
#
#  Copyrighted portions of the software are annotated within the source code.
#  Open Source Licenses, included in the source code, apply to the applicable
#  copyrighted portions.  Copyrighted portions of the software are not in the
#  public domain.

######################################
##  ------------------------------- ##
##          help_window.py          ##
##  ------------------------------- ##
##      Writen by: Jason Deters     ##
##      Edited by: Joseph Gutenson  ##
##      Edited by: Chris French     ##
##  ------------------------------- ##
##    Last Edited on:  2025-07-09   ##
##  ------------------------------- ##
######################################

import logging
import os
import subprocess
import sys
import time
import tkinter
import webbrowser
from tkinter import ttk

# Custom Libraries
try:
    from .utilities import JLog
except Exception:
    # Reverse compatibility step - Add utilities folder to path directly
    MODULE_PATH = os.path.dirname(os.path.realpath(__file__))
    ROOT = os.path.dirname(MODULE_PATH)
    PYTHON_SCRIPTS_FOLDER = os.path.join(ROOT, "Python Scripts")
    TEST = os.path.exists(PYTHON_SCRIPTS_FOLDER)
    if TEST:
        UTILITIES_FOLDER = os.path.join(PYTHON_SCRIPTS_FOLDER, "utilities")
        sys.path.append(UTILITIES_FOLDER)
    else:
        ARC_FOLDER = os.path.join(ROOT, "arc")
        UTILITIES_FOLDER = os.path.join(ARC_FOLDER, "utilities")
        sys.path.append(UTILITIES_FOLDER)
    import JLog

# Internal Imports
try:
    from arc.utils import find_file_or_dir
except:
    from utils import find_file_or_dir

logger = logging.getLogger(__name__)

try:
    help_path = find_file_or_dir(os.getcwd(), "help.txt")

    # read in file and decode weird characters
    with open(help_path, "rb") as file:
        content_bytes = file.read()
    ABOUT_HELP_TEXT = content_bytes.decode("utf-8", errors="ignore")

except Exception as e:
    logger.warn(f"error loading helptext: {e}")
    ABOUT_HELP_TEXT = """To report errors with this program, click the "Report Issue" button or email:  
APT-Report-Issue@usace.army.mil"""


class HelpWindow(object):
    """GUI for the Help Page of the Antecedent Precipitation Tool"""

    def __init__(self):
        self.log = JLog.PrintLog()
        self.ula_ccepted = False
        self.button_labels = []
        self.pdf_buttons = []
        self.youtube_buttons = []
        self.separators = []
        self.num_usage_single = 0
        self.num_usage_watershed = 0
        self.row = 0
        # Find Root Folder
        module_path = os.path.dirname(os.path.realpath(__file__))
        root_folder = os.path.split(module_path)[0]

        # Create Master Frame
        self.master = tkinter.Tk()
        # width = 978
        # height = 735
        # self.master.geometry("{}x{}+431+332".format(width, height))
        self.master.geometry("")
        # self.master.minsize(width, height)
        # self.master.maxsize(1370, 749)
        self.master.resizable(1, 1)
        self.master.title("About the Antecedent Precipitation Tool")

        # Set Window Icon
        try:
            images_folder = os.path.join(root_folder, "images")
            graph_icon_file = os.path.join(images_folder, "Graph.ico")
            self.master.wm_iconbitmap(graph_icon_file)
        except Exception:
            images_folder = os.path.join(sys.prefix, "images")
            graph_icon_file = os.path.join(images_folder, "Graph.ico")
            self.master.wm_iconbitmap(graph_icon_file)

        # Create an additional level of frame just so everything stays together when the window is maximized
        self.primary_frame = ttk.Frame(self.master)
        self.primary_frame.grid()

        # ---FIRST TEXT FRAME---
        self.first_frame = ttk.Frame(self.primary_frame)
        self.first_frame.grid(row=self.row, column=0, sticky="nsew", padx=25, pady=10)
        self.first_text = tkinter.Text(self.first_frame, height=35, width=100)
        self.scrollbar1 = tkinter.ttk.Scrollbar(
            self.first_frame
        )  # making a scrolbar with entry test text field
        self.first_text.config(
            yscrollcommand=self.scrollbar1.set
        )  # setting scrolbar to y-axis
        self.scrollbar1.config(
            command=self.first_text.yview
        )  # setting the scrolbar to entry test textbox
        self.first_text.insert("end", ABOUT_HELP_TEXT)  # Inserting the About/Help Text
        self.first_text.config(state="disabled")
        self.first_text.grid(column=0, row=self.row, sticky="nsew")
        self.scrollbar1.grid(column=1, row=self.row, sticky="nsw")
        self.add_separator(self.first_frame)

        # Create/Grid Close Button
        self.button_close = ttk.Button(
            self.primary_frame,
            text="Close This Window",
            command=self.click_close_button,
        )
        self.button_close.grid(row=self.row, column=0, pady=12, padx=100, sticky="w")

        # Create/Grid Close Button
        self.button_close = ttk.Button(
            self.primary_frame,
            text="Report Issue",
            command=self.click_report_issue_button,
        )
        self.button_close.grid(row=self.row, column=0, pady=12, padx=200, sticky="e")

        # Configure rows/columns
        self.master.geometry("")  # ("+800+400")
        # Configure rows and columns
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)
        # a trick to activate the window (on windows 7)
        self.master.deiconify()
        # End of __init__ method

    def add_separator(self, frame):
        separator = ttk.Separator(frame, orient="horizontal")
        self.row += 1  # give this thing it's own row
        separator.grid(row=self.row, sticky="ew", columnspan=3, pady=3)
        self.row += 1  # set up row for the next thing inevitably being separated
        self.separators.append(separator)

    def click_report_issue_button(self):
        """
        If Outlook present:
        Drafts and email with the current error log as an attachment directed to me

        If Outlook not present:
        Opens the Error Log and requests that users transmit it with their error report.
        """
        self.log.send_log()

    # End of send_log method

    def click_close_button(self):
        self.ula_ccepted = True
        self.master.destroy()  # Close ULA window
        return False

    def run(self):
        self.master.mainloop()


def click_how_to_run_point_button():
    # Define File Path
    # Ensure Exists
    # Popen PDF
    # Announce ready for new input
    print("")
    print("Ready for new input.")
    return


def open_youtube_link(youtube_url):
    # Define File Path
    # Ensure Exists
    # Popen PDF
    # Announce ready for new input
    print("")
    print("Ready for new input.")
    return


if __name__ == "__main__":
    APP = HelpWindow()
    APP.run()
