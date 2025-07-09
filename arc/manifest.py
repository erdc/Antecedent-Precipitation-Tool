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
##           local_norm.py          ##
##  ------------------------------- ##
##     Written by: Chris French     ##
##  ------------------------------- ##
##    Last Edited on:  2025-07-09   ##
##  ------------------------------- ##
######################################

import json
import os
from datetime import datetime


class FirstLevelIndentJSONEncoder(json.JSONEncoder):
    def __init__(self, *args, indent=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.indent = indent
        self._level = 0

    def encode(self, obj):
        if isinstance(obj, dict):
            self._level += 1
            # Handle top-level dictionary with newlines between entries
            if self._level == 1:
                items = []
                for key, value in obj.items():
                    items.append(f'\n"{key}": {self.encode(value)}')
                return "{" + ",".join(items) + "\n}"
            # Handle nested dictionaries without indentation
            else:
                items = []
                for key, value in obj.items():
                    items.append(f'"{key}": {self.encode(value)}')
                return "{" + ",".join(items) + "}"
        elif isinstance(obj, list):
            return "[" + ",".join([self.encode(item) for item in obj]) + "]"
        else:
            return super().encode(obj)


class Manifest:
    def __init__(self):
        """Initialize an empty manifest."""
        self.data = {}
        self.record = {}

    def _get_timestamp(self):
        """Generate timestamp in yyyy-mm-dd,hh-mm format."""
        return datetime.now().strftime("%Y-%m-%d,%H:%M:%S")

    def _read_from_file(self, filename):
        """Read manifest data from file if it exists."""
        if os.path.isfile(filename):
            with open(filename, "r") as f:
                self.record = json.load(f)

    def reset(self):
        self.data = {}

    def update(self, key, value):
        """Update or add a key-value pair to the manifest."""
        self.data[key] = value

    def write_to_file(self, filename):
        """Write the manifest to a JSON file. Creates file if it doesn't exist."""
        self._read_from_file(filename=filename)
        self.record[self._get_timestamp()] = self.data
        with open(filename, "w") as f:
            formatted_json = json.dumps(self.record, cls=FirstLevelIndentJSONEncoder)
            f.write(formatted_json)
        self.reset()


if __name__ == "__main__":

    test = Manifest()
    test.update(key="test", value="string")
    test.update(key="num", value=123)
    test.update(key="bool", value=True)
    test.write_to_file("test.json")
