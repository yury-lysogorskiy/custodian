# coding: utf-8

from __future__ import unicode_literals, division

"""
This module implements specific error handlers for surface VASP runs. These handlers
tries to detect common errors in vasp runs and attempt to fix them on the fly
by modifying the input files.
"""

__author__ = "Shyue Ping Ong, William Davidson Richards, Anubhav Jain, " \
             "Wei Chen, Stephen Dacek"
__version__ = "0.1"
__maintainer__ = "Shyue Ping Ong"
__email__ = "ongsp@ucsd.edu"
__status__ = "Beta"
__date__ = "2/4/13"

import os
import time
import datetime
import operator
import shutil
from functools import reduce
import re

from six.moves import map

import numpy as np

from monty.dev import deprecated
from monty.serialization import loadfn

from math import ceil

from custodian.custodian import ErrorHandler
from custodian.utils import backup
from pymatgen.io.vasp import Poscar, VaspInput, Incar, Kpoints, Vasprun, Oszicar
from pymatgen.transformations.standard_transformations import \
    SupercellTransformation

from custodian.ansible.interpreter import Modder
from custodian.ansible.actions import FileActions
from custodian.vasp.interpreter import VaspModder

VASP_BACKUP_FILES = {"INCAR", "KPOINTS", "POSCAR", "OUTCAR", "OSZICAR",
                     "vasprun.xml", "vasp.out"}

class SurfaceVaspErrorHandler(ErrorHandler):
    """
    Master VaspErrorHandler class that handles a number of common errors
    that occur during VASP runs.
    """

    is_monitor = True

    error_msgs = {
        "tet": ["Tetrahedron method fails for NKPT<4",
                "Fatal error detecting k-mesh",
                "Fatal error: unable to match k-point",
                "Routine TETIRR needs special values"],
        "inv_rot_mat": ["inverse of rotation matrix was not found (increase "
                        "SYMPREC)"],
        "brmix": ["BRMIX: very serious problems"],
        "subspacematrix": ["WARNING: Sub-Space-Matrix is not hermitian in "
                           "DAV"],
        "tetirr": ["Routine TETIRR needs special values"],
        "incorrect_shift": ["Could not get correct shifts"],
        "real_optlay": ["REAL_OPTLAY: internal error",
                        "REAL_OPT: internal ERROR"],
        "rspher": ["ERROR RSPHER"],
        "dentet": ["DENTET"],
        "too_few_bands": ["TOO FEW BANDS"],
        "triple_product": ["ERROR: the triple product of the basis vectors"],
        "rot_matrix": ["Found some non-integer element in rotation matrix"],
        "brions": ["BRIONS problems: POTIM should be increased"],
        "pricel": ["internal error in subroutine PRICEL"],
        "zpotrf": ["LAPACK: Routine ZPOTRF failed"],
        "amin": ["One of the lattice vectors is very long (>50 A), but AMIN"],
        "zbrent": ["ZBRENT: fatal internal in",
                   "ZBRENT: fatal error in bracketing"],
        "pssyevx": ["ERROR in subspace rotation PSSYEVX"],
        "eddrmm": ["WARNING in EDDRMM: call to ZHEGV failed"],
        "edddav": ["Error EDDDAV: Call to ZHEGV failed"]
    }

    def __init__(self, output_filename="vasp.out"):
        """
        Initializes the handler with the output file to check.

        Args:
            output_filename (str): This is the file where the stdout for vasp
                is being redirected. The error messages that are checked are
                present in the stdout. Defaults to "vasp.out", which is the
                default redirect used by :class:`custodian.vasp.jobs.VaspJob`.
        """
        self.output_filename = output_filename
        self.errors = set()

    def check(self):
        incar = Incar.from_file("INCAR")
        self.errors = set()
        with open(self.output_filename, "r") as f:
            for line in f:
                l = line.strip()
                for err, msgs in SurfaceVaspErrorHandler.error_msgs.items():
                    for msg in msgs:
                        if l.find(msg) != -1:
                            # this checks if we want to run a charged
                            # computation (e.g., defects) if yes we don't
                            # want to kill it because there is a change in e-
                            # density (brmix error)
                            if err == "brmix" and 'NELECT' in incar:
                                continue
                            self.errors.add(err)
        return len(self.errors) > 0

    def correct(self):
        backup(VASP_BACKUP_FILES | {self.output_filename})
        actions = []
        vi = VaspInput.from_directory(".")

        if "zbrent" in self.errors:
            vi = VaspInput.from_directory(".")
            ibrion = vi["INCAR"].get("IBRION")
            if ibrion == 2:
                actions.append({"dict": "INCAR",
                                "action": {"_set": {"IBRION": 1}}})
            elif ibrion == 1:
                actions.append({"file": "CONTCAR",
                                "action": {"_file_copy": {"POSCAR"}}})


class SurfacePotimErrorHandler(ErrorHandler):
    """
    Check if a run has excessively large positive energy changes.
    This is typically caused by too large a POTIM. Runs typically
    end up crashing with some other error (e.g. BRMIX) as the geometry
    gets progressively worse.
    """
    is_monitor = True

    def __init__(self, input_filename="POSCAR", output_filename="OSZICAR",
                 dE_threshold=1):
        """
        Initializes the handler with the input and output files to check.
        Args:
            input_filename (str): This is the POSCAR file that the run
                started from. Defaults to "POSCAR". Change
                this only if it is different from the default (unlikely).
            output_filename (str): This is the OSZICAR file. Change
                this only if it is different from the default (unlikely).
            dE_threshold (float): The threshold energy change. Defaults to 1eV.
        """
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.dE_threshold = dE_threshold

    def check(self):
        try:
            oszicar = Oszicar(self.output_filename)
            n = len(Poscar.from_file(self.input_filename).structure)
            max_dE = max([s['dE'] for s in oszicar.ionic_steps[1:]]) / n
            if max_dE > self.dE_threshold:
                return True
        except:
            return False

    def correct(self):
        backup(VASP_BACKUP_FILES)
        vi = VaspInput.from_directory(".")
        potim = float(vi["INCAR"].get("POTIM", 0.5))
        ibrion = int(vi["INCAR"].get("IBRION", 0))
        if potim < 0.2 and ibrion != 3:
            actions = [{"dict": "INCAR",
                        "action": {"_set": {"IBRION": 3,
                                            "SMASS": 0.75}}}]
        elif potim < 0.1:
            actions = [{"dict": "INCAR",
                        "action": {"_set": {"SYMPREC": 1e-8}}}]
        else:
            actions = [{"dict": "INCAR",
                        "action": {"_set": {"POTIM": potim * 0.5}}}]

        VaspModder(vi=vi).apply_actions(actions)
        return {"errors": ["POTIM"], "actions": actions}


class SurfaceFrozenJobErrorHandler(ErrorHandler):
    """
    Detects an error when the output file has not been updated
    in timeout seconds. Changes ALGO to Normal from Fast
    """

    is_monitor = True

    def __init__(self, output_filename="vasp.out", timeout=21600):
        """
        Initializes the handler with the output file to check.
        Args:
            output_filename (str): This is the file where the stdout for vasp
                is being redirected. The error messages that are checked are
                present in the stdout. Defaults to "vasp.out", which is the
                default redirect used by :class:`custodian.vasp.jobs.VaspJob`.
            timeout (int): The time in seconds between checks where if there
                is no activity on the output file, the run is considered
                frozen. Defaults to 21600 seconds, i.e., 6 hour.
        """
        self.output_filename = output_filename
        self.timeout = timeout

    def check(self):
        st = os.stat(self.output_filename)
        if time.time() - st.st_mtime > self.timeout:
            return True

    def correct(self):
        backup(VASP_BACKUP_FILES | {self.output_filename})

        vi = VaspInput.from_directory('.')
        actions = []

        if vi["INCAR"].get("ALGO", "Normal") == "Fast":
            actions.append({"dict": "INCAR",
                        "action": {"_set": {"ALGO": "Normal"}}})
        else:
            actions.append({"dict": "INCAR",
                        "action": {"_set": {"SYMPREC": 1e-8}}})

        VaspModder(vi=vi).apply_actions(actions)

        return {"errors": ["Frozen job"], "actions": actions}



class SurfacePositiveEnergyErrorHandler(ErrorHandler):
    """
    Check if a run has positive absolute energy.
    If so, change ALGO from Fast to Normal or kill the job.
    """
    is_monitor = True

    def __init__(self, output_filename="OSZICAR"):
        """
        Initializes the handler with the output file to check.
        Args:
            output_filename (str): This is the OSZICAR file. Change
                this only if it is different from the default (unlikely).
        """
        self.output_filename = output_filename

    def check(self):
        try:
            oszicar = Oszicar(self.output_filename)
            if oszicar.final_energy > 0:
                return True
        except:
            pass
        return False

    def correct(self):
        # change ALGO = Fast to Normal if ALGO is !Normal
        vi = VaspInput.from_directory(".")
        algo = vi["INCAR"].get("ALGO", "Normal")
        if algo.lower() not in ['normal', 'n']:
            backup(VASP_BACKUP_FILES)
            actions = [{"dict": "INCAR",
                        "action": {"_set": {"ALGO": "Normal"}}}]
            VaspModder(vi=vi).apply_actions(actions)
            return {"errors": ["Positive energy"], "actions": actions}
        #Unfixable error. Just return None for actions.
        elif algo == 'Normal':
            potim = float(vi["INCAR"].get("POTIM", 0.5)) / 2.0
            actions = [{"dict": "INCAR",
                        "action": {"_set": {"POTIM": potim}}}]
            VaspModder(vi=vi).apply_actions(actions)
            return {"errors": ["Positive energy"], "actions": actions}
        else:
            return {"errors": ["Positive energy"], "actions": None}