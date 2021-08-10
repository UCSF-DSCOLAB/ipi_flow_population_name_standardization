from __future__ import print_function

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 18:04:24 2019

@author: llupinjimenez
"""
import argparse
import numpy as np
import os
import pandas as pd
import xml.etree.ElementTree as ET
import sys
import argparse
from gates_fromWsp import pop_gates
from config_loader import config_loader

############################################
###CODE BORROWED FROM AUTOGATING PIPELINE###
############################################

# stain aliases, and dictionaries for comparing the names between fcs and tube names...
STAIN_SYNONYMNS = {"treg" : (["treg", "1", 1, "Stain 1", "stain 1"]),
                    "nktb" : (["nktb", "2", 2, "Stain 2", "stain 2"]),
                    "sort" : (["sort", "3", 3, "Stain 3", "stain 3"]),
                    "dc" : (["dc", "4", 4, "Stain 4", "stain 4"]),
                    "innate" : (["innate", "5", 5, "Stain 5", "stain 5"]),

                    "Stain 1" : (["treg", "1", 1, "Stain 1", "stain 1"]),
                    "Stain 2" : (["nktb", "2", 2, "Stain 2", "stain 2"]),
                    "Stain 3" : (["sort", "3", 3, "Stain 3", "stain 3"]),
                    "Stain 4" : (["dc", "4", 4, "Stain 4", "stain 4"]),
                    "Stain 5" : (["innate", "5", 5, "Stain 5", "stain 5"]),

                    "stain 1" : (["treg", "1", 1, "Stain 1", "stain 1"]),
                    "stain 2" : (["nktb", "2", 2, "Stain 2", "stain 2"]),
                    "stain 3" : (["sort", "3", 3, "Stain 3", "stain 3"]),
                    "stain 4" : (["dc", "4", 4, "Stain 4", "stain 4"]),
                    "stain 5" : (["innate", "5", 5, "Stain 5", "stain 5"]),

                    "1" : (["treg", "1", 1, "Stain 1", "stain 1"]),
                    "2" : (["nktb", "2", 2, "Stain 2", "stain 2"]),
                    "3" : (["sort", "3", 3, "Stain 3", "stain 3"]),
                    "4" : (["dc", "4", 4, "Stain 4", "stain 4"]),
                    "5" : (["innate", "5", 5, "Stain 5", "stain 5"]),

                    "v11_1" : (["treg", "1", 1, "Stain 1", "stain 1"]),
                    "v11_2" : (["nktb", "2", 2, "Stain 2", "stain 2"]),
                    "v11_3" : (["sort", "3", 3, "Stain 3", "stain 3"]),
                    "v11_4" : (["dc", "4", 4, "Stain 4", "stain 4"]),
                    "v11_5" : (["innate", "5", 5, "Stain 5", "stain 5"]),

                    1 : (["treg", "1", 1, "Stain 1", "stain 1"]),
                    2 : (["nktb", "2", 2, "Stain 2", "stain 2"]),
                    3 : (["sort", "3", 3, "Stain 3", "stain 3"]),
                    4 : (["dc", "4", 4, "Stain 4", "stain 4"]),
                    5 : (["innate", "5", 5, "Stain 5", "stain 5"])}

STAIN_STANDARD_CELL = {"treg" : "treg", "nktb" : "nktb", "sort" : "sort", "dc" : "dc", "innate" : "innate",
                    "Stain 1" : "treg", "Stain 2" : "nktb", "Stain 3" : "sort", "Stain 4" : "dc", "Stain 5" : "innate",
                    "stain 1" : "treg", "stain 2" : "nktb", "stain 3" : "sort", "stain 4" : "dc", "stain 5" : "innate",
                    "1" : "treg", "2" : "nktb", "3" : "sort", "4" : "dc", "5": "innate",
                    1 : "treg", 2 : "nktb", 3 : "sort", 4 : "dc", 5 : "innate"}

STAIN_STANDARD_STAIN = {"treg" : "Stain 1", "nktb" : "Stain 2", "sort" : "Stain 3", "dc" : "Stain 4", "innate" : "Stain 5",
                    "Stain 1" : "Stain 1", "Stain 2" : "Stain 2", "Stain 3" : "Stain 3", "Stain 4" : "Stain 4", "Stain 5" : "Stain 5",
                    "stain 1" : "Stain 1", "stain 2" : "Stain 2", "stain 3" : "Stain 3", "stain 4" : "Stain 4", "stain 5" : "Stain 5",
                    "1" : "Stain 1", "2" : "Stain 2", "3" : "Stain 3", "4" : "Stain 4", "5": "Stain 5",
                    1 : "Stain 1", 2 : "Stain 2", 3 : "Stain 3", 4 : "Stain 4", 5 : "Stain 5"}

STAIN_STANDARD_INT = {"treg" : 1, "nktb" : 2, "sort" : 3, "dc" : 4, "innate" : 5,
                    "Stain 1" : 1, "Stain 2" : 2, "Stain 3" : 3, "Stain 4" : 4, "Stain 5" : 5,
                    "stain 1" : 1, "stain 2" : 2, "stain 3" : 3, "stain 4" : 4, "stain 5" : 5,
                    "1" : 1, "2" : 2, "3" : 3, "4" : 4, "5": 5,
                    1 : 1, 2 : 2, 3 : 3, 4 : 4, 5 : 5}

# standard stain name definitions...this should be adaptable and rearrangeable in the scripts.
# STAINS_V11_4_STAINS is the current standard for the stain naming.
# ex IPIGYN001_t1_Stain 1, IPICRC002_n1_Stain 3, ...
STAINS_V11_4_CELLS = ["treg", "nktb", "sort", "dc", "innate"]
STAINS_V11_4_STAINS = ["Stain 1", "Stain 2", "Stain 3", "Stain 4", "Stain 5"]
STAINS_V11_4_NUMS = ["1", "2", "3", "4", "5"]
STAINS_V11_4_ALL = STAINS_V11_4_CELLS + STAINS_V11_4_STAINS + STAINS_V11_4_NUMS

# Privileged characters for training/testing/lineage separation...

# population separations
# ex. population singlecell2@singlecell@time@Stain 1_root
LINEAGE_SEP = "@"
ORDER_LINEAGE_SEP = "&&"

##########################
###END OF BORROWED CODE###
##########################

def get_pops_fromWsp(wsp_path = None, config = None):
    assert wsp_path

    popGates = pop_gates(wsp_file = wsp_path,
                         config = config)
    # popGates.parse_gates()
    # including all samplenodes
    popGates.parse_gates_anyWsp(do_ignore_gate = True)
    return popGates


# The types of .csv groupings that can be done.
#  wsp - all the tubes in a single wsp are grouped together in a single .csv.
#  tube - each individual tube in a wsp becomes its own .csv.
#  stain - If a stain entry (either value or key if dict) is found in the tube name, those under the same stain are grouped into a single .csv.
#  all - All the wsp files in the wsp_dir are grouped together in a single .csv
GROUP_CSV_TYPES = set(["wsp", "tube", "stain", "all"])

def get_wsp_counts(wsp_dir,
                   stain_types,
                   group_csv = "wsp",
                   all_tubes_name = None,
                   do_dir_descend = True,
                   start_pop = None,
                   do_order_pops = False,
                   config = None,
                   str_include = ""):
    import glob

    assert group_csv in GROUP_CSV_TYPES
    stain_dfs = {}

    if do_dir_descend:
        wsp_paths = []
        for root, dirs, files in os.walk(wsp_dir):
            for file in files:
                if file.endswith(".wsp") and all([b in file for b in str_include]):
                     wsp_paths.append(os.path.join(root, file))
    else:
        wsp_paths = [b for b in os.listdir(wsp_dir) if ".wsp" in b]

    print("wsp_paths: %s"%str(wsp_paths))
    for wsp_path in wsp_paths:
        wsp_name = os.path.basename(wsp_path)
        print(wsp_name)
        wsp_name_short = ".".join(wsp_name.split(".")[:-1])

        my_pop = get_pops_fromWsp(wsp_path, config = config)

        from pprint import pprint

        for type_key in sorted(my_pop.uniquePops.keys()):
            stain = None

            if group_csv == "all":
                if not all_tubes_name:
                    stain = "all_tubes"
                else:
                    stain = all_tubes_name
            elif group_csv == "wsp":
                stain = wsp_name_short
            elif group_csv == "tube":
                stain = "%s--%s"%(wsp_name_short, type_key)
            elif group_csv == "stain":
                for b in stain_types:
                    if ("_"+str(b) in type_key.lower()) or (str(b)+"_" in type_key.lower()):
                        if isinstance(stain_types, dict):
                            stain = stain_types[b]
                        else:
                            stain = b
                        break
                if not stain:
                    print(" >>>%s not corresponding to any searched for stains. continuing...<<<"%type_key)
                    continue
            else:
                raise

            print(" %s, stain: %s"%(type_key, stain))
            if stain not in stain_dfs:
                stain_dfs[stain] = pd.DataFrame()
            df = stain_dfs[stain]

            for pop in [p for p in my_pop.uniquePops[type_key].keys() if (start_pop in p.split(LINEAGE_SEP) or start_pop is None)]:
                if start_pop is None:
                    pop_truncate = pop
                else:
                    pop_truncate = LINEAGE_SEP.join(pop.split(LINEAGE_SEP)[:pop.split(LINEAGE_SEP).index(start_pop)+1])

                if do_order_pops:
                    pop_truncate = ORDER_LINEAGE_SEP.join(sorted(pop_truncate.split(LINEAGE_SEP)))

                pop_info = my_pop.uniquePops[type_key][pop]
                flowjo_count = pop_info["flojo_count"]
                df.loc[pop_truncate,type_key] = flowjo_count

    return stain_dfs

def get_save_pop_counts(wsp_dir,
                        out_dir,
                        stain_types = STAIN_STANDARD_STAIN,
                        group_csv = "tube",
                        start_pop = None,
                        config = None):

    stain_dfs = get_wsp_counts(wsp_dir = wsp_dir,
                   stain_types = stain_types,
                   group_csv = group_csv,
                   all_tubes_name = os.path.basename(wsp_dir).replace(".wsp",""),
                   start_pop = start_pop,
                   config = config)

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # experiment_name = os.path.basename(wsp_dir)
    # out_experiment_dir = os.path.join(out_dir, experiment_name)

    # if not os.path.isdir(out_experiment_dir):
        # os.mkdir(out_experiment_dir)

    for stain in stain_dfs.keys():
        stain_dfs[stain].to_csv(os.path.join(out_dir, "%s.csv"%str(stain)))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Output WSP populations with standardized names')
    parser.add_argument('--wspdir', '-id', type=str,
                        default="/tmp/my-wsps",
                        help='Directory to check for .wsp files', dest='wspdir')

    parser.add_argument('--outdir', '-od', type=str,
                        default='/tmp/my-csv-files',
                        help='Path to save csvs to', dest='outdir')

    arguments = parser.parse_args()

    get_save_pop_counts(arguments.wspdir, arguments.outdir, config=config_loader())