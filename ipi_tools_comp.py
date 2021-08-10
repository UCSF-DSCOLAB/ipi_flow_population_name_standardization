from __future__ import print_function

import pandas as pd
import os
import sys
import numpy as np
import xml.etree.ElementTree as ET
import time
import pprint
import scipy.stats as stats
from sklearn.neighbors.kde import KernelDensity
from scipy.optimize import curve_fit
import fcsparser
import re

from pop_consistency_nameChanges import get_pop_alias

STAIN_STANDARD_STAIN = {"treg" : "Stain 1", "nktb" : "Stain 2", "sort" : "Stain 3", "dc" : "Stain 4", "innate" : "Stain 5",
                    "Stain 1" : "Stain 1", "Stain 2" : "Stain 2", "Stain 3" : "Stain 3", "Stain 4" : "Stain 4", "Stain 5" : "Stain 5",
                    "stain 1" : "Stain 1", "stain 2" : "Stain 2", "stain 3" : "Stain 3", "stain 4" : "Stain 4", "stain 5" : "Stain 5",
                    "1" : "Stain 1", "2" : "Stain 2", "3" : "Stain 3", "4" : "Stain 4", "5": "Stain 5",
                    1 : "Stain 1", 2 : "Stain 2", 3 : "Stain 3", 4 : "Stain 4", 5 : "Stain 5"}

FINAL_TUBENAME_FORM = "SAMPLE_TISSUE_ASSAY_STAIN"
############################
###WSP AND FCS OPERATIONS###
############################

def get_facs_param_name(param):
    #return param
    return param.replace("Comp-","")

# Written to handle most IPI gatings...
def ipi_wsp_getComp(wsp_file, 
                    #stain_stain_types = None, 
                    #stain_cell_types = None,
                    stain_pop_parse = None,
                    stain_standard_stain = None,
                    experiment_channel_alias = None,
                    standard_channels = True, 
                    tissue_type = None, 
                    bypass_tissue_type = False):
    """
    Returns comp matrices with stain alias for each stain in the sample used in FlowJo.

    input: wsp_file (path to .wsp),
            stains_in_wsp (stains to look for .wsp comp matrices in, defaults to all stains)
            tissue_type (tissue type to look for while parsing)
    output: comp_mxs (dictionary of the compensation matrices for each stain number)
    """
    
    if tissue_type == None:
        tissue_type = "NOT_DEFINED"
        bypass_tissue_type = True

    tree = ET.parse(wsp_file)
    root = tree.getroot()

    # these are in the same order, so we can iterate over
    #  and find the corresponding matrices using the index
    samples_elements = [r for r in root.findall("./SampleList/Sample/") if
                        "name" in r.attrib.keys() and
                        "spillover" in r.tag]
    sample_identifiers = [r for r in root.findall("./SampleList/Sample/") if
                          "uri" in r.attrib.keys() and
                          r.tag == "DataSet"]

    # comp identifiers and value elements should be the same size...
    assert len(samples_elements) == len(sample_identifiers)


    # identifying the names by index which they appear in the wsp
    dataset_names = {}
    spillover_names = {}

    # identifying the corresponding stain panel, because there is ambiguity in the naming....

    # the identifier found in the uri,
    #  if any of the corresponding names are in the uri
    for i, ident in enumerate(sample_identifiers,0):
        for stain_parse in stain_pop_parse:
            if stain_parse in ident.get("uri").lower():
                if "_"+tissue_type.lower() in ident.get("uri").lower() or bypass_tissue_type:
                    base_name = os.path.basename(ident.get("uri")).lower()
                    dataset_names[i] = [base_name, stain_parse]

    # what sample ids associate with what stains, so the right comps are applied?
    for i, sample in enumerate(samples_elements,0):
        sample_info = sample.attrib
        sample_num = sample.get("sampleID")

        # acquisition or ipi corresponding in tube name
        stain_name_lower = sample_info["name"].lower()
        if sample_num is not None:
            print(stain_name_lower, sample_num)

            if "ipi" in stain_name_lower:
                # the cell staindard is probably in the name too...
                for stain_parse in config.experiment.STAINS_V11_4_CELLS:
                    if "_"+stain_parse in stain_name_lower:
                        if "_"+tissue_type.lower() in ident.get("uri").lower() or bypass_tissue_type:
                            spillover_names[i] = [stain_name_lower, stain_standard_stain[stain_parse]]

            elif sample_num in [str(s) for s in [1,2,3,4,5]]:
                spillover_names[i] = [stain_name_lower, stain_standard_stain[sample_num]]

            else:
                print(">>>...naming specifications not met. Continuing...<<<")
                print(sample_info["name"].lower())
                #raise Exception("Sample did not meet the naming specifications (whatever the fuck they were in the first place...)")
        else:
            #stain_name_lower may not clearly represent a certain stain...
            spillover_names[i] = [stain_name_lower, "n/a"]

    comp_mxs = {}
    
    for i in dataset_names.keys():
        lasers = []
        values = []
        # corresponding element
        sample = samples_elements[i]

        if i in spillover_names.keys():
            sample_cellName = dataset_names[i][1]
            sample_uriBase = dataset_names[i][0]

            stain_name = stain_standard_stain[sample_cellName]

            #should check each laser, not assume that the matrix is equivalent
            for sf in sample:
                if "parameters" in sf.tag:
                    for ssf in sf:
                        things = ssf.items()
                        for thing in things:
                            # need better way to do this...future may be more lasers?
                            if "name" in thing[0]:
                                #print(thing[1]),
                                lasers.append(thing[1])

                if "spillover" in sf.tag:
                    for ssf in sf:
                        #WORKS for one stain
                        things = ssf.items()
                        for thing in things:
                            # need better way to do this...future may be more lasers?
                            if "name" in thing[0]:
                                #print(thing[1]),
                                lasers.append(thing[1])
                            elif "value" in thing[0]:
                                values.append(float(thing[1]))

            # weird that it works sometimes...got rid of other req to proceed into if statement
            found_stain = False

            for stain_type in stain_pop_parse:
                if str(stain_type) == stain_name:
                    stain_name = stain_standard_stain[stain_type]
                    found_stain = True
                    break

            assert found_stain

            stain = stain_standard_stain[stain_type]

            if len(values) > 0:
                comp_array_np = np.reshape(values,(len(lasers),len(lasers)))
                comp_array_df = pd.DataFrame(comp_array_np, index = lasers, columns = lasers)
                if standard_channels:
                    lasers = [experiment_channel_alias[stain][l] for l in lasers if l in experiment_channel_alias[stain].keys()]
                    comp_array_df = comp_array_df.rename(index = experiment_channel_alias[stain], columns = experiment_channel_alias[stain])
                    comp_array_df = comp_array_df.loc[lasers][lasers]
                comp_mxs[stain] = [comp_array_df , lasers]

    return comp_mxs

###################################
###FILE CLEANUP AND ORGANIZATION###
###################################

def name_cleanup_func(f, 
                      file_type = None, 
                      do_print_fnew = False,
                      do_remove_extension = True,
                      do_remove_parentheses =True,
                      do_lowercase = False):
    """
    Returns edited file name
    
    input: f
    output: f_new
    """
    
    f_new = f
    
    # removing anything after a period
    if do_remove_extension:
        f_new = os.path.basename(os.path.splitext(f_new)[0])
    
    # sometimes notes are put in between parentheses in file naming for IPI...remove chars before the parentheses
    if do_remove_parentheses:
        def remove_brackets(test_str):
            ret = ''
            skip1c = 0
            skip2c = 0
            for i in test_str:
                if i == '[':
                    skip1c += 1
                elif i == '(':
                    skip2c += 1
                elif i == ']' and skip1c > 0:
                    skip1c -= 1
                elif i == ')'and skip2c > 0:
                    skip2c -= 1
                elif skip1c == 0 and skip2c == 0:
                    ret += i
            return ret
            
        f_new = remove_brackets(f_new)
    
    if do_lowercase:
        f_new = f_new.lower()
    
    if do_print_fnew:
        print("%s -> %s"%(f, f_new))
        
    return f_new

def ipi_name_edit(f, file_type = None, start_str = "IPI", do_print_fnew = False):
    """
    Returns edited file name, to match IPI formatting
    
    input: f (file name)
    output: f_new (IPI changed file name)
    """
    
    f_new = f
    f_new = f_new.replace(" ","").replace("copy","").replace("Specimen_001_","").replace("_biex","")
    
    # sometimes notes are put in between parentheses in file naming for IPI...remove chars before the parentheses
    if "(" and ")" in f:
       f_new = f_new[f_new.index(")")+1:]
    
    f_new = f_new[f_new.index(start_str):]
    
    if do_print_fnew:
        print("%s -> %s"%(f, f_new))
        
    return f_new
    
def get_ipi_flow_files(target_dir):
    """
    Regular get_flow_files function called with the argument cleanup_func set to ipi_rename_edit
    """
    
    return get_flow_files(target_dir, cleanup_func = ipi_name_edit)

  
def get_flow_files(target_dir, 
                    name_conditions = [], 
                    path_conditions = [],
                    in_dict = {},
                    file_type = None,
                    cleanup_func = lambda x, *args, **kwargs: x, 
                    sort_by = len):
    """
    Root dir to find fcs/wsp names in. All the files need to satisfy the name_conditions and the path_conditions...
    
    input: target_dir, 
            name_conditions = [] (conditions that need to be satisfied for the name of the file), 
            path_conditions = [] (conditions that need to be satisfied for the path of the file),  
            in_dict = {} (use in recursive step, checking each subdirectory), 
            cleanup_func = name_cleanup_func, 
            sort_by = len (the "len" function for sorting the target_dir files is default)
    output: in_dict
    """
    
    file_names = list(sorted(os.listdir(target_dir), key = lambda x: sort_by(x)))
    for f in file_names:
        file_path = os.path.join(target_dir, f)
        
        if os.path.isdir(file_path):
            in_dict.update(get_flow_files(file_path, 
                                          name_conditions = name_conditions, 
                                          path_conditions = name_conditions,
                                          file_type = file_type))
        else:
            #print(f, file_path, name_conditions, all([c in str(f) for c in name_conditions]), all([c in str(file_path) for c in path_conditions]))
            # if all the conditions are not satisfied, the file_name/file_path should not go into the dictionary...
            if all([c in str(f) for c in name_conditions])*all([c in str(file_path) for c in path_conditions]):
                f_new = cleanup_func(f, file_type)
                in_dict[f_new] = file_path
                    
    #print(">>>", in_dict.keys())
    return in_dict

def get_ipi_sample_dict(target_dir = None, 
                        stain_dict = {'_1': 'Stain 1',
                                     '_2': 'Stain 2',
                                     '_3': 'Stain 3',
                                     '_4': 'Stain 4',
                                     '_5': 'Stain 5',
                                     'Stain 1': 'Stain 1',
                                     'Stain 2': 'Stain 2',
                                     'Stain 3': 'Stain 3',
                                     'Stain 4': 'Stain 4',
                                     'Stain 5': 'Stain 5',
                                     'dc': 'Stain 4',
                                     'innate': 'Stain 5',
                                     'nktb': 'Stain 2',
                                     'sort': 'Stain 3',
                                     'treg': 'Stain 1',
                                     'stain 1': 'Stain 1',
                                     'stain 2': 'Stain 2',
                                     'stain 3': 'Stain 3',
                                     'stain 4': 'Stain 4',
                                     'stain 5': 'Stain 5',},
                        stsd = {}, 
                        condition_strs = set(["IPI"]),
                        start_str = "",
                        depth = 0,
                        ignore_strs = set(['IGNORE', 'DONTUSE', 'Rainbows']),
                        fcs_name_format = "patient.tissue.stain",
                        wsp_name_format = "patient",
                        fcs_wsp_share = "patient",
                        fcs_share = "patient",
                        wsp_share = "patient",
                        cleanup_func = name_cleanup_func,
                        #cleanup_func = ipi_name_edit,
                        do_assert_wsp = True,
                        descriptor_num = 3,
                        **kwargs):
    """
    Returns dictionary of samples, separated by tissue and stain.
    
    input: target_dir = None, 
            stsd = {}, 
            condition_strs = set(["IPI"]),
            depth = 0,
            ignore_strs = set(["DONTUSE","DONT","TUBES", "IGNORE","Rainbows","TEST"]),
                        do_assert_wsp = True
    output: stsd (sample tissue stain dictionary)
    """
    return get_sample_dict(target_dir = target_dir, 
                           stain_dict = stain_dict,
                           stsd = stsd, 
                           condition_strs = condition_strs,
                           start_str = start_str,
                           depth = depth,
                           ignore_strs = ignore_strs,
                           cleanup_func = cleanup_func,
                           do_assert_wsp = do_assert_wsp,
                           descriptor_num = descriptor_num,
                           fcs_name_format = fcs_name_format,
                           wsp_name_format = wsp_name_format,
                           fcs_wsp_share = fcs_wsp_share,
                           fcs_share = fcs_share,
                           wsp_share = wsp_share,
                           **kwargs)

# flag if does not follow inputted format wsp/fcs 
def get_sample_dict(target_dir = None, 
                    stain_dict = None,
                    stsd = {}, 
                    depth = 0,
                    condition_strs = set([]),
                    ignore_strs = set(["Rainbows", "IGNORE", "._"]),
                    ignore_dir_strs = set([]),
                    condition_dir_strs = set([]),
                    start_str = "",
                    remove_str = "",
                    descriptor_num = 0,
                    info_sep = "-",
                    sep_symbols = [".","__","_","-", " "],
                    fcs_name_format = None,
                    wsp_name_format = None,
                    fcs_wsp_share = None,
                    fcs_share = None,
                    wsp_share = None,
                    stain_override = "",
                    cleanup_func = name_cleanup_func,
                    do_cleanup = True,
                    do_assert_wsp = True,
                    do_assert_fcs = False,
                    do_assert_formatExact = True,
                    do_accept_all_fcsNames = False,
                    do_detail = False,
                    ):
    """
    Returns dictionary of samples, separated by tissue and stain.
    
    input: target_dir = None, 
           stain_dict = None,
           stsd = {}, 
           depth = 0,
           condition_strs = set([]),
           ignore_strs = set(["Rainbows", "IGNORE", "._"]),
           ignore_dir_strs = set([]),
           condition_dir_strs = set([]),
           start_str = "",
           remove_str = "",
           descriptor_num = 3,
           info_sep = "_",
           fcs_name_format = None, (this refers to the format that the .fcs sample file names are expected to be in. Ex. PATIENT.TISSUE.RUN.STAIN.fcs)
           wsp_name_format = None, (this refers to the format that the .wsp file names are expected to be in. If each .fcs corresponds one to one to .wsp, then it is the same.
                                    If the .wsp is shared between multiple .fcs's, then this should be the descriptor that shared between the files. Ex. PATIENT.wsp)
           fcs_wsp_share = None, (the shared desciptor between fcs's and wsp's)
           fcs_share = None, (the shared desciptor between fcs's)
           wsp_share = None, (the shared desciptor between wsp's)
           stain_override = "",
           cleanup_func = name_cleanup_func,
           do_cleanup = True,
           do_assert_wsp = True,
           do_assert_formatExact = True,
           do_accept_all_fcsNames = False,
           do_detail = False,
    output: stsd (sample tissue stain dictionary)
    """
    # we need to find the relevant information from the file name, to associate the .fcs's with the correct .wsp's...
    
    fcs_name_format_split = fcs_name_format.replace(".fcs","")
    wsp_name_format_split = wsp_name_format.replace(".wsp","").replace(".fcs","")
    
    def get_name_share(file_name, name_format, name_share, format = "", do_print_share = False):
        """
        Get the files that share the format that is being searched for.
        This ties the files that should be identified to be the same type, dependent on how the file name string is structured.
        
        input: file_name, 
               name_format,
               name_share, 
               format = ""
        output: 
        """
        if do_print_share:
            print("finding share in ''%s'', using ''%s''"%(file_name, name_format), end = "")
            
        if name_format is not None:
            name_format_split = name_format[:]
            for sep_symbol in sep_symbols:
                name_format_split = name_format_split.replace(sep_symbol, info_sep)
            name_format_split = name_format_split.split(info_sep)
            
        file_name_split = file_name.replace(format,"")
        for sep_symbol in sep_symbols:
                file_name_split = file_name_split.replace(sep_symbol, info_sep)
        file_name_split = file_name_split.split(info_sep)
        
        if do_assert_formatExact:
            if len(file_name_split) != len(name_format_split):
                if do_detail:
                    print(" %s does not match %s format in length. Continuing..."%(str(file_name_split), str(name_format_split)))
                return False
        
        # now returning the string to tie the fcs-wsp together
        fcs_wsp_share_name = []
        try:
            for fws in name_share.split(info_sep):
                fcs_wsp_share_name.append(file_name_split[name_format_split.index(fws)])
        except IndexError:
            if do_detail:
                print(">>>Error in indexing fcs_name_split. Continuing...<<<")
            return False
        
        name_share = info_sep.join(fcs_wsp_share_name)
        if do_print_share:
            print(" -> ''%s''"%(name_share))
        return name_share
        
    def get_fcs_share(file_name, name_format = fcs_name_format_split, name_share = fcs_wsp_share, format = ".fcs"):
        return get_name_share(file_name, name_format, name_share, format)
    
    def get_wsp_share(file_name, name_format = wsp_name_format_split, name_share = fcs_wsp_share, format = ".wsp"):
        return get_name_share(file_name, name_format, name_share, format)
       
        
    assert target_dir
    assert stain_dict
    if isinstance(target_dir, list):
        for file_path in target_dir:
            stsd.update(get_sample_dict(target_dir = file_path, 
                                        stain_dict = stain_dict,
                                        stsd = stsd,
                                        depth = depth + 1, 
                                        condition_strs = condition_strs,
                                        ignore_strs = ignore_strs,
                                        ignore_dir_strs = ignore_dir_strs,
                                        condition_dir_strs = condition_dir_strs,
                                        start_str = start_str,
                                        remove_str = remove_str,
                                        descriptor_num = descriptor_num,
                                        info_sep = info_sep,
                                        fcs_name_format = fcs_name_format,
                                        wsp_name_format = wsp_name_format,
                                        fcs_share = fcs_share,
                                        wsp_share = wsp_share,
                                        fcs_wsp_share = fcs_wsp_share,
                                        stain_override = stain_override,
                                        cleanup_func = cleanup_func,
                                        do_cleanup = do_cleanup,
                                        do_assert_wsp = do_assert_wsp,
                                        do_assert_fcs = do_assert_fcs,
                                        do_accept_all_fcsNames = do_accept_all_fcsNames))
    
    file_names = list(sorted(os.listdir(target_dir)))
    
    # do .fcs's first, then other files (potenital directories), then finally .wsp'...
    for f in sorted(file_names, key = lambda x: 0 if ".fcs" in x else 1 if ".wsp" in x else 2):
        file_path = os.path.join(target_dir, f)
        # if directory, run recursively, else proceed to add...
        if os.path.isdir(file_path):
            file_name = os.path.basename(file_path)
            if not any([s in file_name for s in ignore_dir_strs]) and all([s in file_name for s in condition_dir_strs]):    
                stsd.update(get_sample_dict(target_dir = file_path, 
                                            stain_dict = stain_dict,
                                            stsd = stsd,
                                            depth = depth + 1, 
                                            start_str = start_str,
                                            remove_str = remove_str,
                                            condition_strs = condition_strs,
                                            ignore_strs = ignore_strs, 
                                            ignore_dir_strs = ignore_dir_strs,
                                            condition_dir_strs = condition_dir_strs,
                                            descriptor_num = descriptor_num,
                                            info_sep = info_sep,
                                            fcs_name_format = fcs_name_format,
                                            wsp_name_format = wsp_name_format,
                                            fcs_share = fcs_share,
                                            wsp_share = wsp_share,
                                            fcs_wsp_share = fcs_wsp_share,
                                            stain_override = stain_override,
                                            cleanup_func = cleanup_func,
                                            do_cleanup = do_cleanup,
                                            do_assert_wsp = do_assert_wsp,
                                            do_assert_fcs = do_assert_fcs,
                                            do_accept_all_fcsNames = do_accept_all_fcsNames))
        else:
            try:
                # if "._" not in f and start_str in f and all([s not in f for s in ignore_strs]) and \
                                                       # all([s in f for s in condition_strs]):
                if (start_str in f and "._" not in f) or start_str == "":
                    
                    # removing .fcs and .wsp, normalizing...
                    if start_str != "":
                        f = f[f.index(start_str):]
                    
                    if do_cleanup:
                        f_cleaned = cleanup_func(f)
                    else:
                        f_cleaned = f
                    
                    # does it satisfy the descriptor_num requirement, and has none of the ignore_strs/all of the condition_strs in f_cleaned?
                    if ".fcs" in f and start_str in f_cleaned and \
                                       not any([s in f for s in ignore_strs]) and \
                                       all([s in f for s in condition_strs]):
                                       
                        f_cleaned = f_cleaned.replace(".fcs","")
                        
                        # shared naming between wsp and fcs, in our case patient
                        f_wsp_fcs_share_name = get_fcs_share(f_cleaned, name_format = fcs_name_format, name_share = fcs_wsp_share)
                        # shared naming between fcs's, in this case patient+tissue
                        fcs_patient_tissue_share_name = get_fcs_share(f_cleaned, name_format = fcs_name_format, name_share = fcs_share)
                        # shared naming between stains
                        stain_share = get_fcs_share(f_cleaned, name_format = fcs_name_format, name_share = "STAIN")
                        
                        # all of these above have to be defined correctly to add an fcs to the dictionary
                        if not (f_wsp_fcs_share_name and fcs_patient_tissue_share_name and stain_share):
                            if do_detail:
                                print(">>>%s not valid .fcs! continuing...<<<"%f_cleaned)
                            continue

                        # is the stain_share found valid for the stains we are considering?
                        if not (str(stain_share) in STAIN_STANDARD_STAIN):
                            if do_detail:
                                print(">>>%s is not a valid stain in STAIN_STANDARD_STAIN! Continuing...<<<"%stain_share)
                            continue
                            
                        # # getting the index of where the IPI name starts
                        # f_cleaned = f_cleaned[f_cleaned.index(start_str):]
                        
                        # finding the right stain and breaking...
                        #  if stain_override is str other than "", then we set all fcs's to this stain type.
                        fcs_core_name_stain = None
                        if stain_override is None:
                            for b in stain_dict.keys():
                                if b.lower() in f_cleaned.lower() or do_accept_all_fcsNames: #this is if it isn't parsable
                                    fcs_core_name_stain = f_wsp_fcs_share_name + info_sep + b
                                    if do_detail:
                                        print("  %s - %s"%(f_cleaned,b))
                                    break
                            stain_alias = stain_dict[b]
                        else:
                            fcs_core_name_stain = stain_override
                            stain_alias = stain_override
                            
                        if isinstance(fcs_core_name_stain, type(None)):
                            if do_detail:
                                print(">>>fcs_core_name_stain not set! continuing...<<<")
                            continue
                            
                        # creating nested dictionary to get correct .fcs paths for stains...
                        if f_wsp_fcs_share_name not in stsd.keys():
                            stsd[f_wsp_fcs_share_name] = {}
                            
                        if "fcs" not in stsd[f_wsp_fcs_share_name].keys():
                            stsd[f_wsp_fcs_share_name]["fcs"] = {}
                            
                        if fcs_patient_tissue_share_name not in stsd[f_wsp_fcs_share_name]["fcs"].keys():
                            stsd[f_wsp_fcs_share_name]["fcs"][fcs_patient_tissue_share_name] = {}
                            
                        if stain_share not in stsd[f_wsp_fcs_share_name]["fcs"][fcs_patient_tissue_share_name].keys():
                            stsd[f_wsp_fcs_share_name]["fcs"][fcs_patient_tissue_share_name][stain_share] = file_path
                            
                    # assuming the longer name is the one that we want to use, because it has more descriptors?...
                    #  not sure what would be a good way to do this.
                    elif ".wsp" in f and start_str in f_cleaned:
                        f_cleaned = f_cleaned.replace(".wsp","")
                        wsp_share_name = get_wsp_share(f_cleaned, name_format = wsp_name_format, name_share = wsp_share)
                        
                        if not wsp_share_name:
                            if do_detail:
                                print(">>>%s not valid .wsp! continuing...<<<"%f_cleaned)
                            continue
                        
                        # # what FCS_SHARE's does this wsp correspond to? 
                        # for key in stsd.keys():
                            # fcs_share_name = get_fcs_share(key, name_format = fcs_share, name_share = fcs_wsp_share)
                            # wsp_share_name = get_wsp_share(f_cleaned, name_format = wsp_name_format, name_share = fcs_wsp_share)
                            # if fcs_share_name == wsp_share_name:
                                # f_wsp_fcs_shareName = get_fcs_share(f_cleaned, name_format = fcs_share, name_share = fcs_share)
                                # stsd[f_wsp_fcs_shareName]["wsp"] = file_path
                                
                        f_wsp_fcs_shareName = get_fcs_share(f_cleaned, name_format = wsp_name_format, name_share = wsp_share)
                        if f_wsp_fcs_shareName not in stsd:
                            stsd[f_wsp_fcs_shareName] = {}
                        if "wsp" not in stsd[f_wsp_fcs_shareName]:
                            stsd[f_wsp_fcs_shareName]["wsp"] = file_path
                                
            except Exception as e:
                print(">>>error: %s<<<"%e)
                raise

    if do_assert_wsp and depth == 0:
        stsd_keys = stsd.keys()
        del_keys = set([])
        for samp in stsd_keys:
            if "wsp" not in stsd[samp]:
                if do_detail:
                    print(">>>''wsp'' not in stsd[%s]\ntarget_dir: %s\nDeleting samp key...<<<"%(samp, target_dir))
                del_keys.add(samp)
                continue
        for samp in del_keys:
            del stsd[samp]
    
    if do_assert_fcs and depth == 0:
        stsd_keys = stsd.keys()
        del_keys = set([])
        for samp in stsd_keys:
            if "fcs" not in stsd[samp]:
                if do_detail:
                    print(">>>''fcs'' not in stsd[%s]\ntarget_dir: %s\nDeleting samp key...<<<"%(samp, target_dir))
                del_keys.add(samp)
                continue
        for samp in del_keys:
            del stsd[samp]
    return stsd

def get_file_dict(target_dir = None, 
                  sep_symbols = [".","__","_","-"],
                  info_sep = ".",
                  file_formats = None):
    
    assert file_format is not None
    
    path_name = os.listdir(target_dir)
    
    file_format_splits = []
    for file_format in file_formats:
        file_format_split = file_format[:]
        for sep_symbol in sep_symbols:
            file_format_split = file_format_split.replace(sep_symbol, info_sep)
        file_format_split = file_format_split.split(info_sep)
        file_format_splits.append(file_format_split)
        
    def check_match(file_format_split):
        pass
    
    for path in paths:
        if os.path.join(target_dir, path_name):
            return get_file_dict(target_dir = target_dir, 
                  sep_symbols = sep_symbols,
                  info_sep = info_sep,
                  file_format = file_format)
        else:
            pass
    
ALL_STAINS = ["Stain 1","Stain 2","Stain 3","Stain 4", "Stain 5"]
ALL_STAINS_PRIORITY = ["Stain 1","Stain 4","Stain 2","Stain 3","Stain 5"]
default_stain = "Stain 1"

def get_channel_alias(channel_name, stain = None, channel_dict = None, logging = None):
    """
    Get channel alias with input channel and corresponding stain
    """
    # if something is invalid about the stain or the channel_dict isn't provided
    if stain not in ALL_STAINS or channel_dict is None:
        if logging is not None:
            logging.warning(">>> stain invalid or channel_dict not provided! Returning unaltered channel_name <<<")
        return channel_name
        
    try:
        return channel_dict[stain][channel_name]
    except KeyError:
        #print(">>> stain %s : channel_name %s not found in channel_dict! Replacing with occurance from other stain panel <<<"%(stain, channel_name))

        if logging is not None:
            #logging.warning(">>> stain %s : channel_name %s not found in channel_dict! Replacing with occurance from other stain panel <<<"%(stain, channel_name))
            pass
            
        for stain_test in ALL_STAINS_PRIORITY:
            if channel_name in channel_dict[stain_test]:
                #print((">>> replacement from stain %s : channel_name %s : alias %s"%(stain_test, channel_name, channel_dict[stain_test][channel_name])))
                #logging.warning(">>> replacement from stain %s : channel_name %s : alias %s"%(stain_test, channel_name, channel_dict[stain_test][channel_name]))
                return channel_dict[stain_test][channel_name]
                
        if logging is not None:
            if logging is not None:
                print(">>> Replacement not found for any stain! Returning unaltered channel_name <<<")
                logging.warning(">>> Replacement not found for any stain! Returning unaltered channel_name <<<")
            
        return channel_name
    
def fcs_custom_write(filename, 
                     chn_names, 
                     data,
                     text_kw_pr={},
                     endianness=None,
                     compat_chn_names=False,
                     compat_copy=False,
                     compat_negative=False,
                     compat_percent=False,
                     backslash_replace = "--",
                     compat_max_int16=10000):
    
    """Write numpy data to an .fcs file (FCS3.0 file format)
    Parameters
    ----------
    filename: str or pathlib.Path
        Path to the output .fcs file
    ch_names: list of str, length C
        Names of the output channels
    data: 2d ndarray of shape (N,C)
        The numpy array data to store as .fcs file format.
    text_kw_pr: dict
        User-defined, optional key-value pairs that are stored
        in the primary TEXT segment
    endianness: str|None
        Set to "little" or "big" to explicitly define the byte 
        order used. If None, the endianness is inherited from the
        $BYTEORD key in text_kw_pr
    compat_chn_names: bool
        Compatibility mode for 3rd party flow analysis software:
        The characters " ", "?", and "_" are removed in the output
        channel names.
    compat_copy: bool
        Do not override the input array `data` when modified in
        compatibility mode.
    compat_negative: bool
        Compatibliity mode for 3rd party flow analysis software:
        Flip the sign of `data` if its mean is smaller than zero.
    compat_percent: bool
        Compatibliity mode for 3rd party flow analysis software:
        If a column in `data` contains values only between 0 and 1,
        they are multiplied by 100.
    compat_max_int16: int
        Compatibliity mode for 3rd party flow analysis software:
        If a column in `data` has a maximum above this value,
        then the display-maximum is set to 2**15.
    Notes
    -----
    - These commonly used unicode characters are replaced: "µ", "²"
    - If the input data contain NaN values, the corresponding rows
      are excluded due to incompatibility with the FCS file format.
    """
    
    import pathlib
    import struct
    import warnings

    import numpy as np

    from fcswrite._version import version

    # Put this in a fresh dict since we modify it later in the function
    _text_kw_pr = text_kw_pr.copy()
    # Drop the keys that will need to be filled at write-time by this function
    for k in ['__header__', 
              '$BEGINANALYSIS', '$ENDANALYSIS', '$BEGINSTEXT', '$ENDSTEXT', 
              '$BEGINDATA', '$ENDDATA', '$DATATYPE', '$MODE', 
              '$NEXTDATA', '$TOT', '$PAR']:
        _text_kw_pr.pop(k, None)

    filename = pathlib.Path(filename)
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=float)
    # remove rows with nan values
    nanrows = np.isnan(data).any(axis=1)
    if np.sum(nanrows):
        msg = "Rows containing NaNs are not written to {}!".format(filename)
        warnings.warn(msg)
        data = data[~nanrows]

    if endianness == "little":
        # use little endian
        byteord = '1,2,3,4'
    elif endianness == "big":
        # use big endian
        byteord = '4,3,2,1'
    else:
        try:
            byteord = _text_kw_pr.pop('$BYTEORD')
        except KeyError:
            raise ValueError('Cannot have `endianness=None` if `$BYTEORD` is not a key in '
                               '`text_kw_pr`')
        else:
            if byteord not in ('4,3,2,1', '1,2,3,4'):
                raise ValueError('`text_kw_pr["$BYTEORD"]` can only be one of `1,2,3,4` or '
                                 '`4,3,2,1`.')

    msg = "length of `chn_names` must match length of 2nd axis of `data`"
    assert len(chn_names) == data.shape[1], msg

    rpl = [["µ", "u"],
           ["²", "2"],
           ]

    if compat_chn_names:
        # Compatibility mode: Clean up headers.
        rpl += [[" ", ""],
                ["?", ""],
                ["_", ""],
                ]

    for ii in range(len(chn_names)):
        for (a, b) in rpl:
            chn_names[ii] = chn_names[ii].replace(a, b)

    # Data with values between 0 and 1
    pcnt_cands = []
    for ch in range(data.shape[1]):
        if data[:, ch].min() >= 0 and data[:, ch].max() <= 1:
            pcnt_cands.append(ch)
    if compat_percent and pcnt_cands:
        # Compatibility mode: Scale values b/w 0 and 1 to percent
        if compat_copy:
            # copy if requested
            data = data.copy()
        for ch in pcnt_cands:
            data[:, ch] *= 100

    if compat_negative:
        toflip = []
        for ch in range(data.shape[1]):
            if np.mean(data[:, ch]) < 0:
                toflip.append(ch)
        if len(toflip):
            if compat_copy:
                # copy if requested
                data = data.copy()
            for ch in toflip:
                data[:, ch] *= -1

    # DATA segment
    data1 = data.flatten().tolist()
    DATA = struct.pack('>%sf' % len(data1), *data1)

    # TEXT segment
    ## note that "/" separates each new keyword argument in the text segment!
    header_size = 256

    TEXT = '/$BEGINANALYSIS/0/$ENDANALYSIS/0'
    TEXT += '/$BEGINSTEXT/0/$ENDSTEXT/0'
    # Add placeholders for $BEGINDATA and $ENDDATA, because we don't
    # know yet how long TEXT is.
    TEXT += '/$BEGINDATA/{data_start_byte}/$ENDDATA/{data_end_byte}'
    TEXT += '/$BYTEORD/{0}/$DATATYPE/F'.format(byteord)
    TEXT += '/$MODE/L/$NEXTDATA/0/$TOT/{0}'.format(data.shape[0])
    TEXT += '/$PAR/{0}'.format(data.shape[1])
    # Add fcswrite version
    TEXT += '/fcswrite version/{0}'.format(version)

    # Check for content of data columns and set range
    for jj in range(data.shape[1]):
        # Set data maximum to that of int16
        if (compat_max_int16 and
            np.max(data[:, jj]) > compat_max_int16 and
                np.max(data[:, jj]) < 2**15):
            pnrange = int(2**15)
        # Set range for data with values between 0 and 1
        elif jj in pcnt_cands:
            if compat_percent:  # scaled to 100%
                pnrange = 100
            else:  # not scaled
                pnrange = 1
        # default: set range to maxium value found in column
        else:
            pnrange = int(abs(np.max(data[:, jj])))
        # TODO:
        # - Set log/lin
        # Using pop on _text_kw_pr will remove it from the dict if it exists else it will use a 
        # default value . 
        data_str = '/'.join(['',
                            '$P{0}B'.format(jj+1),
                            str(_text_kw_pr.pop('$P{0}B'.format(jj+1), '32')),
                            '$P{0}E'.format(jj+1),
                            str(_text_kw_pr.pop('$P{0}E'.format(jj+1), '0,0')),
                            '$P{0}N'.format(jj+1),
                            str(_text_kw_pr.pop('$P{0}N'.format(jj+1), chn_names[jj])),
                            '$P{0}R'.format(jj+1),
                            str(_text_kw_pr.pop('$P{0}R'.format(jj+1), pnrange)),
                            '$P{0}D'.format(jj+1),
                            str(_text_kw_pr.pop('$P{0}D'.format(jj+1), 'Linear')),
                            '$P{0}G'.format(jj+1),
                            str(_text_kw_pr.pop('$P{0}G'.format(jj+1), '1')),
                            ])
        TEXT += data_str

    # Finally, add any remaining, additional key-value pairs provided by the user.
    ## any instance of "/" is changed to -, since / is reserved for the separation
    for key in sorted(_text_kw_pr.keys()):
        #TEXT += '/{0}/{1}'.format(key, text_kw_pr[key])
        TEXT += '/{0}/{1}'.format(key.replace('/', backslash_replace), 
                                  text_kw_pr[key].replace('/', backslash_replace))

    TEXT += '/'

    # SET $BEGINDATA and $ENDDATA using the current size of TEXT plus padding.
    text_padding = 47  # for visual separation and safety
    data_start_byte = header_size + len(TEXT) + text_padding
    data_end_byte = data_start_byte + len(DATA) - 1
    TEXT = TEXT.format(data_start_byte=data_start_byte,
                       data_end_byte=data_end_byte)
    # Pad TEXT segment with spaces until data_start_byte
    lentxt = len(TEXT)
    TEXT = TEXT.ljust(data_start_byte - header_size, " ")
    
    # HEADER segment
    ver = 'FCS3.0'

    textfirst = '{0: >8}'.format(header_size)
    textlast = '{0: >8}'.format(lentxt + header_size - 1)

    # Starting with FCS 3.0, data segment can end beyond byte 99,999,999,
    # in which case a zero is written in each of the two header fields (the
    # values are given in the text segment keywords $BEGINDATA and $ENDDATA)
    if data_end_byte <= 99999999:
        datafirst = '{0: >8}'.format(data_start_byte)
        datalast = '{0: >8}'.format(data_end_byte)
    else:
        datafirst = '{0: >8}'.format(0)
        datalast = '{0: >8}'.format(0)

    anafirst = '{0: >8}'.format(0)
    analast = '{0: >8}'.format(0)

    HEADER = '{0: <256}'.format(ver + '    '
                                + textfirst
                                + textlast
                                + datafirst
                                + datalast
                                + anafirst
                                + analast)
    # Write data
    with filename.open("wb") as fd:
        fd.write(HEADER.encode("ascii", "replace"))
        fd.write(TEXT.encode("ascii", "replace"))
        fd.write(DATA)
        fd.write(b'00000000')
        
def write_channelFixed_fcs(channel_dict_loc,
                           stain_version,
                           stain_use, 
                           fcs_path, 
                           fcs_out_path, 
                           marker_dict = None,
                           meta_update = None,
                           error_info_csv_path = None,
                           do_print = False,
                           do_ignore_existing = True,
                           logging = None,
                           sep = ","):
    
    if not os.path.exists(fcs_out_path) or do_ignore_existing:
        pass
    else:
        if do_ignore_existing:
            print("''%s'' already is existing path, but continuing..."%fcs_out_path)
        else:
            raise Exception("''%s'' already is existing path!"%fcs_out_path)
    
    
    if error_info_csv_path is not None:
        if not os.path.exists(error_info_csv_path):
            error_info_csv = pd.DataFrame(columns = ["fcs_path",
                                                     "fcs_out_path",
                                                     "stain_version", 
                                                     "stain_use",
                                                     "original_channels",
                                                     "new_channels",
                                                     "unmatched_channels",
                                                     "markers",
                                                     "marker_dict",
                                                    ])
        else:
            error_info_csv = pd.read_csv(error_info_csv_path, index_col = 0, sep = sep)
    
    # creating channel dict from stain_channelAlias_channelStandard.csv table
    # creating channel dict from stain_channelAlias_channelStandard.csv table
    if isinstance(channel_dict_loc, str):
        channel_dict_df = pd.read_csv(channel_dict_loc)
        channel_dict = {}
        for i in channel_dict_df.index:
            stain, fluor_alias, fluor_standard = \
                channel_dict_df.loc[i][["stain", "fluor_alias","fluor_standard"]].values.tolist()
                
            if stain not in channel_dict:
                channel_dict[stain] = {}
            if fluor_alias not in channel_dict[stain]:
                channel_dict[stain][fluor_alias] = fluor_standard
            else:
                if channel_dict[stain][fluor_alias] != fluor_standard:
                    print("%s already in channel_dict[%s]!"%(fluor_alias, stain))
                    print(" prev channel_dict[stain][fluor_alias]:", channel_dict[stain][fluor_alias])
                    print(" new fluor_standard:", fluor_standard)
                    print()
                    #raise Exception("%s already in channel_dict[%s]!"%(fluor_alias, stain))
                    
    elif isinstance(channel_dict_loc, dict):
        channel_dict = channel_dict_loc
    else:
        raise Exception("channel_dict_loc type %s not supported"%(type(channel_dict_loc)))


    # renaming parameter names in the fcs files...
    meta, data = fcsparser.parse(fcs_path)
    try:
        meta.pop("$SRC")
    except:
        if logging is not None:
            logging.warning(">>>$SRC key not in fcs meta?<<<")
        print(">>>$SRC key not in fcs meta?<<<")
    
    if meta_update is None:
        meta_update = {}
    meta_old = {}
    param_marker_change = {}
    
    # unfound channels
    unmatched_channels = []
    # original and new channels
    original_channels = []
    new_channels = []
    # markers named
    markers = []
    
    # fluor channels
    param_meta_reg = re.compile(r"\$P.+N")
    for key in sorted(meta.keys()):
        if bool(re.match(param_meta_reg, key)):
            param_update = get_channel_alias(meta[key], stain_use, channel_dict, logging)
            if do_print:
                print("%s: %s -> %s"%(key, meta[key], param_update))
            # new updated channel key
            meta_update[key] = param_update
            # storing old key
            meta_old[key] = meta[key]
            # what markers have been changed, and to what
            param_marker_change[meta[key]] = meta_update[key] 
            original_channels.append(meta[key])
            new_channels.append(param_update)
            
            # replaces parameter naming with the naming from the marker_dict, and uses the standard population naming changes on that value
            if marker_dict is not None:
                key_marker = key[:-1]+"S"
                if key[:-1]+"S" in sorted(meta.keys()):
                    if param_update in marker_dict:
                        marker_update = get_pop_alias(marker_dict[param_update])
                    else:
                        marker_update = get_pop_alias(meta[key_marker])
                        if logging is not None:
                            logging.warning(">>>channel %s not found in the marker_dict! Using %s...<<<"%(param_update, marker_update))
                        unmatched_channels.append(meta[key])
                        
                    meta_update[key_marker] = marker_update
                    meta_old[key_marker] = meta[key_marker]
                    param_marker_change[meta[key_marker]] = marker_update
                    markers.append(marker_update)
                    #print(meta[key_marker], "(%s)"%param_update, "->", marker_update)
            else:
                markers = "markers unchanged"
                unmatched_channels = "marker_dict unprovided"
                
    # parameters channels
    # param_meta_reg = re.compile(r"\$P.+S")
    # for key in meta.keys():
        # if bool(re.match(param_meta_reg, key)):
            # if do_print:
                # print("%s: %s -> %s"%(key, meta[key], get_channel_alias(meta[key], stain_use, channel_dict, logging)))
            # meta_update[key] = get_pop_alias(meta[key])
            # meta_old[key] = meta[key]
            # param_marker_change[meta[key]] = meta_update[key]
        
    
    # renaming acquisition compensation matrix if not named correctly, and creating key "SPILL" if it does not exist...
    if "SPILL" not in meta:
        if logging is not None:
            logging.warn(">>>SPILL not in meta! Finding alternative spill...<<<")
        possible_spills = [k for k in meta.keys() if "spill" in k.lower()]
        
        if possible_spills != []:
            other_spill_key = possible_spills[0]
            if len(possible_spills) > 1:
                if logging is not None:
                    logging.warn(">>>multiple spills! Choosing first spill: %s<<<"%other_spill_key)
            else:
                if logging is not None:
                    logging.warn(">>>Choosing other spill: %s<<<"%other_spill_key)
                
            meta["SPILL"] = meta[other_spill_key]
        else:
            if logging is not None:
                logging.warn(">>>No spill found! Skipping spill renaming<<<")
    
    # with newly created SPILL key-vaue, now changing the params
    if "SPILL" in meta:
        spillover = meta["SPILL"]
        spillover_list = spillover.split(",")
        
        # first comma separated value in the spill should correspond to the number of parameters...
        try:
            param_num = int(spillover_list[0])
            for i, s in enumerate(spillover_list[1:param_num+1],1):
                spillover_list[i] = get_channel_alias(s, stain_use, channel_dict, logging)
        except ValueError:
            for i, s in enumerate(spillover_list,0):
                spillover_list[i] = get_channel_alias(s, stain_use, channel_dict, logging)
                # if stain_use in channel_dict:
                    # if s in channel_dict[stain_use]:
                        # spillover_list[i] = channel_dict[stain_use][s]
                    # #spillover_list[i] = get_channel_alias(meta[key], stain_use, channel_dict, logging)
                # else:
                    # if logging is not None:
                        # logging.warning(">>>%s not in channel_dict!<<<"%stain_use)
                    # print(">>>%s not in channel_dict!<<<"%stain_use)
        
        meta_update["SPILL"] = ",".join(spillover_list)
        meta_old["SPILL"] = ",".join(spillover_list)
    
    # renaming data columns...
    col_rename = {}
    for c in data.columns:
        if do_print:
            print("%s, %s -> %s"%(stain_use, c, get_channel_alias(c, stain_use, channel_dict, logging)))
        col_rename[c] = get_channel_alias(c, stain_use, channel_dict, logging)
    data = data.rename(col_rename, axis = 1)
    
    meta.update(meta_update)
    
    # now to resave the fcs path
    print("saving new fcs to ''%s''..."%fcs_out_path)
    fcs_custom_write(fcs_out_path, 
                     chn_names = data.columns.values, 
                     data = data.values, 
                     text_kw_pr = meta)
                     
    if error_info_csv_path is not None:
        error_info_csv.loc[os.path.basename(fcs_out_path)] = [fcs_path, #"fcs_path",
                                                          fcs_out_path, #"fcs_out_path",
                                                          stain_version,
                                                          stain_use, #"stain_use",
                                                          str(original_channels),
                                                          str(new_channels),
                                                          str(unmatched_channels),
                                                          str(markers),
                                                          str(marker_dict),
                                                         ]
            
    if error_info_csv_path is not None:
        error_info_csv.to_csv(error_info_csv_path, sep = ",")
    
    return param_marker_change

def get_fcs_meta_data(fcsdir,
                      logging = None,
                      meta_relevant_keys = ["tubename", "date", "etim", "experiment name", "export time", "export user name", "cytometer config name", "cytometer config create date", "fil", "COUNTS", "SPILL", "SPILL_EXISTS"]):
    """
    Returns meta data within the .fcs flow files.
    It searches a directory recursively for .fcs files, then sets the values in the dataframe to the meta data key value, ignoring case, spacing, and "$"
    """
    
    fcs_dict = {}
    fcs_dict = get_flow_files(fcsdir, name_conditions = [".fcs"], in_dict = fcs_dict)
    
    file_keys = ["fcs_file_name", "fcs_file_path"]
    fcs_metadata_df = pd.DataFrame(index = np.arange(1, len(fcs_dict.keys())+1), columns = file_keys+meta_relevant_keys)
    
    for i, fcs_name in enumerate(sorted(fcs_dict.keys()), 1):
    
        if logging is not None:
            logging.info("%i : %s"%(i, fcs_name))
            
        print("%i : %s"%(i, fcs_name))
        fcs_path = fcs_dict[fcs_name]
        fcs_metadata_df.loc[i]["fcs_file_name"] = fcs_name
        fcs_metadata_df.loc[i]["fcs_file_path"] = fcs_path
        meta, data = fcsparser.parse(fcs_path)
        
        # renaming acquisition compensation matrix if not named correctly, and creating key "SPILL" if it does not exist...
        if "SPILL" not in meta:
            if logging is not None:
                logging.warn(">>>SPILL not in meta! Finding alternative spill...<<<")
            possible_spills = [k for k in meta.keys() if "spill" in k.lower()]
            
            if possible_spills != []:
                other_spill_key = possible_spills[0]
                if len(possible_spills) > 1:
                    if logging is not None:
                        logging.warn(">>>multiple spills! Choosing first spill: %s<<<"%other_spill_key)
                else:
                    if logging is not None:
                        logging.warn(">>>Choosing other spill: %s<<<"%other_spill_key)
                    
                meta["SPILL"] = meta[other_spill_key]
            else:
                if logging is not None:
                    logging.warn(">>>No spill found! Skipping spill renaming<<<")
        
        for rk in meta_relevant_keys:
            if rk == "COUNTS":
                fcs_metadata_df.loc[i][rk] = data.shape[0]
            elif rk == "SPILL":
                fcs_metadata_df.loc[i][rk] = meta["SPILL"]
            elif rk == "SPILL_EXISTS":
                fcs_metadata_df.loc[i][rk] = "SPILL" in meta.keys()
            else:
                for k in meta.keys():
                    if k.lower().replace(" ","").replace("$","") == rk.replace(" ",""):
                        fcs_metadata_df.loc[i][rk] = meta[k]
        
        if logging is not None:
            logging.info(pprint.pformat(fcs_metadata_df.loc[i]))
            
    return fcs_metadata_df
        

FLOW_STAINS = ['dc', 'innate', 'nktb', 'sort', 'treg']
STAIN_NUMBERS = {
    'stain1': 'treg',
    'stain2': 'nktb',
    'stain3': 'sort',
    'stain4': 'dc',
    'stain5': 'innate'
}

def apply_simple_fixes(sample_name):
        updated_sample_name = sample_name.replace(
            'ttreg', 'treg').replace(
            '.', '_').replace(
            ' ', '_').replace(
            'LIUNG', 'LUNG').replace(
            'SARC', 'SRC').replace(
            'IPIPCRC', 'IPICRC').replace(
            'IPILHEP', 'IPIHEP').replace(
            'BMEL', 'MELB').replace(
            'GBA', 'GBM').replace(
            'IPIPHEP', 'IPIHEP').replace(
            'IPIIKID', 'IPIKID').replace(
            'IPIPGYN', 'IPIGYN')
        if not updated_sample_name.startswith('IPI'):
            if updated_sample_name.startswith('IP'):
                updated_sample_name = updated_sample_name.replace('IP', 'IPI')
            else:
                updated_sample_name = 'IPI' + updated_sample_name
        stain = updated_sample_name.split('_')[-1]
        if stain in STAIN_NUMBERS.keys():
            updated_sample_name = updated_sample_name.replace(
                stain, STAIN_NUMBERS[stain])
        stain = updated_sample_name.split('_')[-1]
        if stain not in FLOW_STAINS and any(stain.startswith(valid_stain) for valid_stain in FLOW_STAINS):
            updated_sample_name = re.match(
                r'(?P<sample_name>IPI.*[dc|innate|nktb|sort|treg]).*', updated_sample_name).group('sample_name')
        # Should be a 3 digit number after the indication
        patient_id = updated_sample_name.split('_')[0]
        code_regex = re.compile(r'IPI[a-z]+(?P<code>[0-9]+)', re.IGNORECASE)
        if code_regex.match(patient_id):
            code = code_regex.match(patient_id).group('code')
            if len(code) > 3 and code[0] == '0':
                updated_sample_name = updated_sample_name.replace(
                    code, code[1::])
            elif len(code) < 3:
                updated_sample_name = updated_sample_name.replace(
                    code, code.rjust(3, '0'))
        return updated_sample_name

if __name__ == "__main__":
    """
    Outputs fcs with fixed channel names.
    
    Ex. python fcs_channel_renaming.py "<stain_channelAlias_channelStandard.csv location>" "<Stain #>" "<fcs_in_path>" "<fcs_out_path>"
    """
    write_channelFixed_fcs(*sys.argv[1:])
    