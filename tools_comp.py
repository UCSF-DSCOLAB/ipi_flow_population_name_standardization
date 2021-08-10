from __future__ import print_function, unicode_literals, division

import fcswrite
import pandas as pd
import os
import numpy as np
import xml.etree.ElementTree as ET
import time
import pprint
import scipy.stats as stats
from sklearn.neighbors.kde import KernelDensity
from scipy.optimize import curve_fit
import re

#######################
###CANNED ALGORITHMS###
#######################

class custom_return_algorithm:
    """
    Custom algorithm that can take place of majority of ml learning algorithms, mimicking sklearn alg function calls.
    """
    
    def __init__(self, name = None, return_function = None, params = {}):
        """
        name (name of algorithm), return_function (custom function to run prediction on with input. default is trivial_algReturn -- f(input) = constant, default f(input) = 0),
        params (parameters for return_function
        """
        
        if return_function == None:
            self.return_function = self.trivial_algReturn
            self.params = {}
        else:
            self.return_function = return_function
            self.params = params
            
    def predict(self, data):
        return self.return_function(data, **self.params)
        
    def get_params(self):
        return self.params
        
    def trivial_algReturn(input_df, return_val = 0.):
        return np.tile(return_val, input_df.shape[0])
        
        
class quadrant_alg(custom_return_algorithm):

    def __init__(self, name = "quadrant_alg", params = {}):
        assert "params_split" in params
        if "mid_type" not in params:
            params["mid_type"] = "mean"
            
        super().__init__(return_function = self.quadrant_algReturn, params = params)
        self.name = name
        
    def quadrant_algReturn(self, data, params_split, mid_type):
        assert all([param in data.columns for param in params_split])
        
        pred_arr = np.tile(1., data.shape[0])
        for param in params_split:
            if mid_type == "mean":
                mid_val = np.mean(data[param].values)
            elif mid_type == "median":
                mid_val = np.median(data[param].values)
            
            # is param greater than mid_val when 1, or less than when 0?
            param_mask = np.array(((2.*params_split[param] - 1)*(data[param].values - mid_val)) >= 0) * 1. 
            pred_arr = pred_arr * param_mask
        
        return pred_arr
    
##############################
###TRANSFORMS AND FUNCTIONS###
##############################

def power_transform(x, shift = 0, a1 = 1, b1 = 1/2., use_sign = True):
    """
    input: x, shift, a1, b1
    output: a1*(x-shift)**(b1)
    """
    
    x_sign = 1
    if use_sign:
        x_sign = np.sign(x)
    x = np.abs(x)

    return x_sign*a1*(x-shift)**(b1)

def log_transform(x, a1 = 1, shift = 1, base = np.e, use_sign = True):
    """
    input: x, a1 = 1, shift = 1, base = np.e, use_sign = True
    output: x_sign*a1*np.log(x+shift)/np.log(base)
    """
    
    x_sign = 1
    if use_sign:
        x_sign = np.sign(x)

    x = np.abs(x)

    return x_sign*a1*np.log(x+shift)/np.log(base)
    
def pol1_transform(vals, x1, x2, y1 = None, y2 = None):
    """
    Linear transformation solving for order 1 polynomial f satisfying f(x1) = y1 and f(x2) = y2
    
    input: vals (data to transform), x1, x2, y1, y2 (linear transformation solving f(x1) = y1 and f(x2) = y2)
    output: f(vals) (transformed input values)
    """
    
    if isinstance(x1, tuple) and isinstance(x2, tuple):
        (x1, y1), (x2, y2) = x1, x2
    
    vals = np.array(vals,dtype = float)
    
    if (x2 == x1 and y2 != y1) or (x2 != x1 and y2 == y1):
        return np.tile(np.nan, vals.shape)
    elif x2 == x1 and y2 == y1:
        return vals
        
    [x1,x2,y1,y2] = np.array([x1,x2,y1,y2], dtype = float).tolist()
    return ((y2-y1)/(x2-x1))*vals+(y1-(y2-y1)/(x2-x1)*x1)
    
def inv_log_transform(x, a1 = 1, shift = 1, base = np.e, use_sign = True):
    """
    input: x, a1 = 1, shift = 1, base = np.e, use_sign = True
    output:
    """
    
    x_sign = 1
    if use_sign:
        x_sign = np.sign(x)

    return x_sign*np.e**(x_sign*x*np.log(base)/a1)-shift

def trivial_transform(x):
    return x

def lin_return_x(y, p1, p2):
    """
    Return x coord of point on line p1>p2 with y coord y]

    input: y (the output), p1, p2 (two 2-uple points to represent the line)
    output: calculated x coordinate
    """
    
    return (p2[0]-p1[0])/(p2[1]-p1[1])*(y - p1[1])+p1[0]

def calc_std(column_vals, scalar = 0, indices = None):
    """
    input: column_vals, scalar = 0, indices = None
    output: series of variances or single std value
    """

    column_values = series_column.values
    if not isinstance(indices, list) and isinstance(column_vals, pd.Series):
        indices = series_column.index.values

    meta_vals = np.tile(np.std(column_values[indices]), np.shape(column_values)[0])
    if not scalar:
        return pd.Series(meta_vals, index = series_column.index.values)
    else:
        return np.var(column_values[indices])

def calc_mean(column_vals, scalar = 0, indices = None):
    """
    input: column_vals, scalar = 0, indices = None,
    output: array of means or single mean value"""

    column_values = series_column
    # uses all indices in column values, otherwise takes mean of only index in indices
    if not isinstance(indices, list) and isinstance(column_vals, pd.Series):
        indices = series_column.index.values

    meta_vals = np.tile(np.mean(column_values[indices]), np.shape(column_values)[0])
    if not scalar:
        return pd.Series(meta_vals, index = series_column.index.values)
    else:
        return np.mean(column_values[indices])

# fitting functions
def gauss(x,c,mu,sigma):
    """General gaussian distribution function, unnormalized."""
    return c*np.e**(-1./2.*((x-mu)/sigma)**2)

def double_gauss(x,c1,mu1,sigma1,c2,mu2,sigma2):
    """two peaked distribution function, unnormalized."""
    return gauss(x,c1,mu1,sigma1)+gauss(x,c2,mu2,sigma2)
    
def get_histMeanSTD(x, y, min_y, dev_const = .01):
    binned_avs = np.array(x)
    n_hist = np.array(y)
    
    bool_use = n_hist > min_y
    average_inBin = float(np.sum(n_hist[bool_use])/len(n_hist[bool_use]))

    bin_mult_n = np.mean(binned_avs[bool_use]*n_hist[bool_use])
    weighted_mean = bin_mult_n/average_inBin
    
    # each point in a bin is considered to be at the same value
    deviation = np.sqrt(np.sum((binned_avs[bool_use]-weighted_mean)**2*n_hist[bool_use])/sum(n_hist[bool_use]))+dev_const
    return weighted_mean, deviation
    
def double_gauss_p0(x, y, min_y = .01):
    """Returns reasonable p0 for initial fitting guess, and boundaries on popt"""
    # assumes x and y are "binning values"
    binned_avs = np.array(x)
    n_hist = np.array(y)
    weighted_mean, deviation = get_histMeanSTD(binned_avs, n_hist, min_y)
    
    p0 = [1., weighted_mean - deviation/2., deviation,
           .8, weighted_mean + deviation/2., deviation]
    p0_bounds = [[0., weighted_mean - 1.5*deviation, 0.00, 
                      0., weighted_mean - 1.5*deviation, 0.00],
                    [n_hist.max(), weighted_mean + 4.*deviation, deviation, 
                      n_hist.max(), weighted_mean + 4.*deviation, deviation]]
    return p0, p0_bounds, weighted_mean, deviation
    
def triple_gauss(x,c1,mu1,sigma1,c2,mu2,sigma2,c3,mu3,sigma3):
    """three peaked distribution function, unnormalized."""
    return gauss(x,c1,mu1,sigma1)+gauss(x,c2,mu2,sigma2)+gauss(x,c3,mu3,sigma3)
    
def linear_landmarks(x, x0, x1):
    """Transforms input values based on landmarks provided.
    For example, linear_landmarks([0,1,2,3,4], 1, 2) -> [-1,0,1,,2,3]
                 linear_landmarks([0,1,2,3,4], 2, 4) -> [-1,-.5,0,.5,1]
    
    input: x, x0 (origin, x' = 0), x1 (unit = 1, x' = 1)
    output: transformed_x
    """
    x = np.array(x)
    return pol1_transform(x, (x0,0.), (x1,1.))
    
TRANSFORM_FUNCTIONS = {
                "power" : power_transform,
                "biex" : log_transform,
                "trivial" : trivial_transform,
             }

############################
###WSP AND FCS OPERATIONS###
############################

ELEMENTS = ['H', 'He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og',]
SYMBOLS = ['{','}','(',')','[',']','.',',',':',';','+','-','*','/','&','|','<','>','=','~','_']

def get_cytof_param_name(param, method = "isotope"):

    #The equivalence is based off the element/isotope mass. We parse the name to get just this mass, and return that mass str
    if method == "isotope":
    
        # to get the isotope mass
        prev_dig = False
        digit_strs = []
        dig_str = ""
        for b in param:
            if b.isdigit():
                dig_str += b
                prev_dig = True
            elif not b.isdigit() and prev_dig:
                digit_strs.append(dig_str)
                dig_str = ""
                prev_dig = False
        if prev_dig:
            digit_strs.append(dig_str)
        
        # were there any digits representing isotopes?
        if len(digit_strs) > 0:
            isotope = sorted(digit_strs, key = lambda x: -len(x))[0]
        else:
            return param
        
        prev_char = False
        symbols = []
        symbol_str = ""
        for b in param:
            if not b.isdigit() and b not in SYMBOLS:
                symbol_str += b
                prev_char = True
            elif (b.isdigit() or b in SYMBOLS) and prev_char:
                symbols.append(symbol_str)
                symbol_str = ""
                prev_char = False
        if prev_char:
            symbols.append(symbol_str)
            
        # to get the isotope element symbol
        found = False
        for elem in ELEMENTS:
            if elem in [e for e in symbols]:
                found = True
                break
        if not found:
            raise Exception(">>>Element symbol not found!<<<")
            
        return elem+isotope
    else:
        print(">>>method %s not implemented. returning original param<<<")
        return param
        
def wsp_getComp(wsp_file, 
                stain_pop_parse = None,
                stain_standard_stain = None,
                experiment_channel_alias = None,
                standard_channels = True, 
                tissue_type = None, 
                bypass_tissue_type = False,
                experiment_type = "cytof",
                do_ignore_sample_num = True):
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
    if experiment_type == "facs":
        samples_elements = [r for r in root.findall("./SampleList/Sample/") if
                            "name" in r.attrib.keys() and
                            "spillover" in r.tag]
        sample_identifiers = [r for r in root.findall("./SampleList/Sample/") if
                              "uri" in r.attrib.keys() and
                              r.tag == "DataSet"]
    elif experiment_type == "cytof":
        samples_elements = [r for r in root.findall("./SampleList/Sample/") if 
                            "name" in r.attrib.keys()]
        sample_identifiers = [r for r in root.findall("./SampleList/Sample/") if
                              "uri" in r.attrib.keys()] 
    else:
        print(">>>experiment_type not yet supported!<<<")

    # comp identifiers and value elements should be the same size...
    print("len(samples_elements):", len(samples_elements))
    print("len(sample_identifiers):", len(sample_identifiers))
    try:
        assert len(samples_elements) == len(sample_identifiers)
    except:
        print([sample.attrib["name"].lower() for sample in samples_elements])
        print([ident.get("uri").lower() for ident in sample_identifiers])
        raise Exception("len(samples_elements) == len(sample_identifiers)")
        
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
        if (sample_num is not None) and (not do_ignore_sample_num):
            print(stain_name_lower, sample_num)

            if "ipi" in stain_name_lower:
                # the cell staindard is probably in the name too...
                for stain_parse in stain_pop_parse:
                    if "_"+stain_parse in stain_name_lower:
                        if "_"+tissue_type.lower() in ident.get("uri").lower() or bypass_tissue_type:
                            spillover_names[i] = [stain_name_lower, stain_standard_stain[stain_parse]]

            elif sample_num in [str(s) for s in [1,2,3,4,5]]:
                spillover_names[i] = [stain_name_lower, stain_standard_stain[sample_num]]

            else:
                print(">>>naming specifications not met for %s. continuing...<<<"%sample_info["name"].lower())
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
    
def fsc_change(fcs_data, data_name = None, do_norm_fsc = True, do_plot = False):
    """
    Checks the fcs_data forward scatter to see if the names have been changed (an issue between versions 11.4-11.5)
    Currently, the switch is made when the slope of the normed fsch v fsca (width vs area) is greater than the slope of normed fscw v fsca (height vs area)
    
    input: fcs_data
    output: fcs_data (either with appropriate column renames, or original), bool (whether column names changed or not)
    """

    # good estimation that works most of the time?
    fsca_temp, fscw_temp, fsch_temp = fcs_data["FSC-A"], fcs_data["FSC-W"], fcs_data["FSC-H"]
    fsca_temp_mask = fsca_temp < 250000
    fscw_temp_mask = fscw_temp < 250000
    fsch_temp_mask = fsch_temp < 250000
    valid_mask = fsca_temp_mask * fscw_temp_mask * fsch_temp_mask
    fsca_temp = fsca_temp[valid_mask]
    fscw_temp = fscw_temp[valid_mask]
    fsch_temp = fsch_temp[valid_mask]
    
    if do_norm_fsc:
        fsca_temp = linear_landmarks(fsca_temp, fsca_temp.min(), fsca_temp.max())
        fscw_temp = linear_landmarks(fscw_temp, fscw_temp.min(), fscw_temp.max())
        fsch_temp = linear_landmarks(fsch_temp, fsch_temp.min(), fsch_temp.max())
        
    # norm1 = np.sum((fscw_temp-fsca_temp)/np.linalg.norm([fscw_temp,fsca_temp],axis = 0))
    # norm2 = np.sum((fsch_temp-fsca_temp)/np.linalg.norm([fscw_temp,fsca_temp],axis = 0))
    # m1 = np.sum((fscw_temp>fsca_temp))
    # m2 = np.sum((fsch_temp>fsca_temp))
    m1 = np.std(fscw_temp)/np.std(fsca_temp) * stats.pearsonr(fscw_temp, fsca_temp)[0]
    m2 = np.std(fsch_temp)/np.std(fsca_temp) * stats.pearsonr(fsch_temp, fsca_temp)[0]
    
    if do_plot:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2,1)
        axs[0].scatter(fsca_temp, fscw_temp, s = 2, alpha = .3)
        axs[0].set_title("fscw_temp v fsca_temp; m=%.4f"%m1)
        axs[1].scatter(fsca_temp, fsch_temp, s = 2, alpha = .3)
        axs[1].set_title("fsch_temp v fsca_temp; m=%.4f"%m2)
        plt.show()
        
    # this is to determine if the name has been changed, which appears between 11.4-11.5
    if m2 > m1:
        print(">>>FSC-W and FSC-H switched. %s<<<"%str(data_name))
        return fcs_data.rename(columns={"FSC-W":"FSC-H", "FSC-H":"FSC-W"}), True
    else:
        print(">>>no fsc change.<<<")
        return fcs_data, False

def fcs_getComp(fcs_file, experiment_channel_alias = None, give_lasers = True, standard_channels = True, stain = None):
    """
    Gets acquisition defined compensation matrix from the fcs_file, with the associated laser names for each row/column in the matrix.

    input: fcs_file (path to .fcs file), give_lasers (bool)
    output: compMx_fcs (compensation matrix from acquisition), laser_names (if give_lasers == True)
    """
    meta, _data = get_fcs(fcs_file)
    print("_data.columns", _data.columns)
    del _data
    
    comp_info = meta["SPILL"].split(",")
        
    rows_cols = int(comp_info[0])
    comp_info = comp_info[1:]
    laser_names = comp_info[:rows_cols]

    laser_values = [float(v) for v in comp_info[rows_cols:]]

    compMx_fcs = pd.DataFrame(np.reshape(laser_values, (rows_cols,rows_cols)),
                              index = laser_names, columns = [l for l in laser_names])
    
    print("before standardization:", compMx_fcs.columns)
    if standard_channels:
        use_lasers = [l for l in compMx_fcs.index if l in experiment_channel_alias[stain].keys()]
        compMx_fcs = compMx_fcs.loc[use_lasers][use_lasers]
        compMx_fcs = compMx_fcs.rename(index = experiment_channel_alias[stain], columns = experiment_channel_alias[stain])
        laser_names = [experiment_channel_alias[stain][l] for l in use_lasers]
    print("after standardization:", compMx_fcs.columns)
    
    if give_lasers:
        return compMx_fcs, laser_names
    else:
        return compMx_fcs

def get_fcs(fcs_file, 
            do_use_fcsparser = True, 
            do_cleanup = False,
            key_changes = {"$SPILLOVER" : "SPILL",
                       "SPILLOVER" : "SPILL",
                       "$SPILL" : "SPILL",
                       "SPILL" : "SPILL"}):
    """
    Returns the fcs.data and fcs.meta

    input: fcs_file (path to .fcs file)
    output: fcs_meta (dictionary of meta information from fcs), fcs_data (pd df of data)
    """

    if do_use_fcsparser:
        import fcsparser
        meta, data = fcsparser.parse(fcs_file)
    else:
        from FlowCytometryTools import FCMeasurement
        fcs = FCMeasurement(ID='test', datafile=fcs_file)
        meta, data = fcs.meta, fcs.data
        
    for k in meta.keys():
        if k in key_changes.keys():
            meta[key_changes[k]] = meta.pop(k)
            
    # changing the names that start with $ to without, and removing unnecessary types...
    if do_cleanup:
        str_keys = []
        for k in meta.keys():
            if "$" == k[0]:
                newk = k[1:]
                meta[newk] = meta.pop(k)
                
        for key in meta.keys():
            if type(meta[key]) in set([type(u''), str, set]):
                str_keys.append(key)
                
        meta = {k : meta[k] for k in str_keys}
        
    return meta, data

def get_param_dict(meta = None,
                   fcs_file = None, 
                   do_use_fcsparser = True, 
                   do_cleanup = False, 
                   do_fullInfo = False,
                   stain_fluor_dict_key = "N",
                   stain_fluor_dict_value = "S",
                   stain_channel_alias = None):
    """
    Returns experiment parameter dictionary for single .fcs file.
    
    input: **kwargs (see get_fcs)
    output: param_dict OR stain_fluor_dict (param_dict if do_fullInfo is True, otherwise stain_fluor_dict)
    """
    if isinstance(meta, type(None)):
        meta, _data = get_fcs(fcs_file = fcs_file, do_use_fcsparser = do_use_fcsparser, do_cleanup = do_cleanup)
        del _data
        
    parameter_infos = {k : m for k, m in meta.items() if "$P" in k}
    param_dict = {}
    for key in parameter_infos.keys():
        try:
            pi = int(key[key.index("$P")+2:-1])
            if pi not in param_dict:
                param_dict[pi] = {}
            param_dict[pi][key[-1]] = parameter_infos[key]
        except ValueError:
            pass
        except:
            raise
    if do_fullInfo:
        return param_dict
    
    # now returning dict for Stain marker -> Fluorescent marker only
    stain_fluor_dict = {}
    for pi in param_dict:
        # try except for key existence inside the param_dict...
        
        try:
            key = param_dict[pi][stain_fluor_dict_key]
            # if key is in stain_channel_alias, then we replace key with the associated value
            if not isinstance(stain_channel_alias, type(None)):
                if key in stain_channel_alias:
                    key = stain_channel_alias[key]
            value = param_dict[pi][stain_fluor_dict_value]
            
            if stain_fluor_dict_value in param_dict[pi]:
                stain_fluor_dict[key] = param_dict[pi][stain_fluor_dict_value]
            else:
                stain_fluor_dict[key] = key
        except KeyError:
            pass
            #print(">>>error on entry %s. continuing..."%(str(pi)))
    return stain_fluor_dict
    
#####################
###WRITE FCS FILES###
#####################

def write_mask_fcs(fcs_file = None, out_path = None, mask = None):
    """
    Writes new fcs_file with mask applied. The shape of the fcs file is now smaller in the sample direction (n in (n x m) sized fcs matrix) than the original.

    input: fcs_file (original location of fcs_file), out_path (location to write fcs file to), mask (1/0 array )
    output: (none)
    """
    
    meta, data = get_fcs(fcs_file)
    # applying mask to array
    data = data.loc[mask.astype(bool)]
    
    fcswrite.write_fcs(chn_names = data.columns.values, text_kw_pr = meta, data = data.values,
                    filename = out_path, compat_chn_names = False,
                    compat_percent = False)
                    
def add_value_fcs(fcs_data, mask = None, head = "%",
                        mask_name = None, mult = 1.):
    """
    Writes new column to fcs data containing the new column mask information (such as pop filter)
    Note: Each mask value is multiplied by ''mult'' to overcome the flooring that the fcs writer automatically applies

    input: fcs_data, mask = None, head = FCS_VAL_SEP, mask_name = None (new added value name), mult = 1.
    output: new_data
    """
    
    new_data = fcs_data
    new_data[head+mask_name] = mask * mult
    return new_data

def fcs_custom_write(filename, chn_names, data,
                      text_kw_pr={},
                      endianness=None,
                      compat_chn_names=False,
                      compat_copy=False,
                      compat_negative=False,
                      compat_percent=False,
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

    #LLJ edit -- pop $SRC because can cause errors if wrong unicode characters are in the str...
    text_kw_pr.pop("$SRC")
    
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
    for key in sorted(_text_kw_pr.keys()):
        TEXT += '/{0}/{1}'.format(key, text_kw_pr[key])

    TEXT += '/'

    # SET $BEGINDATA and $ENDDATA using the current size of TEXT plus padding.
    text_padding = 47  # for visual separation and safety
    data_start_byte = header_size + len(TEXT) + text_padding
    data_end_byte = data_start_byte + len(DATA) - 1
    TEXT = TEXT.format(data_start_byte=data_start_byte,
                       data_end_byte=data_end_byte)
    lentxt = len(TEXT)
    # Pad TEXT segment with spaces until data_start_byte
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

def write_value_fcs(fcs_data = None, meta = None, out_path = None, fsc_switch = False):
    """
    Write new fcs with input fcs_data pandas dataframe.

    input: fcs_data (pd dataframe), out_path (path to write .fcs to),
            fsc_switch (defaults to false; only use if the fcs_data has not seen the pop_gates pipeline and has not already been switched)
    output: (None)
    """
    
    if isinstance(fcs_data, pd.DataFrame):
        new_data = fcs_data
    else:
        raise(">>>FCS must be pandas DataFrame object to be written to<<<")

    if fsc_switch:
        new_data, _change = fsc_change(new_data)
    
    if isinstance(meta, type(None)):
        meta = {}
        
    # prev fcswrite version...
    # fcswrite.write_fcs(chn_names = new_data.columns.values, data = new_data.values,
                        # filename = out_path, compat_chn_names = False,
                        # compat_percent = False)
    # current fcs write version
    # fcswrite.write_fcs(chn_names = new_data.columns.values, text_kw_pr = meta, data = new_data.values,
                       # filename = out_path, compat_chn_names = False,
                       # compat_percent = False)
                       
    fcs_custom_write(filename = out_path, 
                     chn_names = new_data.columns.values, 
                     data = new_data, 
                     text_kw_pr = meta)

###################
###SPECIAL GATES###
###################

def timeGate(input_data, params = ["Time", "FSC-A"], period = .001, every_other = 1, give_other = False):
    """
    Since time is so fucked, this should estimate pretty well the expected time gate values.
    Time changes between one measurement to the next can be 0!
    However, the rate is taken over an window, so the average over a long enough window should be ok.

    input: input_data, params = ["Time", "FSC-A"],
            period = .001 (the initial event assumed change), every_other = 1 (skip amount), give_other = False (other parameters used during calculation)
    output: in_gate (mask of events that pass through thte gate), np.array(indices)
    """

    original_indices = input_data.index.values
    original_data = input_data.values

    # convolve lengths....if they are too small, the script encounters an error
    tp = 1000./every_other
    tp2 = 2000./every_other
    #longer periods
    tp3 = tp * 10.
    tp4 = tp2 * 10.
    cp_avg_fsca = [1./tp, int(tp)]
    cp_avg_rate = [1./tp2, int(tp2)]
    cp_avg_fsca_long = [1./tp3, int(tp3)]
    cp_avg_rate_long = [1./tp4, int(tp4)]

    # change it so it is variacne from a longer average
    assert (max(tp,tp2) <= len(input_data.index.values))
    #input_data = input_data.iloc[:][input_data[params[1]] <= 250000]

    fscaConvolve = np.array([v for v in np.convolve(np.tile(*cp_avg_fsca), input_data.iloc[:][params[1]].values, mode = "same")])
    avg_fsca = np.transpose([input_data.iloc[:][params[0]].values, fscaConvolve])

    fscaConvolve_long = np.array([v for v in np.convolve(np.tile(*cp_avg_fsca_long), input_data.iloc[:][params[1]].values, mode = "same")])
    avg_fsca_long = np.transpose([input_data.iloc[:][params[0]].values, fscaConvolve_long])

    timeDiffs = np.append([period],np.diff(input_data.iloc[:][params[0]].values))
    #print("num of change rates == 0.: ", sum([t == 0. for t in timeDiffs]))

    timeDiffsConvolve = np.convolve(np.tile(*cp_avg_rate), timeDiffs, mode = "same")
    rateDiffsConvolve = np.array([1./float(t) for t in timeDiffsConvolve])

    timeDiffsConvolve_long = np.convolve(np.tile(*cp_avg_rate_long), timeDiffs, mode = "same")
    rateDiffsConvolve_long = np.array([1./float(t) for t in timeDiffsConvolve_long])

    avg_rate = np.transpose([input_data.iloc[:][params[0]].values, rateDiffsConvolve])
    avg_rate_long =  np.transpose([input_data.iloc[:][params[0]].values, rateDiffsConvolve_long])

    # statistical calculations
    mean_avg_rate = np.mean(rateDiffsConvolve)
    mean_avg_fsca = np.mean(fscaConvolve)

    mid_avg_rate = np.median(rateDiffsConvolve)

    std_avg_rate = np.std(rateDiffsConvolve)
    std_avg_fsca = np.std(fscaConvolve)

    max_avg_rate = np.max(rateDiffsConvolve)
    max_avg_fsca = np.max(fscaConvolve)

    min_avg_rate = np.min(rateDiffsConvolve)
    min_avg_fsca = np.min(fscaConvolve)

    max_fsca = np.max(input_data.iloc[:][params[1]].values)
    min_fsca = np.min(input_data.iloc[:][params[1]].values)

    # rate limits
    lower_avg_fsca = mean_avg_fsca-4.*std_avg_fsca
    upper_avg_fsca = mean_avg_fsca+6.*std_avg_fsca

    lower_avg_rate = mid_avg_rate-1.*std_avg_rate
    upper_avg_rate = mid_avg_rate+1.*std_avg_rate

    lower_fsca = min_fsca+10000
    upper_fsca = max_fsca-5000

    boundaries = {"lower_avg_fsca" : lower_avg_fsca,
                    "upper_avg_fsca" : upper_avg_fsca,
                    "lower_avg_rate" : lower_avg_rate,
                    "upper_avg_rate" : upper_avg_rate,
                    "lower_fsca" : lower_fsca,
                    "upper_fsca" : upper_fsca}

    print("lower_avg_fsca, upper_avg_fsca, lower_avg_rate, upper_avg_rate, mid_avg_rate, lower_fsca, upper_fsca:")
    print(lower_avg_fsca, upper_avg_fsca, lower_avg_rate, upper_avg_rate, mid_avg_rate, lower_fsca, upper_fsca)


    #print("calculating indices that satisfy time criteria...")
    in_gate = np.tile(0, len(list(input_data.index.values)))
    indices = []
    i = 0
    df_indices = original_indices
    while i < len(input_data.index.values):
        #print(i)
        #print(np.shape(avg_fsca))
        stime = time.time()
        this_avg_fsca = avg_fsca[i,1]
        this_avg_rate = avg_rate[i,1]
        this_avg_fsca_long = avg_fsca_long[i,1]
        this_avg_rate_long = avg_rate_long[i,1]

        this_fsca = original_data[i,1]

        '''
        #print(avg_rate.iloc[i][params[0]], mean_avg_rate)
        #dev_time = float(avg_rate.iloc[i][params[0]] - mean_avg_rate)/std_avg_rate
        #print("fsca, lower_fsca, dev_time:", avg_fsca.iloc[i][params[1]], dev_time)

        # for changing long average fsca
        lower_avg_fsca = this_avg_fsca_long - 4.*std_avg_fsca
        upper_avg_fsca = this_avg_fsca_long + 6.*std_avg_fsca
        lower_avg_rate = this_avg_rate_long - 1.*std_avg_rate
        upper_avg_rate = this_avg_rate_long + 1.*std_avg_rate

        if this_avg_fsca >= lower_avg_fsca and this_avg_fsca <= upper_avg_fsca:
            if this_avg_rate >= lower_avg_rate and this_avg_rate <= upper_avg_rate:
                if this_fsca >= lower_fsca and this_fsca <= upper_fsca:
                    in_gate[i] = 1
                    indices.append(int(df_indices[i]))'''

        if this_avg_fsca >= lower_avg_fsca and this_avg_fsca <= upper_avg_fsca:
            if this_avg_rate >= lower_avg_rate and this_avg_rate <= upper_avg_rate:
                if this_fsca >= lower_fsca and this_fsca <= upper_fsca:
        #if True:
                    in_gate[i] = 1
                    indices.append(int(df_indices[i]))

        tot_time = time.time()-stime
        if i % 50000 == 0:
            print(i, float(tot_time), sum(in_gate), this_avg_fsca, this_avg_rate, this_fsca, in_gate[i])

        i+=1

    if give_other:
        return in_gate, np.array(indices), boundaries, avg_rate, avg_rate_long, avg_fsca, avg_fsca_long

    return in_gate, np.array(indices)

def timeGate_simple(input_data, params = ["Time", "FSC-A"], give_other = False):
    """
    Time gate that only considers the upper/lower fsca for exclusion...

    input: input_data, params = ["Time", "FSC-A"], std_above_min = 2, std_below_max = .5,
            every_other = 1 (skip amount), give_other = False (other parameters used in computation)
    output: in_gate, np.array(df_indices[pass_mask]) (original input_data indices with mask applied)
    """

    original_indices = input_data.index.values
    lower_avg_fsca = input_data.iloc[:][params[1]].values.min() + 15000
    upper_avg_fsca = input_data.iloc[:][params[1]].values.max() - 100

    df_indices = original_indices
    #print("calculating indices that satisfy time criteria...")
    pass_mask = ((input_data.iloc[:][params[1]].values >= lower_avg_fsca) * (input_data.iloc[:][params[1]].values <= upper_avg_fsca))
    in_gate = np.tile(0, len(list(input_data.index.values)))
    in_gate[pass_mask] = 1

    return in_gate, np.array(df_indices[pass_mask])

def timeGate2(input_data, params = ["Time", "FSC-A"], every_other = 1, give_other = False, period = .01, fastAvg = 500, medAvg = 2000, slowAvg = 10000, stds = 3.):
    """
    Time gate that only considers the upper/lower fsca for exclusion...
    
    input: input_data, params = ["Time", "FSC-A"], std_above_min = 2, std_below_max = .5,
            every_other = 1 (skip amount), give_other = False (other parameters used in computation)
    output: in_gate, np.array(df_indices[pass_mask]) (original input_data indices with mask applied)
    """

    original_indices = input_data.index.values

    # convolve lengths....if they are too small, the script encounter
    # tp2 is the one compared against both tp and tp3
    tp = int(fastAvg/every_other)
    tp2 = int(medAvg/every_other)
    tp3 = int(slowAvg/every_other)

    #longer periods
    cp_avg_rw = [1./float(tp), tp]
    cp_avg_rw2 =[1./float(tp2), tp2]
    cp_avg_rw3 =[1./float(tp3), tp3]

    # change it so it is variacne from a longer average
    assert (max([tp,tp2,tp3]) <= len(input_data.index.values))
    #input_data = input_data.iloc[:][input_data[params[1]] <= 250000]

    fscaConvolve = np.convolve(np.tile(*cp_avg_rw), input_data.iloc[:][params[1]].values, mode = "same")/\
                        np.convolve(np.tile(*cp_avg_rw), np.tile(1, len(input_data.iloc[:][params[1]].values)), mode = "same")

    fscaConvolve2 = np.convolve(np.tile(*cp_avg_rw2), input_data.iloc[:][params[1]].values, mode = "same")/\
                        np.convolve(np.tile(*cp_avg_rw2), np.tile(1, len(input_data.iloc[:][params[1]].values)), mode = "same")

    fscaConvolve3 = np.convolve(np.tile(*cp_avg_rw3), input_data.iloc[:][params[1]].values, mode = "same")/\
                        np.convolve(np.tile(*cp_avg_rw3), np.tile(1, len(input_data.iloc[:][params[1]].values)), mode = "same")
    #avg_fsca = np.transpose([input_data.iloc[:][params[0]].values, fscaConvolve])

    timeDiffs = np.append([period],np.diff(input_data.iloc[:][params[0]].values))
    #print("num of change rates == 0.: ", sum([t == 0. for t in timeDiffs]))

    timeConvolve = np.convolve(np.tile(*cp_avg_rw), timeDiffs, mode = "same")/\
                        np.convolve(np.tile(*cp_avg_rw), np.tile(1, len(timeDiffs)), mode = "same")

    timeConvolve2 = np.convolve(np.tile(*cp_avg_rw2), timeDiffs, mode = "same")/\
                        np.convolve(np.tile(*cp_avg_rw2), np.tile(1, len(timeDiffs)), mode = "same")

    timeConvolve3 = np.convolve(np.tile(*cp_avg_rw3), timeDiffs, mode = "same")/\
                        np.convolve(np.tile(*cp_avg_rw3), np.tile(1, len(timeDiffs)), mode = "same")

    # statistical calculations
    #mean_avg_fscaConv = np.mean(fscaConvolve)

    # this is a constant...if the flow rate is changed, even though the std of a window should increase, it will with below...
    std_avg_fscaConv2_all = np.std(fscaConvolve2)
    std_avg_timeConv2_all = np.std(timeConvolve2)

    # square of fast - slow, to get the higher bound on std
    fscaSquare13 = np.square(fscaConvolve - fscaConvolve3)
    timeSquare13 = np.square(timeConvolve - timeConvolve3)

    fscaSquare23 = np.square(fscaConvolve2 - fscaConvolve3)
    timeSquare23 = np.square(timeConvolve2 - timeConvolve3)


    std_avg_fscaConv = np.sqrt(np.convolve(np.tile(*cp_avg_rw), fscaSquare23, mode = "same") /\
                            np.convolve(np.tile(*cp_avg_rw), np.tile(1, len(fscaSquare23)), mode = "same"))
    std_avg_timeConv = np.sqrt(np.convolve(np.tile(*cp_avg_rw), timeSquare23, mode = "same") /\
                            np.convolve(np.tile(*cp_avg_rw), np.tile(1, len(timeSquare23)), mode = "same"))

    std_avg_fscaConv = np.concatenate([np.tile(std_avg_fscaConv[-1],tp2), std_avg_fscaConv[:-int(tp2)]], axis = 0)
    std_avg_timeConv = np.concatenate([np.tile(std_avg_timeConv[-1],tp2), std_avg_timeConv[:-int(tp2)]], axis = 0)

    # rate limits....should use constant ones?
    std_balance = .15
    fsca_std = stds * (std_balance * std_avg_fscaConv2_all + (1 - std_balance) * std_avg_fscaConv)
    time_std = stds * (std_balance * std_avg_timeConv2_all + (1 - std_balance) * std_avg_timeConv)

    lower_avg_fsca = fscaConvolve2 - fsca_std
    upper_avg_fsca = fscaConvolve2 + fsca_std

    lower_avg_time = timeConvolve2 - time_std
    upper_avg_time = timeConvolve2 + time_std

    #lower_fsca = input_data.iloc[:][params[1]].values.min() + 25000
    balance = .2
    lower_fsca = (balance*(fscaConvolve3-2.*stds*std_avg_fscaConv)+(1.-balance)*(input_data.iloc[:][params[1]].values.min() + 10000.))
    upper_fsca = input_data.iloc[:][params[1]].values.max() - 100.

    df_indices = original_indices
    #print("calculating indices that satisfy time criteria...")
    pass_mask = ((input_data.iloc[:][params[1]].values >= lower_fsca) * (input_data.iloc[:][params[1]].values <= upper_fsca) *\
                    (fscaConvolve >= lower_avg_fsca) * (fscaConvolve <= upper_avg_fsca) *\
                    (timeConvolve >= lower_avg_time) * (timeConvolve <= upper_avg_time))

    in_gate = np.tile(0, len(list(input_data.index.values)))
    in_gate[pass_mask] = 1
    if give_other:
        return in_gate, np.array(df_indices[pass_mask]), fscaConvolve, lower_avg_fsca, upper_avg_fsca, timeConvolve, lower_avg_time, upper_avg_time

    return in_gate, np.array(df_indices[pass_mask])

def gate_pass(input_data):
    """
    Just passes forward all the data and indices; a completely open permissive gate.
    
    input: input_data,
    output: in_gate (1 array length of input_data), np.array(original_indices)
    """
    
    original_indices = input_data.index.values
    in_gate = np.tile(1, len(list(input_data.index.values)))

    return in_gate, np.array(original_indices)

##################
##PREPROCESSING###
##################

# processing raw/comped data for data input
def preprocessing1(data, auto_skip = True,
                   pop_meta = False,
                   parent_indices = None,
                   zscore = True,
                   meta_processing = \
                           {"mn" : np.mean,
                            "st" : np.std,
                            "md" : np.median}):
    """
    input: dataframe (probably the raw/comped input df seen by techs), auto_skip, pop_meta, parent_indices, zscore, meta_processing
    output: dataframe (with additional meta values)
            ex. z score single channels, differences
    """

    indices = data.index.values
    columns = data.columns.values

    out_data = pd.DataFrame(data.values, columns = data.columns, index = data.index)

    # time mean and std means nothing
    startt = time.time()

    divisor = 1000
    if auto_skip:
        skip_all = int(float(np.shape(out_data.values)[0])/divisor)
        if parent_indices:
            skip_par = int(float(np.shape(out_data.loc[parent_indices].values)[0])/divisor)
        else:
            skip_par = 1
    else:
        skip_all, skip_par = 1, 1

    for c in columns:
        if "Time" not in c:
            for mp in meta_processing:

                # first on all, second no mask if pop_meta
                # the calculations take a while, so skipping amount to speed up
                col_name = c + "__" + mp + "_all"
                meta_val = meta_processing[mp](out_data.iloc[::skip_all][c].values)
                out_data[col_name] = meta_val

                if pop_meta:
                    col_name_parent = c + "__" + mp + "_par"
                    meta_val_par = meta_processing[mp](out_data.loc[parent_indices].iloc[::skip_par][c].values)
                    out_data[col_name_parent] = meta_val_par

    return out_data

def preprocessing2(data,
                   subsample = None,
                   zscore = True,
                   output_raw = False,
                   rolling_windows = [15,30, -1]):
    """
    input: dataframe (probably the raw/comped input df seen by techs), subsample, zscore, output_raw, rolling_windows
    output: dataframe (with additional meta values)
        ex. z score single channels, differences
    """
    print("preprocessing2...")

    orig_indices = data.index.values
    orig_columns = data.columns.values

    out_data = pd.DataFrame(data.values, columns = orig_columns, index = orig_indices)
    # subsampling, so the preprocessing step is quicker...
    if subsample != None:
        pass
    else:
        subsample = 1

    # skipping to make calculation simpler
    out_data = out_data.iloc[::subsample]

    # time diff...
    # we are considering the time_diff as part of the original data...
    # should be time independent!
    time_diff = np.diff(out_data["Time"].values)
    time_diff = np.append(time_diff[[0]],time_diff)
    out_data["Time%sdiff"%PRM_PRM_SEP] = time_diff
    elapse_time = out_data["Time"].values[-1] - out_data["Time"].values[0]
    out_data = out_data.drop("Time", axis = 1)

    # now, out_data includes all the operatable columns
    # sort data so it is easier to match columns later...

    drop_columns = out_data.columns.values

    # rolling zscore, defined from the second value in rolling_zscore
    if rolling_windows != None:
        for rw in rolling_windows:
            if rw == -1:
                index_window = len(out_data.index.values)
                window_conv = np.tile(1./index_window,index_window)
            else:
                # rolling score represents the seconds that should be in the score window....
                # set the window width to that value, based on the maximum/minimum time measured
                av_time_spacing = (elapse_time)/(out_data.index.values[-1] - out_data.index.values[0])
                index_spacing = float(np.sum(np.diff(out_data.index.values))/(len(out_data.index.values)-1.))
                index_window = int(rw/(av_time_spacing*index_spacing))+1
                #print(av_time_spacing, index_spacing, index_window)
                try:
                    assert index_window > 2
                except:
                    print(">>>ERROR: INDEX WINDOW IS TOO SMALL. CONTINUING WITH index_window = 2...<<<")
                    index_window = 2

                window_conv = np.tile(1./index_window,index_window)

            # done on raw columns still...
            for i, c in enumerate(drop_columns, 0):
                #print(c, i, " ; ", end = "")
                if rw != -1:
                    c_r_mean = np.convolve(window_conv, out_data[c].values, mode = "same")
                    # convolved values near beginning are smaller, since only part of the convolution is applied
                    c_conv_shift = np.convolve(window_conv, np.tile(1., len(out_data[c].values)), mode = "same")
                    c_r_mean = c_r_mean / c_conv_shift
                else:
                    c_r_mean = np.tile(np.mean(out_data[c].values), len(out_data[c].values))

                c_r_sigma = np.sqrt((out_data[c].values-c_r_mean)**2/index_window)
                c_r_zs_val = (out_data[c].values - c_r_mean)/c_r_sigma

                out_data[c+PRM_CALC_SEP+"rM#%s"%str(rw)] = np.nan_to_num(c_r_mean)
                out_data[c+PRM_CALC_SEP+"rS#%s"%str(rw)] = np.nan_to_num(c_r_sigma)
                out_data[c+PRM_CALC_SEP+"rZ#%s"%str(rw)] = np.nan_to_num(c_r_zs_val)
            #print()

    # keep or remove raw data for training....
    if not output_raw:
        out_data = out_data.drop(drop_columns, axis = 1)

    return out_data[sorted(out_data.columns.values)]
    
def preprocess_function(name):
    """
    Returns correct preprocessing function based on input preprocess name.
    
    input: name
    output: preprocess function
    """
    preprocess_funcs = set(["preprocessing1","preprocessing2","preprocessing3"])
    assert name in preprocess_funcs
    if name == "preprocessing1":
        return preprocessing1   
    if name == "preprocessing2":
        return preprocessing2   
    if name == "preprocessing3":
        return preprocessing3   
        
## Currently used preprocessing function...
def preprocessing3(data,
                   subsample = None,
                   output_raw = False,
                   naming_sep_dict = {},
                   transform_type = "log",
                   rolling_windows = [16, -1],
                   rw_ignore_cols = ["Time"],
                   laser_ignore_cols = ["Time","FSC","SSC"],
                   process_types = ["z","s"],
                   kde_useCols = ["all"],
                   kde_events = 1024,
                   include_diff = False,
                   gauss_useCols = "laser_columns",
                   gauss_fitTails = ["rZ#-1","rZ#16"],
                   gauss_bins = 1000,
                   deprecated_version = -1,
                   precision = None,
                   test_plots = False,
                   test_plot_sampleName = None
                   ):
    """
    Preprocessing the input data to be used for the algorithm.
    
    This is done to normalize the sample data across samples and indications, so the data is less prone to variance between individual experimental settings.

    The assumption is that any extra information presented to the algorithm that is not directly related to the population gating for any sample in general
        should not be included as input for training. This may lead to increased bias in the training stage of the algorithm.
    
    Based on the input parameters, a transformation is applied to each of the input data. Then, using rolling windows for subsets of the original data,
        z-scores, standard deviations, and 
    
    input: data,
           subsample = None (if integer, the output data will only be a subsample that takes every subsample integer event), 
           output_raw = False (if True, the raw data will also be included into the output. Typically false, because increases bias),
           transform_type = "log" (the transform to be applied to the laser_columns, everything that doesn't contain any of the strings in laser_ignor_cols below),
           rolling_windows = [16, -1], (The amount of time in seconds to take a rolling window average. If -1, then it computes the averages/medians from the full data input),
           rw_ignore_cols = ["Time"] (list of columns to ignore when doing the rolling window computations),
           laser_ignore_cols = ["Time","FSC","SSC"] (list of columns to ignore for the laser column transformations),
           process_types = ["z","s"] (possible string entries include from processing to include into the output_data:
                "m" - mean for value of the parameter over the rw,
                "u" - median...,
                "s" - sigma, sqrt(std)...,
                "z" - z-score using mean...,
                "zu" - z-score using median...,
                "su" - sigma, sqrt(std) using median...,
                "kde" - kernel density estimation (the average density of points around a given point)...,
                "gauss" - normalization using peak fitting using double gaussian fitting estimation...
                "der" - normalization using peak fitting using abs(grad) minimization and double grad maximization (local maximums)...)
           kde_useCols = ["all"] (kernel density estimation columns to use. if "all", uses all the columns and constructs the density in the dimensionality of that space)...
           kde_events = 1024 (number of events to consider for kde),
           include_diff = False (if True, includes the normalized error for gaussian fitting estimation),
           gauss_useCols = "laser_columns" (columns to use in normalization binary fitting. See peak_fitting func),
           gauss_fitTails = ["rZ#-1","rZ#16"] (the tails to check for in the peak fitting),
           gauss_bins = 1000 (the number of bins to use for the gaussian fitting to construct curve to fit double gauss to),
           deprecated_version = -1 (deprecated; used for compatibility with previous preprocessing3 versions),
           precision = None (if integer, sets output values to this integer of decimal precision),
           test_plots = False (if True, tries to construct plots of the gaussian peak fitting),
           test_plot_sampleName = None (str title of plot to sample name)
    output: dataframe with meta computed values in other columns
    """
    
    assert all([p in set(["m","u","s","z","zu","su","kde","gauss","der"]) for p in process_types])
    assert transform_type in set(["log","power","trivial"])
    
    print("preprocessing3...")
    if transform_type == "log":
        transform = log_transform
    elif transform_type == "power":
        transform = power_transform
    elif transform_type == "trivial" or transform_type == None:
        transform = trivial_transform
    else:
        raise(">>>Transform type not implemented<<<")

    # new dataframe to add column to...
    out_data = data

    # subsampling, so the preprocessing step is quicker...
    if subsample != None and isinstance(subsample, int):
        print(" subsampled by %s..."%(str(subsample)))
        out_data = out_data.iloc[::subsample]
    else:
        print(" default subsampling of 0:None:1...")
        subsample = 1

    # time diff...
    # we are considering the time_diff as part of the original data...
    # should be t independent! however, could depend on dt/d(event) == dt/de, due to changes in flow rates...
    #  however, it ould not be if there are consistent errors in acquisition/etc at certain periods during the acquisition.
    time_diff = np.diff(out_data["Time"].values)
    # duplicating first dt/de value twice. the time_diff array has original size out_data.shape[0] - 1 -> changed out_data.shape[0]
    time_diff = np.append(time_diff[[0]],time_diff)
    # adding the time diff to the out_data matrix
    out_data["Time%sdiff"%naming_sep_dict["PRM_PRM_SEP"]] = time_diff
    # total elapsed time from the first event to the last event
    elapse_time = out_data["Time"].values[-1] - out_data["Time"].values[0]
    av_time_spacing = (elapse_time)/(out_data.index.values[-1] - out_data.index.values[0])
    out_data = out_data.drop("Time", axis = 1)

    # now, out_data includes all the operatable columns that potentially have effect on prediction
    # sort data so it is easier to match columns later...

    # all columns up to this point will be dropped if output_raw bool value is False/0
    # other special arrangements of columns
    # original columns + Time diff column
    first_columns = out_data.columns.values
    
    # columns to perform rolling window analysis on
    rw_columns = [c for c in out_data.columns.values if not any ([b in c for b in rw_ignore_cols])]
    
    # all columns not having to do with scatter or time
    laser_columns  = [c for c in out_data.columns.values if not any([b in c for b in laser_ignore_cols])]
    
    # transforms the laser columns by the transformation chosen in the arguments
    #  (if transform == None, it can be consiere as the trivial transformation out_data -> out_data)
    if transform != None:
        out_data[laser_columns] = transform(out_data[laser_columns])

    # rolling zscore, defined from the second value in rolling_zscore
    if rolling_windows != None:
        print(" rws...")
        for rw in rolling_windows:
            print("  %i"%rw)
            rw_zscore_names = []
            # if rw == -1, the whole acquisition matrix is used as the window
            # otherwise, the value rw as a time float
            if rw == -1:
                index_window = len(out_data.index.values)
                window_conv = np.tile(1./index_window,index_window)
            else:
                # rolling score represents the seconds that should be in the score window....
                # set the window width to that value, based on the maximum/minimum time measured

                # this will be something other than 1 if we have used subsampling...
                index_spacing = float(np.sum(np.diff(out_data.index.values))/(len(out_data.index.values)-1.))
                # delta t/(dt/de * de) -> unitless index window spacing
                index_window = int(rw/(av_time_spacing*index_spacing))

                # index window must be greater than 2 (2 different indices considered in as window)
                try:
                    assert index_window > 2
                except:
                    index_window = 2
                    print(">>>ERROR: INDEX WINDOW IS TOO SMALL. continuing with default index_window = %i...<<<"%index_window)

                window_conv = np.tile(1./index_window,index_window)

            for i, c in enumerate(rw_columns, 0):
                # mean M, median U, sigma S, z-score Z
                rw_mean_name = c+naming_sep_dict["PRM_CALC_SEP"]+"rM#%s"%str(rw)
                rw_median_name = c+naming_sep_dict["PRM_CALC_SEP"]+"rU#%s"%str(rw)
                rw_std_name = c+naming_sep_dict["PRM_CALC_SEP"]+"rS#%s"%str(rw)
                rw_zscore_name = c+naming_sep_dict["PRM_CALC_SEP"]+"rZ#%s"%str(rw)
                rw_stdu_name = c+naming_sep_dict["PRM_CALC_SEP"]+"rSU#%s"%str(rw)
                rw_zscoreu_name = c+naming_sep_dict["PRM_CALC_SEP"]+"rZU#%s"%str(rw)

                if rw != -1:
                    # c_r_mean = np.convolve(window_conv, out_data[c].values, mode = "same")
                    # # convolved values near beginning and end are smaller, since only part of the convolution is applied to get convolutions of the same size
                    # c_conv_shift = np.convolve(window_conv, np.tile(1., len(out_data[c].values)), mode = "same")
                    # c_r_mean = c_r_mean / c_conv_shift
                    c_r_mean = out_data[c].rolling(index_window, min_periods = 1, center = True).mean()
                    c_r_median = out_data[c].rolling(index_window, min_periods = 1, center = True).median()
                else:
                    c_r_mean = np.tile(np.mean(out_data[c].values), len(out_data[c].values))
                    c_r_median = np.median(out_data[c].values)

                # rolling window mean changes, -1 window mean is constant
                # the original values stay the same, but the median and consequently s and z change.

                # deprecated version had the index_window denominator, which is not the reasonable solution for normalization.
                # the index_window should be in the 
                if deprecated_version == 1:
                    c_r_sigma = np.sqrt((out_data[c].values-c_r_mean)**2/index_window)
                    c_r_zs_val = (out_data[c].values - c_r_mean)/c_r_sigma
                    c_r_sigmau = np.sqrt(np.sum((out_data[c].values-c_r_median)**2)/index_window)
                    c_r_zsu = (out_data[c].values - c_r_median)/c_r_sigma
                else:
                    c_r_sigma = np.sqrt(np.sum((out_data[c].values-c_r_mean)**2)/len(out_data[c].values))
                    c_r_zs = (out_data[c].values - c_r_mean)/c_r_sigma
                    c_r_sigmau = np.sqrt(np.sum((out_data[c].values-c_r_median)**2)/len(out_data[c].values))
                    c_r_zsu = (out_data[c].values - c_r_median)/c_r_sigma

                #rw_zscore_names.append(rw_zscore_name)
                if "m" in process_types:
                    out_data[rw_mean_name] = np.nan_to_num(c_r_mean)
                if "u" in process_types:
                    out_data[rw_median_name] = np.nan_to_num(c_r_median)
                if "s" in process_types:
                    out_data[rw_std_name] = np.nan_to_num(c_r_sigma)
                if "su" in process_types:
                    out_data[rw_stdu_name] = np.nan_to_num(c_r_sigmau)
                if "z" in process_types:
                    out_data[rw_zscore_name] = np.nan_to_num(c_r_zs)
                if "zu" in process_types:
                    out_data[rw_zscoreu_name] = np.nan_to_num(c_r_zsu)
                    
        # gauss takes a double gaussian distribution, and fits it to the processed data above.
        #  based off the fitting parameters, "landmarks" are chosen to match each mu1 and mu2 of the fit.
        #  then, all the datapoints are transformed such that mu1 -> 0 and mu2 -> 1, linearly.
        #  this normalizes the "peaks" of the distributions 
        #
        # der takes the derivative, and estimates the values that equal 0 (which are said to be the peaks of the function
        #  takes the top two, and does the same transformation that gauss does
        
        # peak_fitting if "gauss" or "der" in input arguments
        out_data = peak_fitting(out_data,
                                 naming_sep_dict,
                                 process_types,
                                 laser_ignore_cols,
                                 gauss_useCols, 
                                 gauss_fitTails, 
                                 gauss_bins, 
                                 test_plots,
                                 include_diff,
                                 do_inter_processing = False
                                 )
                  
        # only considers the closeness of the laser columns....
        if "kde" in process_types:
            silvermann_bandwidth = lambda sigma, n: ((4*sigma**5.)/(3*n))**(1/5.)

            print(" kdes...")
            kde_points = kde_events
            kde_subsample = min(100, int(out_data.shape[0]/kde_points))
            kde_columns = laser_columns

            # kernel density distribution on subsamples data...
            sigmas = []
            for kde_col in kde_columns:
                sigma = np.std(out_data[[kde_col]].values)
                sigmas.append(sigma)
                if "laser_columns" in kde_useCols:
                    n = len(out_data[[kde_col]].values)
                    h = silvermann_bandwidth(sigma, n)

                    print("  %s : bandwidth %s"%(str(kde_col), str(h)))
                    fit_data = np.sort(out_data.iloc[:kde_points*kde_subsample:kde_subsample][[kde_col]].values)
                    kde = KernelDensity(kernel='linear', bandwidth=h, leaf_size = 40).fit(fit_data)

                    #print("  density estimate...")
                    density_est_data = out_data[[kde_col]].values
                    ind_kde_name = kde_col+naming_sep_dict["PRM_CALC_SEP"]+"kde"
                    kde_est = kde.score_samples(density_est_data)
                    out_data[ind_kde_name] = np.where((1 - np.isfinite(kde_est)), 0, kde_est)

            if "all" in kde_useCols:
                all_sigma = np.sqrt(np.sum([s**2 for s in sigmas]))
                n = int(out_data.shape[0])
                h = silvermann_bandwidth(all_sigma, n)

                print("  all : bandwidth %s"%(str(h)))
                fit_data = out_data.iloc[:1024*kde_subsample:kde_subsample][kde_columns].values
                kde = KernelDensity(kernel='linear', bandwidth=h, leaf_size = 40).fit(fit_data)

                all_kde_name = "ALL"+naming_sep_dict["PRM_CALC_SEP"]+"kde"
                density_est_data = out_data[kde_columns].values
                kde_est = kde.score_samples(density_est_data)
                out_data[all_kde_name] = np.where((1 - np.isfinite(kde_est)), 0, kde_est)
                # kde_scores = kde.score_samples(out_data[rw_zscore_names])
                # print(kde_scores.shape)
                # print(kde_scores)

                #[m, n] = out_data.values.shape
                #print(np.tile(np.sum(out_data.values*out_data.values, axis = 1),m))# + (m+1)*(np.sum(out_data.values*out_data.values)) - 2*out_data.values*(np.tile(np.sum(out_data.values,axis = 1),m)))


    # keep or remove raw data for training....
    if not output_raw:
        out_data = out_data.drop(first_columns, axis = 1)

    # sort columns normally, so easier to align in future...
    
    if isinstance(precision, int):
        print("Truncating to %i decimals..."%precision)
        out_data.iloc[:,:] = np.around(out_data.values,precision)
    return out_data[sorted(out_data.columns.values)]
    
    
def interprocess_function(name):
    """
    Returns correct interprocessing function based on input interprocessing name inputted.
    
    input: name
    output: interprocessing function
    """
    interproc_funcs = set(["peak_fitting", "transform"])
    assert name in interproc_funcs
    if name == "peak_fitting":
        return peak_fitting   
    elif name == "transform":
        return transform       
    
    
def transform(data,
              naming_sep_dict = {},
              tcols = [],
              tsmxs = [],
              tails = [],
              test_plots = False, 
              do_inter_processing = False,
              unit = "degrees",
              precision = None
              ):
    
    def rot_mx(rad, unit):
        if unit == "degree":
            rad = rad * 2*np.pi/360.
        elif unit == "rad":
            pass
        else:
            raise("unit not valid.")
        return np.array([[np.cos(rad), np.sin(-rad)],[np.sin(rad), np.cos(rad)]])
        
    # assert the number of rotation parameter vectors matches the number of different rotations given...
    assert len(tcols) == len(tsmxs)
    
    # assert the dimensionality of each rotation parameter vector matches the dimensionality of all the matrices in tsmxs...
    for tcol, tmxs in zip(tcols, tsmxs):
        for i in np.arange(0, len(tmxs)):
            if isinstance(tmxs[i], float):
                tmxs[i] = rot_mx(tmxs[i], unit)
            tmxs[i] = np.array(tmxs[i])
            assert np.shape(tmxs[i])[1] == len(tcol)
    
    columns = data.columns.values    
    out_data = data.copy()
    
    # we want to rotate each of the pairs of values 
    for i, (tmxs, tcol) in enumerate(zip(tsmxs, tcols),0):
        
        #each set of tcol tails must be the same, and exist in tails...
        for tail in tails:
            params_use = []
            
            # want to get the associated parameters in columns that satisfy the tcol and tails
            for c in columns:
                for p in tcol:
                    if p in c and tail == c.split(naming_sep_dict["PRM_CALC_SEP"])[-1]:
                        params_use.append(c)
            
            # did we find the associated processing name for each of the params in tcol?
            assert len(params_use) == len(tcol)
            
            # creating the param_transform_names to use in the output
            param_transform_names = [p+naming_sep_dict["INTER_PROC_SEP"]+"TR"+str(i) for p in params_use]
            transform_vals = data[params_use].copy().values
            
            # now applying each transformation tmx onto the values
            for tmx in tmxs:
                assert tmx.shape[1] == transform_vals.T.shape[0]
                transform_vals = np.dot(tmx, transform_vals.T)
            
            for ti, tn in enumerate(param_transform_names, 0):
                out_data[tn] = transform_vals[ti,:]
    
        if test_plots:
            import matplotlib.pyplot as plt
            valsold = out_data[params_use].values
            valsnew = out_data[param_transform_names].values
            
            plt.scatter(valsold[:,0], valsold[:,1])
            plt.scatter(valsnew[:,0], valsnew[:,1])
            plt.show()
    
    if isinstance(precision, int):
        print("Truncating to %i decimals..."%precision)
        out_data.iloc[:,:] = np.around(out_data.values,precision)
    
    return out_data
            
      
def peak_fitting(data,
                 naming_sep_dict = {},
                 process_types = ["gauss"],
                 laser_ignore_cols = ["Time","FSC","SSC"],
                 cols = [], 
                 tails = [], 
                 gauss_bins = 1000, 
                 test_plots = False, 
                 include_diff = False,
                 do_inter_processing = False,
                 precision = None,
                 ):
    """
    Shifts selected data columns to new values that is normalize to double gaussian peaking, such that mu's parameters of each gaussian describe normalizing function.
    
    input: data, 
             process_types = ["gauss"], 
             cols = [], 
             tails = [], 
             gauss_bins = 1000, 
             test_plots = False, 
             include_diff = False,
             do_inter_processing = False
                 
    output: out_data (with peak fitted normalized event values) 
    """
    
    out_data = data.copy()
    
    if "gauss" in process_types or "der" in process_types:
        if test_plots:
            import matplotlib.pyplot as plt
            
        print(" gauss/der fit...")
        
        use_cols = None
        found_col = False
        if isinstance(cols, str):
            if "all" in cols:
                print("  all...")
                use_cols = out_data.columns.values
                found_col = True
            elif "laser_cols" in cols:
                print("  laser_cols...")
                use_cols = [c for c in out_data.columns.values if not any([b in c for b in laser_ignore_cols])]
                found_col = True
            else:
                print("  >>>%s cols str conditional is not implemented (use list if specific parameter names)<<<"%cols)
        elif isinstance(cols, list):
            print("  %s..."%str(cols))
            use_cols = [c for c in out_data.columns.values if any([b in c for b in cols])]
            found_col = True
        else:
            print("  >>>%s is not a valid cols input<<<"%str(cols))
            
        if not found_col:
            print("  >>>use_cols not defined, check naming<<<")
            
        for gauss_fitTail in tails:
            print("   %s"%gauss_fitTail)
            use_preprocess_columns = [c for c in use_cols if \
                            naming_sep_dict["PRM_CALC_SEP"] in c and \
                            gauss_fitTail == c.split(naming_sep_dict["PRM_CALC_SEP"])[-1]]

            if test_plots:
                row_cols_plot = int(np.sqrt(len(use_preprocess_columns)))+1
                f, axs = plt.subplots(row_cols_plot,row_cols_plot, figsize = (18,10))
                plt.subplots_adjust(wspace = .05, hspace = 1.2)
            
            # lpc laser processing columns
            for ilpc, lpc in enumerate(use_preprocess_columns,0):
                
                if do_inter_processing:
                    lpc_name = lpc + naming_sep_dict["INTER_PROC_SEP"]
                else:
                    lpc_name = lpc
                    
                n_hist, bin_boundaries = np.histogram(out_data[lpc].values, bins=gauss_bins)
                # len bin boundaries is one more than n_hist in each bin...
                binned_avs = np.convolve([1/2.,1/2.], bin_boundaries, mode="same")[:-1]
                n_hist = n_hist/n_hist.max()
                
                if "gauss" in process_types:
                    
                    p0, p0_bounds, weighted_mean, deviation = double_gauss_p0(binned_avs, n_hist)
                    #print(p0, p0_bounds)
                    
                    # the actually shifted gaussian fit normalization
                    lpc_gauss_name = lpc_name+naming_sep_dict["CALC_CALC_SEP"]+"gauss"
                    # the error of the gaussian curve from the points used for the double gauss fit
                    lpc_gauss_name_diff = lpc_name+naming_sep_dict["CALC_CALC_SEP"]+"gauss"+naming_sep_dict["CALC_CALC_SEP"]+"diff"
                    
                    try:
                        n_hist_max_ind = n_hist.tolist().index(n_hist.max())
                        bool_fitTo = np.array((binned_avs > (binned_avs[n_hist_max_ind] - 2.0*deviation)) * \
                                        (binned_avs < (binned_avs[n_hist_max_ind] + 4.*deviation)), dtype = bool)
                        #bool_fitTo = np.tile(True, np.shape(binned_avs))
                        popt, pcov = curve_fit(double_gauss, binned_avs[bool_fitTo], n_hist[bool_fitTo], p0 = tuple(p0), bounds = tuple(p0_bounds))
                        
                        pred_gauss = double_gauss(binned_avs, *popt)
                        # normalized sum of area between curves
                        diff = np.sum((np.abs(pred_gauss - n_hist)))/np.sum(n_hist)
                        
                        (c1,mu1,sigma1,c2,mu2,sigma2) = popt
                        if mu1 > mu2:
                            mumax, mumin = mu1, mu2
                        else:
                            mumax, mumin = mu2, mu1
                        
                        # if they are found to be too close, there is probably an error...adjustment to 
                        if abs(mumax - mumin) < deviation:
                            mumax = mumin + deviation
                        
                        out_data[lpc_gauss_name] = linear_landmarks(out_data[lpc].values, mumin, mumax)
                        if include_diff:
                            out_data[lpc_gauss_name_diff] = diff
                        
                        if test_plots:
                            ax = axs[int(ilpc/row_cols_plot), int(ilpc%row_cols_plot)]
                            ax.scatter(binned_avs, n_hist, color = "blue", label = "actual", s = 2, alpha = .6)
                            ax.scatter(binned_avs, pred_gauss, color = "red", label = "pred", s = 1, alpha = .6)
                            ax.plot([mumin,mumin],[0,1], color = "orange")
                            ax.plot([mumin + deviation,mumin + deviation],[0,1], color = "green", linewidth = 2, alpha = .7)
                            ax.plot([mumax,mumax],[0,1], color = "purple")
                            ax.set_title("%s\ndiff: %f\npopt: %s"%(lpc,diff, str(["%.3f"%p for p in [mumin, mumax]])))
                            #ax.set_title("%s\ndiff: %f"%(lpc,diff))
                            #ax.legend()
                    # error in fitting the function to the data...
                    except Exception as e:
                        mumin, mumax = weighted_mean, weighted_mean+deviation
                        out_data[lpc_gauss_name] = linear_landmarks(out_data[lpc].values, mumin, mumax)
                        if include_diff:
                            out_data[lpc_gauss_name_diff] = 1.
                        
                # if grad is ~0 and grad_grad < 0, representing local max in a function...
                if "der" in process_types:
                    lpc_gauss_name = lpc_name+naming_sep_dict["CALC_CALC_SEP"]+"der"
                    # av grad and grad_grad computation
                    conv_int = 20
                    conv_array = np.tile(float(1./conv_int),conv_int)
                    n_hist_ave = np.convolve(conv_array, n_hist, mode = "same")/\
                                    np.convolve(conv_array, np.tile(1., len(n_hist)), mode = "same")
                    n_hist_grad = np.gradient(n_hist_ave)
                    n_hist_grad = np.array([n_hist_grad[0]]+n_hist_grad)
                    n_hist_grad = np.convolve(conv_array, n_hist_grad, mode = "same")/\
                                    np.convolve(conv_array, np.tile(1., len(n_hist_grad)), mode = "same")
                    n_hist_grad_grad = np.gradient(n_hist_grad)
                    n_hist_grad_grad = np.array([n_hist_grad_grad[0]]+n_hist_grad_grad)
                    n_hist_grad_grad = np.convolve(conv_array, n_hist_grad_grad, mode = "same")/\
                                    np.convolve(conv_array, np.tile(1., len(n_hist_grad_grad)), mode = "same")
                    
                    #print(n_hist_grad.shape, n_hist_grad_grad.shape)
                    
                    # grad shifts from + to - and grad_grad is significantly below 0 
                    bool_peakConditions1 = (n_hist_grad > 0) * \
                                            (np.append(n_hist_grad[1:],[n_hist_grad[-1]]) <= 0) * \
                                            (n_hist_grad_grad < 0.)
                                            #(n_hist_grad_grad < (-np.std(n_hist_grad_grad)/3.)) \
                                            
                    # grad_grad shifts from - to + and grad is + OR shifts from + to - and grad is -
                    bool_peakConditions21 = (n_hist_grad_grad < 0) * (np.append(n_hist_grad_grad[1:],[n_hist_grad_grad[-1]]) >= 0) *\
                                                (n_hist_grad < -np.std(n_hist_grad))
                    bool_peakConditions22 = (n_hist_grad_grad > 0) * (np.append(n_hist_grad_grad[1:],[n_hist_grad_grad[-1]]) <= 0) *\
                                                (n_hist_grad > np.std(n_hist_grad))
                    bool_peakConditions = np.array((bool_peakConditions1+bool_peakConditions21+bool_peakConditions22), dtype = bool)
                    
                    peak_indices = list(np.where(bool_peakConditions)[0])
                                            
                    #print(n_hist_ave[peak_indices])
                    sorted_peak_indices = sorted(peak_indices, key = lambda x: -n_hist_ave[x])
                    #print([[s,n_hist_ave[s]] for s in sorted_peak_indices])
                    ps = binned_avs[sorted_peak_indices[:20]]
                    _weighted_mean, deviation = get_histMeanSTD(binned_avs, n_hist)
                    
                    mu1, mu2 = ps[0], ps[1]
                    
                    if mu1 > mu2:
                        mumax, mumin = mu1, mu2
                    else:
                        mumax, mumin = mu2, mu1
                    
                    # if they are found to be too close, there is probably an error...adjustment to 
                    if abs(mumax - mumin) < deviation:
                        mumax = mumin + deviation
                    
                    out_data[lpc_gauss_name] = linear_landmarks(out_data[lpc].values, mumin, mumax)
                    
                    if include_diff:
                        out_data[lpc_gauss_name_diff] = diff
                    if test_plots:
                        ax = axs[int(ilpc/row_cols_plot), int(ilpc%row_cols_plot)]
                        ax.scatter(binned_avs, n_hist_ave/np.abs(n_hist_ave).max(), color = "blue", label = "actual", s = 2, alpha = .6)
                        ax.scatter(binned_avs, n_hist_grad/np.abs(n_hist_grad).max(), color = "red", label = "actual", s = 2, alpha = .6)
                        ax.scatter(binned_avs, n_hist_grad_grad/np.abs(n_hist_grad_grad).max(), color = "green", label = "actual", s = 2, alpha = .6)
                        for i, p in enumerate(ps,0):
                            if i < 2:
                                use_color = "purple"
                            else:
                                use_color = "gray"
                            ax.plot([p,p],[0,1.], color = use_color, alpha = 1./(i+1))
                        ax.set_title(lpc)
                        
                if test_plots:
                    plt.suptitle("%s"%(gauss_fitTail))
                    plt.show()
    
    if isinstance(precision, int):
        print("Truncating to %i decimals..."%precision)
        out_data.iloc[:,:] = np.around(out_data.values,precision)
    
    return out_data
            
def inter_processing1(data,
                      naming_sep_dict = {},
                      pop = None, 
                      stain = None, 
                      stain_version_templatePops = None, 
                      pop_interprocessing = None,
                      do_include_final_tail = True):
    """
    Performs processing in between training steps on the input data.
    The interprocessing for each individual population (or populations sharing some sort of naming convention) is defined in definitions.py
    
    This is done dependent on the population and stain....each population has different processing steps.
    
    Ex. 
    (seen in config.yaml file...)
    POP_INTERPROCESSING:
      live*: *INTERPROCESS_TRANSFORM1
    INTERPROCESS_TRANSFORM2: &INTERPROCESS_TRANSFORM1
     - kwargs:
         cols: all
         tails: [rZ#2, rZ#64]
         test_plots: false
         precision: 3
       name: PF1
       process_type: peak_fitting
     - kwargs:
         tcols: [[FSC-A,FSC-W],[FSC-A,FSC-H]]
         tsmxs: [[45.0],[45.0]]
         tails: [rZ#2, rZ#64]
         test_plots: false
         unit: degree
         precision: 3
       name: ROT1
       process_type: transform
    
    This configuration for interprocessing will look for populations that match the key "live*" (meaning all the populations that start with live)
    Then, only for these populations, the input metaparameter dataframe will be modified so peak fitting is applied to preprocessed meta parameters
     where the end of the parameter name ends in "rZ#2" or "rZ#64"
     
    Then, it will pass these new parameters off to the next step, which will apply a 45 degree (+pi/2) rotation too the two parameters pairs (FSC-A FSC-W) and 
      (FSC-A, FSC-H).
      
    Precision means that the float will be truncated to 3 decimal precision.
    
    input: data,
            naming_sep_dict = {},
            pop = None, 
            stain = None, 
            pop_template_loc = POP_TEMPLATE_LOC, 
            pop_interprocessing = POP_INTERPROCESSING,
            do_include_final_tail = True (found in the naming_sep_dict above)
    output: out_data (interprocessed data)
    """
    print("inter_processing1 for pop %s..."%pop)
    
    # all the potential pops that can be interprocessed (those that exist in the template location)...
    potential_pops = stain_version_templatePops[stain]
    
    out_data = data.copy()
    original_columns = set(out_data.columns.values.tolist())
    
    # what populations does the key fit into?
    # checks each key in pop_interprocessing, and sees if the key fits into the pop name...
    #  subkeys are those between astrices. For example, the key ''*this*key*'' will parse into the keys [''this'',''key'']
    #  if there are no leading/trailing astrices, then that first/last key must be at the beginning/end of the name,
    #   for example, ''*this*key'' will fit into ''1thisotherkey'' or ''11thisotherotherkey'', but not ''1thisotherkeytoo''
    # when the key is == "*", this means that the key will fit into each population (as if it was in the preprocessing step, but the computations are done each iteration of inter_processing1
    for pop_interprocess in pop_interprocessing.keys():
        # the pops_satisfying is different for each pop_interprocess, so we have this withing the for loop.
        # the populations satisfying the string requirements...we look through all the interprocessing pops, to make sure which ones satisfy the string requirement
        pops_satisfying = []
        if "*" in pop_interprocess:
            astrix_list = np.where(np.array(list(pop_interprocess)) == "*")[0].tolist()
            if 0 not in astrix_list:
                astrix_list = [-1] + astrix_list
            if len(pop_interprocess)-1 not in astrix_list:
                astrix_list = astrix_list + [len(pop_interprocess)]
           
            # possible populations in stain template
            for potential_pop in potential_pops:
                # start and end of pop around * are in potential_pop? add to the pops_use list
                
                all_satisfy = True
                for i in np.arange(len(astrix_list)-1):
                    # there is no astrix at beginning of pop_interprocess str...
                    if astrix_list[i] == -1:
                        this_satisfy = False
                        if pop_interprocess[0:astrix_list[i+1]] in potential_pop:
                            if potential_pop.index(pop_interprocess[0:astrix_list[i+1]]) == 0:
                                this_satisfy = True
                        all_satisfy = all_satisfy * this_satisfy
                        
                    if astrix_list[i+1] == (len(pop_interprocess)):
                        this_satisfy = False
                        if pop_interprocess[astrix_list[i]:] in potential_pop:
                            # starting position of key should be the length of population string minus lenght of key
                            check_start = len(pop_interprocess)-len(pop_interprocess[astrix_list[i]:])
                            if potential_pop.index(pop_interprocess[astrix_list[i]:]) == check_start:
                                this_satisfy = True
                        all_satisfy = all_satisfy * this_satisfy
                    else:
                        all_satisfy = all_satisfy * (pop_interprocess[astrix_list[i]+1:astrix_list[i+1]] in potential_pop)
                
                if all_satisfy:
                    pops_satisfying.append(potential_pop)  
        else:
            if pop_interprocess in potential_pops: 
                pops_satisfying.append(potential_pop)
        
        # for pop in pops_satisfying the pop_interprocess key in the for loop....
        if pop in pops_satisfying or pop == "*":
            # passed, using interproces_calcs for the pop_interprocess being used...
            print(" inter_process ''%s'' on pop ''%s'' running..."%(pop_interprocess, pop))
            interprocess_calcs = pop_interprocessing[pop_interprocess]
            
            # iterates throug heach step in the interprocess_calcs config...
            for interprocess_calc in interprocess_calcs:
                process_name = interprocess_calc["name"]
                print("  -%s"%process_name)
                proc = interprocess_function(interprocess_calc["process_type"])
                kwargs = interprocess_calc["kwargs"]
                
                old_cols = set([c for c in out_data.columns.values])
                # interprocessing calculations
                out_data = proc(out_data, naming_sep_dict, do_inter_processing = True, **kwargs)
                output_columns = set(out_data.columns.values)
                # new column order, with the process_name tail added to the end
                new_cols = sorted(list(set(output_columns - old_cols)))
                new_cols_nameDict = {n : n+naming_sep_dict["INTER_NAME_SEP"]+process_name for n in new_cols}
                out_data = out_data.rename(new_cols_nameDict, axis = 1)
    
    output_columns = set(out_data.columns.values)
    
    # what are new columns that did not exist in the original output_columns?
    # anything special to do to these new columns ie add a tail
    if original_columns != output_columns:
        if do_include_final_tail:
            new_cols = sorted(list(set(output_columns - original_columns)))
            new_cols_nameDict = {n : n+naming_sep_dict["INTER_FINAL_TAIL"] for n in new_cols}
            out_data = out_data.rename(new_cols_nameDict, axis = 1)
    else:
        print(">>>no column changes done in inter_processing!<<<")
    
    return out_data
    
def data_pred_prep(data, 
                   preprocess_columns = None, 
                   interprocess_columns = None, 
                   default_preprocess_val = 0.,
                   naming_sep_dict = {},
                   pop = None,
                   stain = None, 
                   stain_version_templatePops = None,
                   pop_interprocessing = None,
                   do_interProcessing = True,):
                   
    # consolidated operations   
    assert not isinstance(preprocess_columns, type(None)) and not isinstance(interprocess_columns, type(None))
    print("len(preprocess_columns), len(interprocess_columns): %i, %i"%(len(preprocess_columns), len(interprocess_columns)))
    
    out_data = data.copy()
    try:
        out_data = out_data[preprocess_columns]
    # some of the keys do not exist in out_data? Replace with 0's...
    except KeyError:
        for key in preprocess_columns:
            if key not in out_data.columns:
                out_data[key] = default_preprocess_val
        out_data = out_data[preprocess_columns]   
    
    # the out_data has the same metaparameters as was used before the interprocessing done in the training step, so we do not use a try/except block here
    if do_interProcessing:
        # interprocessing...
        #logging.info("interprocessing...")
        out_data = inter_processing1(out_data,
                                     naming_sep_dict = naming_sep_dict,
                                     pop = pop, 
                                     stain = stain, 
                                     stain_version_templatePops = stain_version_templatePops,
                                     pop_interprocessing = pop_interprocessing)
        out_data = out_data[interprocess_columns]
        
    return out_data
    
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
    
def get_flow_files(target_dir, 
                    name_conditions = [], 
                    path_conditions = [], 
                    in_dict = {}, 
                    file_type = None,
                    cleanup_func = name_cleanup_func, 
                    sort_by = len):
    """
    Root dir to find fcs/wsp names in. All the files need to satisfy the name_conditions and the path_conditions...
    
    input: target_dir, 
            name_conditions = [] (conditions that need to be satisfied for the name of the file), 
            path_conditions = [] (conditions that need to be satisfied for the path of the file),  
            in_dict = {} (use in recursive step, checking each subdirectory), 
            file_type = None,
            cleanup_func = name_cleanup_func, 
            sort_by = len (the "len" function for sorting the target_dir files is default)
    output: in_dict
    """
    
    file_names = list(sorted(os.listdir(target_dir), key = lambda x: sort_by(x)))
    
    temp_dict = {}
    for f in file_names:
        file_path = os.path.join(target_dir, f)
        
        if os.path.isdir(file_path):
            in_dict.update(get_flow_files(file_path, name_conditions, path_conditions, temp_dict, file_type))
        else:
            # if all the conditions are not satisfied, the file_name/file_path should not go into the dictionary...
            if (all([c(str(f)) for c in name_conditions]) and all([c(str(file_path)) for c in path_conditions])):
                f_new = cleanup_func(f, file_type)
                temp_dict[f_new] = file_path
                    
    in_dict.update(temp_dict)
    #print(">>>", in_dict.keys())
    return in_dict
    
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
                    descriptor_num = 3,
                    info_sep = "_",
                    fcs_name_format = None,
                    wsp_name_format = None,
                    fcs_wsp_share = None,
                    fcs_share = None,
                    wsp_share = None,
                    stain_override = "",
                    cleanup_func = name_cleanup_func,
                    do_cleanup = True,
                    do_assert_wsp = True,
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
    wsp_name_format_split = wsp_name_format.replace(".wsp","").replace(".wspt","")
    
    def get_fcs_share(file_name, name_format = fcs_name_format_split, name_share = fcs_wsp_share):
        name_format_split = name_format.split(info_sep)
        file_name_split = file_name.replace(".fcs","").split(info_sep)
        if do_assert_formatExact:
            if len(file_name_split) != len(name_format_split):
                return False
        
        # now returning the string to tie the fcs-wsp together
        fcs_wsp_share_name = []
        for fws in name_share.split(info_sep):
            fcs_wsp_share_name.append(file_name_split[name_format_split.index(fws)])
        return info_sep.join(fcs_wsp_share_name)
        
    def get_wsp_share(file_name, name_format = wsp_name_format_split, name_share = fcs_wsp_share):
        name_format_split = name_format.split(info_sep)
        file_name_split = file_name.replace(".wsp","").split(info_sep)
        if do_assert_formatExact:
            if len(file_name_split) != len(name_format_split):
                return False
        
        # now returning the string to tie the fcs-wsp together
        fcs_wsp_share_name = []
        for fws in name_share.split(info_sep):
            fcs_wsp_share_name.append(file_name_split[name_format_split.index(fws)])
        return info_sep.join(fcs_wsp_share_name)
        
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
                                        do_accept_all_fcsNames = do_accept_all_fcsNames))
            
    file_names = list(sorted(os.listdir(target_dir)))
    
    # do .fcs's first, then other files (potenital directories), then finally .wsp'...
    for f in sorted(file_names, key = lambda x: 0 if ".fcs" in x else 2 if ".wsp" in x else 1):
        file_path = os.path.join(target_dir, f)
        # if directory, run recursively...
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
                    
                    # # the name needs to include all these parts
                    # # if length is longer, it just takes the first few...
                    # if len(f_cleaned.split(info_sep)) >= descriptor_num or ".wsp" in f:
                        # pass
                        # #f_wsp_fcs_shareName = f_cleaned.split(info_sep)[0]
                    # else:
                        # print(">>>skipping %s<<<"%f_cleaned)
                        # continue
                        
                    # does it satisfy the descriptor_num requirement, and has none of the ignore_strs/all of the condition_strs in f_cleaned?
                    if ".fcs" in f and start_str in f_cleaned and \
                                       len(f_cleaned.split(info_sep)) >= descriptor_num and \
                                       not any([s in f for s in ignore_strs]) and \
                                       all([s in f for s in condition_strs]):
                        #fcs_core_name = f_cleaned.split(info_sep)[0] + info_sep + f_cleaned.split(info_sep)[1].lower()
                        f_cleaned = f_cleaned.replace(".fcs","")
                        f_wsp_fcs_share_name = get_fcs_share(f_cleaned, name_format = fcs_name_format, name_share = fcs_share)
                        # print(f_wsp_fcs_share_name)
                        
                        if not f_wsp_fcs_share_name:
                            if do_detail:
                                print(">>>%s not valid .fcs! continuing...<<<"%f_cleaned)
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
                            
                        if f_cleaned not in stsd[f_wsp_fcs_share_name]["fcs"].keys():
                            stsd[f_wsp_fcs_share_name]["fcs"][f_cleaned] = {}
                            
                        if stain_alias not in stsd[f_wsp_fcs_share_name]["fcs"][f_cleaned].keys():
                            stsd[f_wsp_fcs_share_name]["fcs"][f_cleaned][stain_alias] = file_path
                    
                    # assuming the longer name is the one that we want to use, because it has more descriptors?...
                    #  not sure what would be a good way to do this.
                    elif ".wsp" in f and start_str in f_cleaned:
                        f_cleaned = f_cleaned.replace(".wsp","")
                        wsp_share_name = get_wsp_share(f_cleaned, name_format = wsp_name_format, name_share = wsp_share)
                        
                        if not wsp_share_name:
                            if do_detail:
                                print(">>>%s not valid .wsp! continuing...<<<"%f_cleaned)
                            continue
                        
                        # what FCS_SHARE's does this wsp correspond to? 
                        for key in stsd.keys():
                            fcs_share_name = get_fcs_share(key, name_format = fcs_share, name_share = fcs_wsp_share)
                            wsp_share_name = get_wsp_share(f_cleaned, name_format = wsp_name_format, name_share = fcs_wsp_share)
                            if fcs_share_name == wsp_share_name:
                                f_wsp_fcs_shareName = get_fcs_share(key, name_format = fcs_share, name_share = fcs_share)
                                stsd[f_wsp_fcs_shareName]["wsp"] = file_path
                            #if wsp_share_name not in stsd.keys():
                            #    stsd[f_wsp_fcs_shareName] = {}
                            #if "wsp" not in stsd[f_wsp_fcs_shareName].keys():
                            #    stsd[f_wsp_fcs_shareName]["wsp"] = file_path
                            #else:
                            #    if len(file_path) > len(stsd[f_wsp_fcs_shareName]["wsp"]):
                            #        stsd[f_wsp_fcs_shareName]["wsp"] = file_path
                        
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
            if "fcs" not in stsd[samp]:
                if do_detail:
                    print(">>>''fcs'' not in stsd[%s]\ntarget_dir: %s\nDeleting samp key...<<<"%(samp, target_dir))
                del_keys.add(samp)
                continue
        for samp in del_keys:
            del stsd[samp]
        
    return stsd

def print_valid_samples(get_samples_func = get_sample_dict, 
                        dir_folder = None, 
                        stain_dict = None, 
                        **kwargs):
    """
    Print valid available samples in the dir_folder nesting.
    
    input: get_samples_func = get_sample_dict, 
                        dir_folder = None, 
                        stain_dict = None, 
                        **kwargs
    output: sorted(sample_list) (also prints sorted and shuffled sample, with length)
    """
    from random import shuffle
    
    if not dir_folder:
        dir_folder = TRAIN_FROM_DIR
        print(">>>using TRAIN_FROM_DIR<<<")
    
    print(dir_folder)
    
    sample_pathsDict = get_samples_func(target_dir = dir_folder, 
                                        stain_dict = stain_dict, 
                                        **kwargs)
    
    sample_list = list(sample_pathsDict.keys())
    print("sorted:")
    pprint.pprint(sorted(sample_list))
    print()
    print("shuffled:")
    shuffle(sample_list)
    pprint.pprint(list(sample_list))
    print("len: %i"%len(sample_list))
    return sorted(sample_list)

def print_sample_fcs_num(get_samples_func = get_sample_dict, 
                        dir_folder = None, 
                        stain_dict = None,
                        **kwargs):
    """
    Print valid available samples, and the number of fcs within.
    
    input: get_samples_func = get_sample_dict, 
                        dir_folder = None, 
                        stain_dict = None, 
                        **kwargs
    output: (None; prints sorted and number of fcs within each)
    """                 
    sample_pathsDict = get_samples_func(dir_folder, stain_dict)
    
    for samp in sorted(sample_pathsDict.keys()):
        print(samp, sum([len(sample_pathsDict[samp]["fcs"][b]) for b in sample_pathsDict[samp]["fcs"]]))

from ipi_tools_comp import *
#
#
#
           
#############
###TESTING###
#############

def test_preprocessing():
    import matplotlib.pyplot as plt
    import random 
    import fcsparser
    import pickle
    import config_loader
    config = config_loader.config_loader()
    
    # choose random fcs in gated_good...
    gate_dir_path = r"C:\Users\llupinjimenez\Documents\UCSF_datascience\FloJo_stuff\samples_wsp_fcs\gated_good"
    fcs_default = r"C:\Users\llupinjimenez\Documents\UCSF_datascience\FloJo_stuff\samples_wsp_fcs\gated_good\IPIGYN081\IPIGYN081_T1_treg.fcs"
    found = False
    fcs_file = None
    if 0:
        while not found:
            sample_dirs = os.listdir(gate_dir_path)
            random.shuffle(sample_dirs)
            #print(sample_dirs)
            try:
                sample_dir = os.path.join(gate_dir_path, sample_dirs[0])
                fs = os.listdir(sample_dir)
                random.shuffle(fs)
                for f in fs:
                    if any([b in f and ".fcs" in f and "treg" in f for b in STAINS_V11_4_CELLS]):
                        fcs_file = os.path.join(sample_dir, f)
                        found = True
            except:
                print("Trying another sample...")
                continue
        print(fcs_file)
    else:
        fcs_file = fcs_default
    _meta, fcs_data = fcsparser.parse(fcs_file)
    
    preprocessing3_kwargs = {
                   "subsample" : None,
                   "output_raw" : False,
                   # can be none...
                   "transform_type" : "log",
                   # -1 rolling window represents the whole sample space...
                   #rolling_windows = [4, 8, 16, -1], -> used in first cluster processing
                   "rolling_windows" : [2], 
                   #process_types = ["z","s","kde"], -> used in first cluster processing
                   "process_types" : ["z"],
                   #"process_types" : ["z","s","zu","su", "kde"],
                   #"process_types" : ["z","s","zu","su"],
                   "gauss_bins" : 1000,
                   "include_diff" : True,
                   "gauss_useCols" : "laser_cols",
                   "gauss_fitTails" : ["rZ#2",],
                   "kde_useCols" : ["all", "laser_columns"],
                   "deprecated_version" : -1,
                   }
                   
                   
    #preprocessing3_kwargs = PREPROCESSING3_KWARGS
    #print(preprocessing3_kwargs)
                   
    stime = time.time()
    a = preprocessing3(fcs_data, 
                       test_plots = False, 
                       naming_sep_dict = config.definitions.__dict__, 
                       test_plot_sampleName = fcs_file, 
                       **preprocessing3_kwargs)
    
    pprint.pprint(a.columns.values)
    #print(a[[c for c in a.columns if "time" in c.lower()]])
    
    print(len(a.columns.values))
    test_pop_name = "singlecells2@singlecells@time@Stain 1_ROOT"
    #test_pop_name = "singlecells2@singlecells@time@Stain 1_ROOT"
    #test_pop_name = "DOESN'TEXIST@notgrans@cd45+@live@singlecells2@singlecells@time@Stain 1_ROOT"
    #test_pop_name = "testthismfmeormeout"
    
    a = inter_processing1(a, 
                          naming_sep_dict = config.definitions.__dict__, 
                          pop = test_pop_name, 
                          stain = "Stain 1",
                          pop_interprocessing = config.process_args.POP_INTERPROCESSING,
                          stain_version_templatePops = pickle.load(open(config.paths.POP_TEMPLATE_LOC, "rb")))
                          
    print(len(a.columns.values))
    return(a)
    #a = preprocessing3(fcs_data.iloc[::],)
    lookup_str = "Red A-A"
    a_cols_use = [b for b in a.columns.values if lookup_str in b]
    rc = int(np.sqrt(len(a_cols_use)))+1
    
    f, axs = plt.subplots(rc, rc, figsize=(10,10))
    f.subplots_adjust(hspace=.5,wspace=.8)
    for i, col in enumerate(a_cols_use,0):
        ax_tup = (int(i/rc),i%rc)
        axs[ax_tup].hist(a[col].values, bins = 1000)
        axs[ax_tup].set_title(col.split(PRM_CALC_SEP)[1:])
    plt.suptitle("..."+"/"+os.path.basename(fcs_file)+"\n"+"...with ''%s''"%lookup_str)
    plt.show()
    
    print(stime - time.time())
    
    print(a[[c for c in a.columns if "A-A" in c and "rZ" in c]])

def test_timeGate():
    #fcs_file = r"C:\Users\llupinjimenez\Documents\UCSF_datascience\FloJo_stuff\samples_wsp_fcs\IPIGYN097_T1_treg_001.fcs"
    sample_dir = r"C:\Users\llupinjimenez\Documents\UCSF_datascience\FloJo_stuff\samples_wsp_fcs\tech_gated"
    fcs_paths = []
    for sample in os.listdir(sample_dir):
        fcs_dir = os.path.join(sample_dir, sample)
        for fcs in os.listdir(fcs_dir):
            if ".fcs" in fcs:
                fcs_paths.append(os.path.join(fcs_dir, fcs))
    for fcs_file in fcs_paths:
    #fcs_file = r"C:\Users\llupinjimenez\Documents\UCSF_datascience\FloJo_stuff\samples_wsp_fcs\tech_gated\IPICRC062\IPICRC062_T1_treg.fcs"
        params = ["Time", "FSC-A"]
        print(fcs_file)
        fcs_stuff = FCMeasurement(ID='test', datafile=fcs_file)
        meta, data = fcs_stuff.meta, fcs_stuff.data
        print(meta)
        print(data)
        print(data.columns)

        every_other = 1
        until = None
        data = data.iloc[:until:every_other,:]

        if 0:
            in_gate, indices, boundaries, avg_rate, avg_rate_long, avg_fsca, avg_fsca_long = timeGate(data, period = 1, every_other = every_other, params = params, give_other = 1)

            print(avg_rate)

            fig, ax1 = plt.subplots(figsize = (18,9))
            ax2 = ax1.twinx()

            ax1.scatter(data[[params[0]]].values, data[[params[0]]].values, s = .2, color = "gray", alpha = .2)
            ax1.scatter(data[[params[0]]].iloc[in_gate == 1], data[[params[1]]].iloc[in_gate == 1], s = .2, color = "blue", alpha = .8)
            ax1.scatter(data[[params[0]]].iloc[in_gate != 1], data[[params[1]]].iloc[in_gate != 1], s = .2, color = "red", alpha = .4)
            ax1.plot([avg_fsca[0,0], avg_fsca[-1,0]], [boundaries["lower_fsca"], boundaries["lower_fsca"]], linestyle = ":", color = "green")
            ax1.plot([avg_fsca[0,0], avg_fsca[-1,0]], [boundaries["upper_fsca"], boundaries["upper_fsca"]], linestyle = ":", color = "green")

            ax1.plot([avg_fsca[0,0], avg_fsca[-1,0]], [boundaries["lower_avg_fsca"], boundaries["lower_avg_fsca"]], linestyle = "-", color = "red")
            ax1.plot([avg_fsca[0,0], avg_fsca[-1,0]], [boundaries["upper_avg_fsca"], boundaries["upper_avg_fsca"]], linestyle = "-", color = "red")
            ax1.plot(avg_fsca[:,0], avg_fsca[:,1], color = "red", alpha = 1.0)
            #ax1.plot(avg_fsca_long.iloc[:,0], avg_fsca_long.iloc[:,1], color = "purple", alpha = 1.0)

            ax2.plot([avg_rate[0,0], avg_rate[-1,0]], [boundaries["lower_avg_rate"], boundaries["lower_avg_rate"]], linestyle = "-", color = "orange")
            ax2.plot([avg_rate[0,0], avg_rate[-1,0]], [boundaries["upper_avg_rate"], boundaries["upper_avg_rate"]], linestyle = "-", color = "orange")
            ax2.plot(avg_rate[:,0], avg_rate[:,1], color = "orange", alpha = 1.0)
            #ax2.plot(avg_rate_long.iloc[:,0], avg_rate_long.iloc[:,1], color = "yellow", alpha = 1.0)


            ax1.set_ylabel("FSC-A fluorescence", color = "red")
            ax2.set_ylabel("Rate of flow", color = "orange")
            fig.tight_layout()
            plt.show()

        else:
            in_gate, indices, fscaConvolve, lower_avg_fsca, upper_avg_fsca, timeConvolve, lower_avg_time, upper_avg_time = timeGate2(data, give_other = 1)
            fig, ax1 = plt.subplots(figsize = (18,9))
            ax2 = ax1.twinx()

            sum_in_gate = np.sum(in_gate)

            ax1.scatter(data[[params[0]]].iloc[in_gate == 1], data[[params[1]]].iloc[in_gate == 1], s = .2, color = "blue", alpha = .8)
            ax1.scatter(data[[params[0]]].iloc[in_gate != 1], data[[params[1]]].iloc[in_gate != 1], s = .2, color = "red", alpha = .4)
            #ax1.plot([data[params[0]].min(), data[params[0]].max()], [lower_avg_fsca, lower_avg_fsca], color = "orange")
            #ax1.plot([data[params[0]].min(), data[params[0]].max()], [upper_avg_fsca, upper_avg_fsca], color = "orange")
            ax1.plot(data[[params[0]]], fscaConvolve, color = "green")
            ax1.plot(data[[params[0]]], lower_avg_fsca, color = "orange")
            ax1.plot(data[[params[0]]], upper_avg_fsca, color = "orange")

            ax2.plot(data[[params[0]]], timeConvolve, color = "purple")
            ax2.plot(data[[params[0]]], lower_avg_time, color = "pink")
            ax2.plot(data[[params[0]]], upper_avg_time, color = "pink")

            ax1.set_ylabel("FSC-A fluorescence", color = "green")
            ax2.set_ylabel("Rate of flow", color = "purple")
            plt.title("%s out of %s"%(sum_in_gate, len(in_gate)))
            fig.tight_layout()
            plt.show()

if __name__ == "__main__":
    #test_timeGate()
    test_preprocessing()
    #a = get_flow_files(PRED_DIR)
    #pprint.pprint(a)