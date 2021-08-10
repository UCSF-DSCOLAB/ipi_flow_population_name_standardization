from __future__ import print_function

import numpy as np
import os
import pprint
import xml.etree.ElementTree as ET
import pickle
import copy
import time
import matplotlib.path as mpltPath
import pandas as pd
from config_loader import config_loader

try:
    from importlib import reload
except:
    pass

#wsp_getComp is IPI specific...
from tools_comp import wsp_getComp, fsc_change, get_fcs, fcs_getComp, TRANSFORM_FUNCTIONS, timeGate_simple, get_facs_param_name, get_cytof_param_name
from pop_consistency_nameChanges import get_pop_alias

class pop_gates:
    """
    Class for constructing nested dictionary for gate populations, and for running fcs file through gates constructed previously.

    (__init__) -
    input: self, wsp_file = None, fcs_file = None, root = None, name = None
    """
    def __init__(self,
                 config = None,
                 wsp_file = None,
                 fcs_file = None,
                 root = None,
                 name = None,
                 tissue_type = None,
                 group_node_check = "str_parse",
                 print_detail = 0,
                 root_str = "ROOT"):
        """
        Initializes with wsp_file and fcs_file paths, and optional name, and nulled variables
        """
        if wsp_file == None:
            print(">>>wsp_file not defined for this object!<<<")

        self.wsp_file = wsp_file
        self.fcs_file = fcs_file

        if isinstance(config, str):
            config = config_loader(config)

        self.config = config
        self.name = name
        self.gates = None # information parsed from wsp about the population gates and names
        self.families = None # just alias information, useful for getting gating hierarchy easily
        self.uniquePops = None # all the populations, without structure --
        self.populations = None # the population AFTER the python gating from wsp gates
        self.comps = None
        self.fsc_wh_switch = None
        self.tissue_type = tissue_type
        self.root_str = root_str

        if group_node_check == "str_parse":
            self.group_node_check = lambda check_str, use_type: use_type.lower() in check_str.lower()
        elif isinstance(group_node_check, type(None)):
            self.group_node_check = lambda check_str, use_type: True
        elif group_node_check == "IPI":
            self.group_node_check = lambda check_str, use_type: (use_type.lower() in use_type.lower() and \
                    "malformed" not in check_str.lower() and \
                    "_atto" not in check_str.lower())
        elif group_node_check == "true" or group_node_check == "ignore":
            self.group_node_check = lambda check_str, use_type: True
        else:
            self.group_node_check = group_node_check

        self.print_detail = print_detail

        def print_func(print_str, this_detail, **kwargs):
            if self.print_detail >= this_detail:
                print(print_str, **kwargs)

        self.print_func = print_func
    # pop alias
    # ex. Q3: CD8a+ , CD4- ==> Q3-- CD8a+ , CD4- (+ >> parents
    def set_pop_alias(self, pop_alias, sep = "@", do_lowercase = True, do_nospaces = True):
        """
        Sets the population name alias to use for dictionary keys, population name, etc.
        Each population gate in the pop_alias is split into each individual gate pop by str sep, and the name_changes are done for each pop then rejoined.

        input: self, pop_alias, sep = "@", make_alias = True
        output: name_change(pop_alias)
        """
        return get_pop_alias(pop_alias)


    def get_pop_alias(self, pop_alias, stain = None):
        """
        Returns alias if exists in uniquePops, otherwise raises error

        input: pop_alias
        output: pop_alias (if exists, otherwise raises error)
        """
        assert stain is not None

        if pop_alias in self.uniquePops[stain].keys():
            return pop_alias
        else:
            raise("pop_alias name doesn't exist in self.unique_pops")

    def parse_gates(self,
                    do_use_groupNode = False, # GroupNode is what is imported from the template, SampleNode is what is gated on the specific fcs's
                    do_check_groupNode = True, # whether if no pops in either SampleNode/GroupNode (depending on what is checked first) are found, to check group node
                    do_check = True,
                    do_first_stain = False):
        """
        Sets sample id, comp matrices, for each stain in the sample used in FloJo.
        Also gets families, which is already included in gates (but easier to look through individually and see heirarchy)

        input: (None)
        output: (None)
        """

        self.print_func("parse_gates from wsp file...", 1)
        tree = ET.parse(self.wsp_file)
        # root node
        root = tree.getroot()

        # jumps straight to the group node elements that have an owning group
        # owning group can be stain name

        # if ".wspt" in self.wsp_file:
        if do_use_groupNode:
            self.print_func(" parsing for .wspt with GroupNode...", 2)
            group_nodes = [r for r in root.findall("./Groups/GroupNode") if "stain" in r.attrib["name"].lower()]
        else:
            self.print_func(" parsing for .wspt with SampleNode..", 2)
            group_nodes = [r for r in root.findall("./SampleList/Sample/SampleNode") if "name" in r.attrib.keys() and "owningGroup" in r.attrib.keys()]

        stains = {}
        families = {}
        self.uniquePops = {}
        self.populations = {}
        cell_types_added = []

        for group_node in group_nodes:
            # checking each of the specific cell types
            # this is working with the SampleNode subtrees, because it seems to be right...?
            # checks each cell type in stain_synonymns, and breaks when type is found
            #print(group_node.attrib["name"])

            stain_found = False

            # is the root node the right for the tissue time in question?
            if self.tissue_type == None:
                right_tissue = True
            else:
                # the tissue name check is the SAMPLE_SEP+tissue_type.lower(), set at initiation of the class
                right_tissue = ((self.config.definitions.SAMPLE_SEP+self.tissue_type.lower()) in group_node.attrib["name"].lower())

            # (str(cell_type).lower() in group_node.attrib["name"].lower() and \
                    # "malformed" not in group_node.attrib["name"].lower() and \
                    # "_atto" not in group_node.attrib["name"].lower())

            # finding the correct gating for the
            indiv_index = self.config.definitions.FCS_SHARE.split(self.config.definitions.INFO_SEP).index("INDIV")
            if self.config.experiment.ASSAY == "cytof":
                indiv = self.name.split(self.config.definitions.INFO_SEP)[indiv_index]
                indiv_num = indiv[2:]
                ## TO FIX FOR GENERAL CASE, OR REMOVE WHEN NAMING IN WSP IS CORRECT!
                if "pop" in group_node.attrib["name"].lower():
                    pop_index = group_node.attrib["name"].lower().index("pop")
                    pop_num = group_node.attrib["name"].lower()[pop_index+3:pop_index+5]
                    if pop_num.isdigit() and indiv_num.isdigit():
                        if int(pop_num) == int(indiv_num):

                            print("chosen equivalence: %s, %s"%(group_node.attrib["name"], self.name))
                            pass
                        else:
                           continue
                    else:
                        continue
                else:
                    continue

            for cell_type in self.config.experiment.POP_PARSE_STAINS:
                #print(cell_type)
                if (self.group_node_check(group_node.attrib["name"], str(cell_type)) and right_tissue) or not do_check:
                    self.print_func(" ''%s'' found as stain type for ''%s''"%(cell_type, group_node.attrib["name"]), 2)
                    stain_found = True
                    cell_types_added.append(cell_type)
                    # standardize naming
                    stain_name = self.config.experiment.STAIN_STANDARD_STAIN[cell_type]

                    # creating dictionary, which will contain nested dictionaries
                    # and gate data
                    stains[stain_name] = {}
                    families[stain_name] = {}
                    self.uniquePops[stain_name] = {}

                    for subpop in group_node:
                        if "Subpopulations" == subpop.tag:
                            # first subpopulation in the recursion parser is the "root" cell stain name (ex. Stain 1_ROOT).
                            # oldest member of the family heirarchy

                            # now all the ancestry information is in the name....
                            #print("     "+node.tag, stain_name)

                            self.get_wspGateInfo(stain_name,
                                                 subpop,
                                                 stains[stain_name],
                                                 flojo_alias = stain_name,
                                                 lineagei = 1,
                                                 fam = families[stain_name])

                    # once finds a stain synonymn name that works, then breaks from loop since the stain is already parsed.
                    break

            # if you only want to look at one of the group_nodes, say if all the nodes are duplicated heirarchy...
            if do_first_stain:
                break

        self.gates = stains
        self.families = families

        # probably had to use the other node type...
        if self.uniquePops == {} and do_check_groupNode:
            print(" >>>parseGates gave empty dict. Trying with do_use_groupNode = %s...<<<"%str(not do_use_groupNode))
            self.parse_gates(do_use_groupNode = not do_use_groupNode, do_check_groupNode = not do_check_groupNode)

    def parse_gates_anyWsp(self, do_ignore_gate = True):
        """
        Sets sample id, comp matrices, for each stain in the sample used in FloJo.
        Also gets families, which is already included in gates (but easier to look through individually and see heirarchy)

        This differs because it ignores the naming of the stains, so all the group nodes are used...

        input: (None)
        output: (None)
        """
        tree = ET.parse(self.wsp_file)
        # root node
        root = tree.getroot()

        group_nodes = [r for r in root.findall("./SampleList/Sample/SampleNode")]
        #group_nodes2 = [r for r in root.findall("./Groups/GroupNode")]
        #group_nodes = group_nodes1 + group_nodes2

        stains = {}
        families = {}
        self.uniquePops = {}
        self.populations = {}
        cell_types_added = []

        for group_node in group_nodes:

            group_node_name = group_node.attrib["name"].lower()
            if True:

                # naming may be different for each wsp..
                stain_name = group_node_name
                # creating dictionary, which will contain nested dictionaries
                # and gate data
                stains[stain_name] = {}
                families[stain_name] = {}
                self.uniquePops[stain_name] = {}

                for node in group_node:

                    if "Subpopulations" == node.tag:
                        # first subpopulation in the recursion parser is the "root" cell stain name (ex. Stain 1_ROOT).
                        # oldest member of the family heirarchy

                        # now all the ancestry information is in the name....
                        #print("     "+node.tag, stain_name)

                        self.get_wspGateInfo(stain_name,
                                             node,
                                             stains[stain_name],
                                             flojo_alias = stain_name+"_ROOT",
                                             lineagei = 1,
                                             fam = families[stain_name],
                                             do_ignore_gate = do_ignore_gate)

        self.gates = stains
        self.families = families

    def get_wspGateInfo(self,
                        stain,
                        subpop,
                        gate_dict,
                        flojo_alias,
                        lineagei = 1,
                        do_ignore_gate = False,
                        fam = None):
        """
        Populate dictionary with populations, subpopulations, meta data
        parent data, etc.

        input: self,
               stain,
               subpop,
               gate_dict,
               parent,
               flojo_alias,
               lineagei = 1,
               fam = None
        output: (None)
        """
        # base case is population has no subpops, doesn't recurse
        def get_afterBracket(temp_str):
            try:
                bracket_ind = None
                bracket_ind = temp_str.index("}")
                if isinstance(bracket_ind,int):
                    return temp_str[bracket_ind+1:]
                else:
                    return temp_str
            except ValueError:
                return temp_str

        # returns the short names after brackets and the associated value?
        def short_name(in_list):
            in_list = list(in_list)
            for l in range(len(in_list)):
                in_list[l] = (get_afterBracket(in_list[l][0]), in_list[l][1])
            return in_list

        for pop in [p for p in subpop if p.tag == 'Population']:

            # new flojo_alias to use for renaming...
            new_flojo_alias = pop.attrib["name"] + self.config.definitions.LINEAGE_SEP + flojo_alias

            #print(new_flojo_alias, "depth:", lineagei)
            pop_alias = self.set_pop_alias(new_flojo_alias)

            #print("%s\n %s\n  %s ->\n    %s"%(pop.attrib["name"], parent, new_flojo_alias, pop_alias))

            # using the alias name
            fam[pop_alias] = {}

            # print gating tree, with appropriate tab number of spaces for heirarchical level
            self.print_func(" ".join([" "*lineagei, str(lineagei), pop.attrib["name"], pop.attrib["count"], ";", "alias:", pop_alias.split(self.config.definitions.LINEAGE_SEP)[0], "parent:", pop_alias.split(self.config.definitions.LINEAGE_SEP)[1]]),
                            2)

            gate_dict[pop_alias] = {"sub" : {}}
            gate_dict[pop_alias]["data"] = {"flojo_alias" : new_flojo_alias,
                                            "attrib" : pop.attrib,
                                            "gate" : {},
                                            "flojo_count" : int(pop.attrib["count"])}

            if not do_ignore_gate:
                # only one gate per pop
                gates = pop.findall("Gate")
                gate = gates[0]

                if gate:
                    # Gate Type (Rectangle, Polygon?
                    gate_type = get_afterBracket(gate[0].tag)
                    # gate other info, list form for the rest
                    gate_meta = gate[0].items()
                    #gate dimensions
                    gate_dict[pop_alias]["data"]["gate"]["gate_type"] = gate_type
                    gate_dict[pop_alias]["data"]["gate"]["gate_meta"] = gate_meta

                    if "RectangleGate" == gate_type or "CurlyQuad" == gate_type:
                        '''
                        example mx:
                        [[[('name', 'Comp-Violet D-A')],
                          [('min', '282.592191788883'), ('max', '262856.6549833364')]],
                         [[('name', 'Comp-YG A-A')],
                          [('min', '-469.10738629322145'), ('max', '299011.9436905912')]]]
                        '''
                        gate_info = []
                        for gate_dims in gate[0]:
                            gate_dimensions= gate_dims.items()
                            gate_dimensions = short_name(gate_dimensions)
                            #gate name?
                            gate_name = gate_dims[0].items()
                            gate_name = short_name(gate_name)
                            gate_info.append([gate_name, gate_dimensions])

                        # sometimes only a single parameter is used (just a choice on one dimension?)
                        # we can catch this and try to fix it
                        if np.shape(gate_info) in set([(2,2),(2,2,1,2)]):
                            x_var = gate_info[0][0][0][1]
                            y_var = gate_info[1][0][0][1]
                        # doesn't follow standard form...
                        elif np.shape(gate_info) == (1,2):
                            # we will use the same duplicate dimension, but set the bounds the same...

                            x_var = gate_info[0][0][0][1]
                            x_min = float(gate_info[0][1][0][1])
                            x_max = float(gate_info[0][1][1][1])

                            # Any parameter should be ok, we use time because it is the most general, and shold exist for most cytometry runs...
                            y_var = "Time"
                            y_min = -np.inf
                            y_max = np.inf
                            gate_info = [[[('name', x_var)],
                                          [('min', x_min), ('max', y_min)]],
                                         [[('name', y_var)],
                                          [('min', y_min), ('max', y_max)]]]
                        else:
                            raise(">>>Not allowed gate_info dimension<<<")

                    elif "PolygonGate" == gate_type:
                        gate_info = []
                        for gate_dims in gate[0]:
                            for gate_dim in gate_dims:
                                gate_info.append(short_name(gate_dim.items())[0])
                        x_var = gate_info[0][1]
                        y_var = gate_info[1][1]

                    # elif "CurlyQuad" == gate_type:
                        # gate_info = []
                        # for gate_dims in gate[0]:
                            # for gate_dim in gate_dims:
                                # gate_info.append(short_name(gate_dim.items())[0])
                        # x_var = gate_info[0][1]
                        # y_var = gate_info[1][1]

                    # None of the relevant gates exist...
                    else:
                        self.print_func(" >>>gate_type not %s implemented. continuing...<<<"%gate_type, 2)
                        xvar, yvar = None, None

                    gate_dict[pop_alias]["data"]["gate"]["gate_dimensions"] = gate_info

                if "PolygonGate" == gate_type:
                    points = self.poly_gate(gate_info, pop_name = pop_alias)
                elif "RectangleGate" == gate_type:
                    points = self.rectangle_gate(gate_info, pop_name = pop_alias)
                elif "CurlyQuad" == gate_type:
                    points = self.rectangle_gate(gate_info, pop_name = pop_alias)
                else:
                    points = None

                self.uniquePops[stain][pop_alias] = {"flojo_alias" : new_flojo_alias,
                                                       "gate_type" : gate_type,
                                                       "flojo_count" : int(pop.attrib["count"]),
                                                       "params_to_gate" : [x_var,y_var],
                                                       "polygon" : points}
            else:
                self.uniquePops[stain][pop_alias] = {"flojo_alias" : new_flojo_alias,
                                                       "flojo_count" : int(pop.attrib["count"])}

            subpopulations = [s for s in pop if s.tag == 'Subpopulations']

            new_parent = pop_alias
            new_flojo_parent = new_flojo_alias

            # if no subpopulations under current population
            if subpopulations:
                for spop in subpopulations:
                    self.get_wspGateInfo(stain,
                                         spop,
                                         gate_dict[pop_alias]["sub"],
                                         flojo_alias = new_flojo_parent,
                                         lineagei = lineagei+1,
                                         fam = fam[pop_alias],
                                         do_ignore_gate = do_ignore_gate)
            else:
                pass

    def run_gating(self,
                   df_data = None,
                   other_data = None,
                   stain_type = None,
                   flojo_count = None,
                   use_transform = False,
                   standard_channels = True,
                   do_get_mask = False):
        """
        Stores dictionary of population names, with associated information, index locations, and mask if

        input: self,
               df_data = None,
               other_data = None,
               stain_type = None,
               flojo_count = None,
               use_transform = True,
               do_get_mask = False
        output: (none)
        """

        assert stain_type
        self.print_func("run_gating on stain_type %s..."%stain_type, 1)

        if isinstance(df_data, type(None)):
            self.print_func(" >>>df_data not provided. Loading %s"%self.fcs_file, 2)
            _meta, df_data = get_fcs(self.fcs_file)
            flowjo_root_count = len(df_data.index)
        else:
            flowjo_root_count = len(get_fcs(self.fcs_file)[1].index)

        self.populations[stain_type] = {}
        stain_names = self.gates.keys()
        cell_stain_name = self.config.experiment.STAIN_STANDARD_STAIN[stain_type]
        self.populations[stain_type][cell_stain_name+self.config.definitions.SAMPLE_SEP+self.root_str] = {"py_count" : len(df_data.index.values),
                                                                 "flojo_count" : flowjo_root_count,
                                                                 "indices" : df_data.index.values,
                                                                 "parent_indices" : df_data.index.values,
                                                                 "parents" : [],
                                                                 "polygon" : None}

        stain_name = self.config.experiment.STAIN_STANDARD_STAIN[stain_type]

        # if multiple entry pops for whatever reason
        # get_popCounts_index saves only the index, while get_popCounts_mask saves the index and the mask array (therefore takes more memory...)
        #  default is to get the index only, then creating the mask at a later step...

        for pop in self.gates[stain_name].keys():
            if do_get_mask:
                self.get_popCounts_mask(stain_type,
                                        self.gates[stain_name][pop],
                                        pop,
                                        df_data,
                                        df_data.index.values,
                                        np.tile(1, len(df_data.index.values)),
                                        standard_channels = standard_channels,
                                        use_transform = use_transform)
            else:
                self.get_popCounts_index(stain_type,
                                         self.gates[stain_name][pop],
                                         pop,
                                         df_data,
                                         df_data.index.values,
                                         standard_channels = standard_channels,
                                         use_transform = use_transform)

        self.populations[stain_type]["COMP_DATA"] = df_data
        self.populations[stain_type]["RAW_DATA"] = other_data

        # root of stain
        if not flojo_count:
            flojo_count = -1

    def poly_gate(self, info = None, pop_name = None):
        """
        Return list of vertex points constructed via the polygate in FlowJo

        input: info = None, pop_name = None'
        output: points_df
        """
        x_var = info[0][1]
        y_var = info[1][1]
        i = 2
        points = []

        # Some are switched
        if 0:
        #if pop_name in ["Time"]:
            while i < len(info):
                points.append([float(info[i+1][1]), float(info[i][1])])
                i += 2
        else:
            while i < len(info):
                points.append([float(info[i][1]), float(info[i+1][1])])
                i += 2

        points_df = pd.DataFrame(points, columns = [x_var, y_var])

        return points_df

    def rectangle_gate(self, info = None, pop_name = None):
        """
        Return list of points constructed by the rectanglegate in FlowJo.
        Some of these gates are actual quad gates....for these, take q1 tup-right, q2 top-left, q3 bottom-right, q4 bottom-left

        """

        # brack header before max/min that should be stripped
        def get_afterBracket(temp_str):
            try:
                bracket_ind = None
                bracket_ind = temp_str.index("}")
                if isinstance(bracket_ind,int):
                    return temp_str[bracket_ind+1:]
                else:
                    return temp_str
            except ValueError:
                return temp_str

        # some only have a max, or a min...
        x_var = info[0][0][0][1]
        y_var = info[1][0][0][1]
        x_max, y_max = float(1e10), float(1e10)
        x_min, y_min = float(-1e10), float(-1e10)

        #try except blocks for those
        try:
            x_min = float(info[0][1][0][1])
            x_max = float(info[0][1][1][1])

        except IndexError:
            #print("x error: ", get_afterBracket(info[0][1][0][0]))
            if get_afterBracket(info[0][1][0][0]) == "min":
                x_min = float(info[0][1][0][1])
                x_max = float(1e10)
            if get_afterBracket(info[0][1][0][0]) == "max":
                x_min = float(-1e10)
                x_max = float(info[0][1][0][1])
        try:
            y_min = float(info[1][1][0][1])
            y_max = float(info[1][1][1][1])

        except IndexError:
            #print("y error: ", get_afterBracket(info[1][1][0][0]))
            if get_afterBracket(info[1][1][0][0]) == "min":
                y_min = float(info[1][1][0][1])
                y_max = float(1e10)
            if get_afterBracket(info[1][1][0][0]) == "max":
                y_min = float(-1e10)
                y_max = float(info[1][1][0][1])

        # Some are switched...?

        p1 = (x_min,y_min)
        p2 = (x_min,y_max)
        p3 = (x_max,y_max)
        p4 = (x_max,y_min)

        points = pd.DataFrame([p1,p2,p3,p4], columns = [x_var, y_var])

        # if any([q in pop_name for q in ["q1--","q2--","q3--","q4--"]]):
            # print(pop_name, ":\n", points)

        return points

    def get_popCounts_mask(self,
                           stain,
                           node_dict,
                           pop,
                           df_data,
                           indices,
                           mask,
                           standard_channels = True,
                           use_transform = True,
                           transform_name = "biex"):
        """
        For each population, it checks the df_data from the parent population, and checks for the points that lie within the drawn polygon gate.

        The method iterates through each population node in the original xml dictionary, and recursively calls itself with the lower subpopulation.
        Terminates when there are no more sub populations.

        This method differs from get_popCounts_index because it also stores the masking array. This means that the final self.populations dict will take up more memory.
        input: self,
                  stain,
                  node_dict,
                  pop,
                  df_data,
                  indices,
                  standard_channels = True,
                  use_transform = True,
                  transform_name = "biex"
        output: (None)
        """

        # node_dict is the node in that is chosen...
        assert transform_name in TRANSFORM_FUNCTIONS
        transform = TRANSFORM_FUNCTIONS[transform_name]

        pop_name = pop
        #pop_name = node_dict["data"]["attrib"]["name"]
        gate_type = node_dict["data"]["gate"]["gate_type"]
        flojo_count = node_dict["data"]["flojo_count"]

        # parents are now in the name, but the name must be correct with the upper level parent names
        parents = []
        for i in np.arange(1,len(pop.split(self.config.definitions.LINEAGE_SEP))-1):
            parents.append(self.config.definitions.LINEAGE_SEP.join(pop.split(self.config.definitions.LINEAGE_SEP)[i:]))

        self.print_func(" %s ==> gate_type: %s, flojo_count: %s, num_parents %s"\
                            %(self.uniquePops[stain][pop_name]["flojo_alias"],
                              gate_type,
                              str(flojo_count),
                              len(parents)),
                        2)

        # pop_name_alias should be pop_name AND have the same ancestry...
        pop_name_alias = self.get_pop_alias(pop_name, stain)

        # polygon defining points...
        points = self.uniquePops[stain][pop_name]["polygon"].copy()

        # if any columns are in the fsc switch set to be done for the sample
        if any([c in ["FSC-W", "FSC-H"] for c in points.columns]) and self.fsc_wh_switch:
            points = points.rename(columns={"FSC-W":"FSC-H", "FSC-H":"FSC-W"})

        # Some names have Comp- in them for some reason, while others dont.
        # We also want to change the parameter channel names to the normal standard names
        #  for consistency across datasets.

        # initial change before plugging into STAIN_CHANNEL_ALIAS
        if self.config.experiment.ASSAY == "facs":
            points.columns = param_names = [get_facs_param_name(param) for param in points.columns]
        elif self.config.experiment.ASSAY == "cytof":
            points.columns = param_names = [get_cytof_param_name(param) for param in points.columns]

        if standard_channels:
            param_names = [self.config.definitions.STAIN_CHANNEL_ALIAS[stain][l] for l in param_names]
            points = points.rename(index = self.config.definitions.STAIN_CHANNEL_ALIAS[stain], columns = self.config.definitions.STAIN_CHANNEL_ALIAS[stain])

        # probably can be done faster
        py_pop = 0
        new_mask, new_indices = [], []

        # some of the original template gating may not be available for the specific stain
        # even though it appears in the .wsp....
        if not all([p in df_data.columns.values for p in param_names]):
            self.print_func(" >>>Not all parameters exist in the dataframe parameter space for %s. \
                                 continuing with 0 indices...<<<"%pop_name, 2)
            in_gate, new_mask, new_indices = 0, [], []
        else:
            param_vals_df = df_data.iloc[:][param_names]

            # time is the current pop
            if pop_name.split(self.config.definitions.LINEAGE_SEP)[0] == "time":
                self.print_func("",2)
                self.print_func(" >>>SPECIAL TIME GATE RUNNING...<<<", 2)

                #in_gate, new_indices = gate_pass(df_data.loc[indices][:])
                in_gate, new_indices = timeGate_simple(df_data.loc[indices][:])

                self.print_func(" Time new indices len: %i"%(len(new_indices)), 2)
                self.print_func(" >>>SPECIAL TIME GATE DONE<<<", 2)
                self.print_func("",2)

            # using matplotlib for mpltPath.Path.contains_points, which runs more quickly than shapely
            else:
                startt = time.time()

                if use_transform:
                    self.print_func(" >>>USING TRANSFORM %s<<<"%transform_name, 2)

                    param_vals_df_transform = param_vals_df
                    points_transform = points
                    # transform should be in fluorescence channels, not scattering channels...
                    for c in [c for c in points_transform.columns if c in self.config.experiment.CHANNELS]:
                        points_transform[c] = transform(points_transform[c])
                        param_vals_df_transform[c] = transform(param_vals_df_transform[c])

                    indices_paramVals = param_vals_df_transform.values
                    path = mpltPath.Path(points_transform.values)
                    indices = param_vals_df_transform.index.values
                else:
                    self.print_func(" >>>NO TRANSFORM APPLIED<<<", 2)

                    indices_paramVals = param_vals_df.values
                    path = mpltPath.Path(points.values)
                    indices = param_vals_df.index.values

                # mask of points that lie withing the drawn boundary.
                # points are already given in the orientation which the inner boundary is on the correct side...
                new_mask_mplt = path.contains_points(indices_paramVals)

                new_mask = new_mask_mplt
                new_indices = np.array(indices)[new_mask_mplt]
                tott = time.time() - startt

        # population alias, function similar to above method but now checking ancestry for sameness...

        #print("pop_name, gate param_names, gate_type: ", pop_name_alias, param_names, gate_type)
        py_flojo_diff = len(new_indices) - node_dict["data"]["flojo_count"]

        self.print_func(("  ", "py_pop: ", len(new_indices), np.sum(new_mask), "\n",
              "  ", "flojo_count: ", node_dict["data"]["flojo_count"], "\n",
              "  ", "(py_pop - flojo_count): ", py_flojo_diff), 2)
        if flojo_count > 0:
            self.print_func(("  ", "(py_flojo_diff / flojo_count): ", np.around(py_flojo_diff/float(flojo_count),4)),2)
        else:
            self.print_func(("  ", "(py_flojo_diff / flojo_count): n/a (flojo_coujnt == 0)"),2)
        self.print_func("",2)

        # Deprecated...cannot go into transforms easily.
        # Fluor parameters are all done on biexponential scale
        #gate_scales = {p : wsp_getTransforms(p) for p in param_names}

        # Time gate from flojo doesn't really make sense.....need to re-export for each value
        # in FloJo, the gating polygon vertices are incorrect, or scaled in a strange way

        # parents are in the name!

        self.populations[stain][pop_name_alias] = {"original_gate_type" : gate_type,
                                    "params_to_gate" : param_names,
                                    "param_scaling" : {},
                                    "polygon" : self.uniquePops[stain][pop_name]["polygon"],
                                    "parent_polygons" : {p : self.populations[stain][p]["polygon"] for p in parents},
                                    "py_count" : len(new_indices),
                                    "indices" : new_indices,
                                    "mask" : new_mask,
                                    "parent_fun" : indices,
                                    "flojo_count" : node_dict["data"]["flojo_count"],
                                    "time_gate" : ("Time" in pop_name)}


        for subpop in node_dict["sub"].keys():
            self.get_popCounts_mask(stain, node_dict["sub"][subpop], subpop, df_data, new_indices, new_mask, use_transform = use_transform)

    def get_popCounts_index(self,
                            stain,
                            node_dict,
                            pop,
                            df_data,
                            indices,
                            standard_channels = False,
                            use_transform = False,
                            do_specialTime = False,
                            transform_name = "biex"):
        """
        For each population, it checks the df_data from the parent population, and checks for the points that lie within the drawn polygon gate.

        The method iterates through each population node in the original xml dictionary, and recursively calls itself with the lower subpopulation.
        Terminates when there are no more sub populations.

        This method differs from get_popCounts_mask because it DOES NOT store the masking array. This means that the final self.populations dict will take up less memory.
        input: self,
                  stain,
                  node_dict,
                  pop,
                  df_data,
                  indices,
                  standard_channels = True,
                  use_transform = True,
                  transform_name = "biex",
                  do_mask = None
        output: (None)
        """

        # node_dict is the node in that is chosen...
        # indices are the actual index pointers, not the index in the df_data
        assert transform_name in TRANSFORM_FUNCTIONS
        transform = TRANSFORM_FUNCTIONS[transform_name]

        pop_name = pop
        #pop_name = node_dict["data"]["attrib"]["name"]
        gate_type = node_dict["data"]["gate"]["gate_type"]
        flojo_count = node_dict["data"]["flojo_count"]

        # parents are now in the name, but the name must be correct with the upper level parent names
        parents = []
        for i in np.arange(1,len(pop.split(self.config.definitions.LINEAGE_SEP))-1):
            parents.append(self.config.definitions.LINEAGE_SEP.join(pop.split(self.config.definitions.LINEAGE_SEP)[i:]))

        self.print_func(" %s ==> gate_type: %s, flojo_count: %s, num_parents %s"\
               %(self.uniquePops[stain][pop_name]["flojo_alias"],
                 gate_type,
                 str(flojo_count),
                 len(parents)), 2)

        # pop_name_alias should be pop_name AND have the same ancestry...
        pop_name_alias = self.get_pop_alias(pop_name, stain)

        points = self.uniquePops[stain][pop_name]["polygon"].copy()

        # if any columns are in the fsc switch set to be done for the sample
        if any([c in ["FSC-W", "FSC-H"] for c in points.columns]) and self.fsc_wh_switch:
            points = points.rename(columns={"FSC-W":"FSC-H", "FSC-H":"FSC-W"})

        # if the names are going to be standardize, using the dictionary STAIN_CHANNEL_ALIAS...
        df_data_columns = df_data.columns.values
        # Standard naming for naming with the same alias for polyon parameter recognition and gating...
        if self.config.experiment.ASSAY == "facs":
            df_data.columns = [get_facs_param_name(c) for c in df_data_columns]
            points.columns = param_names = [get_facs_param_name(param) for param in points.columns]
        elif self.config.experiment.ASSAY == "cytof":
            df_data.columns = [get_cytof_param_name(c) for c in df_data_columns]
            points.columns = param_names = [get_cytof_param_name(param) for param in points.columns]

        #if standard_channels:
        #    param_names = [self.config.definitions.STAIN_CHANNEL_ALIAS[stain][l] for l in param_names if l in self.config.definitions.STAIN_CHANNEL_ALIAS[stain]]
        #    points = points.rename(index = self.config.definitions.STAIN_CHANNEL_ALIAS[stain], columns = self.config.definitions.STAIN_CHANNEL_ALIAS[stain])
        #else:
        #    points.columns = param_names = [param for param in points.columns]

        orig_points = points.copy()
        # probably can be done faster
        py_pop = 0
        new_indices = []

        # some of the original template gating may not be available for the specific stain
        # even though it appears in the .wsp....
        if not all([p in df_data.columns.values for p in param_names]):
            self.print_func((" >>>Not all parameters exist in the dataframe parameter space for %s. continuing with 0 indices...<<<"%pop_name),
                            2)
            in_gate, new_indices = 0, []
        else:
            param_vals_df = df_data.iloc[:][param_names]

            # time is the current pop
            if pop_name.split(self.config.definitions.LINEAGE_SEP)[0] == "time" and do_specialTime:
                self.print_func((),2)
                self.print_func((" >>>SPECIAL TIME GATE RUNNING...<<<"),2)

                #in_gate, new_indices = gate_pass(df_data.loc[indices][:])
                in_gate, new_indices = timeGate_simple(df_data.loc[indices][:])

                self.print_func(" Time new indices len: %i"%(len(new_indices)), 2)
                self.print_func(" >>>SPECIAL TIME GATE DONE<<<",2)
                self.print_func("",2)

            # using matplotlib for mpltPath.Path.contains_points, which runs more quickly than shapely
            else:
                startt = time.time()

                if use_transform:
                    self.print_func(" >>>USING TRANSFORM<<<", 2)

                    param_vals_df_transform = param_vals_df
                    points_transform = points
                    for c in [c for c in points_transform.columns if c in self.config.experiment.CHANNELS]:
                        points_transform[c] = transform(points_transform[c])
                        param_vals_df_transform[c] = transform(param_vals_df_transform[c])

                    # python polygon gating
                    indices_paramVals = param_vals_df_transform.loc[indices][:].values
                    path = mpltPath.Path(points_transform.values)
                    new_mask_mplt = path.contains_points(indices_paramVals)

                    new_indices = np.array(indices)[new_mask_mplt]
                    tott = time.time() - startt

                else:
                    self.print_func(" >>>NO TRANSFORM APPLIED<<<",2)

                    indices_paramVals = param_vals_df.loc[indices][:].values
                    path = mpltPath.Path(points.values)
                    new_mask_mplt = path.contains_points(indices_paramVals)

                    new_indices = np.array(indices)[new_mask_mplt]
                    tott = time.time() - startt

        # population alias, function similar to above method but now checking ancestry for sameness...
        py_flojo_diff = len(new_indices) - node_dict["data"]["flojo_count"]

        self.print_func(("  ", "py_pop: ", len(new_indices), "\n",
              "  ", "flojo_count: ", node_dict["data"]["flojo_count"], "\n",
              "  ", "(py_pop - flojo_count): ", py_flojo_diff),2)
        if flojo_count > 0:
            self.print_func(("  ", "(py_flojo_diff / flojo_count): ", np.around(py_flojo_diff/float(flojo_count),4)),2)
        else:
            self.print_func(("  ", "(py_flojo_diff / flojo_count): n/a (flojo_coujnt == 0)"),2)
        self.print_func("",2)

        # Deprecated...cannot go into transforms easily.
        # Fluor parameters are all done on biexponential scale
        #gate_scales = {p : wsp_getTransforms(p) for p in param_names}

        # Time gate from flojo doesn't really make sense.....need to re-export for each value
        # in FloJo, the gating polygon vertices are incorrect, or scaled in a strange way

        # parents are in the name!

        self.populations[stain][pop_name_alias] = {"original_gate_type" : gate_type,
                                    "params_to_gate" : param_names,
                                    "param_scaling" : {},
                                    "polygon" : orig_points,
                                    #"parent_polygons" : {p : self.populations[stain][p]["polygon"] for p in parents},
                                    "py_count" : len(new_indices),
                                    "indices" : new_indices,
                                    "parent_indices" : indices,
                                    "flojo_count" : node_dict["data"]["flojo_count"],
                                    "time_gate" : ("Time" in pop_name)}


        for subpop in node_dict["sub"].keys():
            self.get_popCounts_index(stain, node_dict["sub"][subpop], subpop, df_data, new_indices, use_transform = use_transform)

    def wspCompMxs(self, special_version = None):
        """
        Stores comp mxs to pop_gates as self.comps. See "tools_comp" for the implementation of wsp_getComp.

        input: stains_in_wsp = None, tissue_type (what tissue type is the fcs file?)
        output: self.comps
        """
        if special_version == "IPI":
            self.comps = ipi_wsp_getComp(self.wsp_file,
                                         # stain_stain_types = self.config.experiment.STAINS_STAINS,
                                         # stain_cell_types = self.config.experiment.STAINS_CELLS,
                                         stain_pop_parse = self.config.experiment.POP_PARSE_STAINS,
                                         stain_standard_stain = self.config.experiment.STAIN_STANDARD_STAIN,
                                         experiment_channel_alias = self.config.definitions.STAIN_CHANNEL_ALIAS,
                                         tissue_type = self.tissue_type)
        else:
            self.comps = wsp_getComp(self.wsp_file,
                                     #stain_stain_types = self.config.experiment.STAINS_STAINS,
                                     #stain_cell_types = self.config.experiment.STAINS_CELLS,
                                     stain_pop_parse = self.config.experiment.POP_PARSE_STAINS,
                                     stain_standard_stain = self.config.experiment.STAIN_STANDARD_STAIN,
                                     experiment_channel_alias = self.config.definitions.STAIN_CHANNEL_ALIAS,
                                     tissue_type = self.tissue_type)

        return self.comps

    def comp_transform(self,
                          fcs_file = None,
                          stain_type = None,
                          comp_mx = None,
                          acq_comp = False,
                          standard_channels = True,
                          do_remove_pred = True):

        """
        Return the populations using the gates from the wsp file.
            Note that this gating is MORE ACCURATE than that from the wsp
            due to binning that happens in FlowJo (associateds bins of individual events with a single parameter space vector).
            For sparse gates, this may lead to larger count disparities (comparatively, in terms of percentage from FlowJo counts)

        input: self, fcs_file = None, stain_type = None, stains_in_wsp = None, comp_mx = None, acq_comp = False
        output: fcs_data_comped.iloc[:][column_order], fcs_data.iloc[:][column_order], column_order
        """

        self.print_func("comp_transform transforming the self.fcs_file...", 1)
        # maybe want to comp with wsp mx, but use external fcs not tied to flojo?
        assert not (fcs_file == None and self.fcs_file == None)

        if fcs_file == None and self.fcs_file != None:
            fcs_file = self.fcs_file
        else:
            self.fcs_file = fcs_file

        #get stain type from file name, assuming same structure
        if stain_type == None:
            basename = os.path.basename(fcs_file).split(".")[0].lower()
            for stain_cell in self.config.experiment.STAINS_CELLS:
                if self.config.definitions.SAMPLE_SEP+stain_cell in basename:
                    stain_type = stain_cell
                    self.print_func(" stain_type found: %s"%stain_type)
                    break
        else:
            stain_type = stain_type.lower()
            self.print_func(" stain_type user defined: %s"%stain_type, 2)

        stain = self.config.experiment.STAIN_STANDARD_STAIN[stain_type]

        fcs_meta, fcs_data = get_fcs(fcs_file)

        if do_remove_pred:
            fcs_use_columns = [c for c in fcs_data.columns if self.config.definitions.FCS_VAL_SEP not in c]
            fcs_data = fcs_data[fcs_use_columns]

        if standard_channels:
            # why does everything have an "-A" after it?
            stain_chan_alias = self.config.definitions.STAIN_CHANNEL_ALIAS[stain]
            stain_chan_alias.update({b+"-A" : stain_chan_alias[b] for b in stain_chan_alias})
            stain_chan_alias.update({b.replace("-A","") : stain_chan_alias[b] for b in stain_chan_alias})
            stain_chan_alias.update({b.replace(" ","") : stain_chan_alias[b] for b in stain_chan_alias})
            stain_chan_alias.update({b.replace("-A","") : stain_chan_alias[b] for b in stain_chan_alias})
            fcs_data = fcs_data.rename(columns = stain_chan_alias)

        # issue with FSC-H and FSC-W
        if self.config.experiment.ASSAY == "facs":
            fcs_data, self.fsc_wh_switch = fsc_change(fcs_data)
        elif self.config.experiment.ASSAY == "cytof":
            self.print_func(">>>comp is original for cytof. returning fcs_data...<<<", 2)

            column_order = list(fcs_data.columns.values)
            return fcs_data.iloc[:][column_order], fcs_data.iloc[:][column_order], column_order

        if acq_comp:
            self.print_func(">>>compensation transformations based on ACQUISITON mx from .fcs<<<", 1)
            [use_comp, lasers] = fcs_getComp(self.fcs_file,
                                             experiment_channel_alias = self.config.definitions.STAIN_CHANNEL_ALIAS,
                                             standard_channels = True, stain = stain)
        else:
            assert not self.wsp_file == None
            self.print_func(">>>compensation transformations based on WSP MANUALLY CHANGED mx from .wsp<<<", 1)
            # compensating data, and then running through the gates
            # if custom comp_mx not given, use the one ...
            if comp_mx is None:
                # gets previously parsed comps
                wsp_comps = self.comps

            # name ambiguity...may have to change
            # possible weird change of name
            chosen_key = self.config.experiment.STAIN_STANDARD_STAIN[stain_type]
            #some are different for some reason, have 15 or 17 lasers...sets it to wsp columns
            [use_comp, lasers]= wsp_comps[chosen_key]

        use_comp_invTemp = np.linalg.inv(use_comp)
        use_comp_inv = pd.DataFrame(use_comp_invTemp, columns = lasers, index = lasers)

        fcs_data_laser = fcs_data.iloc[:][lasers]
        fcs_data_notLaser = fcs_data.iloc[:][[c for c in fcs_data.columns if c not in lasers]]
        fcs_compData_laser = pd.DataFrame(np.dot(fcs_data_laser, use_comp_inv), columns = lasers)

        fcs_data_comped = pd.concat([fcs_data_notLaser, fcs_compData_laser], axis = 1)
        column_order = list(fcs_data_comped.columns.values)

        return fcs_data_comped.iloc[:][column_order], fcs_data.iloc[:][column_order], column_order

    def draw_tree(self,
                  file_path = None,
                  dir_path = None,
                  stain = None,
                  file_type = "png",
                  param_dict = None,
                  do_save = True,
                  do_show = False,
                  do_ignore_pops = False,):
        """
        Draws population (wwith or without counts) to file_type and saves it to defined or dir default path, default as .pdf

        input: self, file_path = None, dir_path = None, stain = None, file_type = "pdf"
        output: (None)
        """
        import pydotplus as pydot

        print("Drawing population tree...")
        if isinstance(stain, type(None)):
            stain = sorted(self.populations.keys())[0]
            print(">>>stain undefined! using sorted first stain ''%s''...<<<"%stain)

        if param_dict is None:
            if self.config.definitions.CHAN_PARAM is not None:
                param_dict = self.config.definitions.CHAN_PARAM
            else:
                print(">>>No param dict given. Using default channel names<<<")

        stain_n = stain.replace(self.config.definitions.SAMPLE_SEP+self.root_str,"")

        self.i = 0
        def draw(parent_name, parent_alias, child_name, child_alias):
            # simple un counted pop tree if self.populations is empty, otherwise with counts.
            if parent_alias == stain:
                stain_root = stain+self.config.definitions.SAMPLE_SEP+self.root_str
                if self.populations != {}:
                    py_count_parent = self.populations[stain_n][stain_root]["py_count"]
                    flojo_count_parent = self.populations[stain_n][stain_root]["flojo_count"]
                    parent_label = ("%s\n%s \n Tot py: %i flojo: %i"%(self.name, parent_name, py_count_parent,  flojo_count_parent))
                else:
                    parent_label = ("%s\n%s"%(self.name, parent_name))
                graph.add_node(pydot.Node(parent_alias, label = parent_label))

            else:
                if self.populations != {}:
                    py_count_parent = self.populations[stain_n][parent_name]["py_count"]
                    flojo_count_parent = self.populations[stain_n][parent_name]["flojo_count"]
                pass

            # If the populations are not defined, then we move to creating just hte labels without the percentage counts
            child_label = ""
            if self.populations != {} or do_ignore_pops:
                child_label = "%s"%child_name.split(self.config.definitions.LINEAGE_SEP)[0]
                params_to_gate = self.uniquePops[stain_n][child_name]["params_to_gate"]
                child_label += "\noriginal: %s"%str(params_to_gate)
                flojo_gateParams = self.populations[stain_n][child_name]["params_to_gate"]
                child_label += "\nrenamed: %s"%flojo_gateParams

                if param_dict is not None:
                    if self.config.experiment.ASSAY == "cytof":
                        params_marker_to_gate = [param_dict.get(get_cytof_param_name(param), param)  for param in flojo_gateParams]
                        child_label +=   "\nmarkers: %s"%params_marker_to_gate
                    elif self.config.experiment.ASSAY == "facs":
                        params_marker_to_gate = [param_dict.get(get_facs_param_name(param), param) for param in flojo_gateParams]
                        child_label +=   "\nmarkers: %s"%params_marker_to_gate
                    else:
                        pass

                py_count_child = self.populations[stain_n][child_name]["py_count"]
                flojo_count_child = self.populations[stain_n][child_name]["flojo_count"]
                child_label += "\nCount py, flowjo: %i, %i"%(py_count_child, flojo_count_child)

            else:
                params_to_gate = self.uniquePops[stain_n][child_name]["params_to_gate"]
                child_label += "%s\n%s"%(child_name.split(self.config.definitions.LINEAGE_SEP)[0], str(params_to_gate))

            graph.add_node(pydot.Node(child_alias, label = child_label, fill = "blue"))

            if self.populations != {} and not do_ignore_pops:
                ratios = [-1, -1]
                if py_count_parent > 0:
                    ratios[0] = float(py_count_child)/float(py_count_parent)
                if flojo_count_parent > 0:
                    ratios[1] = float(flojo_count_child)/float(flojo_count_parent)
                if ratios[0] != -1:
                    color = "black"
                    penwidth = ratios[0]*5.+.2
                    style = "filled"
                    arrowhead = "empty"
                else:
                    color = "red"
                    penwidth = 2.
                    style = "dashed"
                    arrowhead = "empty"

                edge = pydot.Edge(parent_alias, child_alias, label = "%.3f, %.3f"%(ratios[0],ratios[1]), penwidth = penwidth, color = color, style = style, arrowhead = arrowhead)
            else:
                color = "black"
                penwidth = 2.
                style = "dashed"
                arrowhead = "empty"
                edge = pydot.Edge(parent_alias, child_alias, label = "", penwidth = penwidth, color = color, style = style, arrowhead = arrowhead)

            graph.add_edge(edge)
            self.i += 1

        def visit(node_dict, parent = None, parent_alias=None, parents_used =None):

            # for some reason there is a problem with colons...replace colons with "-"
            for k in node_dict.keys():
                v = node_dict[k]
                k2 = k

                '''if ":" in k2:
                    k2 = k2.replace(":", "-")
                if " " in k2:
                    k2 = k2.replace(" ", self.config.definitions.SAMPLE_SEP)'''

                parents_used.add(k2)
                if isinstance(v, dict):
                    if parent_alias:
                        draw(parent, parent_alias, k, k2)
                    visit(v, parent = k, parent_alias = k2, parents_used = parents_used)
                else:
                    draw(parent, parent_alias, k2)
                    # drawing the label using a distinct name

        parents_used = set(stain)
        graph = pydot.graphviz.Dot(graph_type='graph')

        visit(self.families[stain_n], parent = stain, parent_alias = stain, parents_used = parents_used)
        print("Num nodes found: %i" %(self.i))

        graph.set_dpi(300)

        if file_path == None:
            if dir_path == None:
                print(self.name)
                #graph.write('%s_%s_popTree.pdf'%(self.name,stain), format = "pdf")
                file_path = '%s_%s_popTree.%s'%(self.name, stain, file_type)
                graph.write('%s_%s_popTree.%s'%(self.name, stain, file_type), format = file_type)
            else:
                file_path = os.path.join(dir_path,'%s_%s_popTree.%s'%(self.name, stain, file_type))
                #graph.write(os.path.join(dir_path,'%s_%s_popTree.pdf'%(self.name,stain)), format = "pdf")
                graph.write(os.path.join(dir_path,'%s_%s_popTree.%s'%(self.name, stain, file_type)), format = file_type)
        else:
            graph.write(file_path, format = "png")

        if isinstance(os.path.splitext(file_path)[1], type(None)):
            file_path_png = file_path+"."+format
        else:
            file_path_png = file_path

        if do_show:
            from PIL import Image
            im = Image.open(file_path_png)
            im.show()
        else:
            pass

        if do_save:
            pass
        else:
            os.remove(file_path_png)

    def print_families(self, stain = None, use_flojo_alias = True, use_lineage = False):
        """
        Print families using spaces for child depth.

        input: self, stain = None, use_flojo_alias = True, use_lineage = False
        output: (None)
        """
        print(self.families.keys())
        def print_node(parent_node, i = 0):
            for key in parent_node.keys():
                if use_flojo_alias:
                    pop_name = self.uniquePops[stain][key]["flojo_alias"]
                else:
                    pop_name = key

                if not use_lineage:
                    pop_name = pop_name.split(self.config.definitions.LINEAGE_SEP)[0]

                print("%s %s : %i"%("|"*i, pop_name, i))

                try:
                    print_node(parent_node[key], i+1)
                except Exception as e:
                    print("Error at ", key)
                    raise e

        assert stain is not None
        print(stain, " : num pops: ", len(self.uniquePops[stain].keys()))
        print_node(self.families[stain], i=1)


    def r_familyHeirarchy(self, stain, family = None):
        """
        Prints same depth family heirarchy in form that is usable by r flow tailoring....

        input: self, stain, family = None
        output: (None)
        """
        if family == None:
            family = self.families

        flojo_names = []
        for key in family.keys():
            flojo_names.append(self.uniquePops[stain][key]["flojo_alias"])

        if len(flojo_names) is not 0:

            if len(flojo_names) > 1:
                print('c(', end = '')
            else:
                print('', end = '')

            for i, fn in enumerate(flojo_names, 1):
                if i == len(flojo_names):
                    print('"%s"'%fn, end = '')
                else:
                    print('"%s",'%fn, end = '')

            if len(flojo_names) > 1:
                print('),', end = '\n')
            else:
                print(',', end = '\n')

            for key in family.keys():
                self.print_familyHeirarchy(stain, family = family[key])

    def get_datas(self):
        """
        Returns the comp data, raw data, and column order if self.wsp_file and self.fcs_file are defined.

        input: None
        output: compedData (pd.DataFrame), fcsData (pd.DataFrame), columns (list)
        """
        assert self.wsp_file != None and self.fcs_file != None
        compedData, fcsData, columns = self.comp_transform(fcs_file = self.fcs_file, acq_comp = True)
        return compedData, fcsData, columns
