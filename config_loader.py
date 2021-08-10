import yaml
import os
import datetime

class config_loader(object):
    def __init__(self, config_loc = None, experiment = "IPI"):
                
        if not config_loc:
            config_loc = "config.yaml"
        
        self.config_loc = os.path.realpath(config_loc)
        
        # string version of yaml...
        with open(self.config_loc,"r") as handle:
            self.YAML = handle.read()
        
        # the python object yaml...
        self.config = yaml.load(self.YAML, Loader = yaml.Loader)
        
        if "start_date" in self.config:
            self.start_date = self.config["start_date"]
        else:
            self.start_date = str(datetime.datetime.now()).split(".")[0].replace(":","-").replace(" ","_") 
            
        if "logging" in self.config:
            self.logging = self.logging_attrs(self.config["logging"])
        if "methods" in self.config:
            self.methods = self.methods_attrs(self.config["methods"])
        if "paths" in self.config:
            self.paths = self.paths_attrs(self.config["paths"])
        if "sample_order" in self.config: 
            self.sample_order = self.sample_order_attrs(self.config["sample_order"])
        if "definitions" in self.config:
            self.definitions = self.definitions_attrs(self.config["definitions"], 
                                                      self.paths.STAIN_CHANNEL_STANDARD_LOC, 
                                                      self.paths.STAIN_CHANNEL_ALIAS_LOC, 
                                                      self.paths.CHAN_PARAM_LOC)
        if "process_args" in self.config:
            self.process_args = self.process_args_attrs(self.config["process_args"])
        if "experiment" in self.config:
            self.experiment = self.experiment_attrs(self.config["experiment"])
        
    def save_yaml(self, output_loc):
        import yaml
        with open(output_loc,"w") as handle:
            yaml.dump(yaml.load(self.YAML, Loader = yaml.Loader), handle)
        
    class logging_attrs():
        def __init__(self, logging):
            self.LOGGING_DETAIL = logging["detail"]
    
    class methods_attrs():
        def __init__(self, methods):
            self.write_version_info = methods.get("write_version_info", {})
            self.process = methods.get("process", {})
            self.train = methods.get("train", {})
            self.predict = methods.get("predict", {})
            self.metrics = methods.get("metrics", {})
            
    class paths_attrs():
        def __init__(self, paths):
            # disk locations
            self.STAINVERSIONINFO_DIR = os.path.normpath(paths["folder_paths"]["STAINVERSIONINFO_DIR"])
            self.HOME_DIR = os.path.normpath(paths["folder_paths"]["HOME_DIR"])
            self.TRAIN_FROM_DIR = os.path.normpath(paths["folder_paths"]["TRAIN_FROM_DIR"])
            self.PRED_ON_DIR = os.path.normpath(paths["folder_paths"]["PRED_ON_DIR"])
            
            self.CHAN_DICT_HEAD = paths["folder_names"]["CHAN_DICT_HEAD"]
            self.CHAN_PARAM_NAME = paths["folder_names"].get("CHAN_PARAM_NAME",None)
            self.VERSION_HEAD = paths["folder_names"]["VERSION_HEAD"]
            self.SAVE_NAME = paths["folder_names"]["SAVE_NAME"]
            self.SAMPLE_START_STR = paths["folder_names"]["SAMPLE_START_STR"]
            
            self.TRAINING_SAVE_DIR = os.path.normpath(os.path.join(self.HOME_DIR, paths["folder_names"]["TRAINING_SAVE_DIR"]))
            self.OUTPUT_PRED_DIR = os.path.normpath(os.path.join(self.HOME_DIR, paths["folder_names"]["OUTPUT_PRED_DIR"]))
            self.OUTPUT_POPRATIOS_DIR = os.path.normpath(os.path.join(self.HOME_DIR, paths["folder_names"]["OUTPUT_POPRATIOS_DIR"]))
            self.LOG_DIR = os.path.normpath(os.path.join(self.HOME_DIR, paths["folder_names"]["LOG_DIR"]))
            self.CONFIG_DIR = os.path.normpath(os.path.join(self.HOME_DIR, paths["folder_names"]["CONFIG_DIR"]))
            self.SAMPLE_ORDER_DIR = os.path.normpath(os.path.join(self.HOME_DIR, paths["folder_names"]["LOG_DIR"]))
            
            self.POPNUMDICT_LOC = os.path.normpath(os.path.join(self.STAINVERSIONINFO_DIR, "%s_popNumDict"%self.VERSION_HEAD))
            self.NUMPOPDICT_LOC = os.path.normpath(os.path.join(self.STAINVERSIONINFO_DIR, "%s_numPopDict"%self.VERSION_HEAD))
            self.POP_TEMPLATE_LOC = os.path.normpath(os.path.join(self.STAINVERSIONINFO_DIR, "%s_uniquePops"%self.VERSION_HEAD))
            self.STAIN_CHANNEL_STANDARD_LOC = os.path.normpath(os.path.join(self.STAINVERSIONINFO_DIR, "%s_stain_chans_dict"%self.CHAN_DICT_HEAD))
            self.STAIN_CHANNEL_ALIAS_LOC = os.path.normpath(os.path.join(self.STAINVERSIONINFO_DIR, "%s_stain_alias_dict"%self.CHAN_DICT_HEAD))
            self.CHAN_PARAM_LOC = os.path.normpath(os.path.join(self.STAINVERSIONINFO_DIR, paths["folder_names"]["CHAN_PARAM_NAME"]))
            
    class sample_order_attrs():
        def __init__(self, sample_order):
            self.TRAINING_SAMPLES = sample_order["TRAINING_SAMPLES"]
            self.TESTING_SAMPLES = sample_order["TESTING_SAMPLES"]
            self.PREDICTING_SAMPLES = sample_order["PREDICTING_SAMPLES"]
    
    class definitions_attrs():
        def __init__(self, definitions, stain_channel_standard_loc, stain_channel_alias_loc, chan_param_loc):
            import pickle

            # Stain Channel Standard Names
            if os.path.exists(stain_channel_standard_loc):
                with open(stain_channel_standard_loc,"rb") as handle:
                        self.STAIN_CHANNEL_STANDARD = pickle.load(handle)
            else:
                print(">>>no stain_channel_standard_loc!<<<")
            
            if os.path.exists(stain_channel_alias_loc):
                with open(stain_channel_alias_loc,"rb") as handle:
                    self.STAIN_CHANNEL_ALIAS = pickle.load(handle)
            else:
                print(">>>no stain_channel_alias_loc!...<<<")
                
            if os.path.exists(chan_param_loc):
                with open(chan_param_loc,"rb") as handle:
                    self.CHAN_PARAM = pickle.load(handle)
            else:
                print(">>>no chan_param_loc!...<<<")
                    
            self.SAMPLE_SEP = definitions["naming"]["pop_gates"]["SAMPLE_SEP"]
            
            self.LINEAGE_SEP = definitions["naming"]["processing"]["LINEAGE_SEP"]
            self.PRM_PRM_SEP = definitions["naming"]["processing"]["PRM_PRM_SEP"]
            self.PRM_CALC_SEP = definitions["naming"]["processing"]["PRM_CALC_SEP"]
            self.CALC_CALC_SEP = definitions["naming"]["processing"]["CALC_CALC_SEP"]
            self.INTER_PROC_SEP = definitions["naming"]["processing"]["INTER_PROC_SEP"]
            self.INTER_NAME_SEP = definitions["naming"]["processing"]["INTER_NAME_SEP"]
            self.INTER_FINAL_TAIL = definitions["naming"]["processing"]["INTER_FINAL_TAIL"]

            # FCS write prediction channel name separations
            # ex. !singlecell2@singlecell@time@Stain 1_root (boolean value if in or out of singlecell 2 gate)
            self.FCS_BOOL_SEP = definitions["naming"]["fcs_output"]["FCS_BOOL_SEP"]
            # ex. %live@singlecell2@singlecell@time@Stain 1_root (raw algorithm output value between 0 and 1 for live gate)
            self.FCS_VAL_SEP = definitions["naming"]["fcs_output"]["FCS_VAL_SEP"]

            # FCS .csv prediction values
            self.DF_VAL_SEP = definitions["naming"]["csv_output"]["DF_VAL_SEP"]
            self.ALG_PRED_TAIL = definitions["naming"]["csv_output"]["ALG_PRED_TAIL"]
            self.FLOWJO_COUNT_TAIL = definitions["naming"]["csv_output"]["FLOWJO_COUNT_TAIL"]
            self.NUM_COUNT_TAIL = definitions["naming"]["csv_output"]["NUM_COUNT_TAIL"]
            self.DENOM_COUNT_TAIL = definitions["naming"]["csv_output"]["DENOM_COUNT_TAIL"]
            self.COUNT_RATIO_TAIL = definitions["naming"]["csv_output"]["COUNT_RATIO_TAIL"]
            
            self.INFO_SEP = definitions["file_reading"]["INFO_SEP"]
            self.FCS_NAME_FORMAT = definitions["file_reading"]["FCS_NAME_FORMAT"]
            self.WSP_NAME_FORMAT = definitions["file_reading"]["WSP_NAME_FORMAT"]
            self.FCS_WSP_SHARE = definitions["file_reading"]["FCS_WSP_SHARE"]
            self.FCS_SHARE = definitions["file_reading"]["FCS_SHARE"]
            self.WSP_SHARE = definitions["file_reading"]["WSP_SHARE"]
            self.STAIN_OVERRIDE = definitions["file_reading"]["STAIN_OVERRIDE"]
            
    class experiment_attrs():
        def __init__(self, experiment):
            
            # for flattening objects
            from numpy import array
            
            self.ASSAY = experiment["ASSAY"]
            self.CHANNELS = experiment["CHANNELS"]
            self.POP_PARSE_STAINS = array(experiment["POP_PARSE_STAINS"]).flatten().tolist()
            self.STAINS_STAINS = experiment["STAINS_STAINS"]
            self.STAINS_ALL = experiment["STAINS_ALL"]
            self.STAIN_STANDARD_STAIN = experiment["STAIN_STANDARD_STAIN"]
            self.STAIN_SYNONYMNS = experiment["STAIN_SYNONYMNS"]
            
    class process_args_attrs():
        def __init__(self, process_args):
            # processing args; see preprocessing3...
            self.PREPROCESSING3_KWARGS = process_args["PREPROCESSING3_KWARGS"]
            self.PREPROCESSING_CALCS = process_args["PREPROCESSING_CALCS"]
            self.TESTING_LIMITS = process_args["TESTING_LIMITS"]
            # * wild card should allow you to do the same processing for each over all the stains, or whatever parents fit that *
            self.POP_INTERPROCESSING = process_args["POP_INTERPROCESSING"]