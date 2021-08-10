This repository contains code to output a CSV file with standardize population names and cell counts, for a directory of WSP files.

To run the code, you first need to install a set of Python libraries, preferably in a virtual environment.

```bash
$ python3 -m venv venv
$ . venv/bin/activate
(venv) $ pip install -r requirements.txt
```

The script to run is `parse_counts.py`, and it takes two arguments:

```bash
 -id, --wspdir: the path to the directory of WSP files.
 -od, --csvdir: the path where you want the individual stain population count CSV files saved to.
```

For example, if you have a WSP located in a directory in this root project like:

```bash
$ ls -alh
total 332K
drwxrwxr-x  7 user user 4.0K Aug 10 14:45 .
drwxrwxr-x 17 user user 4.0K Aug 10 14:21 ..
-rw-rw-r--  1 user user 9.5K Aug 10 14:38 config_loader.py
-rw-rw-r--  1 user user 8.5K Aug 10 14:48 config.yaml
-rw-rw-r--  1 user user  61K Aug 10 14:38 gates_fromWsp.py
drwxrwxr-x  7 user user 4.0K Aug 10 14:21 .git
-rw-rw-r--  1 user user   10 Aug 10 14:47 .gitignore
-rw-rw-r--  1 user user  59K Aug 10 14:38 ipi_tools_comp.py
-rw-rw-r--  1 user user  10K Aug 10 14:40 parse_counts.py
-rw-rw-r--  1 user user  11K Aug 10 14:25 pop_consistency_nameChanges.py
drwxrwxr-x  2 user user 4.0K Aug 10 14:39 __pycache__
-rw-rw-r--  1 user user    0 Aug 10 14:40 README.md
-rw-rw-r--  1 user user  287 Aug 10 14:39 requirements.txt
drwxrwxr-x  2 user user 4.0K Aug 10 14:44 test-inputs
-rw-rw-r--  1 user user 124K Aug 10 14:25 tools_comp.py
drwxrwxr-x  6 user user 4.0K Aug 10 14:22 venv

$ ls -alh test-inputs/
total 4.2M
drwxrwxr-x 2 user user 4.0K Aug 10 14:44 .
drwxrwxr-x 7 user user 4.0K Aug 10 14:45 ..
-rw-rw-r-- 1 user user 4.1M Aug 10 14:43 patient-IPIADR002-flojo_file_processed.wsp
```

You could run:

```bash
(venv) $ python parse_counts.py -id test-inputs -od test-outputs
/home/user/Documents/projects/ucsf/ipi/data_integrity/flow/ipi_flow_population_name_standardization/venv/lib/python3.8/site-packages/sklearn/utils/deprecation.py:143: FutureWarning: The sklearn.neighbors.kde module is  deprecated in version 0.22 and will be removed in version 0.24. The corresponding classes / functions should instead be imported from sklearn.neighbors. Anything that cannot be imported from sklearn.neighbors is now part of the private API.
  warnings.warn(message, FutureWarning)
>>>no stain_channel_standard_loc!<<<
>>>no stain_channel_alias_loc!...<<<
>>>no chan_param_loc!...<<<
wsp_paths: ['test-inputs/patient-IPIADR002-flojo_file_processed.wsp']
patient-IPIADR002-flojo_file_processed.wsp
 60741.fcs, stain: patient-IPIADR002-flojo_file_processed--60741.fcs
 61117.fcs, stain: patient-IPIADR002-flojo_file_processed--61117.fcs
 61176.fcs, stain: patient-IPIADR002-flojo_file_processed--61176.fcs
 61229.fcs, stain: patient-IPIADR002-flojo_file_processed--61229.fcs
 61230.fcs, stain: patient-IPIADR002-flojo_file_processed--61230.fcs
```

Some WSPs may generate "stain" names that are five digit numbers instead of the standard IPI stains (dc, innate, nktb, sort, and treg), as above. In these cases, you can determine the corresponding stain with the following steps:

1. Open up the WSP file.
2. Check for `<SampleNode>` nodes where `name=#`, i.e. `name=61229`. This node will also have a `sampleID` reference. For example, `<SampleNode name="61229.fcs"  annotation=""  owningGroup=""  expanded="0"  sortPriority="10"  count="500000"  sampleID="1" >`.
3. Search for a `<DataSet>` node with the matching `sampleID`, and you will find the right FCS file and stain for the given numeric sequence. i.e. `<DataSet uri="file:IPIADR002_T1_flow_dc.fcs"  sampleID="1" />`