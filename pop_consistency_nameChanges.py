import sys

def get_pop_alias(pop_alias, 
                  sep = "@", 
                  quad_sep = "|",
                  do_lowercase = True, 
                  do_nospaces = True,
                  do_second_changes = True):
    """
    Sets the population name alias to use for dictionary keys, population name consistency, etc.
    Each population gate in the pop_alias is split into each individual gate pop, and the name_changes are done for each pop in the hierarchy are then rejoined.
    
    If you want to do a whole population hierarchy at once (for example, Live@Single Cells 2@Single Cells 1@Time@Stain 1), you can also pass through that whole hierarchy string. 
    You need to define the hierarchy separator character (in the above example, it would be ''sep="@"'')
    
    input: pop_alias, 
           sep = "@", 
           quad_sep = "|", 
           do_lowercase = True, 
           do_nospaces = True
    output: pop_name_change
    """
    def name_change(name, full_lineage = [], do_lowercase = True, do_nospaces = True):
        # weird misspellings, etc...
        
        if do_lowercase:
            # remove spaces, lowercase
            name = name.lower()
        if do_nospaces:
            if "," not in name:
                name = name.replace(" cd", ",cd")
                name = name.replace(" CD", ",CD")
            name = name.replace(" ","")
            
        if ":" in name:
            name = name.replace(":","--").strip()
        if "'" in name:
            name = name.replace("'","")
        if "_1" in name:
            name = name.replace("_1","")
        if "/" in name:
            name = name.replace("/", ".")
        if "ld" == name:
            name = name.replace("ld", "livedead")
        if "live_dead" in name:
            name = name.replace("live_dead", "livedead")
        if "live/dead" in name:
            name = name.replace("live/dead", "livedead")
        if "hla_livedead" in name:
            name = name.replace("hla_livedead", "hladr_livedead")
        if "subset-1" in name:
            name = name.replace("subset-1","subset")
        if "non" in name:
            name = name.replace("non","not")
        if "not-" in name:
            name = name.replace("not-","not")
        if "cd8a" in name:
            name = name.replace("cd8a", "cd8")
        if "pd-1" in name:
            name = name.replace("pd-1", "pd1")
        if "cd80" in name:
            name = name.replace("cd80", "cd90")
        if "cd3+all" in name:
            name = name.replace("cd3+all", "cd3+")
        if "moncytes" in name:
            name = name.replace("moncytes", "monocytes")
        ## CHECK THIS POP
        #if "mono2-macs" in name:
        #    name = name.replace("mono2-macs","mono2")
        if "granunulocytes" in name:
            name = name.replace("granunulocytes", "granulocytes")
        if "ganulocytes" in name:
            name = name.replace("ganulocytes", "granulocytes")
        if "granulocytes_test" in name:
            name = name.replace("granulocytes_test", "granulocytes")
        if "granulocytes" in name:
            name = name.replace("granulocytes", "grans")
        if "ulpb" in name:
            name = name.replace("ulpb", "ulbp")
        if "ilbp" in name:
            name = name.replace("ilbp", "ulbp")
        # if "trem2" in name:
            # name = name.replace("trem2", "b8")
        if "itgb8" in name:
            name = name.replace("itgb8", "b8")
        if "b8hi" in name:
            name = name.replace("b8hi", "b8+")
        if "(tams)" in name:
            name = name.replace("(tams)", "")
        if "(dcs)" in name:
            name = name.replace("(dcs)", "")
        # if "ctla4+" in name:
            # name = name.replace("ctla4+", "ctla4")
        if "ctla-4+" in name:
            name = name.replace("ctla-4+", "ctla4+")
        if "ctla-4-" in name:
            name = name.replace("ctla-4-", "ctla4-")
        if "bdca-" in name:
            name = name.replace("bdca-", "bdca")
        if "nk" in name and "nkcells" not in name:
            name = name.replace("nk", "nkcells")
        if "nkcellscd16+" in name:
            name = name.replace("nkcellscd16+", "nkcells")
        if "neutrophiles" in name:
            name = name.replace("neutrophiles", "neutrophils")
        if "lymphocytes+_1" in name:
            name = name.replace("lymphocytes+_1","lymphocytes")
        if "lymphocytes+" in name:
            name = name.replace("lymphocytes+","lymphocytes")
        if "lineage+cells" in name:
            name = name.replace("lineage+cells", "Lineage+")
        if "ki67subset" in name:
            name = name.replace("ki67subset", "ki67hi")
        if "ki67+" in name:
            name = name.replace("ki67+", "ki67hi")
        if "ki67" in name and "ki67hi" not in name:
            name = name.replace("ki67", "ki67hi")
        if "hla dr" in name:
            name = name.replace("hla dr", "hladr")
        if "hla-dr" in name:
            name = name.replace("hla-dr", "hladr")
        if "hladr+lymphocytes-" in name:
            name = name.replace("hladr+lymphocytes-", "hladr+,lymphocytes-")
        if "hladr-,cd3-cd56+(nk)" in name:
            name = name.replace("hladr-,cd3-cd56+(nk)", "hladr-cd3-,cd56+")
        if "hladr-,cd3-,cd56+(nkqc)" in name:
            name = name.replace("hladr-,cd3-,cd56+(nkqc)", "hladr-cd3-,cd56+")
        if "hladr-,cd3-,cd56+(nk)" in name:
            name = name.replace("hladr-,cd3-,cd56+(nk)", "hladr-cd3-,cd56+")
        if "hladr+" == name:
            name = "cd3-hladr+"
        if "fcerl" in name:
            name = name.replace("fcerl", "fcer1a")
        if "fcer1ahghneutrophils" in name:
            name = name.replace("fcer1ahghneutrophils", "fcer1a_high_neutrophils")
        if "fcer1ahineutrophils" in name:
            name = name.replace("fcer1ahineutrophils", "fcer1a_high_neutrophils")
        if "fcer1alowneutrophils" in name:
            name = name.replace("fcer1alowneutrophils", "fcer1a_low_neutrophils")
        if "fcer1aloneutrophils" in name:
            name = name.replace("fcer1aloneutrophils", "fcer1a_low_neutrophils")
        if "cd64+.cd14+bdca3+" in name:
            name = name.replace("cd64+.cd14+bdca3+", "cd64+,cd14+,bdca3+")
        if "cd45-hla-dr+kii67-" in name:
            name = name.replace("cd45-hla-dr+kii67-", "cd45-,hladr+,ki67-")
        if "cd3+,hladr-cd4+,cd25+,foxp3+(tr)" in name:
            name = name.replace("cd3+,hladr-cd4+,cd25+,foxp3+(tr)", "cd3+,hladr-,cd4+,cd25+,foxp3+(tr)")
        if "foxp3-(th)" in name: ##???? is this ok? some of the samples dont have this
            name = name.replace("foxp3-(th)", "(th)")
        if "cd19+cd20+cd56+cd3-" in name:
            name = name.replace("cd19+cd20+cd56+cd3-", "cd19+,cd20+,cd56+,cd3-")
        if "b-cells" in name:
            name = name.replace("b-cells", "bcells")
        if "bdca1+dcs" in name:
            name = name.replace("bdca1+dcs", "bdca1+cdcs")
        if "bdca3+dcs" in name:
            name = name.replace("bdca3+dcs", "bdca3+cdcs")
        if "cd3-cd19-cd20-cd56-" in name:
            name = name.replace("cd3-cd19-cd20-cd56-", "cd19-cd20-cd56-cd3-")
        if "lineage+" == name:
            name = "cd56+cd19+cd3-"
        if "lineage-cd3-" == name:
            name = "cd56-cd19-cd3-"
        if "q7--cd56+cd19-" == name:
            name = "q7--cd56+cd19-(nkcellst)"
        if "cd4-cd25-" == name: ##careful...
            name = name.replace("cd4-cd25-", "cd4+cd25-")
        if "cd4+cd25-(th)" == name:
            name = "cd4+cd25-"
        if "cd16-notmonocytes" in name:
            name = name.replace("cd16-notmonocytes", "cd16-")
            
        ## full lineage checks
        if "cd3all" in name and ("(tr)" in name or "(th)" in name):
            if len(full_lineage) >= 3:
                if "hladr+" in full_lineage[2] and full_lineage[1] == "q|cd8-cd4+":
                    name = name.replace("cd3all", "cd3+hladr+")
                elif "hladr-" in full_lineage[2] and full_lineage[1] == "q|cd8-cd4+":
                    name = name.replace("cd3all", "cd3+hladr-")
                    
        ## double checks
        if "q|cd56+cd19-(nkcellst)" == name:
            name = "q|cd56+(nkcellst)cd19-"
        if "inthladr_cd14lo-linlo" == name:
            name = "inthladr_cd14lo_linlo"
        if "cd3+cd4+cd25+foxp3+(tr)" == name:
            name = "cd3allcd4+cd25+foxp3+(tr)"   
        if "cd3+cd4+cd25-(th)" == name:
            name = "cd3allcd4+cd25-(th)"
        if "cd3-cd19+cd20+cd56+" == name:
            name = "cd19+cd20+cd56+cd3-"
        if "cd3-cd19-cd20-cd56-" == name:
            name = "cd19-cd20-cd56-cd3-"
        if "cd3+cd4+cd25+foxp3+(tr)" == name:
            name = "cd3allcd4+cd25+foxp3+(tr)"
        if "cd3+cd4+cd25-(th)" == name:
            name = "cd3allcd4+cd25-(th)"
        if "cd16+monocytes" == name:
            name = "cd16+"
        if "cd16-notmonocytes" == name:
            name = "cd16-"
        
        # quadrant gate, so order is consistent...
        # sort is reversed, so non-cd# antibodies appear first

        # if "," in name and "--" in name:
        if "--" in name:
            q_beg, quad_pops= name.split("--")[0], name.split("--")[1]
            quad_pops = ",".join(reversed(sorted(quad_pops.split(","))))
            # q_beg = q_beg.replace("q5","q1")
            # q_beg = q_beg.replace("q6", "q2")
            # q_beg = q_beg.replace("q7", "q3")
            # q_beg = q_beg.replace("q8", "q4")
            # if quadrant number in name doesn't matter
            q_beg = "q"
            
            name = quad_sep.join([q_beg, quad_pops])
        
        #removal of commas, because there are inconsistencies in population names with commas/spaces
        if "," in name:
            name = name.replace(",","")
            
        return name
    
    pop_alias_split = pop_alias.split(sep) 
    pop_hierarchy_change = []
    
    # checking each pop and adding name change to pop_hierarchy_change
    for pop in pop_alias_split:
        pop_hierarchy_change.append(name_change(pop, full_lineage = pop_alias_split, do_lowercase = do_lowercase, do_nospaces = do_nospaces))
    
    ## rerunning with the newly changed lineage to check for specific cases that require more populations to check on
    if do_second_changes:
        pop_hierarchy_change2 = []
        
        for pop in pop_alias_split:
            pop_hierarchy_change2.append(name_change(pop, full_lineage = pop_hierarchy_change, do_lowercase = do_lowercase, do_nospaces = do_nospaces))
        
        pop_name_change = sep.join(pop_hierarchy_change2)
    
    else:
        pop_name_change = sep.join(pop_hierarchy_change)
    
    return pop_name_change

# if __name__ == "__main__":
    # """
    # Converting list of pop_alias strings.
    # """
    
    # args = sys.argv[1:]
    
    # ## converting population naming strings...
    # if isinstance(args[0], str): 
        # pops = args
        # for pop in pops:
            # print("%s -> %s"%(pop, get_pop_alias(pop)))
    