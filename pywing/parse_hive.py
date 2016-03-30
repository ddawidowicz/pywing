import pudb

def parse_hive(infile, num_tables):
    inlog = open(infile, 'rt')

    #preallocate memory
    tables = ['' for i in xrange(num_tables)] #holds table names

    #table_meta = [Partition, Database, Owner, CreateTime, Location, Type]
    inner_list = ['' for i in xrange(6)] #six fields (see above)
    table_meta = [inner_list[:] for i in xrange(num_tables)] #inner_list per table

    var_names = [[] for i in xrange(num_tables)] #holds variable names
    var_types = [[] for i in xrange(num_tables)] #holds variable types
    
    flags = [0, 0, 0] #var_names, partition, meta details
    outer_idx = -1 #which table are we looking at (increments at table name)
    meta_idx = 0 #which meta element we are looking at
    for line in inlog:
        if line.split() == []: #blank line
            continue
        
        line = line.split('\t') #non-blank line, split at tabs
        
        #This line will preceed useful information about the variable and 
        #partition names and should be skipped.
        if 'col_name' in line[0] and '# col_name' not in line[0]:
            continue

        elif '# col_name' in line[0]:
            flags[0] = 1 #start looking for variable names

        elif '# Partition Information' in line[0]:
            flags[0] = 0
            flags[1] = 1 #start looking for partition

        elif '# Detailed Table' in line[0]:
            flags[0] = 0
            flags[2] = 1 #start looking for table information

        # ---------------------- Looking for Table Names --------------------- #
        elif 'Table Name' in line[0]:
            flags = [0, 0, 0] #new table
            outer_idx += 1 #increment table index
            line = line[0].split() #split on white space
            table = line[2].strip()
            tables[outer_idx] = table


        # ----------------------- Looking for Variables ---------------------- #
        #There are two places where '# col_name' is found, right before
        #the actual column (variable) names and right before the partition
        #information. So this ensures that we are looking for variables
        #and not partitions.
        elif flags[0] and not flags[1]: 
            var_names[outer_idx].append(line[0].strip())
            var_types[outer_idx].append(line[1].strip())


        # ----------------------- Looking for Partition ---------------------- #
        elif flags[0] and flags[1]:
            table_meta[outer_idx][0] = line[0].strip()
            var_names[outer_idx].append(line[0].strip()) #add partition to variables
            var_types[outer_idx].append(line[1].strip())
            meta_idx += 1


        # ---------------------- Looking for Meta Details -------------------- #
        elif flags[2]:
            if not flags[1]:
                table_meta[outer_idx][0] = 'Not Partitioned'
                flags[1] = 1
                meta_idx += 1 #no partition, move over one
            if 'Database' in line[0]:
                table_meta[outer_idx][meta_idx] = line[1].strip()
                meta_idx += 1
            elif 'Owner' in line[0]:
                table_meta[outer_idx][meta_idx] = line[1].strip()
                meta_idx += 1
            elif 'CreateTime' in line[0]:
                table_meta[outer_idx][meta_idx] = line[1].strip()
                meta_idx += 1
            elif 'Location' in line[0]:
                table_meta[outer_idx][meta_idx] = line[1].strip()
                meta_idx += 1
            elif 'Table Type' in line[0]:
                table_meta[outer_idx][meta_idx] = line[1].strip()
                meta_idx = 0
                flags[2] = 0

    return [tables, table_meta, var_names, var_types]


def write_results(tables, table_meta, var_names, var_types):
    outlog1 = open('table_info.csv', 'wt')
    outlog2 = open('table_vars.csv', 'wt')
    outlog3 = open('table_types.csv', 'wt')
    for i in xrange(len(tables)):
        outlog1.write('%s,%s,%s,%s,%s,%s,%s\n' % \
                        (tables[i], \
                        table_meta[i][0], table_meta[i][1], \
                        table_meta[i][2], table_meta[i][3], \
                        table_meta[i][4], table_meta[i][5]))
        outlog2.write('%s,' % tables[i])
        outlog3.write('%s,' % tables[i])
        for j in xrange(len(var_names[i])-1): #no comma for last entry
            outlog2.write('%s,' % var_names[i][j])
            outlog3.write('%s,' % var_types[i][j])
        outlog2.write('%s\n' % var_names[i][-1])
        outlog3.write('%s\n' % var_types[i][-1])
    outlog1.close()
    outlog2.close()
    outlog3.close()



if __name__ == '__main__':
    infile = 'hive_desc_out_2.csv'
    num_tables = 5
    [tables, table_meta, var_names, var_types] = parse_hive(infile, num_tables)
    write_results(tables, table_meta, var_names, var_types)
            
