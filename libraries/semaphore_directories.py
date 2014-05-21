#First, we read the directories from the config file into python:
import re, os, inspect

here =os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print here

directories={line.split('=')[0]:line.split('=')[1] for line in open('/opt/semafor-semantic-parser/release/config', 'r').read().split('\n') if '=' in line and 'echo' not in line}

root=directories['SEMAFOR_HOME']

release=root + '/release'

output= root + '/samples/output.txt'

model_dir=root+'/models'

#I want to eventually get around to replacing the MST-parser with the Stanford parser
#(as I need the Stanford tools for my work anyway and I don't want to double parse things).

mst_parser=directories['MST_PARSER_HOME']

mst_port=directories['MST_PORT']

mst_machine = directories['MST_MACHINE']

temp_dir=re.sub(r'\$\{.*\}', str(directories['SEMAFOR_HOME']), directories['TEMP_DIR'])

gold_target=directories['GOLD_TARGET_FILE'].capitalize() if directories['GOLD_TARGET_FILE'].capitalize() !='Null' else None

java_home_bin=directories['JAVA_HOME_BIN']

use_graph_file= True if directories['USE_GRAPH_FILE']=='yes' else False

decoding_type= directories['DECODING_TYPE']
