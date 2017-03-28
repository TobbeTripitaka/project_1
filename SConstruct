import os, sys, urllib

#import mina functioner
from tobbes_handy_functions import write_to_latex, import_csv etc

#import chenagepoint detection
from foo_dir import *

env = Environment(ENV = os.environ)

#Speed up detection of outdated dependencies
env.Decider('MD5-timestamp')


#Function that write LaTeX textfile
def to_tex(label,string='\\textcolor{red}{(No data) }'):
    with open('tex/variables.tex', "a") as myfile:
        myfile.write('\\def \\%s {%s} \n'%(str(label),str(string)))




env.Command(
    source = None,
    target = None,
    action = foo_dir.foo_function,
    a = 1,
    b = 2,
    )
    





















#Exemple of naive detection
in_naive = ['ga_vpt_line_2', 'ga_vpt_line_15', 'ga_vpt_line_16']
#to_tex('NaiveFile',in_naive)

for naive in in_naive:
    env.Command('fig/naive_%s.pdf'%naive ,'data/%s.csv'%naive,'python code/naive_plot.py < $SOURCE > $TARGET')
    Depends('fig/naive%s.pdf'%naive , 'code/naive_plot.py')


#Exemple of online change-point detection

to_tex('conclusion')


#Run offline change-point detection

subsample = 50

file_names = ['ga_vpt_line_2', 
    'ga_vpt_line_12', 
    'ga_vpt_line_13', 
    'ga_vpt_line_14', 
    'ga_vpt_line_15', 
    'ga_vpt_line_16']

for file_name in file_names:
    env.Command(['fig/%s.pdf'%file_name, '%s_testet.csv'%file_name], 'data/%s.csv'%file_name, 
        'python code/offline_m.py < $SOURCE > $TARGET -i %s -s %s'%(file_name, subsample))         
    Depends('fig/%s.pdf'%file_name, 'code/offline_m.py')

    
    
#env.Command(["bar1.dat", "bar2.dat"], "dep.par", "python foo.py --parameters=dep.par")

# Faster subsampled    
#subsample_lo = 500
#file_names_lo = ['ga_vpt_line_2', 
#    'ga_vpt_line_12', 
#    'ga_vpt_line_13', 
#    'ga_vpt_line_14', 
#    'ga_vpt_line_15', 
#    'ga_vpt_line_16']


#for file_name in file_names_lo:
#    env.Command('fig/%s_lo.pdf'%file_name, 'data/%s.csv'%file_name, 
#        'python code/offline_m.py < $SOURCE > $TARGET -i %s -s %s'%(file_name, subsample_lo))         
#    Depends('fig/%s_lo.pdf'%file_name, 'code/offline_m.py')
    
 

#Exclude radiomtric data
#file_names_no_radio = ['ga_vpt_line_2', 'ga_vpt_line_15', 'ga_vpt_line_16']

#for file_name in file_names_no_radio:
#    env.Command('fig/no_radio_%s.pdf'%file_name, 'data/%s.csv'%file_name, 
#        'python code/offline_m.py < $SOURCE > $TARGET -i no_radio_%s -s %s -n "Aus_Ka_ga@ Aus_u_ga@2 Aus_Ka_ga@ Aus_Th_ga@ Aus_Th_ga@" '%(file_name, subsample))        
#    Depends('fig/no_radio_%s.pdf'%file_name, 'code/offline_m.py')
  
#for file_name_no_lo in file_names_no_radio:
#    env.Command('fig/no_radio_%s_lo.pdf'%file_name, 'data/%s.csv'%file_name, 
#        'python code/offline_m.py < $SOURCE > $TARGET -i no_radio_%s -s %s -n "Aus_Ka_ga@ Aus_u_ga@2 Aus_Ka_ga@ Aus_Th_ga@ Aus_Th_ga@" '%(file_name, subsample_lo))        
#    Depends('fig/no_radio_%s_lo.pdf'%file_name, 'code/offline_m.py')
  
 
# Export pdf 
 
 
 
#env.PDF('tex/change_point.pdf', 'tex/change_point.tex')
#env.Command('result/change_point.pdf', 'tex/change_point.pdf', Copy("$TARGET", "$SOURCE"))







#Todo


#Better way to pass parameters to subscripts
#Functions in separate files
# Pass values to Latex, 
#Command std out/in 
#   In: data, pythonscript
#   Out: figure(s), csv, tex, values
        


#Parameter list to numpy
#Boolean column with Nan 
#Column with Terraine name
#top=sum(diff(var()))
#bottom=sum(diff(mean()))
#change = top-bottom

#plot.top
#plot.-bottom
#color_fill = change


#Skip al indexnr and use couln names insted


#Boolean column, no data
#if no data:
#    interpolate
#    plot vline


#If df['DESC][i-1] != df['DESC][i]:
#    vline i
#    print df['DESC'][i]

