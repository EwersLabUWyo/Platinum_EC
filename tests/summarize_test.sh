dir='/Users/waldinian/Project_Data/Platinum_EC/BB-SF/EC/7m/Converted'
glob='TOA5*10Hz*dat'
out='/Users/waldinian/Project_Data/Platinum_EC/BB-SF/EC/7m/summary.pickle'
variable_names='U=Ux_CSAT3B;V=Uy_CSAT3B;W=Uz_CSAT3B;Ts=Ts_CSAT3B;CO2=rho_c_LI7500;H2O=rho_v_LI7500;P=P_LI7500'

summmarization_script='../src/python/ecprocessor/summarization_script.py'

python '../src/python/ecprocessor/summarization_script.py' -d $dir -g $glob -o $out --variable_names $variable_names
