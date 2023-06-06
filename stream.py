import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from scipy.signal import find_peaks
import streamlit as st
import plotly.express as px
from plotly.figure_factory import create_distplot
h1 = '''<center><h1 style="color:blue;"><span style="color:green;">Extract</span> Critical Current</h1></center>'''
h2 = '''<center><h2>Upload <span style="color:rgb(9, 0, 128);background-color: yellow;">.csv</span> file obtained from the <span style="color:rgb(9, 0, 128);background-color: yellow;">.bin</span> file.<br> <span style="color:red">⚠️Do not upload any other csv file.⚠️</span></h2></center>'''
st.markdown(h1,unsafe_allow_html=True)
file = st.file_uploader("Upload .csv file here", type={"csv", "txt"})

if file is not None:
    data = pd.read_csv(file).astype(float)
else:
    st.markdown(h2,unsafe_allow_html=True)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def nearest(lst, k):
    mn = list(abs(lst - k))
    mn_min = np.min(mn)
    return mn.index(mn_min)
    

def find_ic_from_histogram(IC_dict,binsize):
    IC_MAX = {}
    for i in IC_dict.keys():
        n,ic = np.histogram(IC_dict[i],bins = binsize)
        n = list(n)
        ic = list(ic)
        IC_MAX.update({i:ic[n.index(max(n))]})
    return IC_MAX

current_folder = os.getcwd()
try:
    time = data['Time']
    current = data['Current'] - data['Current'].mean()
    voltage = data['Voltage'] - data['Voltage'].mean()
    dv_di = data['dV/dI (mag)']


    plt.plot(voltage,current,label='I-V')
    plt.scatter(np.mean(voltage),np.mean(current),label='Mean i,v',color='red')
    plt.xlabel('Voltage')
    plt.ylabel('Current')
    plt.legend()
    plt.grid()
    st.pyplot(plt.gcf())
    plt.close()

    VTH_PLUS = st.number_input('Enter +VTH(V)',format = "%.7f")
    VTH_MINUS = st.number_input('Enter -VTH(V)',format = "%.7f")
    st.write('+Vth = ', VTH_PLUS,'Volts and -Vth = ', VTH_MINUS,'Volts')
    peaks, _ = find_peaks(current, height=0)
    T = []
    for i in range(len(peaks)-1):
        del_peak = peaks[i+1] - peaks[i]
        T.append(del_peak/4)
    T = int(np.mean(T))
    st.write('Number of datapoints in each phase =',T)
    
    st.divider()
    
    st.markdown('<h5><span style="color:red">Set</span> Histogram Parameters:</h5>',unsafe_allow_html=True)
    binsize= st.slider(
            'Select bins number',
            10,3000,10)
    st.write('Bins:', binsize)
    
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    curr = []
    volt = []


    
    for i in range(0,int(len(current)/T)):
        #print(i*25,(1+i)*25);
        curr.append(current[i*T:(1+i)*T])
        volt.append(voltage[i*T:(1+i)*T])
        
        
    curr = np.array(curr) 
    volt = np.array(volt)
    IC_list = []




    count = 1
    IC_dict = dict({1:[],
                    2:[],
                    3:[],
                    4:[],
        })
    for i in range(len(curr)):
        if count > 4:
            count = 1

        
        if np.mean(volt[i]) > 0:
            vth = VTH_PLUS
        else:
            vth = VTH_MINUS

        IC = curr[i][nearest(volt[i],vth)]

        IC_dict[count].append(IC)
        
        count = count+1
        IC_list.append(IC)
        
    st.divider()
    option = st.selectbox('Plotting Tool',('Matplotlib','Plotly'))
    if option == 'Matplotlib':
        agree = st.checkbox('Show Distribution')
        if agree:
            sns.histplot(IC_dict,bins=binsize,kde=True)
        else:
            sns.histplot(IC_dict,bins=binsize,kde=False)   
        st.pyplot(plt.gcf())
    else:
        import plotly.express as px
        from plotly.figure_factory import create_distplot
        fig = px.histogram(IC_dict,nbins = binsize,color_discrete_sequence=["red", "green", "blue", "goldenrod"])
        fig.update_layout(yaxis_title="Count") 
        fig.update_layout(xaxis_title="Critical Current") 
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    IC_max = pd.DataFrame(find_ic_from_histogram(IC_dict,binsize),index=['Most frequent occurance'])
    
    save_IC = pd.DataFrame(IC_dict)
    st.markdown('<h5><span style="color:red">Ic</span> for each phase:</h5>',unsafe_allow_html=True)
    st.dataframe(save_IC.style.format("{:.8f}"),use_container_width=True)

    now = datetime.now()
    @st.cache_data
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')
    st.download_button(
    label="Download Ic as CSV",
    data=convert_df(save_IC),
    
    file_name = now.strftime("IC__%d_%m_%Y__%H_%M_%S_.csv"),
    mime='text/csv',
    )
    st.markdown('<h5><span style="color:red">Max values Of Ic</span> for each phase obtained from Histogram:</h5>',unsafe_allow_html=True)
    st.write(' ')
    st.dataframe(IC_max.style.format("{:.8f}"),use_container_width=True)
except NameError:
    foot = '''<center><h2><span style="color:blue">No File Uploaded or Some Error occured</span></h2></center>'''
    st.markdown(foot,unsafe_allow_html=True)


st.divider()

footer = '''<center style="color:blue">

<a href="https://github.com/ritartha/critical_current_extract_v1.git">
  <center>  <img src="https://icons.iconarchive.com/icons/limav/flat-gradient-social/48/Github-icon.png" height=50 align="center"></center>
  Github link</a></center>
  ''';
st.markdown(footer,unsafe_allow_html=True)
