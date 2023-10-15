import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

adelie = pd.read_csv('adelie.csv')
adelie['Species'] = 'Adelie'
gentoo = pd.read_csv('gentoo.csv')
gentoo['Species'] = 'Gentoo'
chinstrap = pd.read_csv('chinstrap.csv')
chinstrap['Species'] = 'Chinstrap'



penguins = pd.concat([adelie,gentoo,chinstrap])[['Species','Flipper Length (mm)','Culmen Length (mm)']]



penguins.columns = ['Species','Flipper Length','Bill Length']



for penguin in ['Adelie','Gentoo','Chinstrap']:
    plt.scatter(penguins[penguins['Species'] == penguin]['Flipper Length'],penguins[penguins['Species']==penguin]['Bill Length'],alpha=.75,linewidths=.5,label=penguin)
plt.legend()
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Bill Length (mm)')
plt.title('Penguin Flippers and Bills by Species')




km_model = KMeans(n_clusters = 3, random_state=17).fit(penguins[['Flipper Length','Bill Length']].dropna().values)
def predict(flipper,bill):
  species_name = {0:'Adelie',1:'Gentoo',2:'Chinstrap'}
  return 'Prediction: this penguin belongs to the ' + species_name[list(km_model.predict([[flipper,bill]]))[0]] + ' species!'

fl = st.text_input("Flipper Length")
bl = st.text_input("Bill Length")
bt = st.button("Enter")
if (bt):
   text = predict(int(fl),int(bl))
   st.write(text)
   st.pyplot(plt)
