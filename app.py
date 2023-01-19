import streamlit as st
from transformers import pipeline


st.image('images/hslu.png', caption=None, width=300, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
st.title('Customer Feedback')
st.write('HSLU Course Digital Customer Management')

form = st.form(key='sentiment-form')
user_input = form.text_area('Enter your text')
submit = form.form_submit_button('Submit')


classifier = pipeline("sentiment-analysis")    
result = classifier(user_input)[0]    
label = result['label']    
score = result['score']

if submit:
    classifier = pipeline("sentiment-analysis")
    result = classifier(user_input)[0]
    label = result['label']
    score = result['score']
if label == 'POSITIVE':
        st.success(f'{label} sentiment (score: {score})')
        st.image('images/happy.png', caption=None, width=300, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

else:
    st.error(f'{label} sentiment (score: {score})')
    st.image('images/angry.png', caption=None, width=300, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
