import streamlit as st
from final import generate, translate_text

image, test = st.beta_columns(2)
with(image):
    image = st.empty()
with(test):
    with(st.form('myform')):
        input_text = st.text_input("Ведите текст", 'Большая черная птица')
        submit = st.form_submit_button(label='Отправить')

if submit:
    if input_text == "DIO":
        image.image('Dio.png', caption=input_text)
    else:
        translated = translate_text(input_text)
        img = generate(translated)
        image.image(img, caption=translated)
