import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np


import requests
from io import BytesIO

# Function to download the model from GitHub
@st.cache_resource
def download_model():
    url = 'https://github.com/MH-ML/teacher-searcing/raw/main/model.h5'
    response = requests.get(url)
    open('model.h5', 'wb').write(response.content)
    model = tf.keras.models.load_model('model.h5')
    return model

model = download_model()



# Define a dictionary mapping class indices to class names
class_names = {
    0: 'Asaduzzaman Shaon Junior Instructor (Civil-Wood)',
    1: 'Engineer Mohammad Mosharraf Hossain Chief Instructor (Construction)',
    2: 'Engineer Muhammad Tarekul Islam Chief Instructor (CM)',
    3: 'Engineer Rahmat Ullah Chief Instructor',
    4: 'Manas Badua Instructor (Electrical)',
    5: 'Mashrurul Arefin Instructor (Construction)',
    6: 'Md. Abdul Mannan Instructor (Physics)',
    7: 'Md. Abu Sayed Junior Instructor (Accounting)',
    8: 'Md. Fayezullah (Junior Instructor)',
    9: 'Md. Mahmudul Haqueb Junior Instructor (Mathematics)',
    10: 'Mohammad Abdul Matin Howlader Principal',
    11: 'Mohammad Gias Uddin Instructor (Mathematics)',
    12: 'Mohammad Iqbal Haider Instructor (CSE)',
    13: 'Mohammad Iqbal Hossain Instructor Civil',
    14: 'Mohammad Jewel Ahmed Physical Education Instructor',
    15: 'Mohammad Omar Faruq Chief Instructor',
    16: 'Pankaj Das Instructor (English)',
    17: 'Sajeda Yasmin Instructor (CSE)',
    18: 'Sanjibon Chandra De Instructor (Mathematics) Pending',
    19: 'Suman Chakma Suman Chakma',
    20: 'Tahidul Alam Junior Instructor (Mathematics)'
}


# Define details for each teacher
teacher_details = {
    'Asaduzzaman Shaon Junior Instructor (Civil-Wood)': {
        'Name': 'আসাদুজ্জামান শাওন',
        'Title': 'জুনিয়র ইন্সট্রাক্টর (সিভিল-উড)',
        'Faculty': 'সিভিল-উড',
        'Department': 'সিভিল-উড',
        'Email': 'shaondminor@gmail.com',
        'Batch (BCS)': '০',
        'Mobile': '০১৮৩৬৬৫৮৩৪১'
        
    },
    
    'Engineer Mohammad Mosharraf Hossain Chief Instructor (Construction)' : {
        'Name': 'প্রকৌশলী মোহাম্মদ মোশাররফ হোসেন',
        'Title': 'চিফ ইন্সট্রাক্টর (কন্সট্রাকশন)',
        'Faculty': 'কন্সট্রাকশন',
        'Department': '	কন্সট্রাকশন',
        'Email': 'mushacpi@gmail.com',
        'Batch (BCS)': '০',
        'Mobile': '০১৭১২১৫৩০৬৪'
        
    },
    
    'Engineer Muhammad Tarekul Islam Chief Instructor (CM)' : {
        'Name': 'প্রকৌশলী মুহাম্মদ তারেকুল ইসলাম',
        'Title': 'চিফ ইন্সট্রাক্টর (কম্পিউটার)',
        'Faculty': 'কম্পিউটার',
        'Department': '	কম্পিউটার',
        'Email': 'tareq.bspi@gmail.com',
        'Batch (BCS)': '০',
        'Mobile': '০১৮১১৮৮৮৩৪২'
        
    },
    'Engineer Rahmat Ullah Chief Instructor' : {
        'Name': 'ইঞ্জিনিয়ার রহমত উল্লাহ',
        'Title': 'চিফ ইন্সট্রাক্টর',
        'Faculty': 'অটোমোবাইল',
        'Department': '	অটোমোবাইল',
        'Email': 'engrrahamatullah@yahoo.com',
        'Batch (BCS)': '০',
        'Mobile': '০১৮১৪৯৫৯০৯৪'
        
    },
    
    'Manas Badua Instructor (Electrical)' : {
        'Name': 'মানস বডুয়া',
        'Title': 'ইন্সট্রাক্টর',
        'Faculty': 'ইলেকট্রিক্যাল',
        'Department': '	ইলেকট্রিক্যাল',
        'Email': 'manash74.bspi@gmail.com',
        'Batch (BCS)': '০',
        'Mobile': '০১৮১৫৮৫৩৩৩৮'
        
    },
    
    'Mashrurul Arefin Instructor (Construction)' : {
        'Name': 'মাশরুরুল আরেফীন',
        'Title': 'ইন্সট্রাক্টর',
        'Faculty': 'কন্সট্রাকশন',
        'Department': '	কন্সট্রাকশন',
        'Email': 'mashruruljami@gmail.com',
        'Batch (BCS)': '০',
        'Mobile': '০১৫১৫২২২৭৫৯'
        
    },
    'Md. Abdul Mannan Instructor (Physics)' : {
        'Name': 'মোঃ আবদুল মন্নান',
        'Title': 'ইনস্ট্রাক্টর (ফিজিক্স)',
        'Faculty': 'আনুষাঙ্গিক',
        'Department': '	আনুষাঙ্গিক',
        'Email': 'mannanbspi@gmail.com',
        'Batch (BCS)': '০',
        'Mobile': '০১৮১৬২৩৯২৭৯'
        
    },
    
    'Md. Abu Sayed Junior Instructor (Accounting)' : {
        'Name': 'মোঃ আবু সায়েদ',
        'Title': 'জুনিয়র ইনস্ট্রাক্টর (হিসাববিজ্ঞান)',
        'Faculty': 'আনুষাঙ্গিক',
        'Department': '	আনুষাঙ্গিক',
        'Email': 'sayedsandwip@gmail.com',
        'Batch (BCS)': '০',
        'Mobile': '০১৮৩৬৬৫১৪৪৬'
        
    },
    
    'Md. Fayezullah (Junior Instructor)' : {
        'Name': 'মোঃ ফয়েজউল্লাহ',
        'Title': 'জুনিয়র ইনস্ট্রাক্টর',
        'Faculty': 'অটোমোবাইল',
        'Department': '	অটোমোবাইল',
        'Email': 'powerfoyz93@gamil.com',
        'Batch (BCS)': '০',
        'Mobile': '০১৮৯০৬৭৪৯৩৯'
        
    },
    
    'Md. Mahmudul Haqueb Junior Instructor (Mathematics)' : {
        'Name': 'মোঃ মাহমুদুল হক',
        'Title': 'জুনিয়র ইনস্ট্রাক্টর (গনিত)',
        'Faculty': 'আনুষাঙ্গিক',
        'Department': '	আনুষাঙ্গিক',
        'Email': 'm08203025@gmail.com',
        'Batch (BCS)': '০',
        'Mobile': '০১৭৬২১৫৯২৫৯'
        
    },
    
    'Mohammad Abdul Matin Howlader Principal': {
        'Name': 'মোহাম্মদ আবদুল মতিন হাওলাদার',
        'Title': 'অধ্যক্ষ (অতিরিক্ত দায়িত্ব)',
        'Details': 'মোহাম্মদ আবদুল মতিন হাওলাদার ৩০ মে ২০১৯ সালে উপাধ্যক্ষ হিসেবে বাংলাদেশ সুইডেন পলিটেকনিক ইনস্টিটিউটে যোগদান করেন এবং ২০ জুন ২০১৯ অধ্যক্ষ (অতিরিক্ত দায়িত্ব) হিসেবে দায়িত্বভার গ্রহন করেন। এর পূর্বে তিনি রাজশাহী মহিলা পলিটেকনিক ইনস্টিটিউটে দায়িত্ব পালন করেছেন।',
        "Mobile" : '01827557761'

    },
    
    'Mohammad Gias Uddin Instructor (Mathematics)' : {
        'Name': 'মোহাম্মদ গিয়াস উদ্দীন',
        'Title': 'ইনস্ট্রাক্টর (গনিত)',
        'Faculty': 'আনুষাঙ্গিক',
        'Department': '	আনুষাঙ্গিক',
        'Email': 'mgbspi@gmail.com',
        'Batch (BCS)': '০',
        'Mobile': '০১৮১৯৬০৬১২১'
        
    },
    
    'Mohammad Iqbal Haider Instructor (CSE)' : {
        'Name': 'মোহাম্মদ ইকবাল হায়দার',
        'Title': 'ইনস্ট্রাক্টর',
        'Faculty': 'কম্পিউটার',
        'Department': '	কম্পিউটার',
        'Email': 'parveezapcu@gmail.com',
        'Batch (BCS)': '০',
        'Mobile': '০১৮১২৫৪৬৯৮২'
        
    },
    
    'Mohammad Iqbal Hossain Instructor Civil' : {
        'Name': 'মোহাম্মদ ইকবাল হোসেন',
        'Title': 'ইনস্ট্রাক্টর',
        'Faculty': 'সিভিল-উড',
        'Department': 'সিভিল-উড',
        'Email': 'iqbal.duet82@gmail.com',
        'Batch (BCS)': '০',
        'Mobile': '০১৮১৫৫০০৩২২'
    },
       
    'Mohammad Jewel Ahmed Physical Education Instructor' : {
        'Name': 'মোহাম্মদ জুয়েল আহমেদ',
        'Title': 'ফিজিক্যাল এডুকেশন ইনস্ট্রাক্টর',
        'Faculty': 'আনুষাঙ্গিক',
        'Department': 'আনুষাঙ্গিক',
        'Email': 'juwelahammedrs@gmail.com',
        'Batch (BCS)': '০',
        'Mobile': '০১৭১৭১০৫৯৩৩'
        
    },
    
    'Mohammad Omar Faruq Chief Instructor' : {
        'Name': 'মোহাম্মদ ওমর ফারুক',
        'Title': 'চিফ ইন্সট্রাক্টর (মেকানিক্যাল)',
        'Faculty': 'মেকানিক্যাল',
        'Department': 'মেকানিক্যাল',
        'Email': 'mfarruk2017@gmail.com',
        'Batch (BCS)': '০',
        'Mobile': '০১৬২৫৩৯৫৪৩৬'
        
    },
      
    'Pankaj Das Instructor (English)' : {
        'Name': 'পঙ্কজ দাশ',
        'Title': 'ইনস্ট্রাক্টর (ইংরেজী)',
        'Faculty': 'আনুষাঙ্গিক',
        'Department': 'আনুষাঙ্গিক',
        'Email': 'pangkajdasceitc@gmail.com',
        'Batch (BCS)': '০',
        'Mobile': '০১৬২২৭৯৬৯০৭'
        
    },
    
    'Sajeda Yasmin Instructor (CSE)' : {
        'Name': 'সাজেদা ইয়াসমিন',
        'Title': 'ইনস্ট্রাক্টর',
        'Faculty': 'কম্পিউটার',
        'Department': 'কম্পিউটার',
        'Email': 'symunni85@gmail.com',
        'Batch (BCS)': '০',
        'Mobile': '০১৮৫৫২৪৫৬৯৮'
        
    }, 
    
    'Sanjibon Chandra De Instructor (Mathematics) Pending' : {
        'Name': 'সঞ্জিবন চন্দ্র দে',
        'Title': 'ইনস্ট্রাক্টর (গনিত) স্টেপ থেকে রাজস্বখাতে প্রক্রিয়াধীন',
        'Faculty': 'আনুষাঙ্গিক',
        'Department': 'আনুষাঙ্গিক',
        'Email': 'sanjibanchdey@gmail.com',
        'Batch (BCS)': '০',
        'Mobile': '০১৭১১২৪৫০২০'
        
    },
    
    'Suman Chakma Suman Chakma' : {
        'Name': 'সুমন চাকমা',
        'Title': 'ইনস্ট্রাক্টর',
        'Faculty': 'ইলেকট্রিক্যাল',
        'Department': 'ইলেকট্রিক্যাল',
        'Email': 'sumon8.et@gmail.com',
        'Batch (BCS)': '০',
        'Mobile': '০১৭৯৩৯৩৬৩৩৭'
        
    }, 
    
    'Tahidul Alam Junior Instructor (Mathematics)' : {
        'Name': 'তহিদুল আলম',
        'Title': 'জুনিয়র ইনস্ট্রাক্টর (গণিত)',
        'Faculty': 'আনুষাঙ্গিক',
        'Department': 'আনুষাঙ্গিক',
        'Email': 'towhidul.cu1991@gmail.com',
        'Batch (BCS)': '০',
        'Mobile': '০১৮১৮৬১৬১১৫'
        
    },     
    # Add details for other teachers similarly...
}

# Function to preprocess an image
def preprocess_image(img):
    
    img_height, img_width = 224, 224
    img = img.resize((img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Function to predict the image class
def predict_image_class(img_array):
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_name = class_names.get(predicted_class_index, 'Unknown')
    return predicted_class_name

# Streamlit app
st.markdown(
    """
    <style>
    .reportview-container {
        background: #f0f0f5;
        padding: 20px;
        border-radius: 10px;
    }
    .sidebar .sidebar-content {
        background: #f0f0f5;
        border-radius: 10px;
    }
    .css-18e3th9 {
        background: #fff;
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 20px;
    }
    h1 {
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #444;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown(
    """
    ### Welcome to the BSPI Teacher's Information Searching System By Image!
    Upload an image of a teacher to discover their details instantly.
    - Supported formats: **JPG, JPEG, PNG**
    - The system will process the image and classify it immediately.
    
    ---
    
    **Explore More:** [Visit BSPI's Official Website](https://bspi.polytech.gov.bd/)
    """
)


# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Search Button

if uploaded_file is not None:
    search_buttton = st.button('Search Information')
    if search_buttton:
        with st.spinner("Searching Information.."):
            # Preprocess the image
            img = image.load_img(uploaded_file, target_size=(224, 224))
            img_array = preprocess_image(img)

            # Make a prediction
            predicted_class = predict_image_class(img_array)

            # Display the predicted details
            if predicted_class in teacher_details:
                details = teacher_details[predicted_class]
                st.image(img, caption="Uploaded Image", width=200)
                
                if predicted_class == 'Mohammad Abdul Matin Howlader Principal':
                    st.markdown(
                        f"""
                        **Name:** {details['Name']}

                        **Title:** {details['Title']}

                        **Details:** {details['Details']}
                        
                        **Mobile:** {details['Mobile']}
                        

                        """
                    )
                else:
                    
                    st.markdown(
                        f"""
                        **Name:** {details['Name']}  
                        **Title:** {details['Title']}  
                        **Faculty:** {details['Faculty']}  
                        **Department:** {details['Department']}  
                        **Email:** {details['Email']}  
                        **Batch (BCS):** {details['Batch (BCS)']}  
                        **Mobile:** {details['Mobile']}  
                        """
                    )
            else:
                st.warning("Details not available for the predicted class.")
else:
    st.markdown(
         """
        **Note:** Please upload an image to see the classification result.
        """
        )

st.sidebar.markdown(
    """
    ---
    **About the Model:**
    - The model is trained to recognize BSPI teachers from their images.
    
    **Developed by:**
    - Md.Azizul Hakim
    - Bangladesh sweden polytechnic institute
    - Department : CSE
    - Shift : 2nd
    - Semester : 5th
    - Roll : 688616
    """
)