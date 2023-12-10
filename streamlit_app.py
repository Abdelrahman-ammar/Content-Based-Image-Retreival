import streamlit as st
import pickle
from FlowersRetreival import *
from FlowersRetreival import ResnetPreprocess , get_label
from FlowersRetreival import  GetModel
from FlowersRetreival import Build_tables
from FlowersRetreival import ANN , Cosine_similarity , query_image2 ,file_pathss
from PIL import Image


model = GetModel()

# with open('image_features.pkl', 'rb') as file:
#     loaded_features = pickle.load(file)

# with open('tables_data.pkl', 'rb') as file:
#     loaded_tables = pickle.load(file)


def folder_selector():
    st.sidebar.title("Select Folder")
    st.sidebar.subheader("Enter the folder name containing images")

    folder_name = st.sidebar.text_input("Folder Name")

    if folder_name:
        st.sidebar.write("Selected Folder:", folder_name)
        return folder_name
    return None

folder_name = folder_selector()

if folder_name:
    folder_path = f"{folder_name}/*"
    dataset = file_pathss(folder_path)
    with st.spinner("Extracting Features"):
        images_ds = dataset.map(ResnetPreprocess)
        images_ds = images_ds.batch(32)

        features = model.predict(images_ds)

        labels_ds = dataset.map(get_label)
        labels = []
        for label in labels_ds:
            labels.append(label.numpy().decode('utf-8'))
        

        tables = Build_tables(10,features[0].shape[0],8,features,labels)
     


def file_selector():
    st.title("Select a Query Image")
    st.subheader("Select a Query Image")

    query_image = st.file_uploader("Upload Query Image", type=["jpg", "jpeg", "png"])

    if query_image is not None:
        st.image(query_image, caption="Query Image", use_column_width=True)
        st.write("Image Path:", query_image.name) 
    return query_image

query_image = file_selector()
 
if query_image:
    query_image = f"{folder_name}/{query_image.name}"
    st.write(query_image)
    st.write("Perform actions with the selected query image...")
    num_neighbors = st.number_input("Enter the number of neighbors", min_value=1, value=5, step=1)
    if isinstance(query_image, str): 
            similar_images, feature_query = query_image2(query_image, tables,model)
    else:  
        similar_images, feature_query = query_image2(query_image.read(), tables)
        
    Nearest = ANN(similar_images,feature_query,num_neighbors,features,labels)

    st.write(f"Top {num_neighbors} Nearest Neighbor Images:")
    col_count = 3
    
    for i in range(0, len(Nearest), col_count):
        cols = st.columns(col_count)
        for j in range(col_count):
            idx = i + j
            if idx < len(Nearest):
                image_name = Nearest[idx][0]
                similarity_score = Nearest[idx][1][0][0]  # Extracting the similarity score

                image_path = f"{folder_name}/{image_name}" 
                img = Image.open(image_path)
                cols[j].image(img, caption=f"Name: {image_name}\nSimilarity: {similarity_score:.2f}", use_column_width=True)
else:
     st.write("No images")

