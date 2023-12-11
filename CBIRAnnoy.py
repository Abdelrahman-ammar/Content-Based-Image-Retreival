import numpy as np
import tensorflow as tf
from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing import image
import os
from tensorflow.keras import Sequential,layers
from annoy import AnnoyIndex

ResnetModel = ResNet50(weights="imagenet",include_top=False,input_shape=(224,224,3))
model = Sequential([
    ResnetModel,
    layers.GlobalAveragePooling2D(), #2048
])

def extract_features(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = model.predict(img)
    return features.flatten()


def Cosine_similarity(img1,img2):
    dot_product = np.dot(img1,img2.T)
    img1_norm = np.linalg.norm(img1)
    img2_norm = np.linalg.norm(img2)
    return (dot_product/(img1_norm*img2_norm))



dataset_directory = "D:/test1/ImageRetreival/FLOWERS"

dataset_features = []
dataset_filenames = []
for filename in os.listdir(dataset_directory):
    image_path = os.path.join(dataset_directory, filename)
    features = extract_features(image_path)
    dataset_features.append(features)  
    dataset_filenames.append(filename)


num_trees = 10  # You can adjust this parameter for trade-off between speed and accuracy
feature_dim = len(dataset_features[0])
annoy_index = AnnoyIndex(feature_dim, 'angular') #angular is for cosine similarity metric

for i, feature in enumerate(dataset_features):
    annoy_index.add_item(i, feature)

annoy_index.build(num_trees)

query_image_path = "D:/test1/ImageRetreival/FLOWERS/daisies_00002.jpg"
query_features = extract_features(query_image_path)

top_n = 10 #specifiying the number of most similar neighbours to retreive
similar_indices = annoy_index.get_nns_by_vector(query_features, top_n)

for i, index in enumerate(similar_indices):
    similar_image_filename = dataset_filenames[index]
    similar_image_path = os.path.join(dataset_directory, similar_image_filename)
    # Calculate the similarity using Cosine Similarity
    similarity_score = Cosine_similarity(query_features, dataset_features[index])

    print(f"Similar Image {similar_image_path}: Cosine Similarity - {similarity_score}")