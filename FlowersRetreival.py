import numpy as np
import tensorflow as tf
from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing import image
import os
from numpy.linalg import norm
from tensorflow.keras import Sequential,layers
import matplotlib.pyplot as plt
import pickle

def GetModel():
    ResnetModel = ResNet50(weights="imagenet",include_top=False,input_shape=(224,224,3))
    model = Sequential([
        ResnetModel,
        layers.GlobalAveragePooling2D(), #2048
    ])
    return model

# model = GetModel()

def file_pathss(path):
    return tf.data.Dataset.list_files(path,shuffle=False)

# file_paths = file_pathss("Flowers/*")

def ResnetPreprocess(filepath):
    
    image = tf.io.read_file(filepath)
    
    image = tf.image.decode_jpeg(image, channels=3)
    
    image = tf.image.resize(image,(224,224))
    image = preprocess_input(image)
    
    return image

# images_ds = file_paths.map(ResnetPreprocess)
# images_ds = images_ds.batch(32)


# features = model.predict(images_ds)
# print("Features Extracted")

def get_label(filepath):
    return tf.strings.split(filepath, os.path.sep)[-1]

# labels_ds = file_paths.map(get_label)
# labels = []
# for label in labels_ds:
#     labels.append(label.numpy().decode('utf-8'))

# print("Extracting Labels")

def hash_func(embedding, random_vectors):
    embedding = np.array(embedding)
    
    bools = np.dot(embedding, random_vectors.T) > 0

    return [bool2int(bool_vec) for bool_vec in bools]


def bool2int(x):
    y = np.sum(2**np.where(x)[0])
    return y

def create_projections(dimensions,size,seed):
    np.random.seed(seed)
    return np.random.normal(size=(size,dimensions))


def create_table(dimensions,hash_size,vectors,labels,seed):
    table = {}
    random_projections = create_projections(dimensions,hash_size,seed)
    hashes = hash_func(vectors,random_projections)
    for i, hash_code in enumerate(hashes):
        if hash_code in table:
            table[hash_code].append(labels[i])
        else:
            table[hash_code] = [labels[i]]
    
    return table


def Build_tables(numtables,dimensions,hash_size,vectors,labels):
    tables = []
    for i in range(numtables):
        tables.append(create_table(dimensions,hash_size,vectors,labels,seed=i))
    return tables



# tables = Build_tables(10,2048,8,features,labels)

# print("Build Tables")

def query_image2(file_path, tables,model):
    
    image = ResnetPreprocess(file_path)
    feature_image = model.predict(tf.expand_dims(image, axis=0))
    
    similar_results = {}
    added_images = set()
    
    for i,table in enumerate(tables):
        random_projections = create_projections(2048,8,i)
        bucket = hash_func(feature_image, random_projections)[0]
        
    
        if bucket in table:
            for img in table[bucket]:
                if img not in added_images:
                    if bucket in similar_results:
                        similar_results[bucket].append(img)
                    else:
                        similar_results[bucket] = [img]
                    added_images.add(img)
    return added_images,feature_image

# similar_images,feature_query =query_image2("FLOWERS/daisies_00002.jpg",tables)


def Cosine_similarity(img1,img2):
    dot_product = np.dot(img1,img2.T)
    img1_norm = np.linalg.norm(img1)
    img2_norm = np.linalg.norm(img2)
    return (dot_product/(img1_norm*img2_norm))

def ANN(similar_images, QueryFeature, topn,features,labels):
    similarities = []
    
    QueryFeature = QueryFeature.reshape(1,-1)
    
    for image in similar_images:
        ff = features[labels.index(image)]
        ff_2d = ff.reshape(1,-1)
        
        similarity = Cosine_similarity(ff_2d,QueryFeature)
        similarities.append((image , similarity))
    
    similarities.sort(reverse=True,key=lambda x:x[1])
    top_similar_images = [(image_label , simil*100) for image_label,simil in similarities[:topn]]
    return top_similar_images

# with open('tables_data.pkl', 'wb') as file:
#     pickle.dump(tables, file)

# with open('image_features.pkl', 'wb') as file:
#     pickle.dump(features, file)

# Nearest = ANN(similar_images,feature_query,15)    
# print(Nearest)
