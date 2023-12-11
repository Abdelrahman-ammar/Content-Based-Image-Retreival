# Content-Based-Image-Retreival - Using Aproximate Nearest Neighbours
## Table of Contents 

* [What is this project](https://github.com/Abdelrahman-ammar/Content-Based-Image-Retreival/blob/master/README.md#What-is-this-project)

* [Details](https://github.com/Abdelrahman-ammar/Content-Based-Image-Retreival/blob/master/README.md#Details)

* [Project Structure](https://github.com/Abdelrahman-ammar/Content-Based-Image-Retreival/blob/master/README.md#Project-Structure)


### What is this Project
This project is a Content Based Image Retreival system or it can be called 
Near Duplicated Image Search where it finds from the database or dataset the most similar images based on the query image you are searching for and returns the top number of images you specify.

## Details
The project core is based on (ANN) Aproximate Nearest Neigbours techniques and algorithms <br> 
<br>
The project is implemented on 2 techniques which is LSH (Locality Sensitive Hashing) and Annoy (Approximate Nearest Neighbours Oh Yeah) from Spotify


## Project Structure 
- [FLOWERS](./FLOWERS/) : Contains the dataset where I trained the 2 versions

- [FlowersRetreival](./FlowersRetreival.py/) : Project Version using LSH , contains also Functions and classes used in this project 
    - [Streamlit App](./streamlit_app.py) : Deployment using Streamlit and functions in the [FlowersRetreival](./FlowersRetreival.py)

- [CBIR Annoy](./Annoy.ipynb) : Project Version using the Annoy library from Spotify 

## Notes

Both 2 techniques uses the Cosine Similarity as a metric of distance
<br> 
 To use the GUI write in ```streamlit run streamlit_app.py```

