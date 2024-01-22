# Community-Structures-in-Facebook-Social-Networks

This repository contains datasets, analytical results, and Jupyter notebooks related to the exploration of community structures within Facebook social networks. Each component is aimed at uncovering the intricate connections and clustering patterns that define social circles on Facebook.

## Directory explanations

/facebook: This directory hosts data from 10 distinct ego networks extracted from Facebook. Each ego network is a snapshot of a user's immediate social circle, encapsulating the complex web of friendships and social ties.

/results: Contains 20 CSV files representing the outcomes of community detection algorithms. Each file details the results of processing individual ego networks with two distinct approaches: using only node features and using only the graph structure.

main.ipynb: The central Jupyter notebook containing the core algorithms and methodologies applied for community detection within the Facebook networks.

Combined_Graph_Features.ipynb: A Jupyter notebook that presents a hybrid model, integrating both the structural information of the graph and the features of the nodes to identify community structures.

Understanding_data.ipynb: A visual and analytical guide that provides insights into the datasets. It includes visualizations that help in comprehending the composition and characteristics of the Facebook ego networks.