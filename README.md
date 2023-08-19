
# Deep Learned Image Representations for Efficient Retrieval

**UPDTATE**: This work is now available in the much more polished paper: [Matryoshka Representation Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c32319f4868da7613d78af9993100e42-Abstract-Conference.html) presented at NeurIPS 2022, for which I conducted the image search experiments!

Reverse image search is the problem of efficiently finding similar images to a given query image. Traditional approaches to this problem involve using metadata of each image in the database (eg. tags, descriptions, etc.) and matching the query images metadata to those in the database. However, in this project we propose the use of a deep-learned feature extractor that creates compact and semantically rich representations of images, then performs similarity search across those representations. We find that our approach achieves up to 64.7% MAP@50 on ImageNet with as few as 32 dimensions.

<!-- ## Video

Sorry in advance for the poor audio quality :(

[![Project Presentation](https://user-images.githubusercontent.com/28711661/146103636-4127bb6c-7044-4c68-8a53-11dd5805d605.png)](https://www.youtube.com/watch?v=DKBJ1tnALyY "Project Presentation") -->

## Introduction

Reverse image search engines allow users to find similar images to a given query image. You’ve probably used (or at least heard of) Google’s reverse image search feature in their search engine, which is very popular. However, there are many other applications of image search, ranging from finding similar looking garments, to seeing which celebrities look like you.

There are two key components to the problem of reverse image search: (i) high quality image representations, and (ii) an efficient search structure. We would like our image representations to capture some meaningful aspect of our images, or at least make similar images be close together, and dissimilar images be farther apart. Once we’ve found a good way to represent our images, we can compare those representations in some sort of data structure.

The search structure is only as accurate as brute force search over the representations, which involves computing the similarity (eg. cosine, or l2 distance) between the query vector and each vector in our database. This has a search time of O(dN), where d is the dimension of the representations and N is the number of datapoints (ie. images in our case). With some clever preprocessing, we can make a data structure that has query times of O(d logN) or lower!

In addition to query speed, there are three other main ways to measure the quality of a search engine: accuracy, construction time, and space. In this project, we hope to provide a proof of concept for using deep CNNs to extract compact, and flexible features for image retrieval.

## Related Work

There are two primary approaches to image retrieval.

The first is meta-data based image retrieval. This approach involves comparing the textual meta-data (eg. labels, or descriptions) between images, and finding the image with the most similar meta-data representation [1]. These methods tend to be fast because there are relatively few features to compare for each image. Additionally, each feature is semantically important by itself. This is very useful because it offers a tunable parameter for the search engine if it ever needs to trade accuracy for speed.
Unfortunately, acquiring such meta-data often requires manual curation for each image in the database, which can be prohibitively expensive for databases with billions of images. Moreover, it requires the query image itself to have textual meta-data as well, which is not desirable for our purposes.

The second approach is content-based image retrieval. This approach involves using computer vision techniques to extract features from the images, and then finding the images with the most similar extracted features. Traditionally, content-based features are "hand crafted" and things like average colors, color histograms, and edge point lists. More recent work has focused on the use of pre-trained deep learned models, such as VGG [2], to extract these features. When sufficient labelled training data is available, these methods show significant improvement in accuracy over methods that use hand-crafted features.

The benefits of using a textual meta-data representaion is that they provide quick comparisons and are easily tunable, however, these representations require information that we do not have access to. Content-based image retrieval does not require manually curated features, but can be significantly slower. Is there a way to get the best of both worlds?

## Approach

We generated content-based image representations using the ResNet50 architecture trained on ImageNet-1k. ResNet50 consists of 48 convolutional layers, 1 maxpool layer, and 1 average pool layer. Additionally, many layers have residual connections between them, that look like this:

![image](https://user-images.githubusercontent.com/28711661/146093832-3f183f35-41fb-4660-8cce-11b0e7d24a46.png)

These allow the gradient to flow from the later layers to the earlier ones, and in doing so alleviate gradient vanishing problems. In the second to last layer, an average pool is applied, producing a 2048 dimensional feature vector. Finally, the resulting features are passed through a fully connected layer to get probabilities for each class. We chose to use this architecture because it is close to state of the art in classification tasks. However, we could've used any deep convolutional neural network. What's important for our purposes is that together, all the layers before the linear layer act as a feature extractor for images. This property is common to most deep convolutional nerual networks.

We trained the network using a nested loss function defined as follows:

![image](https://user-images.githubusercontent.com/28711661/146094191-3a90e723-b5a1-4149-8d26-4e209219bf0e.png)

Where L_x is the loss computed using the first x dimensions of the vector produced after the average pool layer and then applying the linear layer to it. The purpose of this loss function is to incentivize the model to compress the most relevant features into the first dimensions. Moreover, loss functions of this sort are incredibly easy to implement, and could be used in almost any architecture.

Once we have the trained model, we can pass any image through it, and it will generate a 2048 dimensional vector where the first 16, 32, 64, etc. dimensions can all approximate the full 2048 dimensional representation. This allows for an image retrieval system to efficiently switch between comparing course or fine granularity features for tradeoffs in accuracy and query time that can be made on the fly.

In order to create an image retrieval system, we pass our entire database of images through this model to get feature representations, and then put the resulting representations into a data structure. We used three different data structures: KD-trees [5], LSH tables [6], and HNSW graphs [7]. Since the focus of this project is on learning the representations rather than the search structure, I will not explain how these work in depth, however, it is important to note that KD-trees find the exact nearest neighbors, while HNSW graphs and LSH tables perform approximate nearest neighbor search.


## Results

### ImageNet Classification
The results that inspired this project were the performance of the different sized representations on the classification task. In this experiment, we trained the model on ImageNet-1k and saw how different feature vector sizes fared in classifying on the ImageNet validation set. Recall, that we are learning all the representations at once, and that each representation uses the same values as the representation before. So for example, the 64 dimensional representations have the same first 32 coordinates as the 32 dimensional representation.

The table below shows the top 1 accuracy for different sized feature representations.

![image](https://user-images.githubusercontent.com/28711661/146094667-f75b70e8-9516-4493-b658-10f9548a8e44.png)

From this table, we can see that there is a clear tradeoff between size of representation and accuracy. This makes a certain amount of sense. After all, the more dimensions a model is allowed to use, the better it can capture the patterns in the data.

Note, that the returns on accuracy are incredibly diminishing with representation size. We can see that using the 16 dimensional representations achieves an accuracy that is about 3% less than that of 2048 dimensions, however, the 16 dimensional representation is over 2 orders of magnitude smaller than the full sized representation.


### ImageNet Retrieval

We realized that this tradeoff between quality of representation and size could be very useful in an image retrieval system. This is because query times are usually proportional to the dimensionality of the representations (because we have to use some compute for each dimension). So, we investigated how various data structures performed-- especially how query times vary -- with different sized representations.

In this experiment, we retrieved from a database of 1.28 million images (all of the ImageNet-1k training set). The data structures compared the model-made representations of these images to find a query image’s 100 nearest neighbors. We computed the accuracy by using the MAP@100 metric\*, we calculated how long it took on average per query, and estimated the size of the index by measuring how much space the pickled data structure took up on disk. The plots below show our results.

#### Accuracy

![image](https://user-images.githubusercontent.com/28711661/146094872-7d12ca45-25f2-4b9c-9b8a-e4e3a59493ef.png)

In the plot above, we measure accuracy in terms of MAP@100\* and see how it varies with different sized feature representations. The KD-tree performs ~1% better in MAP than HNSW graphs and ~3% better than LSH tables. This is expected since KD-trees use exact search, while HNSW graphs and LSH tables only find the approximate nearest neighbors. The stochasticity of LSH and HNSW is also responsible for the spiky curves that we see for both of those data structures since they introduce some noise into the classification.

This plot shows that the diminishing returns in accuracy we observed in *classification* is also present in image *retrieval*. We did expect more features to perform better in terms of image retrieval accuracy, however, the accuracy peaks for KD-trees at around 128 dimensions (and even earlier for the other data structures). This goes against our original hypothesis that more dimensions = better predictions. We suspect this is because 128 dimensions is sufficient for capturing the patterns in ImageNet data, and the rest of the dimensions are picking up noise.

#### Query time

![image](https://user-images.githubusercontent.com/28711661/146095115-0dac537a-db89-4af5-bcac-2bfa92b5c5f4.png)

These plots show query time as a function of dimensions used. The plot on the left compares all three data structures, and as you can see, LSH tables and HNSW graphs blow KD-trees out of the water in terms of search speed. This is because LSH and HNSW sacrifice accuracy for huge boosts in speed. For this reason, we would likely never use a KD-tree for image search (since we rarely care about having the most similar images, and just want decently similar images quickly). The plot on the left compares HNSW and LSH data structures. HNSW has the fastest query time by about an order of magnitude, especially for low dimensional representations.

For KD trees and HNSW tables, query time increases with dimensionality. This is because these data structures have asymptotic query times that are something like O(d logn). We don’t see a constant increase for LSH tables because LSH generates a fixed length bit representation (using random hashes) from the representations we give it. So, in all cases, LSH is comparing 256 bits (the number of hash functions we used), rather than 16, 32, 64, … real numbers. These plots provide further evidence for the hypothesis that using lower dimensional representations in search structures provides much faster queries.


#### Size of Index

![image](https://user-images.githubusercontent.com/28711661/146095244-7424d098-4bc9-41d9-97e6-b9e742fd619e.png)

For completeness we also present plots showing the size of the indexes (left) and the time it takes to construct each index (right). As one might expect, both increase roughly linearly with the number of dimensions (note that the x-axis is logarithmic).


#### Examples

We also present example queries and the nearest neighbors returned:

![image](https://user-images.githubusercontent.com/28711661/146095291-abb19311-6664-437c-85de-e5a40298f044.png)

As you can see, the images are visually quite similar. Moreover, they tend to contain the same objects. The query image on the right is of an oil filter, and all the nearest neighbors returned also contain oil filters. Some of the images even have the text “oil filter”, which would be very helpful for determining what that object actually is, if you didn’t know what the query image was a picture of (like me when I first saw the query image).

The query image on the left depicts jazz players busking. All images returned are of a group of people playing music, and three out of the five are a group of three jazz players busking. Also, notice that the backgrounds of the returned images are very different from that of the query image. This suggests that the featurizer selects features that are relevant to the main subject of the photo, rather than focusing on the background. This makes sense since our featurizer was trained with the objective of identifying objects, rather than locations.


\*In our experiments we evaluate accuracy using the MAP@100 metric similar to what’s presented in [4]. We compute the MAP by first computing average precision (AP@100) as follows:

![image](https://user-images.githubusercontent.com/28711661/146095429-9e67f528-3320-4893-9df0-277aef0efef4.png)

Where P@k is the precision (actual true positive divided by possible number of true positives) of the first k nearest neighbors, and rel(k) is an indicator that was 1 if the kth sample was relevant (in our case a neighbor was considered relevant if it was in the same class as the query vector). MAP@k just takes the mean AP@k for all the queries. In our ImageNet experiment we validated with 5 samples for each class, so a total of 5000 queries were made.


### Places Retrieval

In the previous retrieval pipeline, we use labelled image data to train a feature extractor, generated representations for that same image data to create an image retrieval system, then we tested the performance of that system using never-before-seen images from the ImageNet validation set. In the real world, we should not expect to have labels for all the images in our database. If we did, we could use a meta-data-based approach for image retrieval. A more realistic situation is that we train our feature extractor on a small subset of the images and classes that appear in our database. Then, we use the model to extract features of never-before-seen classes, and perform retrieval on those.

To test our content-based image retrieval system in this scenario, we trained the model on ImageNet-1k, generated representations for images (~1.8 million) from the Places365 dataset, then put those representations in a data structure for retrieval. We queried the data structure to find the 50 nearest neighbors using validation images from Places 365. Specifically, we used 3 randomly selected images from each class for a total of 1095 images as queries. The results for space and time are very similar to that of the ImageNet experiment. However, the accuracy is much worse:
 
![image](https://user-images.githubusercontent.com/28711661/146095625-6b7d3cf1-ba66-45bd-96db-77ea8242b475.png)

As you can see in the plot above, the MAP@50 caps out around 11%, which is ~50% MAP less than what we achieved on ImageNet. These are very bad results, and at first glance might seem to suggest that our technique is ill-suited for a realistic image retrieval system. However, examining the examples for the actual nearest neighbors indicates the source of these low MAP scores may have less to do with poor performance, and more to do with a domain shidft. Although we get very low MAP scores, the nearest neighbors for queries tend to be pretty visually similar. For example consider the nearest neighbors returned for this query image:

![image](https://user-images.githubusercontent.com/28711661/146096009-83a746f1-8b2a-4d77-aa90-fc8f3071caf6.png)

Although all these images are of different places-- and thus have different labels in the Places dataset-- they all depict a fence front and center. We see this in many other examples as well. This indicates that our metric is underestimating the quality of our retrieved images. Recall that the featurizer was trained on ImageNet, and Imagenet labels its images with objects, so the model learns to select features that distinguish objects and disregard the background. Places labels its images with places, and so a model that selects features that distinguish the backgrounds, rather than the primary objects (eg. fence, person) would likely perform much better.

Since the domains that these two datasets sample from (backgrounds vs objects) are so different, we believe that the low MAP scores do not reflect the general technique that we are proposing, and are instead the result of a difficult domain shift.

## Discussion

In this project we hoped to provide a proof of concept for an image representation technique that could be applied in a large-scale, content-based image retrieval system. We showed that our representations allowed for flexibility in choosing between query speed and query accuracy. This tradeoff could be very useful for retrieving from web-scale data. The retrieval system could use coarse granularity features (ie. 16-32 dims) to find a shortlist of nearest neighbors and then use more accurate representations (ie. 256-2048 dims) to find the nearest neighbor among those. The benefits of this technique on web-scale data remains to be tested, however. This would involve testing the retrieval performance on billions of images and millions of classes.

Before signing off, it's worth mentiioning that the accuracy metric we used essentially evaluated whether the returned neighbors were in the same class as the query image. This severely limited the notion of similarity that we were testing for because we did not consider intra-class or inter-class similarity. For example, if you query the system with an image of a siamese cat, then returning another siamese cat image should be a better (ie. more similar) result than returning some other cat, like a tiger. However, our MAP metric does not distinguish between images within the same class in this way.

For a small scale study with limited time, the MAP metric we used was a good choice. However, it would be interesting to design other accuracy measures that took into account the similarity between classes. One way we might do this is by using WordNet to group the labels of ImageNet into sub and super classes. Then, we could use these groupings to define a measure of class similarity by computing the distance from the query’s class to the predicted image’s class in the hierarchy. This metric would give us a better idea as to the quality of images retrieved.

## Acknowledgements

This project is based off research I am doing with Aditya Kusupati, Gary Geng, and Jackson Stokes in the RAIVN Lab at the University of Washington. All the code to generate plots, and performance metrics was written primarily by me with some help from Gary Geng. Most of the code in the nested model was from a previous research project done by Aditya. However, I adapted the model to generate image representations.


## Sources

[1] https://www.google.com/url?q=https://www.researchgate.net/publication/50366286_Metadata_for_Content-Based_Image_Retrieval&sa=D&source=docs&ust=1639102498913000&usg=AOvVaw29DBatE8sLSZgyuVFsE0nw

[2] https://www.ijstr.org/final-print/jun2020/Image-Retrieval-Using-Features-From-Pre-trained-Deep-Cnn.pdf

[3] https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=3320&context=sis_research

[4] https://arxiv.org/pdf/2106.01487.pdf

[5] https://en.wikipedia.org/wiki/K-d_tree

[6] https://en.wikipedia.org/wiki/Locality-sensitive_hashing

[7] https://arxiv.org/abs/1603.09320
