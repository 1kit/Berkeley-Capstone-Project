<center>
    <center>
        <img src = images/classification.jpg width = 70%/>
    </center>
</center>

# <center>SUPERVISED MACHINE LEARNING BASED IP NETWORK TRAFFIC CLASSIFICTION TO DETECT APPLICATIONS</center>

**<center>Venkit</center>**

## Business Understanding
In network traffic classification, it is important to understand the correlation between network traffic and its causal application, protocol, or service group, for example, in facilitating lawful interception, ensuring the quality of service, preventing application choke points, and facilitating malicious behavior identification.
Recently, the utilization of machine learning algorithms has increased in the literature to classify network traffic without the need to access packets’ port numbers or content. It extracts statistical features that represent the behavior of a specific protocol or application flows, which are used for establishing the solution. Supervised, unsupervised, and semi-supervised machine learning frameworks have shown the effectiveness of the traffic classification process, where each one of them has its own strength and weakness.

## Rationale
Large businesses and organizations could potentially use these techniques to detect and prevent cyber-attacks, monitor utilization of networks by various applications and ensure Qualify of Service. This has the potential to save a lot of money and also secure such organizations from malicious cyber attacks.

## Project overview and goals
The goal of this project is to identify effective ways to classify IP network traffic so as to detect applications that are using such packets. 
There are various techniques used for classification. In this project, we want to use statistical features in conjunction with Supervised Machine Learning to classify the causal application.
The goal is to effectively find and train a model that not only increases classification accuracy, but also high precision and recall.

Statistical classification solution relies on the statistical feature extraction in combination with machine learning algorithms when classifying network traffic. 

## Research Question
Can we train a ML model on a time series data that has packet captures and other meta information like packet size, timing between packets, packet header information etc. 

This kind of model would be very useful to identify what type of applications are represented in these packet flows (Example: facebook, google, zoom video etc). This information can be used for generating reports on historic data and performing forensics. It could also be used to identify the traffic and improve SLAs

## Data Sources
The dataset used in this project is sourced from Kaggle and can be accessed [here](https://www.kaggle.com/datasets/jsrojas/ip-network-traffic-flows-labeled-with-87-apps)

The data was collected in a network section from Universidad Del Cauca, Popayán, Colombia by performing packet captures at different hours, during morning and afternoon, over six days (April 26, 27, 28 and May 9, 11 and 15) of 2017. A total of 3.577.296 instances were collected and are currently stored in a CSV (Comma Separated Values) file.

The flow statistics (IP addresses, ports, inter-arrival times, etc) were obtained using [CICFlowmeter](http://www.unb.ca/cic/research/applications.html) ([github](https://github.com/ISCX/CICFlowMeter)). 

The application layer protocol was obtained by performing a DPI (Deep Packet Inspection) processing on the flows with [ntopng](https://www.ntop.org/products/traffic-analysis/ntop/) ([github](https://github.com/ntop/ntopng)).

## Methodology
The industry standard CRISP-DM methodology was followed during this project

### Data Understanding
The first step is the representation of the network traffic in the form of flows, which is the aggregation of packets that share the 5-tuples; source and destination IP addresses, source and destination port numbers, and TCP or UDP protocol. 

The second step is the extraction of statistical features at the packet level or flow level. Packet level feature extraction is conducted over single or aggregate packets, resulting in features as packet length and inter-arrival time. Flow level feature extraction occurs on the entire flow, resulting in features as the total number of packets, total bytes, and the flow duration. 

The dataset has many statistical features at a flow level. Here are some major ones

- Flow duration
- Total number of packets in forward and backward direction
- Total length of packets in forward and backward direction
- Max/Min/Mean/Std-dev packet length in forward and backward direction
- Bytes per second and Packets per second data in flow
- Max/Min/Mean/Std-dev of inter-arrival time for flows.
- Header length data
- Packet flag metadata associated with various flows
- Statistics about size/number of packets sent during initial window in either directions

Following salient points were observed about the dataset

- In all there were `87` features recorded for each flow with a total of `3.5M` flows recorded
- There were no missing information
- `78` unique applications were recorded in the `ProtocolName` column of the the dataset.
- The data is highly unbalanced. Here is a bar plot 
<p align="left">
<img src = images/top20-before-bar.png width = 70%/>
</p>

### Data Preparation
The first step was to make the data balanced with respect to top 20 applications. Based on visual inspection, a random sample of `min( 10000, count )` data from the top 20 applications were taken.
A new pandas data frame from the above was constructed. This led to the following bar plot

<p align="left">
<img src = images/top20-after-bar.png width = 70%/>
</p>

This looks a lot more balanced now.

In addition the following modifications were performed

* A `LabelEncoder` was used to encode the `ProtocolName` column and a new `ProtocolLabel` column was created
* Remove the following unwanted columns (only retain statistically significant features)
    - Flow.ID
    - Source.IP
    - Source.Port
    - Destination.IP
    - Destination.Port
    - Label
    - Timestamp
    - Source.Port
    - Protocol
    - L7 Protocol
    - Flow Duration

* Removed columns that only had 0 values
* Scale the data using `StandardScaler`
* The data was randomly split into train and test sets to facilitate holdout cross-validation with a test size of 20%.

At this point the train and test data has the shape 
`Train:((153714, 50)` 
`Test:(38429, 50))`

### Feature Engineering
The next step was to examine the correlation matrix and heatmap. 
<p align="left">
<img src = images/heatmap.png width = 100%/>
</p>

#### Observation from correlation heatmap

Some of the features are obviously correlated
- For example most of the packet length related features correlate highly. This is not surprising.
- Similarly some of the IAT values (Bwd and Flow IAT.Max) correlate highly. This is also expected.

Some features are highly negatively correlated
- One example is the ACK.Flag.Count to various packet lengths. Here also, there is no surprise, because as packet length increases, the number of acks published for each packet decreases and vice-versa. 

**NOTE: Outside of the above obvious observations, there was nothing much to note.**

The following two feature engineering was performed
1. Using Pricipal Component Analysis (PCA) the dimension was reduced to 10 features
2. Feature selection by sorting the values of Itner Quartile Range (IQR) for all the features. The top 30 features were taken.

A training and holdout test set was obtained for both of the above feature engineering techniques.

### Modeling
The training datasets generated in the previous step was used to build models that classifies the network flows into their causal applications/protocols. The following ML algorithms were evaluated as classifiers.

1. Logistic Regression
2. Decision Tree
3. Support Vector Machines (SVM)
4. K-Nearest Neighbors (KNN)
5. Gaussian Naive Bayes
6. Random Forest Classifier
7. XGBoost Classifier
8. Dummy Classifier

### Evaluation 
The models evaluated were all multi-class classification algorithms. The following five common metrics are used to avaluate a built classifier for a supervised classification problem

1. Accuracy: 

    Indicates the overall accuracy of the model by dividing the correctly classified flows over the total number of flows in a dataset.
    ```
    Accuracy =      #TP + #TN
               --------------------- 
               #TP + #TN + #FP + #FN
    
    #TP = No. of True Positives
    #TN = No. of True Negatives
    #FP = No. of False Positives
    #FN = No. of False Negatives
    ```

2. Precision
    ```
    Precision =       #TP 
                   ----------
                    #TP + #FP
    ```


3. Recall

    ```
    Recall =          #TP 
                   ----------
                    #TP + #FN
    ```
4. F-measure
    ```
    F-measure = 2 x Precision x Recall
                    ------------------
                    Precision + Recall
    ```
5. Receiver Operator Characteristic (RoC) curve

    This represents a tradeoff between precision and recall. ROC curves typically feature true positive rate (TPR) on the Y axis and false positive rate (FPR) on the X axis.
    We have not used this metric in this project because ROC curves are typically used in binary classification, where the TPR and FPR can be defined unambiguously.

### Results

#### 1. Logistic Regression
<p float="left">
<img src = images/lr-pca.png width = 45%/>
<img src = images/lr-corr.png width = 45%/>
</p>
<p float="left">
<img src = images/lr-pca-t.png width = 45%/>
<img src = images/lr-corr-t.png width = 45%/>
</p>

#### 2. Decision Tree
<p float="left">
<img src = images/dt-pca.png width = 45%/>
<img src = images/dt-corr.png width = 45%/>
</p>
<p float="left">
<img src = images/dt-pca-t.png width = 45%/>
<img src = images/dt-corr-t.png width = 45%/>
</p>

#### 3. Support Vector Machines (SVM)
<p float="left">
<img src = images/svm-pca.png width = 45%/>
<img src = images/svm-corr.png width = 45%/>
</p>
<p float="left">
<img src = images/svm-pca-t.png width = 45%/>
<img src = images/svm-corr-t.png width = 45%/>
</p>

#### 4. K-Nearest Neighbors (KNN)
<p float="left">
<img src = images/knn-pca.png width = 45%/>
<img src = images/knn-corr.png width = 45%/>
</p>
<p float="left">
<img src = images/knn-pca-t.png width = 45%/>
<img src = images/knn-corr-t.png width = 45%/>
</p>

#### 5. Gaussian Naive Bayes
<p float="left">
<img src = images/nb-pca.png width = 45%/>
<img src = images/nb-corr.png width = 45%/>
</p>
<p float="left">
<img src = images/nb-pca-t.png width = 45%/>
<img src = images/nb-corr-t.png width = 45%/>
</p>

#### 6. Random Forest Classifier
<p float="left">
<img src = images/rfc-pca.png width = 45%/>
<img src = images/rfc-corr.png width = 45%/>
</p>
<p float="left">
<img src = images/rfc-pca-t.png width = 45%/>
<img src = images/rfc-corr-t.png width = 45%/>
</p>

#### 7. XGBoost Classifier
<p float="left">
<img src = images/xg-pca.png width = 45%/>
<img src = images/xg-corr.png width = 45%/>
</p>
<p float="left">
<img src = images/xg-pca-t.png width = 45%/>
<img src = images/xg-corr-t.png width = 45%/>
</p>

#### 8. Dummay Classifier
<p float="left">
<img src = images/dummy-pca.png width = 45%/>
<img src = images/dummy-corr.png width = 45%/>
</p>
<p float="left">
<img src = images/dummy-pca-t.png width = 45%/>
<img src = images/dummy-corr-t.png width = 45%/>
</p>

**Lets look at the metrics visually**
<p float="left">
<img src = images/lr-pca-v.png width = 45%/>
<img src = images/lr-corr-v.png width = 45%/>
<img src = images/dt-pca-v.png width = 45%/>
<img src = images/dt-corr-v.png width = 45%/>
<img src = images/svm-pca-v.png width = 45%/>
<img src = images/svm-corr-v.png width = 45%/>
<img src = images/knn-pca-v.png width = 45%/>
<img src = images/knn-corr-v.png width = 45%/>
<img src = images/nb-pca-v.png width = 45%/>
<img src = images/nb-corr-v.png width = 45%/>
<img src = images/rfc-pca-v.png width = 45%/>
<img src = images/rfc-corr-v.png width = 45%/>
<img src = images/xg-pca-v.png width = 45%/>
<img src = images/xg-corr-v.png width = 45%/>
<img src = images/dummy-pca-v.png width = 45%/>
<img src = images/dummy-corr-v.png width = 45%/>
</p>

## Conclusions
Looking at the visual representations and observations, here are some high level conclusions:

- In general, across all the models, the feature selection using IQR has performed much better than the PCA mechanism
- The best models are 

   1. XGBoost (Accuracy = 0.68)
   2. Random Forest (Accuracy = 0.67)
   
   Both operating on the features that were selected using the IQR method.
- DecisionTree performed very poorly.
- SVM was the slowest model

### Interpretability
Based on techniques like Feature Importance and SHAP (SHapley Additive exPlanations), the following features were found to be more impactful in determining the applications

- Init_Win_bytes_backward
- Init_Win_bytes_forward
- min_seg_size_forward
- Init_Win_bytes_forward
- Fwd.Packet.Length.Max
- Flow.IAT.Max
- Flow.IAT.Mean

### Business Impact
Misclassifying applications may lead to real money loss. It is hard to quantify the amount of loss since it may involve anything ranging from not able to detect malicious packets to not providing adequate level of service level agreements to the customer. 
To this end, reducing both False-Positives and False-Negatives are important. Hence focus should be on Accuracy. Both precision and recall needs to be maximized.

The models described here could be deployed in the following types of business scenarios:
1. Non critical
2. Best effort 

## Future
The best models above were still not good enough. I am sure, it can be improved with a GridSearch on parameters and some of the following:

- More advanced feature selection techniques could be used in the future, like gain ration (GR) based techniques.
- Also better cross-validation techniques (like N-fold) and Grid search could be employed to tune the model much better.
- Optimize on Multiclass Receiver Operating Characteristics (multi-class RoC)
- Deep learning algorithms, such as Convolution Neural Networks (CNN), have proven their efficiency through the unnecessity of extracting any statistical feature and through their reliance on the employment of the raw network traffic as their input. A future direction could be using deel learning techniques to classify traffic and identify causal applications.

## Outline of project

- [Link to download data](https://www.kaggle.com/datasets/jsrojas/ip-network-traffic-flows-labeled-with-87-apps)
- [Link to main notebook](https://github.com/1kit/Berkeley-Capstone-Project/blob/main/capstone.ipynb)
- [Link to EDA notebook](https://github.com/1kit/Berkeley-Capstone-Project/blob/main/EDA.ipynb)

## Contact and Further Information
Venkit Kasiviswanathan

Email: venkitk@gmail.com 

[LinkedIn](https://www.linkedin.com/in/venkitraman-kasiviswanathan-40a3592/)
