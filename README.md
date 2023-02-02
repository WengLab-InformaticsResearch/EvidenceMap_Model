# EvidenceMap

**Corpora:**    

A gold standard dataset of randomized controlled trial (RCT) abstracts annotated with the EvidenceMap representation including two corpora are provided.      
The “General” corpus includes a broad range of disease domains by randomly selecting 229 RCT article abstracts. 
The “COVID-19” corpus includes 80 randomly selected COVID-19 RCT article abstracts to accommodate the increased demand for related evidence retrieval and synthesis resources during the pandemic. 

The descriptive statistics of these two annotated corpora are listed in Table.

<img width="553" alt="image" src="https://user-images.githubusercontent.com/11466174/174745544-6f043b8c-de6f-44f6-bb76-5fb140341a29.png">


Dependent evidence relationships were used for constructing MEPs, and independent relationships can serve as negative samples for training machine learning based NLP models. 

All annotations were conducted using the web-based interactive annotation tool Brat (https://brat.nlplab.org/). An example abstract with annotations is presented:

<img width="468" alt="image" src="https://user-images.githubusercontent.com/11466174/174746112-2cfe2c50-3fef-4d74-9ef0-bdce6b5001e0.png">

**Pretrained Models:**    
<br>
Download pretrained models [here](https://drive.google.com/file/d/11lmAoUQ1Uyo722SuDnYZvFWpEa13RiXU/view).    
<br>
**Running Environment:**
1. Install tensorflow==2.3    
2. Install bert-for-tf2    
3. Install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_lg-0.4.0.tar.gz    
<br>
**Running the Code:**
1. Unzip model.zip    
2. Move all files under model/    
3. Modify parser configuration parser_config.py    
4. Run examples in wrapper.sh    


<div align="center">Yingcheng Sun, Tian Kang, Chunhua Weng (cw2384@cumc.columbia.edu)</div>
