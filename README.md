# CFDE
Integrating Competitive Framework into Differential Evolution: Comprehensive Performance Analysis and Application in Brain Tumor Detection

## Highlights
• We present an efficient CFDE to handle complex optimization problems.  
• CFDE consists of three tailored components: (1) competitive framework, (2) DE/loser-to-best/loser-to-winner mutation scheme, and (3) random memory initialization.  
• We conduct rigorous numerical experiments in CEC2017, CEC2020, CEC2022, and eight engineering problems to investigate the performance of CFDE against eleven state-of-the-art optimizers.  
• We conduct sensitivity analysis and ablation experiments to comprehensively investigate the performance of CFDE.  
• We further propose the DenseNet-CFDE-ELM model to detect brain tumors in MRI scans.  


## Abstract
This paper presents an efficient and effective optimizer based on the Success History Adaptive DE (SHADE) named Competitive Framework DE (CFDE). We integrate three tailored strategies into CFDE: (1) the competitive framework to identify and prioritize potential individuals, (2) the novel DE/loser-to-best/loser-to-winner mutation scheme to fully leverage the information from the population and competition to construct high-quality offspring individuals, and (3) the random memory initialization to diversify the search patterns of the individual. We conduct comprehensive numerical experiments on CEC2017, CEC2020, CEC2022, and eight engineering problems against eleven state-of-the-art optimizers to confirm the superiority and competitiveness of CFDE. Moreover, the sensitivity experiments on hyperparameters validate the robustness of CFDE, and the ablation experiments practically prove the independent contribution of integrated components. Furthermore, we propose a hybrid model named DenseNet-CFDE-ELM for brain tumor detection, where DenseNet-169 is employed for feature selection and CFDE-optimized Extreme Learning Machine (ELM) classifies the brain tumors in MRI scans. Experimental results on the brain tumor dataset downloaded from Kaggle confirm that the proposed DenseNet-CFDE-ELM achieves improvements in accuracy with 1.794\%, precision with 1.696\%, recall with 1.794\%, and F1 score with 1.812\% against the second-best ResNet-18 model. These results reveal the potential of CFDE in extensive real-world optimization scenarios. The source code of this research can be downloaded from https://github.com/RuiZhong961230/CFDE.
## Citation
@article{Zhong:25,  
title = {Integrating Competitive Framework into Differential Evolution: Comprehensive Performance Analysis and Application in Brain Tumor Detection},  
journal = {Applied Soft Computing},  
volume = {},  
pages = {},  
year = {2025},  
issn = {1568-4946},  
doi = {},  
author = {Rui Zhong and Zhongmin Wang and Yujun Zhang and Junbo Jacob Lian and Jun Yu and Huiling Chen},  
note = {Accept}  
}

## Datasets and Libraries
CEC benchmarks and Engineering problems are provided by opfunu==1.0.0 and enoppy==0.1.1 libraries, respectively, Deep learning models are provided by the Pytorch==2.1.2 library, and the brain tumor dataset is downloaded from Kaggle https://www.kaggle.com/datasets/rm1000/brain-tumor-mri-scans/data.

## Contact
If you have any questions, please don't hesitate to contact zhongrui[a]iic.hokudai.ac.jp


