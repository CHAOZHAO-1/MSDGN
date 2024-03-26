# MSDGN
[MSSP 2023] Mutual-assistance semisupervised domain generalization network for intelligent fault diagnosis under unseen working conditions


## Paper

Paper link: [Mutual-assistance semisupervised domain generalization network for intelligent fault diagnosis under unseen working conditions](https://www.sciencedirect.com/science/article/pii/S0888327022011426#ab010)

## Abstract

Generalizing deep models to unseen working conditions is an essential topic for intelligent fault diagnosis. Existing domain generalization-based fault diagnosis (DGFD) methods usually require sufficient annotated samples from all observed domains during the training phase, while annotating abundant samples is an expensive and difficult task. Therefore, this study proposes a mutual-assistance network for semisupervised domain generalization fault diagnosis (SemiDGFD), where only one source domain is labeled along with several unlabeled source domains. Reliable pseudo labels are assigned to unlabeled data with knowledge assistance from labeled data. Then, an entropy-based sample purification mechanism is designed to improve the quality of pseudo-labeled samples. Finally, pseudo-labeled samples cooperate with real-labeled samples to serve as the input of a low-rank decomposition, which discovers domain invariance against domain shift. Extensive diagnostic experiments demonstrate that the proposed method can obtain higher precision than other popular SemiDGFD methods and achieve comparable performance with up-to-date fully-labeled DGFD methods.


##  Proposed Network 


![image](https://github.com/CHAOZHAO-1/MSDGN/blob/main/IMG/F1.png)

##  BibTex Citation


If you like our paper or code, please use the following BibTex:

```

@article{zhao2023mutual,
  title={Mutual-assistance semisupervised domain generalization network for intelligent fault diagnosis under unseen working conditions},
  author={Zhao, Chao and Shen, Weiming},
  journal={Mechanical Systems and Signal Processing},
  volume={189},
  pages={110074},
  year={2023},
  publisher={Elsevier}
}

```
