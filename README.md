<p align="center"><img width="100%" src="fig/name.png" /></p>

--------------------------------------------------------------------------------

# Geometric matching is Powerful Self-supervised Pre-trainer!

[NEWS!]**This paper has been accpeted by CVPR 2023! The basic code on [pytorch](https://github.com/YutingHe-list/GVSL) has been opened!**

[NOTE!!]**The code will be gradually and continuously opened!**

Learning inter-image similarity is crucial for 3D medical images self-supervised pre-training, due to their sharing of numerous same semantic regions. However, the lack of the semantic prior in metrics and the semantic-independent variation in 3D medical images make it challenging to get a reliable measurement for the inter-image similarity, hindering the learning of consistent representation for same semantics. We investigate the challenging problem of this task, i.e., learning a consistent representation between images for a clustering effect of same semantic features. We propose a novel visual similarity learning paradigm, Geometric Visual Similarity Learning, which embeds the prior of topological invariance into the measurement of the inter-image similarity for consistent representation of semantic regions. To drive this paradigm, we further construct a novel geometric matching head, the Z-matching head, to collaboratively learn the global and local similarity of semantic regions, guiding the efficient representation learning for different scale-level inter-image semantic features. Our experiments demonstrate that the pre-training with our learning of inter-image similarity yields more powerful inner-scene, inter-scene, and global-local transferring ability on four challenging 3D medical image tasks.


<p align="center"><img width="80%" src="fig/fig.png" /></p>

## Paper
This repository provides the official PyTorch implementation of GVSL in the following papers:

**Geometric Visual Similarity Learning in 3D Medical Image Self-supervised Pre-training** <br/> 
[Yuting He](http://19951124.academic.site/?lang=en), [Guanyu Yang*](https://cse.seu.edu.cn/2019/0103/c23024a257233/page.htm), Rongjun Ge, Yang Chen, Jean-Louis Coatrieux,  Boyu Wang, [Shuo Li](http://www.digitalimaginggroup.ca/members/shuo.php) <br/>
Southeast University <br/>
**IEEE/CVF Conference on Computer Vision and Pattern Recognition 2023**<br/>

## Acknowledgments

This research was supported by the Intergovernmental Cooperation Project of the National Key Research and Development Program of China(2022YFE0116700), CAAI-Huawei MindSpore Open Fund and Scientific Research Foundation of Graduate School of Southeast University(YBPY2139). We thank the Big Data Computing Center of Southeast University for providing the facility support. 
