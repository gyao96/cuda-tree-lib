<p style="text-align:center">
    <img src="https://cs184.eecs.berkeley.edu/cs184_sp17_content/article_images/12_.jpg" width="800px" />
</p>

#CUDA-based Spatial Hierarchical Data Structure for Computer Graphics Acceleration

> Gang Yao, Zixi Cai, Hsin-pei Lee, Chuan-Jiun Yang

### Motivation
In modern computer graphics applications, spatial hierarchical data structures are widely used to accelerate calculation. Particularly, it is almost a requirement to take advantage of the bounding volume hierarchy (BVH) or spatial OC-tree to implement efficient ray-tracing based rendering pipeline. These data structures also have a wide range of use cases like collision detection, particle simulation, surface reconstruction and voxel-based global illumination. 

Before long, the common practice is to build the acceleration data structures with CPU before launching any graphics pipeline on GPU. However, while this may be sufficient for rendering a single frame, most real-time applications cannot afford to leverage workload frequently between the CPU and GPU. Thus, constructing these data structures dynamically on the fly and running them in parallel on GPU should be the preferred method of how such applications should be implemented. 

### Project Overview
Our project will focus primarily on implementing the BVH data structures on CUDA and then benchmark its performance with the physically-based rendering pipeline. This includes the cost of constructing the BVH and updating it. 
Given an unsorted list of geometric primitives, constructing the BVH sequentially typically takes O(NlogN) time and O(N) space. Lauterbach and others were the first to present a fast method for constructing so-called linear BVHs, which was later improved by Pantaleoni and Luebke, as well as Garanzha and others. (Karras et al) We plan to expand based on that and implement the parallel version of it using Karras’s algorithm. It is expected to achieve parallelism across nodes and levels and cut down time-complexity to O(Nlog(h)), where h is the depth of the tree. 
Updating only the bounding primitive can result in a degradation of the quality of the BVH, and in some scenes will result in a dramatic deterioration of rendering performance. The typical method to avoid this degradation is to rebuild the BVH when a heuristic determines the tree is no longer efficient, but this rebuild results in a disruption of the interactive system response. (Ize et al) We plan to try and tackle this issue with the following method: 1. Remove and insert nodes of the BVH in parallel and modify shared memory on GPU. 2. Reconstruct the BVH completely in parallel with the rendering process by sharing part of the resources on GPU. Experiments and tuning is expected to be done in order to evaluate which is better.

### Deliverables
* Minimum: A packed library containing a parallel implementation of the BVH data structure that is ready to use in any CUDA application.
* Expected: Fine-tuned CUDA-based BVH data structure with support for dynamically updating nodes and optimized for real-time ray tracing. 
* Bonus: Add point-based octree and KD tree to the library.

### Benchmarking
We will benchmark the performance of: 
- The time of constructing BVH for a scene with N primitives with CPU only and our CUDA implementation.
- The time of rendering a single frame with CPU-based BVH construction and our method.
- The time of rendering a sequence of frames with primitives moving in the scene. Compare between CPU-based BVH construction and our method.
### Reference
- Ize, Thiago, Ingo Wald, and Steven G. Parker. "Asynchronous BVH construction for ray tracing dynamic scenes on parallel multi-core architectures." In Proceedings of the 7th Eurographics conference on Parallel Graphics and Visualization, pp. 101-108. Eurographics Association, 2007.
- Karras, Tero. "Maximizing parallelism in the construction of BVHs, octrees, and k-d trees." In Proceedings of the Fourth ACM SIGGRAPH/Eurographics conference on High-Performance Graphics, pp. 33-37. 2012.

