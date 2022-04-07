# Dynamics-of-social-contagion-on-signed-network  
**符号网络上的传播过程研究(硕士论文项目)**

**摘要**

传统的社会或自然传播动力学研究集中在类似于“多米诺”骨牌一样的简单传播过程，如疾病扩散、舆论蔓延等，这类过程主要依靠节点之间的成对交互。而当下研究表明，成对的交互不足于刻画复杂的传播过程，传播会由于节点间漫长的链路而发生湮灭，这与复杂传播现象不符。复杂的传播行为更需要考虑到组间的传播效果对于个体之间观点或信息的接受、采纳的强化影响，例如时尚观点流行、口碑营销等。本文将需要依靠多个传播源才能激活的过程定义为高阶传播过程。此外，考虑高阶的传播过程适用于刻画个体好恶的主观情绪与态度强相关的社会传播行为。然而在现实世界中，不管是个体与个体，还是个体与组织之间的关系都是存在差异或强弱差距，因此在符号网络上展开对于复杂的社会传播现象的研究更能考虑到深层次的社会关系的影响，更贴近现实，也很有必要。  

## 代码说明  
[SignedNetwork.py](https://github.com/zaoshangqichuang/-thesis-Dynamics-of-social-contagion-on-signed-network/blob/main/%E4%BB%A3%E7%A0%81/SignedNetwork.py)  
构建符号网络，网络结构可设置re、ws、er、ba  
可选择两种机制分配负边：  
* 随机配置：随机撒负边  
* 优先配置：优先配置ws网络长程边、优先配置ba网络高介数边  

[me_analysis.py](https://github.com/zaoshangqichuang/-thesis-Dynamics-of-social-contagion-on-signed-network/blob/main/%E4%BB%A3%E7%A0%81/me_analysis.py)  
平均场方程的理论计算  

[BA_me_analysis.py](https://github.com/zaoshangqichuang/-thesis-Dynamics-of-social-contagion-on-signed-network/blob/main/%E4%BB%A3%E7%A0%81/BA_me_analysis.py)  
异质网络的异质平均场方程计算

[SIS_sim.py](https://github.com/zaoshangqichuang/-thesis-Dynamics-of-social-contagion-on-signed-network/blob/main/%E4%BB%A3%E7%A0%81/SIS_sim.py)  
SIS网络模拟  

[pick_edges.py](https://github.com/zaoshangqichuang/-thesis-Dynamics-of-social-contagion-on-signed-network/blob/main/%E4%BB%A3%E7%A0%81/pick_edges.py)  
随机挑选边、点

