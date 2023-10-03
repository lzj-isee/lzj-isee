title: 基于知识库的LLM问答系统参考资料
date: 2023-07-1 22:13:00
description: 基于知识库的LLM问答系统参考资料，一些开源的项目的链接。
tags:
- collections
- LLM
---

<center><img src="https://github.com/imClumsyPanda/langchain-ChatGLM/blob/master/img/langchain+chatglm.png" width="80%" /></center>
技术路线图总览：<https://github.com/imClumsyPanda/langchain-ChatGLM/blob/master/img/langchain+chatglm.png>。

## 关于LLM的相关项目
- `ChatGLM-6B`项目的GitHub地址：<https://github.com/THUDM/ChatGLM-6B>，这个部署特别方便，也可以参考学习如何使用`gradio`构建webui。`ChatGLM-6B`算是中文LLM里比较好用的了。
- `ChatGLM2-6B`项目的GitHub地址：<https://github.com/THUDM/ChatGLM2-6B>，部署也很方便，`torch 1.12.1`版本可能无法使用更优的attention算子，半精度下显存占用大约比`ChatGLM-6B`高1G，但是token的数量上限显著升高。
- `GLM10B`项目的Github地址：<https://github.com/THUDM/GLM>，这个没用过。

## 关于知识库检索的LLM的相关项目

### Github项目
- `chatglm-qabot`项目的Github地址：<https://github.com/xinsblog/chatglm-qabot>。特别简单的一个小项目，功能很简陋，但是代码很好理解，主要的步骤都有。
- `langchain-GLM_Agent`项目的Github地址：<https://github.com/jayli/langchain-GLM_Agent>。比`chatglm-qabot`项目稍微复杂点，用到了`langchain`，可以作为进阶学习。
- `langchain-ChatGLM`项目的Github地址：<https://github.com/imClumsyPanda/langchain-ChatGLM>。比`langchain-GLM_Agent`项目复杂，部署很方便，大量用到了`langchain`，给出了技术路线图，可以作为进阶学习。
- `ducoment.ai`项目的Github地址：<https://github.com/GanymedeNil/document.ai>。讲了一些知识库检索系统的难点。
- `DocsGPT`项目的Github地址：<https://github.com/arc53/DocsGPT>，这个好像只能处理一篇文档。
- `quivr`项目的Github地址：<https://github.com/StanGirard/quivr>。用的技术比`langchain-ChatGLM`稍微复杂点，加了一个做summarize的过程。

### 知乎文章
- 知乎上介绍`知识库+LLM`的文章：<https://zhuanlan.zhihu.com/p/628890578>，可以了解相关背景。
- 介绍如何在`ModelScope`上实战`知识库+LLM`的文章：<https://zhuanlan.zhihu.com/p/627698200>，有`ModelScope`使用经验的可以试下。
- `chatglm-qabot`项目的知乎介绍文章：<https://zhuanlan.zhihu.com/p/622418308>，讲的很清楚，代码也好懂。
- 知乎上介绍`知识库+LLM`的文章：<https://www.zhihu.com/question/600243900/answer/3060996737>。这里讲到了一个关于`非对称检索`的问题（QQ检索跟QA检索），可以看下。

### ModelScope上提供的成品模型
- `文本分割`模块：<https://www.modelscope.cn/models/damo/nlp_bert_document-segmentation_chinese-base/summary>。
- `文本嵌入`模块：<https://www.modelscope.cn/models/thomas/text2vec-large-chinese/summary>。
- `文本相似度比较`模块（用来做重排）：<https://www.modelscope.cn/models/damo/nlp_rom_passage-ranking_chinese-base/summary>。

## 其他
- 知乎上介绍`中文检索评测数据集 Multi-CPR`的文章：<https://zhuanlan.zhihu.com/p/600804550>。
- `DecryptPrompt`项目的Github地址：<https://github.com/DSXiangLi/DecryptPrompt>。讲一些关于LLM的论文、数据集、Prompt工程，资料比较全。
- 讲解怎么写`Prompt`：<https://www.promptingguide.ai/zh>。
- `langchain`的官方文档：<https://python.langchain.com/docs/get_started/introduction.html>，地位相当于`torch`，而且官方文档的搜索就是基于LLM的智能问答系统。

## 一点体会
- 大多数知识库检索项目的技术路线图都是类似`langchain-ChatGLM`项目这种的，但是这个技术路线非常简略。
- 比如在做搜索的那一步，大多数路线图就是用的向量检索，但是实际上这样做效果很差。
- 比如在向量检索后加用`文本相似度比较`模型做重排，效果会好很多。
- 进一步，这个问题完全可以借鉴专业的检索系统是怎么做的，召回、粗排、精排。
- 也可以扩充检索信息来源，原来只用向量，那么是否能提取主题词，或是先用模型对分割后的文本块做summarize。
- 另一方面，文本分块也很重要，是否基于规则的分割就够用了（个人体感是不够用的），那选用什么模型能解决这个问题，怎么处理中文、英文、代码混合的情况？
- 那个路线图没说的是，检索出参考文本块后也可以做一个上下文拼接。
- `Prompt`怎么设计比较好，什么程度的LLM能够满足使用？个人体感像`ChatGLM2-6B`这种对`Prompt`的响应就很差，`ChatGPT3.5`就好很多。这个对最终回答质量的影响也非常大，毕竟需要LLM做的就是从一堆杂乱的信息中提取出有用的部分给用户。