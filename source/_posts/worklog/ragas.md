---
title: RAG评测方案
description: 本文讲述如何评测检索增强生成（Retrieval Augmented Generation），以及一些对应的实现体验。
date: 2024-02-09 23:01:00
tags:
- 检索增强生成
- RAG
---

# 背景介绍
检索增强生成（Retrieval Augmented Generation, RAG）是一种结合了检索和生成两种技术的人工智能模型应用。在这种方法中，模型首先检索与给定任务或输入相关的信息或数据，然后使用这些信息生成新的内容或回答。

具体来说，检索增强生成通常包括以下几个步骤：

1. **检索**：根据输入的问题或任务，模型在一个大规模的语料库或知识库中检索最相关的信息片段或数据。
2. **理解**：模型对这些检索到的信息进行理解和分析，以确定它们与输入的关联性和重要性。
3. **生成**：利用检索到的信息，模型生成一个连贯、准确的回答或内容。

因此，对于一个简单的RAG系统，其由一个检索器和一个生成器组成（想象Bing和GPT的结合），首先由检索器找出与问询相关的资料，然后由生成器整合信息给出可信的答案。
在当前时间节点（2024-02-09），私域知识库背景下，RAG多数情况下代指更复杂的系统，通常包含了：

* 问询意图分析
* query改写
* 文本分块、向量化
* 向量检索器、稀疏检索器、重排序
* 引用分析
* Agent

等等环节。这些囊括了额外复杂组件的RAG也被称为Advanced RAG。

> 关于 Advanced RAG 可以参考[此篇文章](https://pub.towardsai.net/advanced-rag-techniques-an-illustrated-overview-04d193d8fec6)

在RAG评测方面，一个典型的例子是[RAGAS这篇论文](https://arxiv.org/abs/2309.15217)，讲述了如何从答案可信度、答案相关度、上下文相关度三个方面评测一个RAG系统核心部分的效果。除RAGAS外，在实践过程中我参考了另一篇叫做[WikiChat](https://arxiv.org/abs/2305.14292)的论文，该论文讲述了如何构建一个效果更好的RAG系统，并以Wikipedia为例实现了一个叫做WikiChat的问答机器人，实现了比单独GPT更好的问答效果。虽然WikiChat并不讨论RAG评测系统构建，但是其论文的附页给出了详细的Prompt，我觉得一定程度上比RAGAS提供的Prompt更靠谱。

# RAGAS评测思路
RAGAS主要从如下三个角度评测一个RAG系统的问答效果：

* Faithfulness：这个概念指的是生成的答案应当基于提供的上下文。这一点很重要，以避免产生无根据的虚构内容，并确保检索到的上下文能够作为生成答案的依据。RAG系统常被用于对生成文本与地面实况来源的事实一致性有高度要求的应用中，例如在法律领域，信息是不断演变的。
* Answer Relevance：这个概念指的是生成的答案应当针对提出的问题。也就是说，答案应当是切题的，能够真正回答用户提出的问题。
* Context Relevance：这个概念指的是检索到的上下文应当是集中的，尽量不包含不相关的信息。这一点很重要，因为向大型语言模型（LLMs）提供过长的上下文段落是有成本的。此外，当上下文段落过长时，大型语言模型往往在利用这些上下文方面的效果会降低，尤其是对于那些出现在上下文段落中间的信息。

RAGAS根据上述的三个角度，评测一个RAG系统在回答用户的提问时，答案answer对提问question的效果。

## Faithfulness评测
RAGAS首先编写了一个Prompt，提供给LLM用户的提问question，以及一个RAG系统对该提问的回答answer，然后要求LLM结合question和answer信息将answer拆分为一种叫做statement东西（我个人认为相当于WikiChat中的claim，而且我感觉WikiChat中的Prompt显然更靠谱）；然后编写另一个Prompt，提供给LLM关于前述question检索得到的上下文（相关资料）context，然后要求LLM判断每个statement是否被包含于context，最后计算的分数是被判定为正面的statement占总statement的比例。
显然，Faithfulness评测的是LLM在根据context回答用户提问时发生幻觉的可能性，这非常重要，因为RAG系统的初衷之一就是降低LLM在回答问题时发生幻觉。我这里提供一条示例，有助于理解statement的概念。

```python
question = "《后赤壁赋》的作者是谁？"
answer = "《后赤壁赋》是北宋文学家苏轼于元丰五年（1082年）作于黄州的散文作品，是《前赤壁赋》的姐妹篇。"
statements = [
    "《后赤壁赋》是北宋文学家苏轼的作品",
    "《后赤壁赋》创作于元丰五年（1082年）", 
    "《后赤壁赋》作于黄州", 
    "《后赤壁赋》是《前赤壁赋》的姐妹篇"
]
```

