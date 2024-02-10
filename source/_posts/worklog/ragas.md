---
title: RAG评测方案
description: 本文讲述如何评测检索增强生成（Retrieval Augmented Generation），以及一些对应的实际实现。
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

## Answer Relavance评测
RAGAS首先编写了一个Prompt，提供给LLM关于一个提问的答案answer，然后要求LLM根据这个answer提出n个与该answer相关的问题`question*`，计算每个`question*`与真正用户的提问question之间的余弦相似度（使用OPENAI的API计算），该分数取均值最为最终的Answer Relevance评分。

> 我非常不理解为什么要这样评测Answer Relevance，考虑到一般来说Embedding模型的cos分数的分布不均衡，很容易导致这个分数存在偏差。而且，一个答案answer可能潜在对应了多个问题question，其中很可能只有少数几个question是和真实question类似的，那么就意味着最终的Answer Relevance很有可能普遍偏低。

## Context Relevance评测
RAGAS首先编写一个Prompt，提供给LLM当前检索出的上下文context以及用户的提问question，要求LLM从context中提取出与该question相关的句子的数量，最终的评分为提取出的句子的数量除以context中总的句子的数量。

# 实现思路
我个人认为RAGAS的大体思路是比较靠谱的，从答案置信度、答案相关度以及上下文相关度几个方面评测一个RAG系统的问答效果。但是，我在实际实现中对RAGAS的某些细节做出修改:

1. RAGAS使用向量模型衡量LLM给出的answer与用户提问question之间的相关度，之所以根据answer让LLM生成`question*`，可能是考虑到OPENAI的向量模型计算QA相似度效果不好的问题，因此将其转化为QQ类型后再计算相似度。在答案answer长度比较长的情况下，一个answer可能对应了多个合理`question*`，导致某些`question*`和question之间的相似度比较低。这方面我在实现中也没有比较好的想法，我的观点是评测answer relevance不是特别重要，LLM不太会出现答非所问的情况，而且一部分异常能通过faithfulness体现。
2. RAGAS将context拆分为句子，然后衡量每个句子是否与question相关。我在实际实现中的做法是，在文档级别，计算检索文档的precision和recall，在段落（chunk）级别，计算检索出的段落与标准answer（由LLM伪造）之间的Rouge-L，衡量检索出的文档内容对答案的覆盖率。Precision、Recall、Rouge-L都是传统指标，虽然不是直接测试context对question的相关度，但是相比LLM评价更稳定，更可预测可解释，也是作为一方面参考。
3. RAGAS测试faithfulness时使用的是检索context衡量statement，我理解这种实现方式解耦了context relevance和faithfulness两个指标，但是出于检索context完整性的考虑，我在实际实现中使用的是groundtruth documents作为context验证statement。

# 生成QA对
评测一个RAG系统首先需要构建一个测试数据集，我下载清洗了23年12月的维基百科快照，从中选择了1000个词条页面作为正样本，9000个词条页面作为负样本，总共1w条样本作为测试知识库。其中，正样本的限制条件为文件大小至少为2k，这是为了避免过短的正文词条无法提取出有效的问题-答案对，负样本则从总数约120w的全体样本中随机选取。

在确定好所有的正样本后，将一篇维基百科词条视作一篇文档document，然后编写Prompt使用LLM (QwenMax) 根据一篇document构建出对应的question和groundtruth answer，使用的Prompt如下：

```python
PROMPT_TEMPLATE = """\
你被要求学习一篇维基百科词条文章的内容，然后根据文章内容总结学习成果，要求为：
1. **根据文章内容**总结提出最多1个问题
2. 如果无法提出问题，请直接回答**无法提取问答**
3. **根据文章内容以及你提出的问题**给出答案
4. 你需要按照如下实例格式返回问题和答案

Q: 马萨诸塞的北雷丁有多少人口
A: 根据2020年美国人口普查的数据，北雷丁的面积为34.90平方千米，当地共有人口15554人，而人口密度为每平方千米446人。

Q: 物理学上宇宙的定义是什么
A: 物理学的宇宙被定义为所有的时间与空间（两者共同称为时空）；这包含了电磁辐射以及物质等所有能量的各种形态，进而组成行星、卫星、恒星、星系以及星系际空间。

Q: 福特汽车公司哪一年创立的
A: 1903年

Q: 明朝第五个皇帝是谁
A: 明宣宗朱瞻基

当前提供的文章是：
{content}

现在，总结你的学习成果，给出一个提问"Q"以及针对该问题的答案"A"
Q: \
```

对每一篇正样本document应用该Prompt，可以得到1000个关于`(question, groundtruth answer, document)`的三元组。

# 待测RAG系统生成答案
对前一步骤中的所有三元组，取出其中的question部分作为RAG的问询输入，然后记录RAG检索到的context（包括id以及对应的内容，一个question可能会检索到多个context），以及LLM关于该question在参考context情况下的回答answer。

# 计算评测指标
## 计算RAG检索的评测指标
对于每一个正样本提供的question与document，待测RAG都根据question检索出了对应的context作为LLM的参考上下文，此时根据document id与context ids可以计算RAG中检索部分在文档级别的检索Precision、Recall、NDCG等指标，衡量RAG系统的检索部分的性能。
对于某些RAG检索策略来说，可能评测Precision是没有意义的，这些RAG系统可能缺乏设定一个阈值过滤检索得到的context环节，只是按照固定的top3或者top5，选取一定数量的检索context提供给LLM；但是对于另一些RAG策略，尤其是使用了rerank并且设定了某个过滤阈值的情况下，RAG的检索部分针对不同的question最终得到的context数量也是不同的，此时评测检索的Precision是有意义的，在前述构建评测数据集的设定中，最优情况是RAG每次都只检索出了一篇文档context，并且该context恰好是真值document。

对于每一个正样本提供的groundtruth answer，可以计算RAG检索出的context与groundtruth answer之间的Rouge-L指标。
Rouge-L计算了两个字符串之间的最长公共子序列的长度与被测字符串长度的比值，可细分为Rouge-L-Precision、Rouge-L-Recall、Rouge-L-F1三种指标。
其中Rouge-L-Precision是最长公共子序列与context的长度之间的比值，衡量了检索出的context在包含答案的情况下是否足够简短；
Rouge-L-Recall是最长公共子序列与groundtruth answer的长度之间的比值，衡量了检索出的context对参考答案的覆盖率；
Rouge-L-F1是二者的几何平均数。

> 说实在我觉得Rouge-L-Precision没什么参考意义，因为一般检索的context长度会固定在某个token数量上限，一般情况下这个上限值都是很大的，意味着一般Rouge-L-Precision都很低，但是这并不意味着检索系统性能很差。相反，检索系统优先应该保障Rouge-L-Recall，将提取与question相关内容交给强性能的LLM完成。

总而言之，对于RAG系统的检索部分，至少可以评测：

1. 文档检索Precision
2. 文档检索Recall
3. 文档检索NDCG
4. 上下文context检索Rouge-L-Recall

等指标。

## 计算RAG检索的Faithfulness
该步骤是为了衡量LLM生成结果的可信程度，我在RAGAS方案的基础上稍作修改，使用真值document而非检索context作为衡量基准。该修改会导致Faithfulness指标实际上包含了检索部分的准确性，不过我觉得也可以简单地对检索Recall非0的样本计算LLM生成结果的Faithfulness，降低检索context不包含答案带来的干扰。
之所以使用真值document而非检索context，主要是考虑到检索context可能只是原本的一个片段，上下文的不完整可能对后续LLM的判断造成影响，读者可自行判断是否使用context或是document。

计算Faithfulness的过程主要包含两个主要步骤：

1. 由LLM将RAG针对提问question给出的回答answer拆分为多个自包含的声明claim（相当于RAGAS中的statement）
2. 对每一个claim，由LLM根据document判断该claim是否叙述正确

> 这里我主要借鉴的WikiChat论文中的Prompt写法

第一个步骤的Prompt是few-show写法，其中的system与一轮对话示例分别是（更多示例可参考WikiChat论文）：

```python

```

