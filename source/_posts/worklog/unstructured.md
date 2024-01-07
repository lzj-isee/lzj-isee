---
title: 基于unstructured解析文档
description: 本文讲述了如何修改unstructured代码以更好地解析文档（中文为主）
date: 2024-01-07 22:45:00
tags:
- unstructured
- RAG文档解析
---

2023年出现了非常多火热的RAG项目，比如开源的[Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat)和[FastGPT](https://github.com/labring/FastGPT)等等，这些项目将LLM与检索技术结合，希望在一定程度上缓解LLM的幻觉问题，并且赋予LLM私域知识库问答的能力。
一般来说，这些项目的第一个环节都是读取并解析文档（比如常见的markdown、pdf、docx等等），将非结构化的文档一定程度上组织为可被检索的形式，为后续的LLM问答环节提供相对完整的、准确的参考资料。
在这些项目中，有些就比较简单地将文档中所有的文字提取出来，但是不做任何结构上的解析，因此整个流程中无法区分标题、正文、表格、页眉、公式、代码等等不同的元素，这导致提供给LLM的参考资料混乱，一定程度上降低了RAG的效果。
另一些项目选择直接接入 Langchain 项目的IO接口，或者直接使用 unstructured 项目（这也是 Langchain 项目的文档读取部分的底层）读取并解析文档。
但是，在本人的实际使用中，发现 unstructured 存在一些不完善的地方，尤其是在中文环境下，很容易发生一些错误的解析。
在一些尝试摸索后，我此在总结一些修改 unstructured 代码的经验，也希望能帮助其他人。

> 所有讨论基础基于 unstructured 的0.11.6版本

# Markdown 解析修改
unstructured 解析 markdown 的流程是首先使用一个 markdown 渲染器将 markdown 文本转化成 html 文本，然后使用 html 解析规则。
所以我这里主要是介绍 markdown 渲染器，以及渲染规则的修改。

## 更换渲染器
unstructured 默认的渲染方式是使用 python 的 markdown 库的 `markdown.markdown`方法（额外开启了插件 `tables`）。
这个库仅支持基础的markdown语法，有一些特殊的用法，比如删除线、latex公式等语法就不支持。
为了解决这些问题，我将默认的`markdown.markdown`渲染替换为python的`MarkdownIt`库，并搭配额外的插件实现这些需求，为了更准确地描述这些修改，我首先给出代码示例：

```python
markdown_ot_parser = MarkdownIt(
    "gfm-like", 
    {"breaks": False, "html": True}
).use(
    functools.partial(dollarmath_plugin, double_inline = True)
).use(
    front_matter_plguin
).use(
    footnote_plugin
).enable("table")

html_text: str = markdown_it_parser.reder(text)
```

这里我建议读者参考markdown-it-py的[官方文档](https://markdown-it-py.readthedocs.io/en/latest/index.html)和插件库的[官方文档](https://mdit-py-plugins.readthedocs.io/en/latest/#)学习相关用法。
上述代码使用了三个插件，分别是`dollarmath_plugin`, `front_matter_plguin`, 和`footnote_plugin`，这里后两个插件可以直接安装使用，第一个插件关乎latex公式的识别，为了达到最优状态，需要修改原代码。

## 插件代码修改
在`dollarmath_plugin`插件的[原实现方案](https://mdit-py-plugins.readthedocs.io/en/latest/_modules/mdit_py_plugins/dollarmath/index.html#dollarmath_plugin)中，行内公式是不会被渲染为一个单独的div的，而是一个span，这导致在后续的html解析环节中很难区分行内与行间由`$$`符号定义的latex公式的区别。
使用`$$`符号通常有两种情况，一个是在markdown的行内使用，一个是在markdown的行间作为一个单独的块使用，两者在视觉渲染上是一样的，都是行间公式，但是前者在html符号中被定义为span，后者是div。
在研究了一些MarkdownIt的渲染规则后，我发现可以简单地增加一行代码解决这个问题，即将

```python
md.block.ruler.before(
    "fence", 
    "math_block", 
    math_block_dollar(allow_labels, label_nomalizer, allow_blank_lines)
)
```

修改为

```python
md.block.ruler.before(
    "fence", 
    "math_block", 
    math_block_dollar(allow_labels, label_nomalizer, allow_blank_lines), 
    options = {"alt": ["paragraph"]}
)
```

这涉及到了MarkdownIt的渲染规则，在写这篇博客的时候我已经不记得详细的解决思路了，这大概是允许渲染器在本是inline的规则中插入一个block，于是将一个伪行间公式转化为一个真正的行间公式。

# Html解析修改
html的解析部分我修改了非常多，基本算是重写了html解析部分，在这里不放出具体的代码，只给出大概思路：

1. 原实现中不会识别一个list-item里面的结构，比如表格或者代码或者latex公式等等，这些会被识别为正文文本（因此失去了结构），这里需要修改为嵌套形式，允许解析一个list-item里的结构，并在解析的同时赋予好parent关系；
2. 需要处理markdown中插入图片的情况，MarkdownIt是能正确渲染的，需要增加html解析中识别html的标签的种类；
3. 增加对latex数学公式的识别；
4. 修改table的解析逻辑，默认文本存储html格式的表格，而非正文形式；
5. 修改判定为title的逻辑（这里要着重讲下，原方案中很容易将一些文本识别为title，我的想法是与其错误地识别为title，不如识别为正文）；
6. 增强了定位link的能力，原方案中只能提取出link，不能定位link在字符串中的位置，我加入了位置识别。

# PDF解析修改
PDF解析部分基本也是重写了，因为某些原因，我只能使用fast模式否则处理时间过长会导致超时，在此处我只给出大致的思路要点：

1. 首先删除原方案中所有解析link的部分，反正也识别不准，而且拖累了运行速度；
2. 修改判定为title的逻辑，此处跟markdown解析修改的逻辑是一样的；
3. 不从`LTContainer`取得文本，防止拿到矢量图中非正文的文字；
4. 修改`LTParams`的`line_margin`参数从0.5至0.3，此修改能将某些次级标题从错误的正文识别为正确的标题；
5. 统计所有文本的字号，取众数的字号为正文字号，所有低于或等于该字号的文字为正文，如果一个block的最小字号大于该字号（严格意义上是Top5字号），那么该block被识别为title；
6. 纯数字，或太短的文本，或不包含alpha文本的block将被过滤，这是为了去除一些短文本的干扰。

# DOCX修改
docx文档的修改很简答，原设计中的绝大多数都能直接复用，唯一要修改的是判定title的逻辑，这里需要修改`_DocxPartitioner`这个类的`_parse_paragraph_text_for_elemet_type`方法，怎么修改随意，大体上来说是限制判定为title的条件。
我的修改是

```python
text = paragraph.text.strip()
if len(text) < 2:
    return None
return NarrativeText
```

# DOC修改
doc文档基本没什么问题，因为基础原理是先将doc文档转化为docx文档，然后调用了docx的解析逻辑。
这里需要注意的是原方案使用了`tempfile`，但是代码写得比较迷，会不断产生临时文件，此处简单地修改下代码就能解决这个问题。

# 其他
目前据我所知，unstructured的xlsx解析部分也是很有问题的，但是我目前还没有修改到这里，因此还没有什么想法。
unstructured这个库还是很优秀的，有些不满足需求的部分需要自己集合实际需求做修改，我上述指出的某些“缺点”可能在其他人的业务中也不是个问题。
