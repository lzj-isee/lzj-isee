---
title: 使用wikiextractor提取wiki页面正文数据
description: 本文讲述了如何使用wikiextractor提取 wiki dump 的正文数据，以及相应的代码修改。
date: 2023-12-09 22:48:00
tags:
- wikiextractor
---

最近参与的一个项目中需要大量中文文档作为测试语料库，这里我首先考虑的是中文的维基百科，不仅在质量方面远优于百度百科，而且官方每个月都会将所有词条打包并提供下载（地址：https://dumps.wikimedia.org/zhwiki/ ）。
挂上梯子之后下载也是很方便，官方从不同层级角度提供了不同的语料库，我需要使用的是所有词条的正文内容，因此选择下载文件`zhwiki-20231201-pages-articles.xml.bz2`。
从文件名中可以看出，该文件记录截止到2023年12月1日的词条内容，`pages-articles`表示记录的是正文部分（压缩包大约2.6G大小），另有一些文件只记录摘要或者元信息，压缩包会比较小。
虽然下载很容易，但是这个压缩包无法直接使用，直接解压之后是一个巨大的`xml`文件，并且带有许多类似于`html`和`markdown`的维基百科自己的语法规则的标记（例如正文中会出现大量的`{{}}`或者`[[]]`这种括号），因此需要一些规则化处理。

# 提取工具
网络上能搜索到很多工具帮助我们从这个巨大的`xml`文件中提取出百科词条的标题、摘要、以及正文部分，其中提及次数比较多的两个主要工具是: 

1. [wikiextractor](https://github.com/attardi/wikiextractor)
2. gensim 的 wikicorpus 库

这两个工具都是基于python的，而且也提供了pip安装，使用比较方便。
在我试用之后，这两个工具分别有一些小问题。`wikiextractor`能够正确提取出词条的正文部分，而且支持导出为`html`格式，但是会去掉花括号`{{}}`标记的内容，造成正文内容缺失，而且章节标题会出现两次，不仅仅是标签`<h2>xx</h2>`中出现章节标题，而且紧跟着的下一行正文依然是章节标题，此外还有一些`html`模式下的特有问题（我严重怀疑是使用`html`模式导出的人太少了，这些问题迟迟没有修复）；而`wikicorpus`会去除掉所有的标点，并且会导出非正文的图片caption内容（如果不介意图片caption混入正文可以参考苏神的方案：https://www.kexue.fm/archives/4176/comment-page-1, 避免了去除标点的问题）。
因此我不得不自己尝试修改代码满足需求，我选择在`wikiextractor`的基础上修改代码。

> 所有的代码修改基于 https://github.com/attardi/wikiextractor 的`v3.0.7`版本，在阅读后文前请注意版本号问题。

> 由于我只需要`html`格式的导出文件，我的所有修改均针对`html`模式下出现的问题，因此不考虑非`html`模式的正确性。

# 代码修改
## 启动参数修改
1. 输出的文件格式仅支持`html`，因此修改启动参数`--html`的默认值为`True`
2. 修改参数`--html-safe`默认值为`False`，无需二次转换

## 强制不识别 link
在我的使用场景中不需要正文出现的 link，因此直接禁止防止出现意外 error
```python
# WikiExtractor.py 文件约 143 行处
def _filepath(self):
    return '%s/wiki_%2d.html' % (self._dirname(), self.file_index) 
```

## 正则匹配避免去除花括号中的内容
此处参考了[苏神的处理方案](https://www.kexue.fm/archives/4176/comment-page-1)
```python
# extract.py 文件约85行处
# 在调用
text = extractor.expandTemplates(text)
# 之前添加
text = re.sub(r"(.){{(?:[^|{}\n]*\|)*?([^|{}\n]+?)}}", r"\1[[\2]]", text)
# 该代码主要是为了避免如下情况提取不到文本
# {{lang|正文}} 或 {{lang|el|正文}}
```

## List 的 Tag 问题
```python
# extract.py 文件约187行处
# from 
listItem = {"*": "<li>%s</li>", "#": "<li>%s</<li>", ";": "<dt>%s</dt>", ":": "<dd>%s</dd>"}
# to
listItem = {"*": "<li>%s</li>", "#": "<li>%s</li>", ";": "<dt>%s</dt>", ":": "<dd>%s</dd>"}
# 这里应该存粹是作者不小心写多了一个"<"符号
```

## 添加或去除一些换行符
```python
# extract.py 文件约196行处，compact函数的最开始位置添加如下代码
text = re.sub("\n==", "\n\n==", text) # 发现结束list解析逻辑需要额外一个换行符，如果不加，可能无法正确解析title
text = re.sub("\n。", "。", text)   # 发现可能会出现错误换行的情况
text = re.sub("\n；", "；", text)   # 发现可能会出现错误换行的情况
if not text.endswith("\n"): # list 解析逻辑需要额外一个换行符
    text += "\n"
```

## 为正文文本添加"P"标签区分段落
默认的导出结果中是没有`<p>`标签的，因此可能无法正确区分段落
```python
# extract.py 文件约243行处
# from
page.append(line.lstrip(":"))
# to 
page.append("<p>" + "&nbsp;"*4 + line.lstrip(":") + "</p>")

# extract.py 文件约289行处
# from
page.append(line)
# to 
page.append("<p>%s</p>"%line)   # 原文这里会有个注释 “first line”

# extract.py 文件约292行处
# from
page.append(line)
# to 
page.append("<p>%s</p>"%line)
```

## 避免解析 list 出错
```python
# extract.py 文件约266行处
# 添加一行代码
if l - 1 < 0 or l - 1 >= len(line): continue
# 实践中源代码解析list可能会有越界错误（当遇到空list的情况）
```

## 避免出现重复 title
```python
# extract.py 文件约286行处
# 删除两行代码，避免生成的 html 中重复出现 title 的情况
for (i, v) in items:
    page.append(v)
# 原代码在处理 title 时，除了生成标记<h2>title</h2>，还会在下一行重复正文 title
```

## 添加繁体转简体功能
```python
# pip install opencc-python-reimplemented
# 然后在类 Extractor 中添加
class Extractor():
    self.cc = OpenCC("t2s")
# extract.py 文件约989行处
# 在代码
text = self.clean_text(text, html_safe == html_safe)
# 后添加逻辑
if isinstance(text, list) and all(isinstance(_x, str) for _x in text):
    for i in range(len(text)):
        text[i] = text[i].replace(r"-{}-", "")  # 这个和转简体没关系，存粹顺手加一个过滤规则
        text[i] = self.cc.convert(text[i])  # opencc 繁体转简体
```

## 修改标题的标记方式
```python
# extract.py 文件约 1008行处
# from 
header += self.title + "\n\n"
footer = "\n</doc>\n"
# to 
header += "<h1>%s</h1>"%(self.cc.convert(self.title)) + "\n"
footer = "</doc>\n\n"
```