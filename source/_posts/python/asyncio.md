title: Python的异步编程asyncio
description: 记录一些Python的异步编程asyncio的例子
tags:
- Python
- asyncio
---

最近做模型部署的时候涉及到了Python的异步编程，在此记录一些典型case。知乎上关于Python异步的介绍已经有很多了，在此不做赘述，可参考:

- <https://zhuanlan.zhihu.com/p/609616282>.
- <https://zhuanlan.zhihu.com/p/632857605>.
- <https://blog.csdn.net/qq_69218005/article/details/129789977>.
- <https://zhuanlan.zhihu.com/p/610881194>.

另外推荐: <https://www.cnblogs.com/wztshine/p/14460847.html>.

## 两个正确使用asyncio的方式

### 文心一言的解释

在很多知乎上关于asyncio的教程中，通常可以看到有两种使用异步编程的方式，分别是`asyncio.run`和`loop.run_until_complete`。其中`asyncio.run`是封装好的，更加简单的启动方式，在`asyncio.run`的代码中也能找到`loop.create_task`和`loop.run_until_complete`的字眼，这里放一段文心一言关于二者区别的解释:


> `asyncio.run()` 和 `loop.run_until_complete()` 是 Python 中两种启动异步操作的常用方法，它们有 一些区别。
> 
> 1. **asyncio.run()**
> 
> `asyncio.run()` 是 Python 3.7 中新增的一个函数，它是为了简化异步编程而设计的。这个函数接受一个协程作为参数，然后在一个默认的事件循环中运行这个协程。这个函数的主要优点是它非常简单易用，你不需要手动创建或管理事件循环。
> 
> 例如：
> 
> ```python
> import asyncio
> 
> async def hello():
>     print('Hello')
>     await asyncio.sleep(1)
>     print('World')
> 
> asyncio.run(hello())
> ```
> 在这个例子中，我们使用 `asyncio.run()` 来运行 `hello()` 协程。
> 
> 2. **loop.run_until_complete()**
> 
> `loop.run_until_complete()` 是一个事件循环的方法，它接受一个协程作为参数，并在该事件循环中运行这个协程，直到协程完成。你需要手动获取事件循环，然后使用这个方法来运行协程。通常，你需要先使用 `asyncio.get_event_loop()` 来获取事件循环，然后调用 `run_until_complete()` 方法。
> 
> 例如：
> 
> ```python
> import asyncio
> 
> async def hello():
>     print('Hello')
>     await asyncio.sleep(1)
>     print('World')
> 
> loop = asyncio.get_event_loop()
> loop.run_until_complete(hello())
> ```
> 在这个例子中，我们使用 `loop.run_until_complete()` 来运行 `hello()` 协程。
> 
> 总的来说，`asyncio.run()` 和 `loop.run_until_complete()` 的主要区别在于：
> 
> * `asyncio.run()` 更加简单，不需要手动获取和设置事件循环，它会自动在默认的事件循环中运行协程。
> * `loop.run_until_complete()` 提供了更多的控制，允许你在特定的事件循环中运行协程，但是你需要手动获取和设置事件循环。

### 不使用异步的例子

上述的解释稍微有点模糊，这里我们假设一个场景。在做模型部署的时候，会抽象出例如，`run`、`download`和`prediction`这几个接口，其中假设`run`是最外层的调用，它接受一个`request_id`（当然实际情况下还有其他字段，例如图片的URL等，这里只是为了简化，用一个`request_id`代表），最终返回模型的推理结果。`download`用来下载要推理的图片，`prediction`是真正的模型推理阶段。当然，我们知道通常的写法python是无法并行的，如果我们就简单地写一个逻辑依次执行`download`和`prediction`，总的推理时间会比较久:

```python
import numpy as np, time
np.random.seed(19)

def prediction(request_id: str) -> float:
    wait_time = np.random.rand() * 3
    print(f"Enter <prediction> with request_id <{request_id}> and wait_time <{wait_time:.3f}>.")
    time.sleep(wait_time)  # 模拟prediction操作
    return wait_time

def download(request_id: str) -> float:
    wait_time = np.random.rand() * 3
    print(f"Enter <download> with request_id <{request_id}> and wait_time <{wait_time:.3f}>.")
    time.sleep(wait_time)  # 模拟下载操作
    return wait_time

def run(request_id: str) -> str:
    print(f"Enter <run> with request_id <{request_id}>.")
    start_time = time.time()
    download(request_id)
    prediction(request_id)
    total_time = time.time() - start_time
    print(f"Finish <run> with request_id <{request_id}> and time <{total_time:.3f}>.")
    return request_id

def main():
    results = [run(str(i)) for i in range(5)]
    return results
  
if __name__ == '__main__':
    start_time = time.time()
    results = main()
    total_time = time.time() - start_time
    print("="* 20 + "\n" + f"Finish with <{total_time:.3f}>.")
    print(f"Returns: \n {results}")
```

上面代码的输出是：

```bash
Enter <run> with request_id <0>.
Enter <download> with request_id <0> and wait_time <0.293>.
Enter <prediction> with request_id <0> and wait_time <2.284>.
Finish <run> with request_id <0> and time <2.577>.
Enter <run> with request_id <1>.
Enter <download> with request_id <1> and wait_time <0.741>.
Enter <prediction> with request_id <1> and wait_time <0.414>.
Finish <run> with request_id <1> and time <1.156>.
Enter <run> with request_id <2>.
Enter <download> with request_id <2> and wait_time <0.994>.
Enter <prediction> with request_id <2> and wait_time <0.249>.
Finish <run> with request_id <2> and time <1.244>.
Enter <run> with request_id <3>.
Enter <download> with request_id <3> and wait_time <2.016>.
Enter <prediction> with request_id <3> and wait_time <2.420>.
Finish <run> with request_id <3> and time <4.437>.
Enter <run> with request_id <4>.
Enter <download> with request_id <4> and wait_time <2.948>.
Enter <prediction> with request_id <4> and wait_time <1.907>.
Finish <run> with request_id <4> and time <4.856>.
====================
Finish with <14.271>.
Returns:
 ['0', '1', '2', '3', '4']
```

### 使用asyncio.run的例子

如果`download`和`prediction`这两个步骤是异步的话:

```python
import asyncio, numpy as np, time
np.random.seed(19)

async def prediction(request_id: str) -> float:
    wait_time = np.random.rand() * 3
    print(f"Enter <prediction> with request_id <{request_id}> and wait_time <{wait_time:.3f}>.")
    await asyncio.sleep(wait_time)  # 模拟prediction操作
    return wait_time

async def download(request_id: str) -> float:
    wait_time = np.random.rand() * 3
    print(f"Enter <download> with request_id <{request_id}> and wait_time <{wait_time:.3f}>.")
    await asyncio.sleep(wait_time)  # 模拟下载操作
    return wait_time

async def run(request_id: str) -> str:
    print(f"Enter <run> with request_id <{request_id}>.")
    start_time = time.time()
    await download(request_id)
    await prediction(request_id)
    total_time = time.time() - start_time
    print(f"Finish <run> with request_id <{request_id}> and time <{total_time:.3f}>.")
    return request_id

async def main():
    tasks = [run(str(i)) for i in range(5)]
    results = await asyncio.gather(*tasks)
    return results
  
if __name__ == '__main__':
    start_time = time.time()
    results = asyncio.run(main())
    total_time = time.time() - start_time
    print("="* 20 + "\n" + f"Finish with <{total_time:.3f}>.")
    print(f"Returns: \n {results}")
```

执行时间是:

```bash
Enter <run> with request_id <0>.
Enter <download> with request_id <0> and wait_time <0.293>.
Enter <run> with request_id <1>.
Enter <download> with request_id <1> and wait_time <2.284>.
Enter <run> with request_id <2>.
Enter <download> with request_id <2> and wait_time <0.741>.
Enter <run> with request_id <3>.
Enter <download> with request_id <3> and wait_time <0.414>.
Enter <run> with request_id <4>.
Enter <download> with request_id <4> and wait_time <0.994>.
Enter <prediction> with request_id <0> and wait_time <0.249>.
Enter <prediction> with request_id <3> and wait_time <2.016>.
Finish <run> with request_id <0> and time <0.543>.
Enter <prediction> with request_id <2> and wait_time <2.420>.
Enter <prediction> with request_id <4> and wait_time <2.948>.
Enter <prediction> with request_id <1> and wait_time <1.907>.
Finish <run> with request_id <3> and time <2.432>.
Finish <run> with request_id <2> and time <3.162>.
Finish <run> with request_id <4> and time <3.945>.
Finish <run> with request_id <1> and time <4.194>.
====================
Finish with <4.195>.
Returns:
 ['0', '1', '2', '3', '4']
```

异步对于`download`阶段来说很好理解，这通常不是一个计算密集的任务，所以使用异步方法可以重叠掉IO等待的时间。对于`prediction`阶段，这虽然是计算密集的，但是可以这么理解：

1. 假设使用的设备的性能大于模型的计算消耗，例如模型接受的batchsize为1，或者batch不大，达不到设备的性能上限
2. 在上述情况下，假设要推理100个request，那么就得依次推理100次，总的时间是100次的叠加
3. 当然也可以在模型导出onnx的时候设置动态batch，攒着多个request一起推理，不过通常部署的时候会用fastapi，这么写比较麻烦
4. 对于openvino、tensorrt（好像onnx也支持），它们都支持异步推理的，所以可以认为`prediction`阶段是可以做成异步的

总之，可以看到，如果两个阶段都是异步的（当然设备的计算性能要充足），那么总的推理时间会降低。

### loop.run_until_complete的例子

这种方式和`asyncio.run`差别不大，直接看代码吧:

```python
import asyncio, numpy as np, time
np.random.seed(19)

async def prediction(request_id: str) -> float:
    wait_time = np.random.rand() * 3
    print(f"Enter <prediction> with request_id <{request_id}> and wait_time <{wait_time:.3f}>.")
    await asyncio.sleep(wait_time)  # 模拟prediction操作
    return wait_time

async def download(request_id: str) -> float:
    wait_time = np.random.rand() * 3
    print(f"Enter <download> with request_id <{request_id}> and wait_time <{wait_time:.3f}>.")
    await asyncio.sleep(wait_time)  # 模拟下载操作
    return wait_time

async def run(request_id: str) -> str:
    print(f"Enter <run> with request_id <{request_id}>.")
    start_time = time.time()
    await download(request_id)
    await prediction(request_id)
    total_time = time.time() - start_time
    print(f"Finish <run> with request_id <{request_id}> and time <{total_time:.3f}>.")
    return request_id
  
if __name__ == '__main__':
    start_time = time.time()
    loop = asyncio.get_event_loop()  
    tasks = [run(str(i)) for i in range(5)]
    results = loop.run_until_complete(asyncio.gather(*tasks))
    total_time = time.time() - start_time
    print("="* 20 + "\n" + f"Finish with <{total_time:.3f}>.")
    print(f"Returns: \n {results}")
```

上述的输出是:

```bash
Enter <run> with request_id <0>.
Enter <download> with request_id <0> and wait_time <0.293>.
Enter <run> with request_id <1>.
Enter <download> with request_id <1> and wait_time <2.284>.
Enter <run> with request_id <2>.
Enter <download> with request_id <2> and wait_time <0.741>.
Enter <run> with request_id <3>.
Enter <download> with request_id <3> and wait_time <0.414>.
Enter <run> with request_id <4>.
Enter <download> with request_id <4> and wait_time <0.994>.
Enter <prediction> with request_id <0> and wait_time <0.249>.
Enter <prediction> with request_id <3> and wait_time <2.016>.
Finish <run> with request_id <0> and time <0.543>.
Enter <prediction> with request_id <2> and wait_time <2.420>.
Enter <prediction> with request_id <4> and wait_time <2.948>.
Enter <prediction> with request_id <1> and wait_time <1.907>.
Finish <run> with request_id <3> and time <2.432>.
Finish <run> with request_id <2> and time <3.163>.
Finish <run> with request_id <4> and time <3.944>.
Finish <run> with request_id <1> and time <4.193>.
====================
Finish with <4.194>.
Returns:
 ['0', '1', '2', '3', '4']
```

差了1ms就别纠结了。

## 反面例子1

在很多Python的异步编程教程中，都会说需要`async`和`await`配合使用，其中`await`标识这个执行是一个**可等待对象**。但是只有`await`是不够的，这里是一个反面例子，把`await asyncio.sleep`改成`time.sleep`，假设`prediction`阶段是阻塞的，此时即使我们调用的是`await prediction(request_id)`，总的调用时间依然很长:

```python
import asyncio, numpy as np, time
np.random.seed(19)

async def prediction(request_id: str) -> float:
    wait_time = np.random.rand() * 3
    print(f"Enter <prediction> with request_id <{request_id}> and wait_time <{wait_time:.3f}>.")
    time.sleep(wait_time)  # 模拟prediction操作，假设这一步是阻塞的
    return wait_time

async def download(request_id: str) -> float:
    wait_time = np.random.rand() * 3
    print(f"Enter <download> with request_id <{request_id}> and wait_time <{wait_time:.3f}>.")
    await asyncio.sleep(wait_time)  # 模拟下载操作
    return wait_time

async def run(request_id: str) -> str:
    print(f"Enter <run> with request_id <{request_id}>.")
    start_time = time.time()
    await download(request_id)
    await prediction(request_id)
    total_time = time.time() - start_time
    print(f"Finish <run> with request_id <{request_id}> and time <{total_time:.3f}>.")
    return request_id

async def main():
    tasks = [run(str(i)) for i in range(5)]
    results = await asyncio.gather(*tasks)
    return results
  
if __name__ == '__main__':
    start_time = time.time()
    results = asyncio.run(main())
    total_time = time.time() - start_time
    print("="* 20 + "\n" + f"Finish with <{total_time:.3f}>.")
    print(f"Returns: \n {results}")
```

上述的输出是:

```bash
Enter <run> with request_id <0>.
Enter <download> with request_id <0> and wait_time <0.293>.
Enter <run> with request_id <1>.
Enter <download> with request_id <1> and wait_time <2.284>.
Enter <run> with request_id <2>.
Enter <download> with request_id <2> and wait_time <0.741>.
Enter <run> with request_id <3>.
Enter <download> with request_id <3> and wait_time <0.414>.
Enter <run> with request_id <4>.
Enter <download> with request_id <4> and wait_time <0.994>.
Enter <prediction> with request_id <0> and wait_time <0.249>.
Finish <run> with request_id <0> and time <0.543>.
Enter <prediction> with request_id <3> and wait_time <2.016>.
Finish <run> with request_id <3> and time <2.559>.
Enter <prediction> with request_id <2> and wait_time <2.420>.
Finish <run> with request_id <2> and time <4.980>.
Enter <prediction> with request_id <4> and wait_time <2.948>.
Finish <run> with request_id <4> and time <7.928>.
Enter <prediction> with request_id <1> and wait_time <1.907>.
Finish <run> with request_id <1> and time <9.836>.
====================
Finish with <9.837>.
Returns:
 ['0', '1', '2', '3', '4']
```

这里关于`事件循环`、`可等待`、`Task`、`Future`、`coroutine object`的相关概念建议参考: <https://zhuanlan.zhihu.com/p/632857605>. 我的理解是：

1. 被`await`修饰的`function`称为`coroutine function`，`coroutine function`返回的是`coroutine object`
2. 当程序执行的时候，遇到`coroutine object`时将会直接继续调用执行，直到该`coroutine object`执行的过程中遇到无法继续执行的事件，这些事件包含`await`一个`Task`（返回的应该也是`Future`）或者`Future`
3. 当`await Future`的时候，当前任务（假设叫A）会`yield`出去并将程序控制权交还给`event loop`，告诉`event loop`当前任务A无法继续执行，`event loop`可以调用**其他任务**继续运行（这里发生的事情应该就是能解释异步如何做到了运行时的“并行”）
4. 当`Future`代表的任务执行完成后，`event loop`将择机再次安排任务A继续运行
5. 例如，`asyncio.sleep`的代码是:
```python
async def sleep(delay, result=None):
    """Coroutine that completes after a given time (in seconds)."""
    if delay <= 0:
        await __sleep0()
        return result

    loop = events.get_running_loop()
    future = loop.create_future()   # 注意到这里创建了一个future
    h = loop.call_later(delay,
                        futures._set_result_unless_cancelled,
                        future, result)
    try:
        return await future
    finally:
        h.cancel()
```
6. 因此，如果我们使用普通的`time.sleep()`，程序依然是顺序执行的，不会有“并行”的效果

## 反面例子2

显然，如果我们去除`download`阶段的异步，那么整个程序将变成最初的顺序执行

```python
import asyncio, numpy as np, time
np.random.seed(19)

async def prediction(request_id: str) -> float:
    wait_time = np.random.rand() * 3
    print(f"Enter <prediction> with request_id <{request_id}> and wait_time <{wait_time:.3f}>.")
    time.sleep(wait_time)  # 模拟prediction操作，假设这一步是阻塞的
    return wait_time

async def download(request_id: str) -> float:
    wait_time = np.random.rand() * 3
    print(f"Enter <download> with request_id <{request_id}> and wait_time <{wait_time:.3f}>.")
    time.sleep(wait_time)  # 模拟下载操作
    return wait_time

async def run(request_id: str) -> str:
    print(f"Enter <run> with request_id <{request_id}>.")
    start_time = time.time()
    await download(request_id)
    await prediction(request_id)
    total_time = time.time() - start_time
    print(f"Finish <run> with request_id <{request_id}> and time <{total_time:.3f}>.")
    return request_id

async def main():
    tasks = [run(str(i)) for i in range(5)]
    results = await asyncio.gather(*tasks)
    return results
  
if __name__ == '__main__':
    start_time = time.time()
    results = asyncio.run(main())
    total_time = time.time() - start_time
    print("="* 20 + "\n" + f"Finish with <{total_time:.3f}>.")
    print(f"Returns: \n {results}")
```

上述代码的输出是:

```bash
Enter <run> with request_id <0>.
Enter <download> with request_id <0> and wait_time <0.293>.
Enter <prediction> with request_id <0> and wait_time <2.284>.
Finish <run> with request_id <0> and time <2.577>.
Enter <run> with request_id <1>.
Enter <download> with request_id <1> and wait_time <0.741>.
Enter <prediction> with request_id <1> and wait_time <0.414>.
Finish <run> with request_id <1> and time <1.156>.
Enter <run> with request_id <2>.
Enter <download> with request_id <2> and wait_time <0.994>.
Enter <prediction> with request_id <2> and wait_time <0.249>.
Finish <run> with request_id <2> and time <1.244>.
Enter <run> with request_id <3>.
Enter <download> with request_id <3> and wait_time <2.016>.
Enter <prediction> with request_id <3> and wait_time <2.420>.
Finish <run> with request_id <3> and time <4.436>.
Enter <run> with request_id <4>.
Enter <download> with request_id <4> and wait_time <2.948>.
Enter <prediction> with request_id <4> and wait_time <1.907>.
Finish <run> with request_id <4> and time <4.856>.
====================
Finish with <14.272>.
Returns:
 ['0', '1', '2', '3', '4']
```

## 将自己的程序包装成可异步的形式

在实际使用中，我们难免会遇到一些函数不是可以直接用`await`修饰的，这些函数内部的实现也不是异步编程，此时有两种写法可以将这些非异步编程的函数包装成异步形式。一种是`asyncio.to_thread`，另一种是`loop.run_in_executor`，`Python3.9`版本之前只能用`loop.run_in_executor`。详细可参考: <https://www.cnblogs.com/wztshine/p/14460847.html>.

```python
# 源代码：可以看出，它获取了当前的上下文变量，然后调用了 run_in_executor() 去使用一个默认的线程，执行。
async def to_thread(func, /, *args, **kwargs):
    loop = events.get_running_loop()
    ctx = contextvars.copy_context()
    func_call = functools.partial(ctx.run, func, *args, **kwargs)
    return await loop.run_in_executor(None, func_call)
```

### 方案1

```python
import asyncio, numpy as np, time
np.random.seed(19)

def prediction(request_id: str) -> float:
    wait_time = np.random.rand() * 3
    print(f"Enter <prediction> with request_id <{request_id}> and wait_time <{wait_time:.3f}>.")
    time.sleep(wait_time)  # 模拟prediction操作，假设这一步是阻塞的
    return wait_time

async def download(request_id: str) -> float:
    wait_time = np.random.rand() * 3
    print(f"Enter <download> with request_id <{request_id}> and wait_time <{wait_time:.3f}>.")
    await asyncio.sleep(wait_time)  # 模拟下载操作
    return wait_time

async def run(request_id: str) -> str:
    print(f"Enter <run> with request_id <{request_id}>.")
    start_time = time.time()
    await download(request_id)
    core = asyncio.to_thread(prediction, request_id)
    await asyncio.create_task(core)
    total_time = time.time() - start_time
    print(f"Finish <run> with request_id <{request_id}> and time <{total_time:.3f}>.")
    return request_id

async def main():
    tasks = [run(str(i)) for i in range(5)]
    results = await asyncio.gather(*tasks)
    return results
  
if __name__ == '__main__':
    start_time = time.time()
    results = asyncio.run(main())
    total_time = time.time() - start_time
    print("="* 20 + "\n" + f"Finish with <{total_time:.3f}>.")
    print(f"Returns: \n {results}")
```

运行的结果是:

```bash
Enter <run> with request_id <0>.
Enter <download> with request_id <0> and wait_time <0.293>.
Enter <run> with request_id <1>.
Enter <download> with request_id <1> and wait_time <2.284>.
Enter <run> with request_id <2>.
Enter <download> with request_id <2> and wait_time <0.741>.
Enter <run> with request_id <3>.
Enter <download> with request_id <3> and wait_time <0.414>.
Enter <run> with request_id <4>.
Enter <download> with request_id <4> and wait_time <0.994>.
Enter <prediction> with request_id <0> and wait_time <0.249>.
Enter <prediction> with request_id <3> and wait_time <2.016>.
Finish <run> with request_id <0> and time <0.545>.
Enter <prediction> with request_id <2> and wait_time <2.420>.
Enter <prediction> with request_id <4> and wait_time <2.948>.
Enter <prediction> with request_id <1> and wait_time <1.907>.
Finish <run> with request_id <3> and time <2.432>.
Finish <run> with request_id <2> and time <3.162>.
Finish <run> with request_id <4> and time <3.944>.
Finish <run> with request_id <1> and time <4.193>.
====================
Finish with <4.195>.
Returns:
 ['0', '1', '2', '3', '4']
```

### 方案2

```python
import asyncio, numpy as np, time
np.random.seed(19)

def prediction(request_id: str) -> float:
    wait_time = np.random.rand() * 3
    print(f"Enter <prediction> with request_id <{request_id}> and wait_time <{wait_time:.3f}>.")
    time.sleep(wait_time)  # 模拟prediction操作，假设这一步是阻塞的
    return wait_time

async def download(request_id: str) -> float:
    wait_time = np.random.rand() * 3
    print(f"Enter <download> with request_id <{request_id}> and wait_time <{wait_time:.3f}>.")
    await asyncio.sleep(wait_time)  # 模拟下载操作
    return wait_time

async def run(request_id: str) -> str:
    print(f"Enter <run> with request_id <{request_id}>.")
    start_time = time.time()
    await download(request_id)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, prediction, request_id)
    total_time = time.time() - start_time
    print(f"Finish <run> with request_id <{request_id}> and time <{total_time:.3f}>.")
    return request_id

async def main():
    tasks = [run(str(i)) for i in range(5)]
    results = await asyncio.gather(*tasks)
    return results
  
if __name__ == '__main__':
    start_time = time.time()
    results = asyncio.run(main())
    total_time = time.time() - start_time
    print("="* 20 + "\n" + f"Finish with <{total_time:.3f}>.")
    print(f"Returns: \n {results}")
```

运行的结果是:

```bash
Enter <run> with request_id <0>.
Enter <download> with request_id <0> and wait_time <0.293>.
Enter <run> with request_id <1>.
Enter <download> with request_id <1> and wait_time <2.284>.
Enter <run> with request_id <2>.
Enter <download> with request_id <2> and wait_time <0.741>.
Enter <run> with request_id <3>.
Enter <download> with request_id <3> and wait_time <0.414>.
Enter <run> with request_id <4>.
Enter <download> with request_id <4> and wait_time <0.994>.
Enter <prediction> with request_id <0> and wait_time <0.249>.
Enter <prediction> with request_id <3> and wait_time <2.016>.
Finish <run> with request_id <0> and time <0.545>.
Enter <prediction> with request_id <2> and wait_time <2.420>.
Enter <prediction> with request_id <4> and wait_time <2.948>.
Enter <prediction> with request_id <1> and wait_time <1.907>.
Finish <run> with request_id <3> and time <2.432>.
Finish <run> with request_id <2> and time <3.162>.
Finish <run> with request_id <4> and time <3.945>.
Finish <run> with request_id <1> and time <4.193>.
====================
Finish with <4.195>.
Returns:
 ['0', '1', '2', '3', '4']
```