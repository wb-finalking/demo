## 使用 PyPI 发布与维护自己的模块

下面简述如何使用 PyPI 发布与维护自己的模块

> CSDN/[python如何发布自已pip项目](http://blog.csdn.net/fengmm521/article/details/79144407)
> 该博客介绍了主要的流程，但是部分操作依据过时了

1. [注册自己的 PyPI 账户](https://pypi.python.org/pypi?%3Aaction=register_form)

2. 创建项目，并发布到 github （并不是必要的步骤，但是建议这么做）

3. 编写 setup.py 文件 (参考)

   ```python
   """
   * Project Name: Tensorflow Template
   * Author: wb
   * Mail: 21721060@zju.edu.cn
   * Created Time:  2018-10-16 11:49:53
   """
   from setuptools import setup, find_packages
   
   install_requires = [
       'bunch',
       # 'tensorflow',  # install it beforehand
   ]
   
   setup(
       name="tensorflow_template",
       version="0.2",
       keywords=("wb", "tensorflow", "template", "tensorflow template"),
       description="A tensorflow template for quick starting a deep learning project.",
       long_description="A deep learning template with tensorflow...",
       license="MIT Licence",
       url="https://github.com/wb-finalking/tensorflow/utils",
       author="wb",
       author_email="21721060@zju.edu.cn",
       packages=find_packages(),
       include_package_data=True,
       platforms="any",
       install_requires=install_requires
   )
   ```

   校验 setup.py

   ```
   > python setup.py check
   ```

4. 创建 `$HOME/.pypirc` 文件保存 PyPI 账户信息

   ```
   [distutils]
   index-servers = pypi
   
   [pypi]
   repository: https://upload.pypi.org/legacy/
   username: wb
   password = ***
   ```

   > 这里使用 : 和 = 都可以

5. 打包并上传

   ```bash
   > python setup.py sdist
   > twine upload dist/wbtools-VERSION.tar.gz
   ```

   如果没有 twine，需要先安装

   ```bash
   pip install twine
   ```

     PS: python setup.py upload 已经弃用

6. 上传成功，就可以在 https://pypi.python.org/pypi 查看刚刚上传的项目了

7. 接下来就可以 pip 自己刚刚发布的项目了（如果你没有改过 pip 源的话）