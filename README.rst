========
anamod
========

.. image:: https://img.shields.io/travis/cloudbopper/anamod.svg
        :target: https://travis-ci.org/cloudbopper/anamod
        :alt: Build status

.. image:: https://readthedocs.org/projects/anamod/badge/?version=latest
        :target: https://anamod.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

--------
Overview
--------

``anamod`` is a python library that implements model-agnostic algorithms for the feature importance analysis of trained black-box models.
It is designed to serve the larger goal of interpretable machine learning by using different abstractions over features to interpret
models. At a high level, ``anamod`` implements the following algorithms:

* Given a learned model and a hierarchy over features, (i) it tests feature groups, in addition to base features, and tries to determine
  the level of resolution at which important features can be determined, (ii) uses hypothesis testing to rigorously assess the effect of
  each feature on the model's loss, (iii) employs a hierarchical approach to control the false discovery rate when testing feature groups
  and individual base features for importance, and (iv) uses hypothesis testing to identify important interactions among features and feature
  groups. More details may be found in the following paper::

    Lee, Kyubin, Akshay Sood, and Mark Craven. 2019. “Understanding Learned Models by
    Identifying Important Features at the Right Resolution.”
    In Proceedings of the AAAI Conference on Artificial Intelligence, 33:4155–63.
    https://doi.org/10.1609/aaai.v33i01.33014155.

* Given a learned temporal or sequence model, it identifies important temporal features and interactions.
  More details may be found in the following paper::

    [In preparation]

``anamod`` supersedes and contains the functionality of the existing library ``mihifepe``, based on the first paper
(https://github.com/Craven-Biostat-Lab/mihifepe).
``mihifepe`` is maintained for legacy reasons but will not receive further significant updates.

``anamod`` uses the ``synmod`` library to generate synthetic data, including time-series data, to test and validate the algorithms
(https://github.com/cloudbopper/synmod).


-----
Usage
-----

[TODO]

See the complete documentation at https://anamod.readthedocs.io

------------
Installation
------------

The recommended installation method is via `virtual environments`_ and pip_.
In addition, you also need graphviz_ installed on your system.

When making the virtual environment, specify python3 (3.5+) as the python executable::

    mkvirtualenv -p python3 anamod

To install the latest stable release::

    pip install anamod

Or to install the latest development version from GitHub::

    pip install git+https://github.com/cloudbopper/anamod.git@master#egg=anamod

.. _pip: https://pip.pypa.io/
.. _virtual environments: https://python-guide-cn.readthedocs.io/en/latest/dev/virtualenvs.html
.. _graphviz: https://www.graphviz.org/

-----------
Development
-----------

Collaborations and contributions are welcome. If you are interested in helping with development, please take a look at:

https://anamod.readthedocs.io/en/latest/contributing.html

-------
License
-------

``anamod`` is free, open source software, released under the MIT license. See LICENSE_ for details.

.. _LICENSE: https://github.com/cloudbopper/anamod/blob/master/LICENSE

-------
Contact
-------

`Akshay Sood`_

.. _Akshay Sood: https://github.com/cloudbopper
