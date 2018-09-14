.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

----------------------
Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/cloudbopper/mihifepe/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

mihifepe could always use more documentation, whether as part of the
official mihifepe docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/cloudbopper/mihifepe/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.

------------
Get Started!
------------

Ready to contribute? Here's how to set up `mihifepe` for local development.

1. Fork the `mihifepe` repo on GitHub.
2. Clone your fork locally::

        git clone git@github.com:your_name_here/mihifepe.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development::

        mkvirtualenv -p python3 mihifepe
        cd mihifepe/
        pip install -r requirements_dev.txt -r requirements.txt

4. Create a branch for local development::

        git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass the pylint/flake8 and the tests
   for all python versions::

        tox

   You may use pyenv_ to install and test with multiple python versions.

.. _pyenv: https://github.com/pyenv/pyenv

6. Commit your changes and push your branch to GitHub::

        git add .
        git commit -m "Your detailed description of your changes."
        git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

-----------------------
Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for Python 3.5, 3.6 and 3.7. Check
   https://travis-ci.org/cloudbopper/mihifepe/pull_requests
   and make sure that the tests pass for all supported Python versions.

----
Tips
----

To run a subset of tests::

    py.test tests.test_mihifepe

To run pylint::

    pylint mihifepe tests

To run flake8::

    flake8 mihifepe tests

---------
Deploying
---------

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in `CHANGELOG.rst`_).
Then run::

    bumpversion patch # possible: major / minor / patch
    git push
    git push --tags

.. _`CHANGELOG.rst`: https://github.com/cloudbopper/mihifepe/blob/master/CHANGELOG.rst

Travis will then deploy to PyPI if tests pass.
