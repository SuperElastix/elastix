Welcome to elastix!

This directory contains the documentation of elastix.

doxygen: In this directory we put files that help to generate the doxygen pages in a style that we like. These pages give a lot of information about the setup of elastix, and can be found at http://elastix.dev/doxygen/index.html.

manual: This directory is a submodule that links to the (most recent commit in the) git repository that contains the LaTeX files and the images needed to generate the manual of elastix.
Be aware that cloning the elastix repository doesn't also clone the submodules. In order to do this the following command should be typed after clone the repo:

  git submodule update --init --remote --merge

to get the most recent version:

  cd dox/manual

  git checkout main

exampleinput: In this directory you can find example images and parameter files, that can help you to get started with elastix. In the directory of this README file you can find scripts that show you how to call elastix.
