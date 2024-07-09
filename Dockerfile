FROM chunglabmit/simpleelastix
#
# The Docker for chunglabmit/simpleelastix
#
# It does the SimpleElastix super build
#
# FROM ubuntu:20.04
# ENV DEBIAN_FRONTEND=noninteractive
#RUN apt-get update
#RUN apt-get install -y tzdata
#RUN echo "America/Boston" > /etc/timezone
#RUN dpkg-reconfigure -f noninteractive tzdata
#RUN apt-get install -y cmake swig python3 python3-dev tcl tcl-dev tk tk-dev
#RUN apt-get install -y git
#RUN git clone https://github.com/SuperElastix/SimpleElastix
#RUN mkdir /build
#RUN cd /build
#RUN cd /build;cmake -D PYTHON_EXECUTABLE=`which python3` ../SimpleElastix/SuperBuild
#RUN cd /build;make -j `nproc`
#RUN cd /build/SimpleITK-build/Wrapping/Python/Packaging;python3 setup.py install
RUN apt-get update
RUN apt-get install -y python3-pip

RUN apt-get install -y \
    libjpeg-dev \
    zlib1g-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libopenjp2-7-dev \
    libtiff5-dev \
    libwebp-dev \
    tk-dev \
    tcl-dev \
    blt-dev
#
# At this point six 1.11.0 is installed and Neuroglancer complains
#
RUN apt-get remove -y python3-six
RUN pip3 install six>=1.12.0
RUN pip3 install tornado==4.5.3 numpy
#
# Install Neuroglancer dev environment
#
RUN apt-get install -y wget
# RUN wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
RUN wget -qO- https://raw.githubusercontent.com/creationix/nvm/v0.33.11/install.sh | bash
RUN git clone --branch v2.0 https://github.com/google/neuroglancer
WORKDIR /neuroglancer
RUN /bin/bash -c "source $HOME/.nvm/nvm.sh && nvm install 10.6.0 && npm i"
# RUN /bin/bash -c "source $HOME/.nvm/nvm.sh && nvm install 10.6.0 && npm i"
# RUN export NVM_DIR="$HOME/.nvm";[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"; nvm install 10.6.0 ; npm i
#
# Install Nuggt and dependencies
#
ADD . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
RUN pip3 install --editable .
RUN pip3 install https://github.com/chunglabmit/mp_shared_memory/archive/master.zip
RUN pip3 install https://github.com/chunglabmit/blockfs/archive/master.zip
RUN pip3 install https://github.com/chunglabmit/precomputed-tif/archive/master.zip
