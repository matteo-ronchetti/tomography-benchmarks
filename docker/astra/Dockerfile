FROM ubuntu:20.04 as BASE

RUN apt-get update
ENV TZ=Europe/Rome
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get install -y build-essential autoconf libtool automake libboost-dev git libxml2 wget && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN /bin/bash Miniconda3-latest-Linux-x86_64.sh -b

RUN wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
RUN /bin/bash cuda_11.1.1_455.32.00_linux.run --toolkit --silent

RUN git clone https://github.com/astra-toolbox/astra-toolbox.git
ENV PATH = $PATH:/root/miniconda3/bin/
RUN conda install numpy scipy
RUN /root/miniconda3/bin/pip install cython
RUN cd astra-toolbox/build/linux && ./autogen.sh && \
    ./configure --with-cuda=/usr/local/cuda-11.1 --with-python --with-install-type=module
RUN cd astra-toolbox/build/linux && make && make install

# Cleanup
RUN conda clean -ayf
RUN cd /usr/local/cuda-11.1/ && rm -r nsight-compute-2020.2.1 nsight-systems-2020.3.4 libnvvp nsightee_plugins
RUN rm /usr/local/cuda-11.1/targets/x86_64-linux/lib/libcusolver*


FROM nvidia/cuda:11.1-runtime-ubuntu20.04
COPY --from=BASE /root/miniconda3 /root/miniconda3
# COPY --from=BASE /usr/local/cuda-11.1 /usr/local/cuda-11.1
ENV PATH = $PATH:/root/miniconda3/bin/
# ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.1/targets/x86_64-linux/lib/