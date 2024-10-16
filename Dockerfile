FROM tensorflow/tensorflow:2.15.0-gpu
# LABEL maintainer="https://github.com/mimbres/neural-audio-fp" 

RUN apt-get update
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y ffmpeg wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN apt-get install -y screen curl grep sed dpkg git tmux nano htop cuda-compiler-12-0 sox && \
    apt-get clean

RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip

RUN mkdir /src && mkdir /src/neuralfp-dataset
COPY . /src/peaknetfp
WORKDIR /src/peaknetfp

# FAISS install through conda
RUN wget --quiet \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh && \ 
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
COPY environment.yml /tmp/
RUN conda env create -f /tmp/environment.yml
RUN conda init

RUN echo "bash /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
RUN echo "conda activate base" >> ~/.bashrc
RUN echo "conda init" >> ~/.bashrc
RUN echo "conda activate fp" >> ~/.bashrc
RUN echo "conda env list" >> ~/.bashrc

RUN bash ~/.bashrc
RUN pip3 install tensorflow[and-cuda]

CMD [ "/bin/bash" ]
