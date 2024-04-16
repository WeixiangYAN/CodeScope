FROM ubuntu:jammy
WORKDIR /root

ENV DEBIAN_FRONTEND noninteractive 
ENV TZ Etc/UTC

RUN apt-get update && \
    apt-get install -y \
        gnupg ca-certificates apt-utils \ 
        build-essential curl unzip && \
    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF 
RUN echo "deb https://download.mono-project.com/repo/ubuntu stable-focal main" | tee /etc/apt/sources.list.d/mono-official-stable.list 
RUN apt-get update && \
    apt-get install -y tzdata

RUN apt-get install -y mono-devel \
        gcc-12 g++-12

RUN apt-get update && \
    apt-get install -y software-properties-common python3.11-dev 

RUN curl -O https://download.oracle.com/java/21/latest/jdk-21_linux-x64_bin.deb && \
    apt-get update && \
    apt-get install -y -f libasound2 libc6-i386 libc6-x32 libxi6 libxtst6 && \
    dpkg -i jdk-21_linux-x64_bin.deb && \
    rm jdk-21_linux-x64_bin.deb

RUN curl -OL https://github.com/JetBrains/kotlin/releases/download/v1.7.20/kotlin-compiler-1.7.20.zip
RUN unzip kotlin-compiler-1.7.20.zip -d /usr/local && \
    rm kotlin-compiler-1.7.20.zip

RUN apt-get install -y python2

RUN curl -OL https://go.dev/dl/go1.19.2.linux-amd64.tar.gz
RUN tar -C /usr/local -xzf go1.19.2.linux-amd64.tar.gz && \
    rm go1.19.2.linux-amd64.tar.gz

RUN curl -OL https://downloads.python.org/pypy/pypy3.9-v7.3.9-linux64.tar.bz2
RUN tar -C /usr/local -xf pypy3.9-v7.3.9-linux64.tar.bz2 && \
    rm pypy3.9-v7.3.9-linux64.tar.bz2

RUN curl -OL https://downloads.python.org/pypy/pypy2.7-v7.3.9-linux64.tar.bz2
RUN tar -C /usr/local -xf pypy2.7-v7.3.9-linux64.tar.bz2 && \
    rm pypy2.7-v7.3.9-linux64.tar.bz2

RUN apt-get install -y clang-14

ARG NODE_VERSION=v16.17.1
ARG NODE_DISTRO=linux-x64
ARG NODE_ZIP=node-$NODE_VERSION-$NODE_DISTRO.tar.xz
RUN curl -OL https://nodejs.org/dist/$NODE_VERSION/$NODE_ZIP
RUN tar -C /usr/local -xJvf $NODE_ZIP && \
    rm $NODE_ZIP

ENV RUSTUP_HOME /opt/rust
ENV CARGO_HOME /opt/rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --no-modify-path
RUN apt-get install -y ruby-full

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing php8.1
RUN apt-get install -y libgmp-dev libmpfr-dev

ENV PATH $PATH:/usr/local/go/bin:/usr/local/kotlinc/bin:/usr/local/node-v16.17.1-linux-x64/bin:/opt/rust/bin

WORKDIR /root

RUN update-alternatives --install /usr/bin/java java /usr/lib/jvm/jdk-21-oracle-x64/bin/java 100 && \
    update-alternatives --install /usr/bin/javac javac /usr/lib/jvm/jdk-21-oracle-x64/bin/javac 100 && \
    update-alternatives --install /usr/bin/jar jar /usr/lib/jvm/jdk-21-oracle-x64/bin/jar 100 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 100 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 100 && \
    update-alternatives --install /usr/bin/pypy2 pypy2 /usr/local/pypy2.7-v7.3.9-linux64/bin/pypy2 100 && \
    update-alternatives --install /usr/bin/pypy3 pypy3 /usr/local/pypy3.9-v7.3.9-linux64/bin/pypy3 100

RUN apt-get install -y python3-pip gperf
RUN pip install flask gunicorn flask-cors gmpy2 Cython jsonlines fire

RUN curl -o libseccomp.tar.gz -L https://github.com/seccomp/libseccomp/releases/download/v2.5.4/libseccomp-2.5.4.tar.gz
RUN tar -xzvf libseccomp.tar.gz && cd libseccomp-2.5.4 && chmod +x configure
WORKDIR /root/libseccomp-2.5.4
RUN ./configure --prefix=/usr --enable-python && make
RUN make install
WORKDIR /root/
RUN rm -rf libseccomp*


ENV RUN_UID 1586
ENV RUN_GID 1586
ENV NUM_WORKERS 16
ENV GUNICORN_PORT 5000
ENV WORKER_CFG_DB /root/worker_cfg_db.csv
ENV LOG_LEVEL info

# RUN groupadd -g ${RUN_GID} runner${RUN_GID} && useradd -M runner${RUN_UID} -g ${RUN_GID} -u ${RUN_UID}

EXPOSE ${GUNICORN_PORT}

RUN mkdir execution_engine
COPY ./execution_engine /root/execution_engine

WORKDIR /root/execution_engine

CMD bash start_engine.sh