FROM huggingface/transformers-pytorch-latest-gpu
 
# Set work directory
WORKDIR /sample_folder
 
# Install PyTorch and related packages
RUN pip3 install pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html

RUN python3 --version


# Install Hugging Face Transformers
RUN pip3 install git+https://github.com/huggingface/transformers.git
RUN pip3 install git+https://github.com/huggingface/peft.git
RUN pip3 install git+https://github.com/huggingface/accelerate.git

RUN pip3 install bitsandbytes==0.42
 
# Install other Python packages
RUN pip3 install tqdm~=4.63.1 
RUN pip3 install spacy~=3.3.0 nltk~=3.7 gensim~=4.2.0 tensorboard~=2.9.0 protobuf~=3.19.0 
RUN pip3 install scikit-learn~=1.1.1 seqeval~=1.2.2 
RUN pip3 install datasets==2.16.1
RUN pip3 install trl
#RUN pip3 install tokenizers==0.14.0
 
RUN pip3 install evaluate==0.4.1
RUN pip3 install rouge_score==0.1.2


# Additional dependencies specified in the original Dockerfile
RUN pip3 install --upgrade nvidia-ml-py3==7.352.0
RUN pip3 install --upgrade wandb==0.15.7
RUN pip3 install chromadb==0.3.29


# Back to default frontend
ENV DEBIAN_FRONTEND=dialog