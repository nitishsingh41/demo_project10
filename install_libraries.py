import os


# Set environment variables and install llama_cpp_python for cuda
os.system('CMAKE_ARGS="-DLLAMA_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python')
# Run pip install for requirements.txt
os.system("pip install -r requirements.txt")
