import os


# Set environment variables
os.system('export CMAKE_ARGS="-DLLAMA_CUBLAS=on"')
os.system('export FORCE_CMAKE="1"')
# Run pip install for requirements.txt
os.system("pip install -r requirements.txt")
