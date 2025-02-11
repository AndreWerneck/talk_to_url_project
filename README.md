# talk_to_url_project
Given a website URL, I will create a conversational interface to talk to URL content

First of all you should create a new virtual environment for the project. You can do it by typing the following commands on terminal:

`python -m venv YOUR-VENV-NAME`

`source OUR-VENV-NAME/bin/activate`

`pip install -r requirements.txt`

Once you're done with this, you need to install the LLM we are gonna use in this project. We are going to use the Mistral 7B Instruct quantized model. It will run in your local machine considerably quickly (even in cpu) thanks to llama-cpp, a library that optimizes LLMs to run in c++ either in CPU or GPU.

For installing it, you just need to run the **llm_install.py** file. You can do it by making 

`python3 llm_install.py`

in the command line. It should take between 25 and 30 minutes to install it. All you need is at least 4GB of disk space and a RAM bigger than 7GB. 

If you arrived here, you're done with the project setup! 

Now, all you have to do is : go to the "use_api.ipynb" file to play with the API. You just need to run

`uvicorn main:api --reaload` 

to start the API server on your localhost.  At the beginning of the notebook you have all the URLs and queries I used as example. Feel free to change them if you wish. 

Have fun!