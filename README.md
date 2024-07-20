# Introduction

Music Blocks is a _Visual Programming Language_ and collection of
_manipulative tools_ for exploring musical and mathematical concepts
in an integrative and entertaining way.

“Music is the arithmetic of sounds as optics is the geometry of light.” — Claude Debussy 

“Mathematics and music are in a sense the same thing, with a different application.” — Roger Sessions

## Music Blocks 

Visit the Music Blocks website for a hands-on experience:
[https://musicblocks.sugarlabs.org](https://musicblocks.sugarlabs.org).

Or download Music Blocks from the [Google Play Store](https://play.google.com/store/apps/details?id=my.musicblock.sugarlab)

Additional background on why we combine music and programming can be found
[here](./WhyMusicBlocks.md).

## Getting Started with Music Blocks Lesson Plan Generator 

There are several projects in Music Blocks, but only some of them have lesson plans. Additionally, only some of the concepts have lesson plans. In this project, we aim to create a chatbot that can generate lesson plans and answer your questions about Music Blocks.

Projects in Music Blocks - 

![alt tag](./images/projects.png)

## Setting Project locally

1. Install a virtual environment of python

   ```bash
   pip install virtualenv
   ```

2. setting up the Python environment
   ```bash
   python3 -m venv .venv
   ```
3. Activate the virtual environment
   
    on windows
   ```bash
   .\.venv\Scripts\activate
   ```
   on macOS and Linux:
   ```bash
   source .venv/bin/activate
   ```

5. Clone the Music Blocks repository to your local machine:

  ```bash
  git clone https://github.com/khadar1020/Musicblocks-AI.git
  ```

5. Navigate to the cloned repository:

 ```bash
 cd Musicblocks-AI
 ```
6. Install Ollama locally on your PC

   Link to the website - [https://ollama.com/](https://ollama.com/).
7. Install llama3 on your PC using ollama and terminal

  ```bash
  ollama pull llama3
  ```
8. Checking whether llama3 is installed or not
   
  ```bash
  ollama run llama3
  ```
9. Also pull nomic-embed-text from ollama

 ```bash
 ollama pull nomic-embed-text
 ```
10. Then install all the requirements for the project
   
  ```bash
  pip install -r requirements.txt
  ```
11. create an .env file in the folder of the project
    
12. Go to terminal of our project and run this command then you receive a CHAINLIT_AUTH_SECRET key place it in the .env file as CHAIN_AUTH_SECRET = "<your_secret_key>"

  ```bash
  chainlit create-secret
  ```
13. Go to this website and create an API key by logging in https://literalai.com/ and place it in the .env file as LITERAL_API_KEY="<your_literal_API_key>"
14. Then run this command to create a vector database this will take time 
  ```bash
  python ingest.py
  ```
15. Then run this final command to run the project locally on your PC 
  ```bash
  chainlit run model.py -w
  ```   
    

## License

Music Blocks AI is licensed under the [AGPL](https://www.gnu.org/licenses/agpl-3.0.en.html)
