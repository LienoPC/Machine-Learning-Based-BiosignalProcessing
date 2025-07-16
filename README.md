# Machine-Learning-Based-BiosignalProcessing
This project is part of a thesis work related to integration of a biosensor-guided feedback loop inside a VR application. This solution is composed by a series of sub-modules that work
toghether to make the entire **WebSocket server** work.

## Features

This project runs a WebSocket server that provides:
- Endpoint route used by the biosensor through the ShimmerAPI (or any other APIs based on the needed biosensor)
- Endpoint route used by the user application to receive predictions by the model
- Data pre-processing and inference through a pre-trained model
- Statistics storage of running experiments

## Prerequisites

- Python 3.8+  
- CUDA-capable GPU (if you want GPU inference; otherwise uses CPU)  
- A build for your machine of [ShimmerAPI](https://github.com/LienoPC/ShimmerBioTransmit-MachineLearning-Interface) to be placed inside the folder ./ShimmerAPI
- Shimmer 3 GSR+ paired with the server machine
  
## Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/LienoPC/Machine-Learning-Based-BiosignalProcessing.git
   cd Machine-Learning-Based-BiosignalProcessing
   ```
2. **Install Python packages**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **Download your model checkpoint**
   Put your .pt file under "Model/SavedModels/" and set _MODEL_NAME_ and _CHECKPOINT_PATH_ inside **WebSocketApp.py**

## Usage

1. **Launch the Server**
    ```bash
    python WebSocketApp.py
    # or
    uvicorn WebSocketApp:app --host 0.0.0.0 --port 8000
    ```
2. **Start Client Application**
   Any client application that uses **FLPlugin** can connect to the websocket _ws://<server>:8000/ws/ubs_ (on local network) to listen for classification results. For testing purpose you can use [Game-Kitchen Scenario](https://github.com/UCC-Multimedia/Pacing-Game-Kitchen-Scenario), a VR Unity App that already integrates a client endpoint.
3. **Start Discovery Loop**
   Pressing _h_ will allow the server to send UDP packages in response to connection requests for _/ws/ubs_ route
   
## Contact
- Alberto Cagnazzo <s327678@studenti.polito.it>
Project Link: https://github.com/LienoPC/Machine-Learning-Based-BiosignalProcessing
