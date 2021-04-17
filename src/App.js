import React, {useRef} from "react";
import './App.css';
import * as tfjs from '@tensorflow/tfjs';
import * as cocossd from "@tensorflow-models/coco-ssd";

import Webcam from "react-webcam";
import { drawRect } from "./utilities";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  const runSmartPhone = async () => {
    const net = await cocossd.load();

    setInterval(() => {
      detectSmartPhone(net);
    }, 100);
  };
 
  const detectSmartPhone = async (net) => {
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {

      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;


      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;

      const obj = await net.detect(video);
      const ctx = canvasRef.current.getContext("2d");
      drawRect(obj, ctx);
    }
  };
  
 runSmartPhone();
  return (
    <div className="App">
      <header className="App-header">
        <Webcam ref={webcamRef} style={
          {
            position:'absolute',
            marginLeft:'auto',
            marginRight:'auto',
            left:0,
            right:0,
            textAlign:'center',
            zIndex:9,
            width:640,
            height:480,
          }
        } />
        <canvas ref={canvasRef} style= {
          {
            position:'absolute',
            marginLeft:'auto',
            marginRight:'auto',
            left:0,
            right:0,
            textAlign:'center',
            zIndex:9,
            width:640,
            height:480,
          }
        }/>
      </header>
    </div>
  );
}

export default App;
