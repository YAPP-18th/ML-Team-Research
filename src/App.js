import React, { createRef, useEffect } from 'react';

import { Hands } from '@mediapipe/hands/hands';
import { Camera } from '@mediapipe/camera_utils/camera_utils';
import * as tfjs from '@tensorflow/tfjs';
import * as cocossd from "@tensorflow-models/coco-ssd";
import {smartPhoneDetection, handDetection} from './userActionDetection';

function App() {
  const canvasElementRef = createRef()
  const videoElementRef = createRef()

  useEffect(async () => {
    const coco =await cocossd.load();
    const hand = new Hands({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
      },
    })
    hand.setOptions({
      maxNumHands: 2,
      minDetectionConfidence: 0.85,
      minTrackingConfidence: 0.85
    })
    const camera = new Camera(videoElementRef.current, {
      onFrame: async () => {
        await hand.send({ image: videoElementRef.current })
        await smartPhoneDetection(coco, videoElementRef.current);
      }, 
      width: 1280,
      height: 720,
    })
   
    hand.onResults(handDetection);
    camera.start()
  }, [])

  return (
    <div className="App">
      <header className="App-header">
        <video
          style={{
            position: 'relative',
            top: '0',
            left: '0',
            right: '0',
            bottom: '0',
          }}
          ref={videoElementRef}
        ></video>
        <canvas
          ref={canvasElementRef}
          style={{
            position: 'absolute',
            left: '0',
            top: '0',
            width: '1280px',
            height: '720px',
          }}
        ></canvas>
      </header>
    </div>
  )
}

export default App