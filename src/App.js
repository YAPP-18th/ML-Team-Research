import React, { createRef, useEffect } from 'react';

import { Hands } from '@mediapipe/hands/hands';
import { HAND_CONNECTIONS } from '@mediapipe/hands';
import { Camera } from '@mediapipe/camera_utils/camera_utils';
// import {
//   drawConnectors,
//   drawLandmarks,
// } from '@mediapipe/drawing_utils/drawing_utils';
import * as tfjs from '@tensorflow/tfjs';
import * as cocossd from "@tensorflow-models/coco-ssd";

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
      minDetectionConfidence: 0.8,
      minTrackingConfidence: 0.8
    })
    const camera = new Camera(videoElementRef.current, {
      onFrame: async () => {
        await hand.send({ image: videoElementRef.current })
        await smartPhoneDetection(coco);
      },
      width: 1280,
      height: 720,
    })
    
    hand.onResults(handDetection)
    camera.start()
  }, [])

  async function smartPhoneDetection (network) {
    const detections = await network.detect(videoElementRef.current);
    detections.forEach(prediction => {
      const text = prediction['class']; 

      if (text === 'cell phone') {
        console.log(text);
      }
    });
  }

  function handDetection(results) {
    const canvasCtx = canvasElementRef.current.getContext('2d')
    const canvasElement = canvasElementRef.current

    canvasCtx.save()
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height)
    canvasCtx.drawImage(
      results.image,
      0,
      0,
      canvasElement.width,
      canvasElement.height
    )
    canvasCtx.lineWidth = 1

    // if (results.multiHandLandmarks) {
    //   for (const landmarks of results.multiHandLandmarks) {
    //     drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS,
    //       {color: '#00FF00', lineWidth: 5});
    //       drawLandmarks(canvasCtx, landmarks, {color: '#FF0000', lineWidth: 1});
    //     }
    //   }

    if (results.multiHandedness !== undefined) {
      if (results.multiHandedness.length === 2) {
        console.log('two hands');
      } else if(results.multiHandedness.length === 1) {
        console.log('one hand');
      } 
    } else{
      console.log('No hand');
    }
    
    canvasCtx.restore()
  }

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