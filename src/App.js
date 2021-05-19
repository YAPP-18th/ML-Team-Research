import React, { createRef, useEffect } from 'react';

import { Hands } from '@mediapipe/hands/hands';
import { HAND_CONNECTIONS } from '@mediapipe/hands';
import { Camera } from '@mediapipe/camera_utils/camera_utils';
import {
  drawConnectors,
  drawLandmarks,
} from '@mediapipe/drawing_utils/drawing_utils';

function App() {
  const canvasElementRef = createRef()
  const videoElementRef = createRef()

  useEffect(async () => {
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
      },
      width: 1280,
      height: 720,
    })
    
    hand.onResults(onResults)
    camera.start()
  }, [])

  function onResults(results) {
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

    if (results.multiHandLandmarks) {
      for (const landmarks of results.multiHandLandmarks) {
        drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS,
          {color: '#00FF00', lineWidth: 5});
          drawLandmarks(canvasCtx, landmarks, {color: '#FF0000', lineWidth: 1});
        }
      }

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