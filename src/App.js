import React, { createRef, useEffect } from 'react';

import { Hands } from '@mediapipe/hands/hands';
import { Camera } from '@mediapipe/camera_utils/camera_utils';
import * as tfjs from '@tensorflow/tfjs';
import * as cocossd from "@tensorflow-models/coco-ssd";

var LeftTargetLandmark = [];
var RightTargetLandmark = [];
var currentUserAction = 'study';
var userActions = {
  'userAction' : {
    'study' : {'time': 0, 'count' : 0},
    'drowsiness' : {'time': 0, 'count' : 0},
    'smartphone' : {'time': 0, 'count' : 0},
    'leave' : {'time': 0, 'count' : 0},}
  };

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
   
    hand.onResults(handDetection);
    camera.start()
  }, [])

  async function smartPhoneDetection (network) {
    const detections = await network.detect(videoElementRef.current);
    detections.forEach(prediction => {
      const text = prediction['class']; 

      if (text === 'cell phone') {
        currentUserAction = 'smartphone';
        userActions['userAction'][currentUserAction]['count'] += 1;
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

    if (results.multiHandedness !== undefined) {
      drowsinessDetection(results);

      if (results.multiHandedness.length === 2 && !(currentUserAction === 'smartphone' || currentUserAction === 'drowsiness')) {
        currentUserAction = 'study';
        userActions['userAction'][currentUserAction]['count'] += 1;
      } else if(results.multiHandedness.length === 1) {
        console.log('one hand');
      } 
    } else{
      currentUserAction = 'leave';
      userActions['userAction'][currentUserAction]['count'] += 1;
    }
    
    canvasCtx.restore()
  }

  function drowsinessDetection(handInfo) {
    var leftFingerDetection = true;
    var rightFingerDetection = true;
    var bothFingerDetection = true;
    var isRightFingerXMove;
    var isRightFingerYMove;
    var isLeftFingerXMove;
    var isLeftFingerYMove;

    if (handInfo.multiHandLandmarks.length === 2) {
      if (RightTargetLandmark !== []) {
        if (RightTargetLandmark[4] !== undefined) {
          isRightFingerXMove = Math.abs((handInfo.multiHandLandmarks[0][4]['x']-RightTargetLandmark[4]['x']).toFixed(2)) > 0;
          isRightFingerYMove = Math.abs((handInfo.multiHandLandmarks[0][4]['y']-RightTargetLandmark[4]['y']).toFixed(2)) > 0;
          rightFingerDetection = isRightFingerXMove && isRightFingerYMove;
        }
      }
      if (LeftTargetLandmark !== []) {
        if (LeftTargetLandmark[4] !== undefined) {
          isLeftFingerXMove = Math.abs((handInfo.multiHandLandmarks[1][4]['x']-LeftTargetLandmark[4]['x']).toFixed(2)) > 0;
          isLeftFingerYMove = Math.abs((handInfo.multiHandLandmarks[1][4]['y']-LeftTargetLandmark[4]['y']).toFixed(2)) > 0;
          leftFingerDetection = isLeftFingerXMove && isLeftFingerYMove;
        }
      }

      bothFingerDetection = rightFingerDetection && leftFingerDetection;
      if (!bothFingerDetection) {
        currentUserAction = 'drowsiness';
        userActions['userAction'][currentUserAction]['count'] += 1;
      }

      RightTargetLandmark = handInfo.multiHandLandmarks[0];
      LeftTargetLandmark = handInfo.multiHandLandmarks[1];

    } else {
      if (handInfo.multiHandedness[0]['label'] === 'Left') {
        if (LeftTargetLandmark !== []) {
          if (LeftTargetLandmark[4] !== undefined) {
            isLeftFingerXMove = Math.abs((handInfo.multiHandLandmarks[0][4]['x']-LeftTargetLandmark[4]['x']).toFixed(2)) > 0;
            isLeftFingerYMove = Math.abs((handInfo.multiHandLandmarks[0][4]['y']-LeftTargetLandmark[4]['y']).toFixed(2)) > 0;
            leftFingerDetection = isLeftFingerXMove || isLeftFingerYMove;
          }
        }
        if (!leftFingerDetection) {
          currentUserAction = 'drowsiness';
          userActions['userAction'][currentUserAction]['count'] += 1;
        }
        LeftTargetLandmark = handInfo.multiHandLandmarks[0];

      } else {
        if (RightTargetLandmark !== []) {
          if (RightTargetLandmark[4] !== undefined) {
            isRightFingerXMove = Math.abs((handInfo.multiHandLandmarks[0][4]['x']-RightTargetLandmark[4]['x']).toFixed(2)) > 0;
            isRightFingerYMove = Math.abs((handInfo.multiHandLandmarks[0][4]['y']-RightTargetLandmark[4]['y']).toFixed(2)) > 0;
            rightFingerDetection = isRightFingerXMove || isRightFingerYMove;

          }
        }

        if (!rightFingerDetection) {
          currentUserAction = 'drowsiness';
          userActions['userAction'][currentUserAction]['count'] += 1;
        }
        
        RightTargetLandmark = handInfo.multiHandLandmarks[0];
      }
    }
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