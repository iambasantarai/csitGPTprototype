"use strict";

const dots = {
  interval: 80,
  frames: ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
};

let frameIndex = 0;
let spinnerInterval;

const startSpinner = () => {
  spinnerInterval = setInterval(() => {
    process.stdout.write(`\r${dots.frames[frameIndex]} Generating answer...`);
    frameIndex = (frameIndex + 1) % dots.frames.length;
  }, dots.interval);
};

const stopSpinner = () => {
  clearInterval(spinnerInterval);
  process.stdout.write(`\r`);
};

export { startSpinner, stopSpinner };
