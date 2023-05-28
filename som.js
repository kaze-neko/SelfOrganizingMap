//! USEFUL LINKS
// https://medium.com/machine-learning-researcher/self-organizing-map-som-c296561e2117
// https://codepen.io/raman-mamedov/pen/JjdNWqX?editors=0010
// https://algowiki-project.org/ru/%D0%A1%D0%B0%D0%BC%D0%BE%D0%BE%D1%80%D0%B3%D0%B0%D0%BD%D0%B8%D0%B7%D1%83%D1%8E%D1%89%D0%B0%D1%8F%D1%81%D1%8F_%D0%BA%D0%B0%D1%80%D1%82%D0%B0_%D0%9A%D0%BE%D1%85%D0%BE%D0%BD%D0%B5%D0%BD%D0%B0_(%D0%B0%D0%BB%D0%B3%D0%BE%D1%80%D0%B8%D1%82%D0%BC_%D0%BA%D0%BB%D0%B0%D1%81%D1%82%D0%B5%D1%80%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D0%B8)


//! BASIC CONFIGURATION
const NUMBER_OF_INPUTS = 3;
const NUMBER_OF_NEURONS_BY_WIDTH = 100;
const NUMBER_OF_NEURONS_BY_HEIGHT = 100;
const NEURON_SIZE = 7;

//! DATA SET
// import { DATA_SET } from "./religion_dataset.js";
// import { DATA_SET } from "./expected_lifetime_dataset.js";
// import { DATA_SET } from "./suicide_rate_dataset.js";
// import { DATA_SET } from "./random_dataset.js";
import { DATA_SET } from "./weather_dataset.js";

//! MATH VARIABLES
// it is T variable in theory of SOM
const NUMBER_OF_TRAINING_STEPS = DATA_SET.length;
// it is t variable in theory of SOM
let numberOfTrainingStep = 0;
// sigma is size of the neighborhood around the winner neuron 
const sigma0 = Math.max(NEURON_SIZE * NUMBER_OF_NEURONS_BY_WIDTH, NEURON_SIZE * NUMBER_OF_NEURONS_BY_HEIGHT) / 5;
let sigma;
// 
const lambda = NUMBER_OF_TRAINING_STEPS / Math.log(sigma0);
// theta is influence rate
let theta;
const learningRate0 = 0.1;
let learningRate;

//! PAGE ELEMENTS
// canvas
let mapElement = document.getElementById("map");
mapElement.width = NEURON_SIZE * NUMBER_OF_NEURONS_BY_WIDTH;
mapElement.height = NEURON_SIZE * NUMBER_OF_NEURONS_BY_HEIGHT;
let context = mapElement.getContext("2d");
// menu
let menuElement = document.getElementById("menu");
menuElement.style.height = `${NUMBER_OF_NEURONS_BY_HEIGHT * NEURON_SIZE}px`;
// buttons
document.querySelector('#teach-button').addEventListener('click', () => {
  som.teach();
});
document.querySelector('#teach-all-button').addEventListener('click', () => {
  som.fullTeach();
});
document.querySelector('#teach-0-button').addEventListener('click', () => {
  som.renderForParam(0);
});
document.querySelector('#teach-1-button').addEventListener('click', () => {
  som.renderForParam(1);
});
document.querySelector('#teach-2-button').addEventListener('click', () => {
  som.renderForParam(2);
});
document.querySelector('#find-winner-for-cusom-input').addEventListener('click', () => {
  let data = JSON.parse(document.querySelector('#custom-input').value.replace(/'/g, '"'));
  let winner = som.findWinner(data);
  som.showNeuron(winner, "black");
});

//! NEURON
class Neuron {
  constructor(x_coordinate, y_coordinate) {
    this.x_coordinate = x_coordinate;
    this.y_coordinate = y_coordinate;
    // random value should be in |w| <= 1/sqrt(NUMBER_OF_INPUTS)
    // basic formula is Math.random() * (max - min) + min;
    this.weights = Array.from({length: NUMBER_OF_INPUTS}, () => Math.random() * ((1 / Math.sqrt(NUMBER_OF_INPUTS)) + (1 / Math.sqrt(NUMBER_OF_INPUTS))) + (-(1 / Math.sqrt(NUMBER_OF_INPUTS))));
  }
  //? function finding the average of all weight of neuron weights
  //? use to color map in grayscale like rgb(average*255, average*255, average*255)
  getAverageWeight() {
    let averageWeight = 0;
    for (let i = 0; i < this.weights.length; i++) {
      averageWeight += this.weights[i];
    }
    // should add (1 / Math.sqrt(NUMBER_OF_INPUTS)) to make the range like 0..1 not like -(1 / Math.sqrt(NUMBER_OF_INPUTS))..(1 / Math.sqrt(NUMBER_OF_INPUTS)) 
    return averageWeight / this.weights.length + (1 / Math.sqrt(NUMBER_OF_INPUTS));
  }
}

//! SELF ORGANIZING MAP
class SOM {
  constructor() {
    // array contains all neurons
    this.map = new Array(NUMBER_OF_NEURONS_BY_WIDTH * NUMBER_OF_NEURONS_BY_HEIGHT);
    for (let i = 0; i < this.map.length; i++) {
      // coordinates calculation
      let x_coordinate = Math.floor(i / NUMBER_OF_NEURONS_BY_WIDTH);
      let y_coordinate = i % NUMBER_OF_NEURONS_BY_WIDTH;
      // neuron initialisation
      this.map[i] = new Neuron(x_coordinate, y_coordinate);
    }
  }
  //? fucntion displaying map with colors
  render() {
    // clear canvas
    context.clearRect(0,0,mapElement.width,mapElement.height);
    // draw all neurons
    for (let i = 0; i < this.map.length; i++) {
      //* use grayscale if number over inputs more than three
      // context.fillStyle = `rgb(${this.map[i].getAverageWeight() * 255}, ${this.map[i].getAverageWeight() * 255}, ${this.map[i].getAverageWeight() * 255})`;
      context.fillStyle = `rgb(${this.map[i].weights[0] * 255}, ${this.map[i].weights[1] * 255}, ${this.map[i].weights[2] * 255})`;
      // x_coordinate = height, y_coordinate = width
      context.fillRect(this.map[i].y_coordinate * NEURON_SIZE, this.map[i].x_coordinate * NEURON_SIZE, NEURON_SIZE, NEURON_SIZE);
    }
  }
  //? function finding and returning winner neuron
  findWinner(input) {
    // this array contains the Euclidean distance from the current input data to each neuron
    let distancesMap = new Array(NUMBER_OF_NEURONS_BY_WIDTH * NUMBER_OF_NEURONS_BY_HEIGHT);
    let minDistanceIndex = 0;
    for (let i = 0; i < distancesMap.length; i++) {
      // Euclidean distance calculation
      let sum = 0;
      for (let j = 0; j < NUMBER_OF_INPUTS; j++) {
        sum += (input[j] - this.map[i].weights[j])**2;
      }
      distancesMap[i] = Math.sqrt(sum);
      // min distance searching 
      if (distancesMap[i] < distancesMap[minDistanceIndex]) {
        minDistanceIndex = i;
      }
    }
    // return win neuron
    return this.map[minDistanceIndex];
  }
  //? function displaying the input neuron colored in a given color
  showNeuron(neuron, color) {
    context.fillStyle = color;
    context.fillRect(neuron.y_coordinate * NEURON_SIZE, neuron.x_coordinate * NEURON_SIZE, NEURON_SIZE, NEURON_SIZE);
  }
  //? function displaying win neurons for all rows of the dataset on map at the same time
  showAllWinners() {
    for (let i = 0; i < DATA_SET.length; i++) {
      this.showNeuron(this.findWinner(DATA_SET[i]), "red");
    }
  }
  //? the network training on the whole dataset function for 
  fullTeach() {
    while (DATA_SET.length > 0) {
      this.teach();
    }
    this.render();
  }
  teach() {
    // get random input and delete it drom dataset
    if (DATA_SET.length > 0) {
      let randomInputIndex = Math.floor(Math.random() * DATA_SET.length);
      let randomInput = DATA_SET[randomInputIndex];
      DATA_SET.splice(randomInputIndex,1);
      // find winner neuron
      let winnerNeuron = this.findWinner(randomInput);
      this.showNeuron(winnerNeuron, "blue");
      // find all nearby neurons for winner neuron
      sigma = sigma0 * Math.exp(-(numberOfTrainingStep / lambda));
      let nearbyMap = [];
      for (let i = 0; i < this.map.length; i++) {
        // find distances between the winner neuron and the others
        let distance = Math.sqrt((this.map[i].x_coordinate - winnerNeuron.x_coordinate)**2 + (this.map[i].y_coordinate - winnerNeuron.y_coordinate)**2);
        // save only nearby neurons
        if (distance < sigma) {
          nearbyMap.push(this.map[i]);
        }
      }
      // iterate over all nearby neurons
      for (let i = 0; i < nearbyMap.length; i++) {
        // find the distance between winner neuron and nearby
        let distance =  Math.sqrt((nearbyMap[i].x_coordinate - winnerNeuron.x_coordinate)**2 + (nearbyMap[i].y_coordinate - winnerNeuron.y_coordinate)**2);
        // calculation theta and learning rate
        theta = Math.exp(-(distance**2 / (2 * sigma**2)));
        learningRate = learningRate0 * Math.exp(-(numberOfTrainingStep / lambda));
        // update nearby neuron weights
        for (let j = 0; j < nearbyMap[i].weights.length; j++) {
          nearbyMap[i].weights[j] += theta * learningRate * (randomInput[j] - nearbyMap[i].weights[j]);
        }
      }
      this.render();
      numberOfTrainingStep++;
    }
  }
  //? render grayscale map for each input parameter
  renderForParam(p) {
    context.clearRect(0,0,mapElement.width,mapElement.height);
    // draw all neurons
    for (let i = 0; i < this.map.length; i++) {
      context.fillStyle = `rgb(${this.map[i].weights[p] * 255}, ${this.map[i].weights[p] * 255}, ${this.map[i].weights[p] * 255})`;
      // x_coordinate = height, y_coordinate = width
      context.fillRect(this.map[i].y_coordinate * NEURON_SIZE, this.map[i].x_coordinate * NEURON_SIZE, NEURON_SIZE, NEURON_SIZE);
    }
  }
}

let som = new SOM();
som.render();
