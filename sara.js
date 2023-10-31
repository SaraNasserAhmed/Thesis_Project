
const axios = require('axios');
const { JSDOM } = require('jsdom');
const fs = require('fs');

// Read the contents of your index.html file
const html = fs.readFileSync('./Decider/index.html', 'utf-8');

// Create a virtual DOM from the HTML
const { window } = new JSDOM(html);

// Access the document and window objects
const document = window.document;



// <p> tags:
let p_tags = document.getElementsByTagName("p");
let p_array = [];

for ( p of p_tags ) {
    p_array.push(p.textContent);
}

// <li> tags:
let li_array = [];
let li_tags = document.querySelectorAll("li");

for ( li of li_tags ) {
    li_array.push(li.textContent);
}


const dataToSend = {
    p: p_array,
    li: li_array
};



const flaskServerURL = 'http://localhost:5000'; 

axios.post(`${flaskServerURL}/receiveData`, dataToSend)
    .then((response) => {
        console.log(response.data); // Response from Flask
        // Handle the response data as needed
    })
    .catch((error) => {
        console.error('Error:', error);
    });