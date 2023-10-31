
// -----------------------------------------------------------------------------------
// TEXT:
// -----------------------------------------------------------------------------------
let textBtn = document.getElementById("text-btn");
let textCard = document.getElementById("text-card");
let body = document.body;


// AJAX Steps:
// -----------
textBtn.addEventListener("click", loadAJAX);

async function loadAJAX(){

    // 1- Create AJAX Request:
    let req = new XMLHttpRequest;

    // 2- Identify the location of the data, true for async:
    req.open("GET", "./data/data.csv", true);

    // 3- Send the request:
    await req.send();

    // 4- When the load is finished, do the following:
    req.onload = () => {
        if(req.status === 200){ //successful
            let ReceivedData = req.responseText;

            const new_p_element = document.createElement("p");
            const new_node = document.createTextNode(ReceivedData);
            new_p_element.appendChild(new_node);
            textCard.appendChild(new_p_element);
        }
    
    };
};









// -----------------------------------------------------------------------------------
// JSON:
// -----------------------------------------------------------------------------------
let jsonBtn = document.getElementById("json-btn");
let jsonCard = document.getElementById("json-card");

jsonBtn.addEventListener("click", function(){
    let xhr = new XMLHttpRequest();
    xhr.open("GET", "./data/data.json", true);
    xhr.send();
    xhr.onload = () => {
        if(xhr.status === 200){ //successful
            let JSONReceivedData = xhr.responseText;
            let jsonData = JSON.parse(JSONReceivedData);
            let htmlTemplate = `<ul class="list-group mt-1">
                <li class="list-group-item">Name: ${jsonData.name} </li>
                <li class="list-group-item">Age: ${jsonData.age} </li>
                <li class="list-group-item">Mobile: ${jsonData.mobile} </li>
                </ul>`;
            jsonCard.innerHTML = htmlTemplate;
        };
    };
});




// -----------------------------------------------------------------------------------
// API:
// -----------------------------------------------------------------------------------
let apiBtn = document.getElementById("api-btn");
let apiCard = document.getElementById("api-card");
let apiURL = "https://jsonplaceholder.typicode.com/users";

apiBtn.addEventListener("click", function() {
    let xhr = new XMLHttpRequest();
    xhr.open("GET",apiURL, true);
    xhr.send();
    xhr.onload = () => {
        if (xhr.status === 200 ){
            let apiData = xhr.responseText;
            let jsonData = JSON.parse(apiData);

            // display:
            let htmlTemplate = "";
            for (let user of jsonData) {
                htmlTemplate += `<ul class="list-group m-3">
                <li class="list-group-item">ID: ${user.id}</li>
                <li class="list-group-item">Name: ${user.name}</li>
                <li class="list-group-item">Email: ${user.email}</li>
                <li class="list-group-item">Street: ${user.address.street}</li>
                </ul>
                `;
            };
            apiCard.innerHTML = htmlTemplate;
            
        }
    }
});
