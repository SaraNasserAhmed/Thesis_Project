
// 1- Extract <p> tags:
// -------------------
let p_tags = document.getElementsByTagName("p");
let p_array = [];

for ( p of p_tags ) {
    p_array.push(p.textContent);
}
// p_array should be passed to the classifier 



// 2- Extract <li> tags:
// ---------------------
let li_array = [];
let li_tags = document.querySelectorAll("li");

for ( li of li_tags ) {
    li_array.push(li.textContent);
}
// li_array should be passed to the classifier 



// 3- Extract <h1> tags:
// ---------------------
let h1_array = [];
let h1_tags = document.getElementsByTagName("h1");

for ( h of h1_tags ) {
    h1_array.push(h.textContent);
}
// h1_array should be passed to the classifier 

