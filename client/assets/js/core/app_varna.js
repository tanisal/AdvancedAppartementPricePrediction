function getRoomsValue() {
var uiRooms = document.getElementsByName("uiRooms");
console.log(uiRooms)
for(var i in uiRooms) {
    if(uiRooms[i].checked) {
        return parseInt(i)+1;
    }
}
return -1; // Invalid Value
}


function onGetEstimatedPrice_varna() {
console.log("Get Estimated Price button clicked");
var rooms = getRoomsValue();
var total_square = document.getElementById("uiSquare");
var floor = document.getElementById("uiFloor");
var build = document.getElementById("uiBuild")
var location = document.getElementById("uiLocations");
var estPrice = document.getElementById("uiEstimatedPrice");

var url = "http://127.0.0.1:5000/predict_appartament_price_varna";
// var url = "/api/predict_home_price_varna";

$.post(url, {
    total_square: parseFloat(total_square.value),
    floor: parseInt(floor.value),
    rooms: rooms,
    location: location.value,
    build: build.value
},function(data, status) {
    console.log(data.estimated_price);
    estPrice.innerHTML = "<h6>The Predicted Price is:  "+ "<span class='result'>"+data.estimated_price.toString()+"€</span>" + "</h6>"
    console.log(status);
});
 }


function onPageLoad_varna() {
    console.log( "document loaded" );
    var url = "http://127.0.0.1:5000/get_location_names_varna"; // 
    // var url = "/api/get_location_names_varna";
    $.get(url,function(data, status) {
        console.log("got response for get_location_names request");
        if(data) {
            var locations = data.locations;
            var uiLocations = document.getElementById("uiLocations");
            $('#uiLocations').empty();
            for(var i in locations) {
                var opt = new Option(locations[i]);
                $('#uiLocations').append(opt);
            }
        }
    });
  }

window.onload = onPageLoad_varna;
