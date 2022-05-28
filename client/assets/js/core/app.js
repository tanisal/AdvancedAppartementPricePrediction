function getFloorValue() {
    var uiFloor = document.getElementsByName("uiFloor");
    for(var i in uiFloor) {
      if(uiFloor[i].checked) {
          return parseInt(i)+1;
      }
    }
    return -1; // Invalid Value
  }

function getRoomsValue() {
var uiRooms = document.getElementsByName("uiRooms");
for(var i in uiRooms) {
    if(uiRooms[i].checked) {
        return parseInt(i)+1;
    }
}
return -1; // Invalid Value
}

function getBuildingValue() {
var uiBuilding = document.getElementsByName("uiBuilding");
for(var i in uiBuilding) {
    if(uiBuilding[i].checked) {
        return parseInt(i)+1;
    }
}
return -1; // Invalid Value
}


function onGetEstimatedPrice() {
console.log("Get Estimated Price button clicked");
var total_square = document.getElementById("uiSquare");
var floors = getFloorValue();
var rooms = getRoomsValue();
var location = document.getElementById("uiLocations");
var estPrice = document.getElementById("uiEstimatedPrice");

var url = "http://127.0.0.1:5000/predict_appartament_price"; //Use this if you are NOT using nginx which is first 7 tutorials
//var url = "/api/predict_home_price"; // Use this if  you are using nginx. i.e tutorial 8 and onwards

$.post(url, {
    total_square: parseFloat(total_square.value),
    floor: floor,
    rooms: rooms,
    location: location.value
},function(data, status) {
    console.log(data.estimated_price);
    estPrice.innerHTML = "<h1>" + data.estimated_price.toString() + " Euro</h1>";
    console.log(status);
});
}


function onPageLoad() {
    console.log( "document loaded" );
     var url = "http://127.0.0.1:5000/get_location_names"; // Use this if you are NOT using nginx which is first 7 tutorials
    //var url = "/api/get_location_names"; // Use this if  you are using nginx. i.e tutorial 8 and onwards
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

window.onload = onPageLoad
