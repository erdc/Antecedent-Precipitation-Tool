<html>
   <head>
   <title>Web WIMP Map</title>
   <meta http-equiv="content-type" content="text/html; charset=utf-8"/>
   <link rel=stylesheet type="text/css" href="http://cyclops.deos.udel.edu/wimp/public_html/style.css">
   <script>
      //function changeScreenSize(w,h)
      //{ window.resizeTo( w,h ) ; } 
   </script>                             
</head>

<!-- <body onload="changeScreenSize(1000,890)"> --> 

<body margin-left:"5%" margin-right:"5%">

<script type='text/javascript' src='http://cyclops.deos.udel.edu/wimp/public_html//jquery-1.3.2.js'></script>
<script language="JavaScript">

   function point_it(event) {
      var x ;
      var y ;
      dr = Math.acos (-1.0)/180.0 ;
      if ( ! event.offsetX )
	x = event.layerX - $(event.target).position().left ;
      else
	x = event.offsetX ;
      if ( ! event.offsetY )
	y = event.layerY - $(event.target).position().top ;
      else
	y = event.offsetY ;
      x = x - 28 ;
      y = y - 30 ;
      sin_y1 = 90.0 - y * 0.5 ;
      sin_y2 = sin_y1 - 0.5 ;
      if ( sin_y1 > 90.0 ) { sin_y1 = 90.0 ; }
      if ( sin_y1 < -90.0 ) { sin_y1 = -90.0 ; }
      if ( sin_y2 > 90.0 ) { sin_y2 = 90.0 ; }
      lat = ( Math.asin (sin_y1/90.0) + Math.asin(sin_y2/90.0) ) * 0.5 ;
      temp = lat/dr ;
      lat = (Math.ceil(temp*2.0)) * 0.5 ;
      if ( lat == -0 ) { lat = 0 ; }
      lon = -180.0 + x * 0.5 ;
      if ( lon > 180.0 ) { lon = 180.0 ; }
      if ( lon < -180.0 ) { lon = -180.0 ; }
      if ( lat > 90.0 ) { lat = 90.0 ; }
      if ( lat < -90.0 ) { lat = -90.0 ; }
      if ( lon >= -180.0 && lon <= 180.0 && lat >= -90.0 && lat <= 90.0 ) {
	document.wimp_lonlat.long.value = lon ;
	document.wimp_lonlat.lati.value = lat ;
      }
   }

</script>

<left>

<?php

global $yname ;

$yname = $_POST['yname'] ;

?>

<p>
<hr>
<font color=blue>
   <h2>Click on the location where you would like to evaluate the Climatic Water Balance.</h2>
</font>
<hr>

<form name="wimp_lonlat" method="post" action="wimp_map_input.php" target="_blank" >
<span style='font-size:16.0pt;color:blue'><b>Average Annual Moisture Index
<sup>4</sup>&nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp
            &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp
  <span style='font-size:12.0pt;color:blue'> 
  Longitude: <input type="text" name="long" size="5" value="0" /> &nbsp
  Latitude: <input type="text" name="lati" size="5" value="0" /> <br>
<input type="image" src="http://cyclops.deos.udel.edu/wimp/public_html/wimp_map.gif" name="wimp_map" onmousemove="point_it(event)" height="420" width="780"/>
<input type="hidden" name='yname' value="<?php echo $yname ?>" </input> 
<br>
</form> 
</span>
<hr>
<p><span style='font-size:16pt;color:blue'>
<a href="http://cyclops.deos.udel.edu/wimp/public_html/index.html">Return to the 
<i>Web</i>WIMP Front Page</a>
<hr>
<address style='margin-left:.05in;text-indent:-.05in;font-size:12pt'><sup><span
style='color:blue;font-style:normal'>4</span></sup><span style='color:blue;
font-style:normal'>Willmott, C.J. and J.J. Feddema, 1992. &nbsp
A more rational climatic moisture index. &nbsp </span>
<b style='mso-bidi-font-weight:normal'><span style='color:blue;mso-bidi-font-style:
normal'>Professional Geographer</span></b><span style='color:blue;font-style:
normal'>, <b style='mso-bidi-font-weight:normal'>44</b>,
84-87.<o:p></o:p></span></address>
</left>
</body>
</html>
