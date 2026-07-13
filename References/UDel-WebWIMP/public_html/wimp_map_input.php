<html xmlns="https://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head>
   <title>Web WIMP Input Information</title>
   <meta http-equiv="content-type" content="text/html; charset=utf-8"/>
   <link rel=stylesheet type="text/css" href="http://cyclops.deos.udel.edu/wimp/public_html/style.css">
   <script>
      //function changeScreenSize(w,h)
      //{ window.resizeTo( w,h ) ; } 
   </script>                             
</head>
<body onload="changeScreenSize(900,1000)" leftmargin="5%" rightmargin="5%"> 

<?php
error_reporting(E_PARSE);

global $yname ;

if ( ! $_POST['_submit_lonlat_'] ) {
   $yname = $_POST['yname'] ;
   $lon = $_POST['long'] ;
   $lat = $_POST['lati'] ;
   $rlon = $lon + 0.25 ;
   $rlat = $lat - 0.25 ;
   $height = set_lon_lat()  ;
   show_data() ;
} else {
   $yname = $_POST['yname'] ;
   $lon = $_POST['Longitude'] ;
   $lat = $_POST['Latitude'] ;
   $yname = urldecode( $yname ) ;
   $rlon = -179.75 + 0.5 * (floor(floor( ($lon + 180.0) * 10.0)/5.0)) ;
   $rlat = 89.75 - 0.5 * (floor(floor((90.0 - $lat)*10.0)/5.0)) ;

   $height = set_lon_lat () ;
   show_data () ;
} 

show_form () ;

//********************************************************************************

function set_lon_lat () {

   global $yname ;
   global $lon, $rlon ;
   global $lat, $rlat ;
   global $height ;

   if ( ! is_numeric ($lon) ) {                                                                                           
      print "<p style='font-size:14.0pt;color:blue'>" ;
      print "Please input a numeric number for Longitude. &nbsp" ;
      print "<INPUT TYPE='button' VALUE='Previous page' onClick='history.back()'>" ;
      die ;
   }                                                                                                                      
   if ( ! is_numeric ($lat) ) {                                                                                           
      print "<p style='font-size:14.0pt;color:blue'>" ;
      print "Please input a numeric number for Latitude. &nbsp" ;
      print "<INPUT TYPE='button' VALUE='Previous page' onClick='history.back()'>" ;
      die ;
   }                                                                                                                      

   $link = mysqli_connect ( '127.01','wimp','','air_precip_clim' ) ;

   if ( ! $link ) {
      die('Could not connect: ' . mysqli_error()) ;
      exit() ;
   }
   if ( ! mysqli_set_charset ( $link, 'utf8' ) ) {
      die('Unable to set database connection encoding. ' . mysqli_error() ) ;
      exit() ;
   }

   //if ( ! mysqli_select_db ( $link, 'air_precip_clim' ) ) {
      //die('Unable to locate the database needed. ' . mysqli_error() ) ;
      //exit() ;
   //}

   $rlon = mysqli_real_escape_string($link, $rlon) ;
   $rlat = mysqli_real_escape_string($link, $rlat) ;

   $query = sprintf("SELECT height FROM height WHERE lon='%s' AND lat='%s'", $rlon, $rlat ) ;

   $results = mysqli_query ( $link, $query ) ;

   $height_dat = mysqli_fetch_assoc( $results )  ;

   $height = $height_dat['height'] ;

   if ( is_null ($height) ) {
      print "<p style='color:blue;font-size:18.0pt'>" ;
      print "<hr>\n" ;
      print "<h1><font color=blue>Information Needed by <i>Web</i>WIMP<br>\n" ;
      print "Project: $yname</font></h1>\n" ;
      print "<hr>\n" ;
      print "<p style='margin:0in;margin-bottom:.0001pt'><span style='font-size:14.0pt;color:blue'><br>\n" ;
      print "Longitude (<i>&#955;</i>) and latitude (<i>&#966;</i>):<o:p></o:p></span></p>\n" ;
      print "<form method='POST' action='$_SERVER[PHP_SELF]'>\n" ;
      print "<p><font color=blue><p style='margin-left:.5in'>\n" ;
      print "The <i>&#955</i> and <i>&#966</i> coordinates for your point are:<br>\n" ;
      print "Longitude: " ; print "<input type='text' name='Longitude' size='8' value=$lon> &nbsp \n" ;
      print "Latitude: " ; print "<input type='text' name='Latitude' size='8' value=$lat>\n" ;
      print "<p style='margin:0in;margin-bottom:.0001pt'><span style='font-size:18.0pt;color:blue'>\n" ;
      print "This location falls on a large body of water. Please select another location." ;
      print "<span style='font-size:14.0pt;color:blue'>" ;
      print "<br><br>Revise the values within the boxes above and then " ;
      print "click here &nbsp" ;
      print "<input type='submit' value='Revise Longitude and Latitude'>\n" ;
      print "<input type='hidden' name='_submit_lonlat_' value=1 >\n" ;
      $yname = urlencode( $yname ) ;
      print "<input type='hidden' name='yname' value=$yname ) >\n" ;
      print "</form> \n" ;
      die () ;
   } else {
      print "<hr>\n" ;
      print "<h1><font color=blue>Information Needed by <i>Web</i>WIMP<br>\n" ;
      print "Project: $yname</font></h1>\n" ;
      print "<hr>\n" ;
      print "<p style='margin:0in;margin-bottom:.0001pt'><span style='font-size:14.0pt;color:blue'>\n" ;
      print "Longitude (<i>&#955;</i>) and latitude (<i>&#966;</i>):<o:p></o:p></span></p>\n" ;
      print "<form method='POST' action='$_SERVER[PHP_SELF]'>\n" ;
      print "<p><font color=blue><p style='margin-left:.5in'>\n" ;
      print "The <i>&#955</i> and <i>&#966</i> coordinates for your point, as well as its elevation, are:<br>\n" ;
      print "Longitude: " ; print "<input type='text' name='Longitude' size='8' value=$lon> &nbsp \n" ;
      print "Latitude: " ; print "<input type='text' name='Latitude' size='8' value=$lat>\n" ;
      print "&nbsp&nbsp Elevation: &nbsp $height m\n" ;
      print "<br>" ;
      print "Note: you may revise  <i>&#955</i> and <i>&#966</i>, if you wish.<br> \n" ;
      print "If you wish to change the values, revise the values within the boxes above and then <br>" ;
      print "click here &nbsp" ; 
      print "<input type='submit' value='Revise Longitude and Latitude'>\n" ;
      print "<input type='hidden' name='_submit_lonlat_' value=1 >\n" ; 
      $yname = urlencode( $yname ) ;
      print "<input type='hidden' name='yname' value=$yname ) >\n" ;
      print "</form> \n" ;
   } 

   mysqli_close($link) ;
   return $height ;
} 

//******************************************************************************

function show_form () {
   
   global $lon,$rlon,$lat,$rlat ;
   global $air_temp,$precip,$height ;
   global $yname ;

   $month_id = array('JA','FE','MR','AP','MA','JU','JL','AU','SE','OC','NV','DE','Ann') ;
   $month_na = array('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Ann') ;
   $d_t = array('-5.0','-4.0','-3.0','-2.0','-1.0','0.0','1.0','2.0','3.0','4.0','5.0','6.0','7.0','8.0','9.0','10.0') ;
   $d_p = array('-50.0','-40.0','-30.0','-20.0','-15.0','-10.0','-5.0','0.0','5.0','10.0','15.0',
                 '20.0','30.0','40.0','50.0') ;

   print "<form method='POST' action='wimp_calc.php' target='_blank'>\n" ;
   print "<p><span style='font-size:14.0pt;color:blue'>\n" ;
   print "<hr> \n" ;
   print "<p><span style='font-size:14.0pt;color:blue'>Soil water-holding capacity (<i style='mso-bidi-font-style:normal'>w</i>*):</span>\n" ;
   print "<p style='margin-left:.5in'>\n" ;
   print "Your <i>w</i>* is set initially at the default value of 150.0 mm.<br>\n" ;
   print "Enter a new value, if you wish:&nbsp \n" ;
   print "<select name=FC size=1>\n" ;
   print "<option>50" ; print "<option>75" ; print "<option>100" ; print "<option>125" ;
   print "<option selected>150" ; print "<option>200" ; print "<option>300" ;
   print "</select>\n" ;

   print "<hr> \n " ;
   print "<p><span style='font-size:14.0pt;color:blue'> The declining availability \n" ;
   print " function (<i>&#946;</i>):</span><span style='color:blue'><o:p></o:p></span></p> \n"  ;
   print "<p style='margin-left:.5in'> \n" ;
   print "Your <i>&#946</i> function is initially curve G on the following graph.<br> \n" ;
   print "You may select a new curve (A, B, C, D, E, F or H), if you wish: \n" ;
   print "<select name='resistance' size='1'> \n" ;
   print "<option>A" ; print "<option>B" ; print "<option>C" ; print "<option>D" ; print "<option>E" ;
   print "<option>F" ; print "<option selected>G" ; print "<option>H" ;
   print "</select> \n" ;
   print "<br><br><br> \n" ;
   print "<img src='http://cyclops.deos.udel.edu/wimp/public_html/wimp/resist.png' align='top'> \n" ;
   print "</p> \n" ;
   print "<hr>" ;

   print "<p><span style='font-size:14.0pt;color:blue'>
   Hypothetical climate change for: &nbsp &nbsp \n"  ;
   print " Longitude: " . $lon . "&nbsp&nbsp Latitude: " . $lat . " <o:p></o:p></span></p> \n" ;
   print "<p style='margin-left:.5in'>If you wish to prescribe a hypothetical change in AIR TEMPERATURE,
   enter the monthly <b><i>or</i></b> annual change in (<sup>o</sup>C)<br>
   (note: the default is no change or 0.0<sup>o</sup> C):<br> \n";

   $air_temp_ann_orig = $air_temp['Ann'] ;
   $precip_ann_orig = $precip['Ann'] ;

   print "<table style='font-size:12.0pt;color:blue'> \n" ;
   print "<tr> \n" ;
   for ( $i = 0 ; $i <= 5 ; $i++ ) {
      print "<td align=left> $month_na[$i] :</td> \n" ;
      print "<td align=right> \n" ;
      print "<select name=TC_$i size=1> \n" ;
      for ( $ii = 0 ; $ii <= 4 ; $ii++ ) {
         print "<option>" ; printf('%.1f', $d_t[$ii]) ;
      }
      print "<option selected>" ; printf ('%.1f', $d_t['5']) ;
      for ( $ii = 6 ; $ii <= 15 ; $ii++ ) {
         print "<option>" ; printf('%.1f', $d_t[$ii]) ;
      }
      print "</select>" ;
      print "</td> \n" ;
      }
      print "</tr> \n" ;
      print "<tr>" ;
   for ( $i = 6 ; $i <= 12 ; $i++ ) {
      print "<td align=left> $month_na[$i] :</td>" ;
      print "<td align=right>" ;
      print "<select name=TC_$i size=1> \n" ;
      for ( $ii = 0 ; $ii <= 4 ; $ii++ ) {                                                                                  
         print "<option>" ; printf('%.1f', $d_t[$ii]) ;                                                                      
      }                                                                                                                     
      print "<option selected>" ; printf ('%.1f', $d_t['5']) ;                                                              
      for ( $ii = 6 ; $ii <= 15 ; $ii++ ) {                                                                                  
         print "<option>" ; printf('%.1f', $d_t[$ii]) ;                                                                     
      }                                                                                                                     
      print "</select>" ;
      print "</td> \n" ;
   }
   print "</tr> \n" ;
   print "</table> \n" ;

   print "<p style='margin-left:.5in'>If you wish to prescribe a hypothetical change in PRECIPITATION,
   enter the monthly <b><i>or</i></b> annual change as a percentage (%)
   (note: the default is no change or 0.0 %):<br> \n" ;

   print "<table style='font-size:12.0pt;color:blue'>" ;
   print "<tr>" ;
   for ( $i = 0 ; $i <= 5 ; $i++ ) {
      print "<td align=left>$month_na[$i]:</td>" ;
      print "<td align=right>" ;
      print "<select name=PMM_$i size=1>" ;
      for ($ii = 0 ; $ii <= 6 ; $ii++ ) {
         print "<option>" ; printf('%.1f',$d_p[$ii]) ;
      }
      print "<option selected>" ; printf('%.1f', $d_p['7']) ;
      for ($ii = 8 ; $ii <= 14 ; $ii++ ) {
         print "<option>" ; printf('%.1f', $d_p[$ii]) ;
      }
      print "</select> \n" ;
      print "</td>" ;
   }
   print "</tr>" ;
   print "<tr>" ;
   for ( $i = 6 ; $i <= 12 ; $i++ ) {
      print "<td align=left>$month_na[$i]:</td>" ;                                                                    
      print "<td align='right'>" ;
      print "<select name=PMM_$i size=1>" ;
      for ($ii = 0 ; $ii <= 6 ; $ii++ ) {
         print "<option>" ; printf('%.1f', $d_p[$ii]) ;
      }
      print "<option selected>" ; printf('%.1f', $d_p['7']) ;
      for ($ii = 8 ; $ii <= 14 ; $ii++ ) {
         print "<option>" ; printf('%.1f',  $d_p[$ii]) ;
      }
      print "</select> \n" ;
      print "</td>" ;                                                                                                     
   }                                                                                                                      
   print "</tr>" ;                                                                                                        
   print "</table>" ;
   print "</p>" ;
   print "<hr>" ;

   print "<p><span style='color:blue'></span><span style='font-size:14.0pt;color:blue'>\n" ;                          
   print "Start over: <o:p></o:p></span>\n" ;                     
   print "<p style='margin-left:.5in'>If you wish to reset all of the information needed by
   <it>Web</it>WIMP to default values, click here " ;
   print "<input type='reset'>" ;
   print "<hr>" ;
   print "<p><span style='font-size:16.0pt;color:blue'> \n" ;
   print "Calculate the monthly water balance &#8212; click here\n" ;
   print "<input type='hidden' name='Latitude' value=$rlat>" ;
   print "<input type='hidden' name='Longitude' value=$rlon>" ; 
   print "<input type='hidden' name='air_temp_ann_orig' value=$air_temp_ann_orig>" ;
   print "<input type='hidden' name='precip_ann_orig' value=$precip_ann_orig>" ;
   print "<input type='hidden' name='height' value=$height>" ;
   //$yname = urlencode( $yname ) ;
   print "<input type='hidden' name='yname' value=$yname ) >\n" ;
   print "<input type='submit' value='Water Balance'>\n" ;
   print "</form>"  ;
   print "</p>" ;
}

function show_data () {

   global $lon,$rlon ;
   global $lat,$rlat ;
   global $height ;
   global $air_temp ;
   global $precip ;
   global $yname ;
   
   $link = mysqli_connect ( '127.01','wimp','','air_precip_clim' ) ;

   if ( ! $link ) {
      die('Could not connect: ' . mysqli_error()) ;
      exit() ;
   }

   if ( ! mysqli_set_charset ( $link, 'utf8' ) ) {
      die('Unable to set database connection encoding. ' . mysqli_error() ) ;
      exit() ;
   }

   //if ( ! mysqli_select_db ( $link, 'air_precip_clim' ) ) {
      //die('Unable to locate the database needed. ' . mysqli_error() ) ;
      //exit() ;
   //}

   $month_id = array('JA','FE','MR','AP','MA','JU','JL','AU','SE','OC','NV','DE','Ann') ;
   $month_na = array('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Ann') ;

   $rlon = mysqli_real_escape_string($link, $rlon) ;
   $rlat = mysqli_real_escape_string($link, $rlat) ;

   $query = sprintf("SELECT * FROM air_temperature  WHERE lon='%s' AND lat='%s'",$rlon, $rlat ) ;

   $results = mysqli_query ( $link, $query ) ;
   $air_temp = mysqli_fetch_assoc( $results ) ;

   $air_temp['Ann'] = 0.0 ;
   for ( $i = 0 ; $i <=11 ; $i++ ) {
      $air_temp['Ann']=$air_temp['Ann'] + $air_temp["$month_id[$i]"]/12.0 ;
   }

   $query = sprintf("SELECT * FROM precipitation  WHERE lon='%s' AND lat='%s'",$rlon, $rlat ) ;
   $results = mysqli_query ( $link, $query ) ;
   $precip = mysqli_fetch_assoc( $results ) ;

   $precip['Ann']=0.0 ;
   for ($i = 0 ; $i <= 11 ; $i++ ) {
      $precip['Ann'] = $precip['Ann'] + $precip["$month_id[$i]"] ;
   }

   print "<br><br>\n" ;
   print "<table border=1 style='font-size:12.0pt;color:blue' > \n" ;
   print "<tr> \n" ;
   print "<td align=left>Month</td> \n" ;
   for ( $i = 0 ; $i <= 12 ; $i++ ) {
         print "<td align=left> $month_na[$i]</td> \n" ;
   }
   print "</tr> \n" ; 
   print "<tr> \n" ;
   print "<td align=left>Air Temperature (<sup>o</sup>C)</td> \n" ;
      for ( $i = 0 ; $i <= 12 ; $i++ ) {
      print "<td align=right>" ; printf ('%.1f',$air_temp[$month_id[$i]]) ; print "</td> \n" ;
   }                                                                 
   print "</tr> \n" ;
   print "<tr> \n" ; 
   print "<td align=left>Precipitation (mm)</td> \n" ;
      for ( $i = 0 ; $i <= 12 ; $i++ ) {                                                                                     
      print "<td align=right>" ; printf ('%.1f',$precip[$month_id[$i]]) ; print "</td> \n" ;
   }                                                                                            
   print "</tr> \n" ;
   print "</table> \n" ;
 
   mysqli_close($link) ;
} 
?>

</body>
</html>
