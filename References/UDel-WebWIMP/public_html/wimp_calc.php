<html>
<head>
   <title>Web WIMP Climatic Water Balance Results</title>
   <meta http-equiv="content-type" content="text/html; charset=utf-8"/>
   <link rel=stylesheet type="text/css" href="http://cyclops.deos.udel.edu/wimp/public_html/style.css">
   </head>
   <script>
      //function changeScreenSize(w,h)
      //{ window.resizeTo( w,h ) ; } 
   </script>                             
</head>

<body onload="changeScreenSize(750,1000)" bgcolor="#FFFFFF" leftmargin="5%" rightmargin="5%">

<?php

   //foreach ( $_POST as $key => $value ) {
     //print $key . " " . $value . "<br>\n" ;
   //}

   $rlat = $_POST['Latitude'] ;
   $rlon = $_POST['Longitude'] ;
   $lat = $rlat + 0.25 ;
   $lon = $rlon - 0.25 ;

   $link = mysqli_connect ( '127.01','wimp','','air_precip_clim' ) ;

   if ( ! $link ) {
      die('Could not connect: ' . mysqli_error()) ;
      exit() ;
   }

   if ( ! mysqli_set_charset ( $link, 'utf8' ) ) {
      die('Unable to set database connection encoding. ' . mysqli_error() ) ;
      exit() ;
   }

   //if ( ! mysql_select_db ( 'air_precip_clim', $link ) ) {
      //die('Unable to locate the database needed. ' . mysql_error() ) ;
      //exit() ;
   //}

   $month_id = array('JA','FE','MR','AP','MA','JU','JL','AU','SE','OC','NV','DE','Ann') ;

   $rlon = mysqli_real_escape_string($link, $rlon) ;
   $rlat = mysqli_real_escape_string($link, $rlat) ; 

   $query = sprintf("SELECT * FROM air_temperature  WHERE lon='%s' AND lat='%s'", $rlon, $rlat ) ;

   $results = mysqli_query ($link, $query ) ;

   $air_temp_orig = mysqli_fetch_assoc( $results ) ;

   $query = sprintf("SELECT * FROM precipitation  WHERE lon='%s' AND lat='%s'", $rlon, $rlat ) ;
   
   $results = mysqli_query ( $link, $query ) ;

   $precip_orig = mysqli_fetch_assoc( $results ) ;

   $air_temp_ann_orig = $_POST['air_temp_ann_orig'] ;
   $precip_ann_orig = $_POST['precip_ann_orig'] ;
   $FC =  $_POST['FC'] ;
   $curve = $_POST['resistance'] ;
   $height = $_POST['height'] ;
   $yname = $_POST['yname'] ;
   $yname = urldecode( $yname ) ;

   print "<hr> \n" ;
   print "<H1><font color='blue'><i>Web</i>WIMP Climatic Water Balance Results <br>" ;
   print "Project Title: $yname </h1>" ;
   print "<hr> \n" ;

   $month_na = array('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Ann') ;

   if ( $lat < 0.0 ) {
      $clat=$lat*(-1.0) ;
      $clat=$clat . 'S' ;
    } else {   
      $clat=$lat . 'N' ;
    }
   if ($lon < 0.0) { 
      $clon=$lon * (-1.0) ;
      $clon=$clon . 'W' ;
   } else {
      $clon=$lon . 'E' ;
   }

   $dt12 = $_POST['TC_12'] ;
   if (abs($dt12) < 0.1 ) {
      $d_tmp = 0.0 ;
      for ( $i = 0 ; $i <= 11 ; $i++ ) {
         $tid='TC_' . $i ;
         $dtt[$i] = $_POST[$tid] ;
         $air_temp[$i] = $air_temp_orig["$month_id[$i]"] + $dtt[$i] ;
         $d_tmp=$d_tmp + $dtt[$i]/12.0 ;
      }
   } else { 
      $d_tmp = $dt12 ;
      for ( $i = 0 ; $i <= 11 ; $i++ ) {
        $tid='TC_' . $i ;                                                                                                    
        $air_temp[$i] = $air_temp_orig["$month_id[$i]"] + $dt12 ;
        $dtt[$i] = $dt12 ;                                                         
      }                                              
   }
   $dtt[12] = $d_tmp ;

   $dp12 = $_POST['PMM_12'] ;
   if (abs($dp12) < 0.1 ) {
      $d_tmp = 0.0 ;
      $d_tmp_sum = 0.0 ;
      for ( $i = 0 ; $i <= 11 ; $i++ ) {
         $pid='PMM_' . $i ;
	 $dpp[$i] = $_POST[$pid] ;
         $precip[$i] = $precip_orig["$month_id[$i]"] * ( 1.0 + $dpp[$i]/100.0) ;
         $d_tmp = $d_tmp + ($precip[$i] - $precip_orig["$month_id[$i]"] ) ;
         $d_tmp_sum = $d_tmp_sum +  $precip_orig["$month_id[$i]"] ;
      }  
      $dpp[12] = $d_tmp/$d_tmp_sum * 100.0 ;
   } else {
      for ( $i = 0 ; $i <=11 ; $i++ ) {
         $precip[$i] = $precip_orig["$month_id[$i]"] * ( 1.0 + $dp12/100.0) ;
         $dpp[$i] = $dp12 ;
      }
      $dpp[12] = $dp12 ;
   } 
  
   $process_id = rand(100000,999999) ;
   $png_file = "$process_id" . '.png' ;
   $ps_file =  "$process_id" . '.ps'  ;
   $output_ps = "/var/www/html/wimp/psfiles/$ps_file" ;
   $output_png = "/var/www/html/wimp/psfiles/$png_file" ;
   $html_file = "/var/www/html/wimp/psfiles/$process_id.html" ;
   $var_list = $lat . " " . $lon . " " . $FC . " " . $curve ;
   $air_temp_list = join(" ",$air_temp) ;
   $dtt_list = join (" ",$dtt) ;
   //print $dtt_list ;
   $precip_list = join(" ",$precip) ;
   $dpp_list = join(" ",$dpp) ;
   //print $dpp_list ;

   $input_list = "echo $var_list $process_id $dtt_list $dpp_list $air_temp_list $precip_list $height" ;
   $program_name = "/var/www/html/wimp/public_html/wimp/wimp" ;
   putenv("PGPLOT_FONT=/usr/local/pgplot/pgplot_src/grfont.dat");
   putenv("PGPLOT_DIR=/usr/local/pgplot");
   print "<p style='color:blue'>\n" ;
   passthru ("$input_list | $program_name") ;
   //echo "$input_list | $program_name";
   //echo  $output_png ;
   system ("/usr/bin/convert -density 300 -crop 0x0 $output_ps $output_png") ;

   if ( file_exists( $output_png )) {
      $html_f = fopen ( $html_file,'x+' ) ;
      fwrite ( $html_f,"<html>\n" ) ;
      fwrite ( $html_f,"<head>\n" ) ;
      fwrite ( $html_f,"<TITLE>Web WIMP Climatic Water Balance Results</TITLE>\n" ) ;
      fwrite ( $html_f,"<LINK REL='stylesheet' TYPE='text/css' HREF='http://cyclops.deos.udel.edu/wimp/public_html/style.css'> \n" ) ;
      fwrite ( $html_f,"<script>" ) ;
      fwrite ( $html_f,"function changeScreenSize(w,h)" ) ;
      fwrite ( $html_f,"{ window.resizeTo( w,h ) ; }" ) ;
      fwrite ( $html_f,"</script>" ) ;
      fwrite ( $html_f,"</head>" ) ;
      fwrite ( $html_f,"<body onload='changeScreenSize(700,880)' bgcolor='#FFFFFF' leftmargin='5%' rightmargin='5%'>" ) ;
      fwrite ( $html_f,"</head>" ) ;
      fwrite ( $html_f,"<body bgcolor=#ffffff>\n" ) ;
      fwrite ( $html_f,"<br> \n" ) ;
      fwrite ( $html_f,"<hr noshade size='3'>\n") ;
      fwrite ( $html_f,"<h2>Water Balance at $clon, $clat<br>\n" ) ; 
      fwrite ( $html_f,"Project Title: $yname</h2>\n" ) ;
      fwrite ( $html_f,"<hr noshade size='3'>\n" ) ;
      fwrite ( $html_f,"<br> \n" ) ;
      fwrite ( $html_f,"<img src=$png_file width='600'> \n" ) ;
      fwrite ( $html_f,"<hr noshade size='3'> \n" ) ;
      fwrite ( $html_f,"<table width='75%' style='font-size:13.0pt;color:blue;font-style=normal'> \n" ) ;
      fwrite ( $html_f,"<tr><td>" ) ;
      fwrite ( $html_f,"Location: <b>$clon &nbsp $clat</b>&nbsp&nbsp Elevation: <b> $height  m</b>" ) ;
      fwrite ( $html_f,"</td></tr>\n" ) ;
      fwrite ( $html_f,"<tr><td>\n" ) ;
      fwrite ( $html_f,"Soil water-holding capacity: <b>$FC  mm</b>" ) ;
      fwrite ( $html_f,"</td></tr>\n" ) ;
      fwrite ( $html_f,"<tr><td>\n" ) ;
      fwrite ( $html_f,"Declining availability function: <b>$curve</b>" ) ;
      fwrite ( $html_f,"</td></tr>\n" ) ;
      fwrite ( $html_f,"<tr><td>\n" ) ;
      $format = sprintf('%.1f', $dtt[12]) ;
      fwrite ( $html_f,"Prescribed average-monthly air-temperature changes: <b>$format<sup>o</sup>C</b>" ) ;
      fwrite ( $html_f,"</td></tr>\n" ) ;
      fwrite ( $html_f,"<tr><td>\n" ) ;
      $format = sprintf('%.1f', $dpp[12]) ;
      fwrite ( $html_f,"Prescribed average-monthly precipitation changes: <b> $format %</b> \n" ) ;
      fwrite ( $html_f,"</td></tr>\n" ) ;
      fwrite ( $html_f,"</table>\n" ) ;
      fwrite ( $html_f,"<hr noshade size='3'>\n" ) ;
      fwrite ( $html_f,"</body>\n" ) ;
      fwrite ( $html_f,"</html> \n" ) ;
      fclose ( $html_f ) ;
      print "<p style='font-size:16.0pt;color:blue'>\n" ;
      print '<a href=http://cyclops.deos.udel.edu/wimp/psfiles/' . $process_id . '.html target=_blank>' ;
      print "Monthly and annual climatic water balance <b>graph</b>" ;
   }
?>
</center>
</body>
</html>
