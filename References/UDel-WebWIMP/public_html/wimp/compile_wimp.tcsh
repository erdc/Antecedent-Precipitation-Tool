#! /bin/tcsh -f

#f95 -xlang=f77 wimp_web.f graph.f -L/usr/local/lib \
#-lpgplot90 -lX11 -lz -lnsl -lsocket /usr/local/lib/libpng.a -o wimp

/usr/local/bin/pgplot wimp_web.f graph.f -o wimp.`date '+%m%d%y'`
