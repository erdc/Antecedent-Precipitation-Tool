      SUBROUTINE GRAPH(LAT,LON,P,APE,AE,SUR,DEF,DST,SMT,SST,index)
      implicit none

      REAL LAT,LON,X(390),P(390)
      REAL APE(390),AE(390),SUR(390),DEF(390),DST(390),SMT(390),SST(390)
      real rect_x(4),rect_y(4),rnd,ymax,xxx,xc,yc,xc1,xc2,yc1,yc2,
     $     ych,xch,yline1(390),yline2(390),xline(390),yline3(390),
     $     yline4(390),pp(390)
      INTEGER IX(392),IY(392),irnd,pgopen,i,icolor(5),ilat,ilon,index
      CHARACTER*5 CHRCTR
      character*5 clat
      character*6 clon
      character*4 element(5)
      character*1 month(13),hemi1,hemi2
      character*30 title
      data month/'J','F','M','A','M','J','J','A','S','O','N','D','J'/
      data icolor/2,3,8,6,4/
      data element/'SURP','+DST','-DST','DEF ','Prec'/

      ilat=nint(lat)
      ilon=nint(lon)

      pp=p

      call pgscr (  6,  0.85,  0.37,  0.05)
      call pgscr (  8,  0.99,  0.85,  0.56)
      call pgscr (  3,  0.74,  0.84,  0.91)
      call pgscr (  2,  0.19,  0.51,  0.74)

      write(clat,'(f5.1)') abs(lat)
      write(clon,'(f6.1)') abs(lon)

      hemi1='N'
      if (ilat.lt.0) hemi1='S'
      hemi2='E'
      if (ilon.lt.0) hemi2='W'
      
c     title='Water Balance at'//clat//hemi1//clon//hemi2
      title=''
      
      call init3

      if (index.eq.1) then
         call pgsvp(0.1,0.85,0.1,0.9)
         call pgswin(0.0,391.0,0.0,300.0)
         call pgwnad(0.0,391.0,0.0,300.0)
      else
         call pgsvp(0.2,0.6,0.25,0.8)
         call pgswin(0.0,391.0,0.0,300.0)
      end if

      call pgslw(4)
      call pgsch(1.2)
      call pgscf(1)
      call pgbox('BCTI',30.0,0,'BCTISNV',0.0,5)
      call pglab('Month',' ',title)

      yc=-20.0
      do i=1,13
         xc=real(i-1)*30.0 + 15.0
         call pgptxt(xc,yc,0.0,0.5,month(i))
      end do

      call pgptxt(-30.0,150.0,90.0,0.5,'mm')
      
      if (index.eq.1) then
         DO I=1,30
            P(I+360)= P(I)
            pp(i+360)=pp(i)
            APE(I+360)= APE(I)
            AE(I+360)= AE(I)
            SUR(I+360)= SUR(I)
            DEF(I+360)= DEF(I)
            DST(I+360)= DST(I)
            SMT(I+360)= SMT(I)
            SST(I+360)= SST(I)
         end do

         P(1)= P(1)*30.0
         pp(1)=pp(1)*30.0
         APE(1)= APE(1)*30.0
         AE(1)= AE(1)*30.0
         SUR(1)= SUR(1)*30.0
         DEF(1)= DEF(1)*30.0
         DST(1)= DST(1)*30.0
         SMT(1)= SMT(1)*30.0
         SST(1)= SST(1)*30.0
         yline4(1)=pp(1)
         IF (SMT(1).GT.0.) THEN
            P(1)= P(1)+SMT(1)
         ELSE
            IF (SST(1).GT.0.) P(1)= 0.
         END IF
      end if

      x(1)=1.0
      call pgbbuf

      do i=2,390
         xline(i-1)=(real(i)+real(i-1))*0.5
         x(i)=real(i)

         if(index.eq.1) then
            P(I)= P(I)*30.0
            pp(i)=pp(i)*30.0
            APE(I)= APE(I)*30.0
            AE(I)= AE(I)*30.0
            SUR(I)= SUR(I)*30.0
            DEF(I)= DEF(I)*30.0
            DST(I)= DST(I)*30.0
            SMT(I)= SMT(I)*30.0
            SST(I)= SST(I)*30.0
         end if

         yline4(i-1)=pp(i-1)

         IF (SMT(I).GT.0.) THEN
            P(I)= P(I)+SMT(I)
         ELSE
            IF (SST(I).GT.0.) P(I)= 0.
         END IF

         yline1(i-1)=ape(i-1)
         yline2(i-1)=p(i-1)
         yline3(i-1)=ae(i-1)

         rect_x(1)=x(i-1)
         rect_x(2)=rect_x(1)
         rect_x(3)=x(i)
         rect_x(4)=rect_x(3)

         if (sur(i-1)> 0.0 .and. sur(i) > 0.0) then
            rect_y(1)=ape(i-1)+sur(i-1)
            rect_y(2)=ape(i-1)
            rect_y(3)=ape(i)
            rect_y(4)=ape(i)+sur(i)
            call pgsci(icolor(1))
            call pgsfs(1)
            call pgpoly(4,rect_x,rect_y)
         end if

         if (def(i-1) > 0.0 .and. def(i) > 0.0) then
            call pgsci(icolor(4))
            rect_y(1)=ape(i-1)
            rect_y(2)=ape(i-1)-def(i-1)
            rect_y(3)=ape(i)-def(i)
            rect_y(4)=ape(i)
            call pgpoly(4,rect_x,rect_y)
         end if

         if (dst(i-1) /= 0.0 .and. dst(i) /= 0.0) then
            call pgsfs(1)
            if (dst(i) < 0.0) then
               call pgsci(icolor(3))
            else
               call pgsci(icolor(2))
            end if

            rect_y(1)=ape(i-1)-def(i-1)
            rect_y(2)=ape(i-1)-def(i-1)+dst(i-1)
            rect_y(3)=ape(i)-def(i)+dst(i)
            rect_y(4)=ape(i)-def(i)
            call pgpoly(4,rect_x,rect_y)
         end if
      end do

      call pgslw(3)
      call pgsci(1)
      call pgmove(xline(1),yline1(1))      
      do i=2,390
         call pgdraw(xline(i-1),yline1(i-1))
      end do

      call pgmove(xline(1),yline2(1))
      do i=2,390
         call pgdraw(xline(i-1),yline2(i-1))
      end do

      call pgmove(xline(1),yline3(1))
      do i=2,390
         call pgdraw(xline(i-1),yline3(i-1))
      end do

      call pgmove(xline(1),yline4(1))
      call pgsci(icolor(5))
      call pgslw(5)
      do i=2,390
         call pgdraw(xline(i-1),yline4(i-1))
      end do
      call pgsci(1)

      if (index.eq.1) then
         call pgsvp(0.85,1.0,0.0,1.0)
         call pgswin(0.0,300.0,0.0,1000.0)
         call pgwnad(0.0,300.0,0.0,1000.0)
      else
         call pgsvp(0.65,0.9,0.25,0.8)
         call pgswin(0.0,300.0,0.0,800.0)
      end if

      xc1=50.0
      xc2=150.0
      call pgqcs(4,xch,ych)

      call pgslw(4)
      call pgsch(1.0)
      do i=1,5
         call pgsci(icolor(i))
         call pgsfs(1)
         if (index.ne.1.and.i.eq.4) call pgsfs(4)
         if (i == 5) then
            call pgsfs(2)
            yc1=50.0
         else
            yc1=real(i-1)*200.0+250.0
         end if
         yc2=yc1+100.0
         call pgrect(xc1,xc2,yc1,yc2)
         if (index.ne.1) call pgsci(1)
         call pgptxt(170.0,
     $        (yc1+yc2)/2.0-0.25*ych,0.0,0.0,element(i))
         if (i == 5) then
            call pgmove(xc1,(yc1+yc2)/2.0)
            call pgdraw(xc2,(yc1+yc2)/2.0)
         end if
         call pgsfs(2)
         call pgsci(1)
         call pgrect(xc1,xc2,yc1,yc2)
      end do
      call pgebuf

      RETURN
      END
c======================================================================
      subroutine init3

c     call pgscr(0,0.4,0.4,0.4)
      call pgscr(0,1.0,1.0,1.0)
      call pgscr(1,0.0,0.0,0.0)

      return
      end
c======================================================================
      subroutine draw_bound (rect_x,rect_y)
      implicit none
      real,dimension(*):: rect_x,rect_y

      call pgsci(1)
      call pgslw(4)
      call pgmove(rect_x(1),rect_y(1))
      call pgdraw(rect_x(4),rect_y(4))
      call pgmove(rect_x(2),rect_y(2))
      call pgdraw(rect_x(3),rect_y(3))
  
      return
      end
