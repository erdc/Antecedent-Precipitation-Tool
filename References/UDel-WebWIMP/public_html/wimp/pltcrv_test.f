C***********************************************************************
      SUBROUTINE PLTCRV(csm,sm)
      implicit none
      
      character csm,choose_func
      real  AFUN,BFUN,CFUN,DFUN,EFUN,FFUN,GFUN,HFUN,XFUN,sm
      EXTERNAL AFUN,BFUN,CFUN,DFUN,EFUN,FFUN,GFUN,HFUN,XFUN
      integer pgopen,id,i,id1
      
      
      if (pgopen('?').le.0) stop
      call pgask(.false.)

c      call pgscr(0,0.4,0.4,0.4)
c      call pgscr(0,0.5,0.5,0.5)
c      call pgpap(8.0,0.6)
      call init2
      call pgsch(1.2)
      call pgslw(5)
      call pgbox ('BNIT',0.0,0,'BCNITV',0.0,0)

      call pgscf(2)
      call pgsch(1.5)
      do i=1,8
         call pgsci(i)
         call draw_func(i)
      end do

      call pgsci(1)
      call pgslw(5)
      call pglab ('Soil Moisture (%)','AE/PE',' ')
      call pgclos

c      id1=0
c      call init1
c      call menu1
c 10   call init1
c      call menu2(id,id1)
c      
c      call init2c

c c     call pgsci(1)
c c     call draw_func(id1)

c      call pgsci(2)
c      call draw_func(id)
c      if (id.eq.9) then
c         call pgclos
c      else
c         id1=id
c         go to 10
c      end if
c      csm=choose_func(id1)
c      sm=real(id1-1)
      write(6,*) csm
      RETURN
      END
C***********************************************************************
C     
C     FUNCTIONS FOR SUBROUTINE PLTCRV
C     
C***********************************************************************
      FUNCTION XFUN(T)
      XFUN= 100.-T
      RETURN
      END
      FUNCTION AFUN(T)
      AFUN= 100.
      RETURN
      END
      FUNCTION BFUN(T)
      BFUN = 100.0*(1.0-EXP(-.068*T))
      RETURN
      END
      FUNCTION CFUN(T)
      CFUN= T
      RETURN
      END
      FUNCTION DFUN(T)
      DFUN= 0.98*EXP(.0679*T)-0.98
      IF (DFUN.GT.100.) DFUN= 100.
      RETURN
      END
      FUNCTION EFUN(T)
      EFUN= 0.98*EXP(0.095*T)-0.98
      IF (EFUN.GT.100.) EFUN= 100.
      RETURN
      END
      FUNCTION FFUN(T)
      FFUN= 10.*EXP(0.078*T)-10.
      IF (FFUN.GT.100.) FFUN= 100.
      RETURN
      END
      FUNCTION GFUN(T)
      GFUN= T*1.4286
      RETURN
      END
      FUNCTION HFUN(T)
      HFUN= T*2.0
      RETURN
      END
c=======================================================================
      SUBROUTINE INIT1
C     
C     Set up graphics device to display menu.
      CALL PGSVP(0.7,1.0,0.2,0.8)
      CALL PGSWIN(0.0,0.5,0.0,1.0)
      RETURN
      END
c=======================================================================
      SUBROUTINE INIT2
C     
C     Set up graphics device to display menu.
c      call pgsvp(0.15,0.65,0.2,0.8)
      call pgvsiz (1.5,4.5,1.5,4.5)
      call pgswin (0.0,100.0,0.0,100.0)
      call pgwnad(0.0,100.0,0.0,100.0)

      RETURN
      END
c======================================================================
      SUBROUTINE MENU1
      use menu_var
      implicit none

c     INTEGER NP, ID
C     
C     Display menu of plots.
c     INTEGER NBOX
c     PARAMETER (NBOX=9)
c     CHARACTER*12 VALUE(NBOX)
c     INTEGER I, JUNK, K
c     REAL X1, X2, Y(NBOX), XX, YY, R
c     CHARACTER CH
c     INTEGER PGCURS
C     
c     DATA VALUE / 'A','B','C','D','E','F','G','H','Exit'/

c     DATA XX/0.5/, YY/0.5/
C     
      X1 = 0.1
      X2 = 0.2
      DO 5 I=1,NBOX
         Y(I) = 1.0 - REAL(I+1)/REAL(NBOX+2)
 5    CONTINUE

      CALL PGSCI(1)
      CALL PGSCH(1.5)
      call pgslw(1)
      CALL PGPTXT(0.0, 1.0, 0.0, 0.0, 
     $     'Select a resistance')
      call pgptxt(0.0,0.92,0.0,0.0,'curve: <A - H>')

      CALL PGSLW(1)
      CALL PGSCH(2.0)
      DO 10 I=1,NBOX
         CALL PGSCI(1)
         CALL PGSFS(1)
         CALL PGCIRC(X1, Y(I), 0.02)
         CALL PGSCI(2)
         CALL PGSFS(2)
         CALL PGCIRC(X1, Y(I), 0.02)
         CALL PGSCI(1)
         CALL PGPTXT(X2, Y(I)-0.02, 0.0, 0.0, VALUE(I))
 10   CONTINUE
      return
      end
c===========================
      subroutine menu2(id,id1)
      use menu_var
      implicit none
      integer id,pgcurs,id1

C     
C     Cursor input.
C     
 20   JUNK = PGCURS(XX, YY, CH)
      IF (ICHAR(CH).EQ.0) GOTO 50
C     
C     Find which box and highlight it
C    
      if (id1.ne.0) then
      call pgsci(1)
      call pgcirc(x1,y(id1),0.02)
      end if
      DO 30 I=1,NBOX
         R = (XX-X1)**2 +(YY-Y(I))**2
         IF (R.LT.(0.03**2)) THEN
            ID = I
            CALL PGSCI(2)
            CALL PGSFS(1)
            CALL PGCIRC(X1, Y(I), 0.02)
            RETURN
         END IF
 30   CONTINUE
      GOTO 20
 50   ID = 0
      RETURN
      END
c++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      subroutine draw_func(i)
      implicit none

      integer i
      real afun,bfun,cfun,dfun,efun,ffun,gfun,hfun
      external afun,bfun,cfun,dfun,efun,ffun,gfun,hfun
      
      call pgsls(1)
      if (i.eq.1) then
         call pgslw(2)
         CALL pgfunx(AFUN,100,0.0,100.0,1)
         call pgslw(5)
         call pgtext(15.0,afun(15.0),'A')
      end if
      if (i.eq.2) then
         call pgslw(2)
         CALL pgfunx(BFUN,100,0.0,100.0,1)
         call pgslw(5)
         call pgtext(15.0,bfun(15.0),'B')
      end if
      if (i.eq.3) then
         call pgslw(2)
         CALL pgfunx(CFUN,100,0.0,100.0,1)
         call pgslw(5)
         call pgtext(80.0,cfun(80.0),'C')
      end if
      if (i.eq.4) then
         call pgslw(2)
         CALL pgfunx(DFUN,100,0.0,100.0,1)
         call pgslw(5)
         call pgtext(50.0,dfun(50.0),'D')
      end if
      if (i.eq.5) then
         call pgslw(2)
         CALL pgfunx(EFUN,100,0.0,100.0,1)
         call pgslw(5)
         call pgtext(40.0,efun(40.0),'E')
      end if
      if (i.eq.6) then
         call pgslw(2)
         CALL pgfunx(FFUN,100,0.0,100.0,1)
         call pgslw(5)
         call pgtext(23.0,ffun(23.0),'F')
      end if
      if (i.eq.7) then
         call pgslw(2)
         CALL pgfunx(GFUN,100,0.0,69.0,1)
         call pgslw(5)
         call pgtext(50.0,gfun(50.0),'G')
      end if
      if (i.eq.8) then
         call pgslw(2)
         CALL pgfunx(HFUN,100,0.0,50.0,1)
         call pgslw(5)
         call pgtext(31.0,hfun(31.0),'H')
      end if

      return
      end
c======================================================================
      function choose_func(id1)
      implicit none

      integer id1
      character choose_func,index(8)

      data index/'A','B','C','D','E','F','G','H'/

      choose_func=index(id1)

      return
      end
c=====================================================================
      module menu_var
      INTEGER NP
C
C Display menu of plots.
      INTEGER NBOX
      PARAMETER (NBOX=9)
      CHARACTER*12 VALUE(NBOX)
      INTEGER I, JUNK, K
      REAL X1, X2, Y(NBOX), XX, YY, R
      CHARACTER CH
      DATA VALUE / 'A','B','C','D','E','F','G','H','Exit'/

      DATA XX/0.5/, YY/0.5/

      end module menu_var
