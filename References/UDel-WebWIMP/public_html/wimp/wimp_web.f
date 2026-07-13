      PROGRAM WIMP
      implicit none
C     
C     **********************************************************************
C     *                                                                    *
C     *         W       W     IIIIIIIII     M       M     PPPPPPPP         *
C     *         W       W         I         MM     MM     P       P        *
C     *         W   W   W         I         M M   M M     P       P        *
C     *         W  W W  W         I         M  M M  M     PPPPPPPP         *
C     *         W W   W W         I         M   M   M     P                *
C     *         WW     WW         I         M       M     P                *
C     *         W       W     IIIIIIIII     M       M     P                *
C     **********************************************************************
C     *                                                                    *
C     *            THE WATER BUDGET INTERACTIVE MODELING PROGRAM           *
C     *                      PERSONAL COMPUTER VERSION                     *
C     *                                                                    *
C     **********************************************************************
      REAL LAT,LON,T(390),P(390),APE(390),D(390),AE(390),ST(390),
     1     DST(390),DEF(390),SUR(390),SF(390),RN(390),SMT(390),SST(390),
     2     TM(13),PM(13),PEM(13),APEM(13),tmptm(13),tmppm(13),
     3     OUT(12,12),SUMM(5),SUMY(5),H(13),DY(13),MON(13),fc,
     4     sm,x,xmin,dist,pcc,heat,dt,dl,dr,clat,clon,rlat,rlon,cc,pi,a,
     $     b,csc,curve_id,tcc(13),pccx(13),p_total1,p_total2
      INTEGER LAND(360,133),IOFF(8),JOFF(8),ilat,ilon,kd,irnd,ier,km,
     $     index,jlat,jlon,i,j,k,n,m,ixx,jyy,ifile,irec,ix,jy,
     $     jjlat,jjlon,idev,pgopen,iii,irow,icol
      integer id_rec(720),iheight
      CHARACTER*1 ICONT,CSM
      CHARACTER*3 IMON(12)
      character*50 filename
      character*80 output_gif,pmm_path,pg_device
      logical ext
      COMMON /RREAL/ LAT,LON,ILAT,ILON,FC
      COMMON /CHARS/ CSM,ICONT
      COMMON /CCHANGE/ TCC,PCCX
      DATA DY/13*15.0/,KD/15/,KM/1/
      DATA MON/1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,1./
      DATA IOFF/-1,0,1,1,1,0,-1,-1/, JOFF/1,1,1,0,-1,-1,-1,0/
      DATA IMON/'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep',
     1     'Oct','Nov','Dec'/
C     
      IRND(X)= INT(SIGN(1.,X)*(ABS(X)+0.5))
C     
C     INITIAL PARAMETERS:
C     
C     FC - FIELD CAPACITY OF THE TOP (ONLY) SOIL LAYER IN MM.
C     THE DEFAULT VALUE IS 150 MM.
C     
C     SM - DETERMINES A DIMENSIONLESS AVAILABILITY FUNCTION WHICH
C     CONTROLS THE REMOVAL OF SOIL MOISTURE BY EVAPOTRANSPIRATION.
C     THE VALUE OF THIS PARAMETER SPECIFIES WHICH ONE OF
C     SEVERAL FUNCTIONS WILL BE USED. THE DEFAULT FUNCTION IS
C     SUCH THAT THE RATE OF EVAPOTRANSPIRATION WILL DECLINE
C     LINEARLY WITH THE RATIO OF ACTUAL TO POTENTIAL MAXIMUM SOIL
C     MOISTURE (FIELD CAPACITY). A GRAPH SHOWING THE AVAILABILITY
C     FUNCTIONS AVAILABLE IN WIMP IS GIVEN BY MATHER (1974: 106).
C     
C     LAT - THE LATITUDE IN DEGREES.
C     
C     LON - THE LONGITUDE IN DEGREES.
C     
C     
C     EXTERNAL ROUTINES USED:
C     
C     CLEAR - A UTILITY ROUTINE WHICH CLEARS THE TERMINAL SCREEN.
C     
C     GRAPHIC SUBROUTINES ARE FROM MICROCOMPATIBLES' GRAFMATIC LIBRARY
C     
C     OPEN NECESSARY FILES:
C     FILE 20 MUST BE OPENED WITH DIRECT ACCESS; UNIT 7 IS A PRINTER
C     THESE STATEMENTS ARE SYSTEM-SPECIFIC AND MAY NEED TO BE MODIFIED
C     
C     SET DEFAULT VALUES OF FIELD CAPACITY AND THE AVAILABILITY FUNCTION
C     
      pi=acos(-1.0)
      dr=pi/180.0
      FC= 150.0
      SM= 2.0
      CSM= 'C'
      LAT= 400.0
      LON= 400.0

      read(5,*) lat,lon,fc,csm,
     $   output_gif,tcc,pccx,
     $   (TM(I),I=1,12),(PM(I),I=1,12),
     $    iheight
    
      IF (ABS(LAT).GT.90.0 .OR. ABS(LON).GT.180.0) then
        WRITE(6,9110)
        stop
      end if

c     irow=int((90.0-lat)*10.0)/5 + 1
c     icol=int((lon+180.0)*10.0)/5 + 1
c     write(6,*) irow,icol,'wimp_test'

c     open(22,file=
c    $   '/export/home/wimp/public_html/wimp/wimp_index.dat',
c    $   form='unformatted',
c    $   access='direct',recl=2880)

c     open (24,file=
c    $     '/export/home/wimp/public_html/wimp/wimp.dat',
c    $     form='unformatted',
c    $     access='direct',recl=96)
c     open (25,file=
c    $ '/export/home/wimp/public_html/wimp/wimp_height.dat',
c    $  form='unformatted',
c    $  access='direct',recl=4)

c     read(22,rec=irow) id_rec
c     irec=id_rec(icol)
c     if (irec == 0) then
c     write(6,*) "<p><span style=font-size:14.0pt;color=blue>"
c     write(6,9020)
c       stop
c     end if

c     read(24,rec=irec) (TM(I),I=1,12),(PM(I),I=1,12)
c     read(25,rec=irec) iheight

      IF (FC.lt.0.0) then
         WRITE(6,9150)
         stop
      end if

      sm=curve_id(csm)

c     if (pccx(13) > 0.0) then
c        pcc =PCCX(13)*0.01+1.0
c        do i=1,12
c          pm(i)=pm(i)*pcc
c        end do
c     else
c
c     p_total1=sum(pm(1:12))
c     do i=1,13
c            pcc = pccx(i)*0.01+1.0
c           pm(i)=pm(i)*pcc 
c     end do
c     p_total2 = sum(pm(1:12))
c     pccx(13) = (p_total2-p_total1) / p_total1*100.0
c
c     end if
c     if (nint(tcc(13)*10.0) /= 0) then
c        do i=1,12
c           tm(i)=tm(i)+tcc(13)
c        end do
c     else
c    
c     tcc(13)=sum(tcc(1:12))/12.0
c     do i=1,13
c           tm(i)=tm(i)+tcc(i)
c     end do

c     end if

C
C***  
C***  EVALUATE THE MONTHLY WATER BALANCE FOR THE SELECTED LOCATION
C***  USING THE SPECIFIED FIELD CAPACITY AND SOIL MOISTURE AVAILABILTY
C***  FUNCTION
C***  
      N= 12
      M= 13
      CALL MATHER(N,M,H,TM,HEAT,A,PEM,APEM,LAT,DL,KD,KM,DT,MON,DY)
C     
C     ESTIMATE DAILY P AND APE USING AREA-PRESERVING SPLINES
C     
      CALL SPLINE(PM,P)
      CALL SPLINE(APEM,APE)
C     
C     ESTIMATE DAILY T USING A PERIODIC CUBIC SPLINE
C     
      CALL CUBICP(TM,T)
C     
C     CONVERT P AND APE TO DAILY VALUES
C     
      K= 1
      DO 40 J=1,12
         DO 30 I=1,30
            P(K)= P(K)/30.
            APE(K)= APE(K)/30.
            K= K+1
 30      CONTINUE
 40   CONTINUE
C     
      N= 360
      M= 390
C     
      CALL DIFF(N,M,T,P,RN,SF,APE,D)
      CALL BAL(N,M,T,P,RN,SF,SMT,SST,ST,D,FC,SM,SUR,DST,DT,KM,IER)

      IF (IER.EQ.1) then 
         write(6,*) "<p><span style=font-size:14.0pt;color=blue>"
         WRITE(6,9220) LAT,LON
         stop
      end if

      IF (IER.EQ.2) then 
         WRITE(6,9250) LAT,LON
         stop
      end if
      CALL EVAPO(N,M,D,AE,APE,P,RN,SMT,SUR,DST,DEF)
C     
C     ROUND OFF TO NEAREST WHOLE NUMBER AND GET TOTALS BEFORE WRITING.
C     
      CALL OUTPUT(TM,PEM,APEM,PM,D,SST,SMT,ST,DST,AE,DEF,SUR,M,OUT,SUMY)
C     
      CALL TABLE(IMON,OUT,SUMY,6,iheight)

      pg_device='/var/www/html/wimp/psfiles/'
     $//trim(output_gif)//'.ps/vcps'
c     write(6,*) pg_device, 'HI'

      if (pgopen(pg_device).le.0)
     $   stop
      CALL GRAPH(LAT,LON,P,APE,AE,SUR,DEF,DST,SMT,SST,1)

      call pgclos

      stop
C     
 9020 FORMAT(' The location that you have selected falls in a large',
     *    ' body of water.',/,' Please select another location.',//)
 9110 FORMAT(' Latitude must be between -90 degrees and 90 degrees;',
     * ' Longitude must be between',/,' -180 degrees and 180 degrees.',
     * '  Please choose values within these ranges.',//)
 9150 FORMAT(/,' Field capacity must be positive and non-zero; please ',
     * 'choose another value: ')
 9220 FORMAT(' A permanent snow cover exists at latitude ',
     * f5.1,' and longitude ',f6.1,'.',///)
 9250 FORMAT(' The Water Balance did not converge after 150 iterations fo
     *r',/,' latitude ',f5.1,' and longitude ',f6.1,///)
      END
C***********************************************************************
      SUBROUTINE MATHER(N,M,H,T,HEAT,A,PE,APE,LAT,DL,KD,KM,DT,MON,DY)
      implicit none
      
      integer n,m,kd,km,i
      real heat,a,dl,alat,xn,decd,sum,dt,smtsum,ratio
      REAL LAT,H(M),T(M),PE(M),APE(M),DAYS(13),MON(M),DY(M)
      DATA DAYS/0.0,31.0,28.0,31.0,30.0,31.0,30.0,31.0,31.0,30.0,31.0,
     1     30.0,31.0/
C     
C     CALCULATE POTENTIAL EVAPOTRANSPIRATION.
C     
C     WHEN LAT IS GREATER THAN 50 DEGS, THE DAYLENGTH CORRECTION
C     REMAINS EQUAL TO THAT FOR 50 DEGS. ALAT IS, THEREFORE,
C     USED AS THE ARGUMENT FOR SUBROUTINE DAY.
C     
      ALAT= LAT
      IF (ABS(ALAT).GE.50.0) ALAT= SIGN(50.0,ALAT)
C     
C     CALCULATE THE HEAT INDEX
C     
      XN= N
      HEAT= 0.0
      DO 30 I= 1,N
         IF (T(I).LE.0.0) GO TO 10
         H(I)= (T(I) / 5.0) ** 1.514
         GO TO 20
 10      H(I)= 0.0
 20      CONTINUE
         HEAT= HEAT + H(I)
 30   CONTINUE
C     
C     NOTE: "A" IS AN EMPIRICALLY DERIVED EXPONENT BASED UPON "HEAT".
C     
 40   CONTINUE
      A= 6.75 / 10.0 ** 7.0 * HEAT ** 3.0-7.71 / 10.0 ** 5.0 * HEAT **
     1     2.0+1.79 / 10.0 ** 2.0 * HEAT+0.49
C     
C     GET INITIAL MONTHLY PE, I.E. BASED UPON 30 DAYS IN
C     A MONTH AND 12 HOURS IN A DAY
C     NOTE: PE(I) AND APE(I) ARE CALCULATED IN MM / MONTH
C     
      DO 70 I= 1,N
         IF (T(I).LE.0.0) GO TO 50
         PE(I)= 16.0 * (10.0 * T(I) / HEAT) ** A
C     
C     CORRECT FOR TEMPERATURES GREATER THAN 26.5 DEG C.
C     SEE THORNTHWAITE (1948) FOR EXPLANATION.
C     
         IF (T(I).GE.26.5) PE(I)= (-415.8547 + 32.2441 * T(I)-0.4325 *
     1        T(I) ** 2.0)
         GO TO 60
 50      PE(I)= 0.0
 60      CONTINUE
         KD= DY(I)
         KM= MON(I)
C     
C     ADJUST PE FOR DAYLENGTH AND THE NUMBER OF DAYS IN A MONTH.
C     
         CALL DAY(ALAT,KD,KM,DT,DECD,DL)
         APE(I)= PE(I) * (DAYS(KM + 1) / 30.0) * (DL / 12.0)
 70   CONTINUE
      RETURN
      END
C***********************************************************************
      SUBROUTINE DAY(LAT,KD,KM,DT,DECD,DL)
      implicit none

      integer kd,km,i
      REAL DAYS(13),sum,dayl,decr,cz,alat,xx,csh,h,smtsum,ratio
      REAL LAT,dt,decd,dl,x
      DATA DAYS/0.0,31.0,28.0,31.0,30.0,31.0,30.0,31.0,31.0,30.0,31.0,
     1     30.0,31.0/
C     
C     CALCULATE THE NUMBER OF HOURS IN A DAY AND THE SOLAR DECLINATION
C     ASSOCIATED WITH THAT DAY. THE INPUT REQUIRED INCLUDES:
C     THE MONTH (KM), THE DAY (KD) AND THE LATITUDE (LAT).
C     
      X= 0.0
      DO 10 I= 1,KM
         X= X + DAYS(I)
 10   CONTINUE
      SUM= X + KD
C     
C     GET THE NUMBER OF DAYS SINCE THE VERNAL EQUINOX (MARCH 21).
C     
      DAYL= SUM - 80.0
      IF (DAYL.LE.0.0) DAYL= 285.0 + SUM
C     
C     CALCULATE THE DECLINATION.
C     
      DECD= 23.45 * SIN(DAYL / 365.0 * 6.2832)
      DECR= DECD * 0.017453
C     
C     CALCULATE THE NUMBER OF HOURS OF DAYLIGHT CORRESPONDING
C     TO DAY KD AND MONTH KM (SEE SELLERS, 1965).
C     
      CZ= COS(1.5708 + 0.01745 * (50.0 / 60.0))
      ALAT= LAT * 0.017453
      XX= COS(DECR) * COS(ALAT)
      IF (XX.LE.0.0) GO TO 20
      CSH= (CZ - SIN(DECR) * SIN(ALAT)) / XX
      H= ACOS(CSH)
      DL= 24.0 * H / 3.1416
      GO TO 30
C     
C     ERROR MESSAGE - DIVIDE BY ZERO OR LESS.
C     
 20   WRITE(6,1000)
C     
 30   CONTINUE
      RETURN
 1000 FORMAT ('0',' ERROR - DIVIDE BY ZERO OR LESS - LAT. ',//,
     *     '  OR THE DECLINATION IS PROBABLY INCORRECT ')
      END
C***********************************************************************
      SUBROUTINE DIFF(N,M,T,P,RN,SF,APE,D)
      implicit none

      integer n,m,i
      REAL T(M),P(M),RN(M),SF(M),APE(M),D(M)
C
C     COMPARE ADJUSTED PE (APE(I)) WITH PRECIPITATION (P(I)) AND
C     DETERMINE FROZEN PRECIPITATION -- "SNOW FALL" (SF(I)) AND
C     LIQUID PRECIPITATION -- "RAINFALL" (RN(I)).
C
      DO 10 I= 1,N
        SF(I)= 0.0
        RN(I)= P(I)
        IF(T(I).GE.-1.0) GO TO 20
        SF(I)= P(I)
        RN(I)= 0.0
   20   D(I)= RN(I) - APE(I)
   10 CONTINUE
C
      RETURN
      END
C***********************************************************************
      SUBROUTINE BAL(N,M,T,P,RN,SF,SMT,SST,ST,D,FC,SM,SUR,DST,DT,KM,IER)
      implicit none

      integer n,m,km,ier,i,k,itc,init2,iflag,np1,l
      REAL ST(M),D(M),SUR(M),DST(M),T(M),P(M),RN(M),SF(M),SMT(M),SST(M)
      real fc,sm,dt,z,z2,xx4,zz4,sfsum,x1,x2,sx,tx,px,sstx,snmlt,smelt,
     $     ds,ra,xx,zz,xx2,zz2,xx3,zz3,smtsum,ratio,a
C     
C     ITERATE FOR SOIL AND SNOW MOISTURE TERMS THAT BALANCE
C     THE WATER BUDGET
C     
C     SUBROUTINE BAL REVISED AND TESTED OCTOBER 1980 - CMR
C     SUBROUTINE BAL REVISED AND TESTED  AUGUST 1981 - CJW
C     SUBROUTINE BAL REVISED AND TESTED   MARCH 1986 - DRL
C     

      IER= 0
      DO 5 I=1,30
         SST(N + I)= 0.0
         ST(N + I)= 0.0
         D(N + I) = D(I)
         P(N + I)= P(I)
         SF(N + I)= SF(I)
         T(N + I)= T(I)
    5 CONTINUE
      SST(1)= SF(1)
      ST(1)= FC*0.5
      DST(1)= 0.0
      K= 0
      ITC= 0
      Z= 0.0
      Z2= 0.0
      XX4= 0.0
      ZZ4= 0.0
 10   CONTINUE
      SFSUM= 0.0
      SMTSUM= 0.0
      INIT2= 2
      IFLAG=0
      NP1= 390
 20   CONTINUE
      IF(IFLAG.EQ.1) K= 0
      DO 80 I= INIT2,NP1
         L=I-1
         IF(IFLAG.EQ.1) L= N
C     
C     MONTHLY BUDGETS (NOTE: THIS IS DONE ON
C     AN APPROXIMATE DAY BY DAY BASIS).
C     
         X1= ST(L)
         X2= SST(L)
         SMT(I)= 0.0
         SUR(I)= 0.0
         SX= ST(L)
C     
C     ESTIMATE WATER IN THE SNOWPACK AND SNOWMELT
C     
         SST(I)= SST(L) + SF(I)
         TX= T(I)
         PX= P(I)
         SSTX= SST(I)
         SNMLT= SMELT(TX,PX,SSTX)
         IF (I.GT.30) THEN
            SMTSUM= SMTSUM+SNMLT
            SFSUM= SFSUM+SF(I)
         END IF
         SST(I)= SST(I) - SNMLT
         SMT(I)= SMT(I) + SNMLT
C     
C     DEMAND OR SURPLUS
C     
         DS= SNMLT + D(I)
C     
C     FIND SOIL MOISTURE STORAGE
C     
 40      RA= RATIO(SX,FC,SM)
         IF(DS.GE.0.0) RA= 1.0
         ST(I)= ST(L) + DS * RA
         IF (ST(I).GE.FC) GO TO 50
         IF(ST(I).LE.0.1) ST(I)= 0.1
         GO TO 60
 50      SUR(I)= SUR(I) + ST(I) - FC
         !write(6,*) SUR(i)
         ST(I)= FC
 60      CONTINUE
         ST(L)= X1
         SST(L)= X2
         DST(I)= ST(I) - ST(L)
         IF(IFLAG.EQ.0)GO TO 80
         Z= ST(I)
         Z2= SST(I)
         GO TO 10
 80   CONTINUE
C     
      K= K + 1
C     
C     TEST FOR NET SNOW ACCUMULATION
C     
      IF (K.GT.1.AND.(SFSUM-SMTSUM).GE.0.1) GO TO 110
C     
C     TESTS FOR BALANCES
C     
      IF (K.GT.50) GO TO 90
      XX= ABS(ST(N + 1) - ST(1))
      ZZ = ABS(Z - ST(1))
      XX2= ABS(SST(N + 1) - SST(1))
      ZZ2= ABS(Z2 - SST(1))
C     
C     ADJUST FOR HIGH LATITUDES OR ELEVATIONS WHERE THERE IS A
C     NET ANNUAL INCREASE IN THE SNOWPACK.
C     
      XX3= ABS(XX2 - XX4)
      ZZ3= ABS(ZZ2 - ZZ4)
      XX4= XX2
      ZZ4= ZZ2
      SST(1)= SST(N + 1)
      ST(1)= ST(N + 1)
C     
      IF (XX.LT.1.0.AND.ZZ.LE.1.0 .AND.
     1     (XX2.LT.1.0.OR.XX3.LT.1.0).AND.(ZZ2.LE.1.0.OR.ZZ3.LE.1.0)
     2     .AND. ITC.GT.0) GO TO 100
      IF (XX.LT.1.0.AND.XX2.LT.1.0) GO TO 90
      GO TO 10
C     
 90   CONTINUE
C     
C     INITIALIZATION FOR SECONDARY BALANCING
C     
      IF(K.GT.50.AND.ITC.EQ.2) GO TO 100
      ITC= ITC + 1
      IFLAG= 1
      INIT2= 1
      NP1= 1
      GO TO 20
 100  CONTINUE
      IF(K.GT.50.AND.ITC.EQ.2) IER= 2
      RETURN
C     
 110  IER= 1
C     
      RETURN
      END
C***********************************************************************
      SUBROUTINE EVAPO(N,M,D,AE,APE,P,RN,SMT,SUR,DST,DEF)
      implicit none
      
      integer n,m,i
      REAL D(M),AE(M),APE(M),P(M),RN(M),SMT(M),SUR(M),DST(M),DEF(M)
C     
C     CALCULATE ACTUAL EVAPOTRANSPIRATION AND DEFICIT.
C     
      DO 10 I= 1,N
         AE(I)= RN(I) + SMT(I) - DST(I) - SUR(I)
         DEF(I)= APE(I) - AE(I)
         IF(DEF(I).LE.0.0) DEF(I)= 0.0
 10   CONTINUE
      RETURN
      END
C***********************************************************************
      SUBROUTINE OUTPUT(T,PE,APE,P,D,SST,SMT,ST,DST,AE,DEF,SUR,M,OUT,SUM
     #     )
      implicit none

      integer m,l,i,k,nchar
      real t,pe,ape,p,d,sst,smt,st,dstae,def,sur,out,sum,rnd,ratio,
     $     dst,ae,a,tcc,pccx
      DIMENSION T(13),PE(13),APE(13),P(13),D(M),ST(M),DST(M),AE(M),
     1     DEF(M),SUR(M),SMT(M),SST(M),OUT(12,12),SUM(5)
C     
      RND(A)= FLOAT(INT(SIGN(1.,A)*(ABS(A)+0.5)))
C     
C     FILL THE OUTPUT ARRAY "OUT", ROUND ALL VALUES, AND SUM THE COLUMNS
C     
      DO 5 L=1,5
         SUM(L)= 0.0
    5 CONTINUE
C     
      DO 20 L=1,12
         OUT(1,L)= T(L)
         OUT(2,L)= RND(PE(L))
         OUT(3,L)= RND(APE(L))
         OUT(4,L)= RND(P(L))
         OUT(6,L)= RND(ST((L)*30))
         OUT(8,L)= AE((L-1)*30+1)
         OUT(9,L)= DEF((L-1)*30+1)
         OUT(10,L)= SUR((L-1)*30+1)
         OUT(11,L)= SMT((L-1)*30+1)
         OUT(12,L)= RND(SST((L)*30))
         DO 10 I=2,30
            OUT(8,L)= AE((L-1)*30+I)+OUT(8,L)
            OUT(9,L)= DEF((L-1)*30+I)+OUT(9,L)
            OUT(10,L)= SUR((L-1)*30+I)+OUT(10,L)
            OUT(11,L)= SMT((L-1)*30+I)+OUT(11,L)
 10      CONTINUE
         OUT(8,L)= RND(OUT(8,L))
         OUT(9,L)= RND(OUT(9,L))
         OUT(10,L)= RND(OUT(10,L))
         OUT(11,L)= RND(OUT(11,L))
         IF (OUT(9,L).EQ.0.0) OUT(8,L)= OUT(3,L)
         OUT(9,L)= OUT(3,L)-OUT(8,L)
         SUM(1)= SUM(1)+OUT(3,L)
         SUM(2)= SUM(2)+OUT(4,L)
         SUM(3)= SUM(3)+OUT(8,L)
         SUM(4)= SUM(4)+OUT(9,L)
         IF (L.EQ.1) GO TO 20
         K= L-1
         OUT(5,L)= OUT(4,L)-OUT(3,L)-OUT(12,L)+OUT(12,K)
         OUT(7,L)= OUT(6,L)-OUT(6,K)
         IF (OUT(10,L).EQ.0.0) GO TO 20
         OUT(10,L)= OUT(4,L)-OUT(8,L)-OUT(7,L)-OUT(12,L)+OUT(12,K)
         IF (OUT(10,L).LT.0.0) OUT(10,L)= 0.0
         SUM(5)= SUM(5)+OUT(10,L)
 20   CONTINUE
      OUT(5,1)= OUT(4,1)-OUT(3,1)-OUT(12,1)+OUT(12,12)
      OUT(7,1)= OUT(6,1)-OUT(6,12)
      IF (OUT(10,1).EQ.0.0) RETURN
      OUT(10,1)= OUT(4,1)-OUT(8,1)-OUT(7,1)-OUT(12,1)+OUT(12,12)
      IF (OUT(10,1).LT.0.0) OUT(10,1)= 0.0
      SUM(5)= SUM(5)+OUT(10,1)
      RETURN
      END
C***********************************************************************
      FUNCTION RATIO(SX,FC,SM)
      implicit none
      
      integer ism
      real ratio,sx,fc,sm
C     
C     SELECT A FUNCTION THAT DESCRIBES THE RESISTANCE
C     OF SOIL WATER TO REMOVAL BY EVAPOTRANSPIRATION
C     
      ISM= INT(SM+0.1)
      GO TO (10,20,30,40,50,60,70), ISM
C     
C     CURVE A
C     
      RATIO= 1.0
      RETURN
C     
C     CURVE B
C     
 10   RATIO = 1.0 - EXP(-6.8 * (SX / FC) )
      RETURN
C     
C     CURVE C
C     
 20   RATIO= SX / FC
      RETURN
C     
C     CURVE D
C     
 30   RATIO= 0.98E-2*EXP(6.79*(SX/FC))-0.98E-2
      IF (RATIO.GT.1.0) RATIO= 1.0
      RETURN
C     
C     CURVE E
C     
 40   RATIO= 0.98E-2*EXP(9.50*(SX/FC))-0.98E-2
      IF (RATIO.GT.1.0) RATIO= 1.0
      RETURN
C     
C     CURVE F
C     
 50   RATIO= 1.0E-1*EXP(7.8*(SX/FC))-0.1
      IF (RATIO.GT.1.0) RATIO= 1.0
      RETURN
C     
C     CURVE G
C     
 60   RATIO= SX / (0.7 * FC)
      IF (RATIO.GE.1.0) RATIO= 1.0
      RETURN
C     
C     CURVE H
C     
 70   RATIO= SX / (0.5 * FC)
      IF (RATIO.GE.1.0) RATIO= 1.0
      RETURN
      END
C***********************************************************************
      FUNCTION SMELT(TX,PX,SSTX)
      implicit none
      
      real smelt,tx,px,sstx
C     
C     COMPUTE SNOWMELT
C     
      SMELT= 2.63 + 2.55 * TX + 0.0912 * TX * PX
      IF(SMELT.LE.0.0) SMELT= 0.0
      IF(SMELT.GE.SSTX) SMELT= SSTX
C     
      RETURN
      END
C***********************************************************************
      SUBROUTINE TABLE(IMON,OUT,SUMY,IFILE,iheight)
      implicit none

      integer ifile,ilat,ilon,jlat,jlon,i,j,ndec,nchar
      REAL OUT(12,12),SUMY(5),LAT,LON,MSTNDX,fc,rnd,sgn,a,tccx,
     $     pccx(13),tcc(13)
      INTEGER IOUT(12),ISUMY(5),iheight
      CHARACTER*1 CSM,ICONT,ALAT,ALON
      CHARACTER*3 IMON(12),cmon(13)
      COMMON /RREAL/ LAT,LON,ILAT,ILON,FC
      COMMON /CHARS/ CSM,ICONT
      COMMON /CCHANGE/ TCC,PCCX

      data cmon/'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug',
     $     'Sep','Oct','Nov','Dec','Ann'/
C     
      RND(A)= FLOAT(INT(SIGN(1.,A)*(ABS(A)+0.5)))
C     
      ALAT= 'N'
      ALON= 'E'
      IF (LAT.EQ.0.0) ALAT= ' '
      IF (LON.EQ.0.0) ALON= ' '
      IF (LAT.LT.0.0) ALAT= 'S'
      IF (LON.LT.0.0) ALON= 'W'
      JLAT= INT(ABS(LAT))
      JLON= INT(ABS(LON))
C     
C     CALCULATE THE MOISTURE INDEX
C     
c     MSTNDX= ( (SUMY(2)/SUMY(1)) -1.0 ) *100.0
      
      if (sumy(1) == 0.0 .and. sumy(2) == 0.0) then
	mstndx = 0.0
	else 
           if (sumy(2) >= sumy(1)) then
	      mstndx = 1.0 - sumy(1)/sumy(2)
           else
              mstndx = sumy(2)/sumy(1) - 1.0
           end if
      end if

C     
C     WRITE THE OUTPUT HEADERS
C     
c     IF (IFILE.EQ.6) CALL CLEAR
c     WRITE(IFILE,9050) JLAT,ALAT,JLON,ALON,FC,CSM,MSTNDX,TCC,PCCX
c     WRITE(IFILE,9050) abs(LAT),ALAT,abs(LON),ALON,FC,CSM,
c     $   MSTNDX,TCC,PCCX

      tcc(13)=(nint(tcc(13)*10.0))/10.0
      WRITE(IFILE,9050) abs(LON),ALON,abs(LAT),ALAT,iheight,
     $ FC,CSM,MSTNDX
      write(6,*) '<p><br><span style="font-size:14.0pt;color:blue">'
      write(6,*) 'Prescribed air temperature changes' 
      write(6,*) '(<sup>o</sup>C)'
      write(6,*) '<table border=1>'
      write(6,*) '<tr>'
      do i=1,13
         write(6,*) '<td align="center">',cmon(i),'</td>'
      end do
      write(6,*) '</tr>'
      write(6,*) '<tr>'
      do i=1,13
         write(6,"('<td align=right><b>',f6.1,'</br></td>')") tcc(i)
      end do
      write(6,*) '</tr>'
      write(6,*) '</table>'
c     write(6,*) '<br>'
      write(6,*) '<p><br><span style="font-size:14.0pt;color:blue">'
cc    write(6,*) '<h4>'
      write(6,*) 'Prescribed precipitation changes (%)'
c     write(6,*) '</h4>'
      write(6,*) '<table border=1>'
      write(6,*) '<tr>'
      do i=1,13
         write(6,*) '<td align="center">',cmon(i),'</td>'
      end do
      write(6,*) '</tr>'
      write(6,*) '<tr>'
      do i=1,13
         write(6,"('<td align=right><b>',f6.1,'</b></td>')") pccx(i)
      end do
      write(6,*) '</tr>'
      write(6,*) '</table>'
c     write(6,*) '<br>'
C     
C     OUTPUT WATER BUDGET RESULTS
C     
      write(6,*) "<p><span style=font-size:16.0pt;color:blue>"
c     write(6,*) "<font size='13.0pt'>"
      write(6,*) "<br>Monthly and annual climatic water"
      write(6,*) 'balance <b>table</b><br>'
c     write(6,*) '<left>'
      write(6,*) '<table border=1>'
      write(6,*) '<tr>'
      WRITE(IFILE,1050) 

      write(6,*) '</tr>'
      DO 20 I=1,12
         DO 10 J=2,12
            IOUT(J)= OUT(J,I)
 10      CONTINUE
         write(6,*) '<tr>'
         WRITE(IFILE,1060) IMON(I),OUT(1,I),(IOUT(J),J=2,12)
         write(6,*) '</tr>'
 20   CONTINUE

      DO 30 I=1,5
         ISUMY(I)= RND(SUMY(I))
 30   CONTINUE

      write(6,*) '<tr>'
      write(6,1070) isumy(1:5)

 1070 format('<td>Total</td>',
     $     '<td align=right>&nbsp</td>',
     $     '<td align=right>&nbsp</td>',
     $     '<td align=right><b>',i5,'</td>',
     $     '<td align=right><b>',i5,'</b></td>',
     $     '<td>&nbsp </td><td>&nbsp </td><td>&nbsp </td>',
     $     '<td align=right><b>',i5,'</b></td>',
     $     '<td align=right><b>',i5,'</b></td>',
     $     '<td align=right><b>',i5,'</b></td>',
     $     '<td>&nbsp </td><td>&nbsp</td>')
      write(6,*) '</tr>'
      write(6,*) '</table>'
      write(6,*) '</center>'
C     
cr      DO 30 I=1,5
crc         ISUMY(I)= RND(SUMY(I))
c 30   CONTINUE
c      WRITE(IFILE,1030) ISUMY
C     
      IF (IFILE.EQ.7) WRITE(IFILE,1010)
C     
      RETURN
 1010 FORMAT(//)
 1030 FORMAT('Yearly Totals (mm):<br>','APE:',I6,
     $     '&nbsp PREC:',i6,
     $     '&nbsp AE:',i6,'&nbsp DEF:',i6,'&nbsp SURP:',i6)

 1050 FORMAT ('<td>MON</td>',
     $'<td><a href=''javascript: var w=window.open',
     $'("http://cyclops.deos.udel.edu/wimp/bad/temp.html", "temp_win",',
     $'"width=400,height=100,status=no,resizable=no");''>TEMP</a></td>',
c
     $'<td><a href=''javascript: var w=window.open',
     $'("http://cyclops.deos.udel.edu/wimp/bad/upe.html", "upe_win",',
     $'"width=400,height=100,status=no,resizable=no");''>UPE</a></td>',
c
     $'<td><a href=''javascript: var w=window.open',
     $'("http://cyclops.deos.udel.edu/wimp/bad/ape.html", "ape_win",',
     $'"width=400,height=100,status=no,resizable=no");''>APE</a></td>',
c
     $'<td><a href=''javascript: var w=window.open',
     $'("http://cyclops.deos.udel.edu/wimp/bad/prec.html", "prec_win",',
     $'"width=400,height=100,status=no,resizable=no");''>PREC</a></td>',
c
     $'<td><a href=''javascript: var w=window.open',
     $'("http://cyclops.deos.udel.edu/wimp/bad/diff.html", "diff_win",',
     $'"width=400,height=100,status=no,resizable=no");''>DIFF</a></td>',
c
     $'<td><a href=''javascript: var w=window.open',
     $'("http://cyclops.deos.udel.edu/wimp/bad/st.html", "st_win",',
     $'"width=400,height=100,status=no,resizable=no");''>ST</a></td>',
c
     $'<td><a href=''javascript: var w=window.open',
     $'("http://cyclops.deos.udel.edu/wimp/bad/dst.html", "dst_win",',
     $'"width=400,height=100,status=no,resizable=no");''>DST</a></td>',
c
     $'<td><a href=''javascript: var w=window.open',
     $'("http://cyclops.deos.udel.edu/wimp/bad/ae.html", "ae_win",',
     $'"width=400,height=100,status=no,resizable=no");''>AE</a></td>',
c
     $'<td><a href=''javascript: var w=window.open',
     $'("http://cyclops.deos.udel.edu/wimp/bad/def.html", "def_win",',
     $'"width=400,height=100,status=no,resizable=no");''>DEF</a></td>',
c
     $'<td><a href=''javascript: var w=window.open',
     $'("http://cyclops.deos.udel.edu/wimp/bad/sur.html", "sur_win",',
     $'"width=400,height=100,status=no,resizable=no");''>SURP</a></td>',
c
     $'<td><a href=''javascript: var w=window.open',
     $'("http://cyclops.deos.udel.edu/wimp/bad/smt.html", "smt_win",',
     $'"width=400,height=100,status=no,resizable=no");''>SMT</a></td>',
c
     $'<td><a href=''javascript: var w=window.open',
     $'("http://cyclops.deos.udel.edu/wimp/bad/sst.html", "sst_win",',
     $'"width=400,height=100,status=no,resizable=no");''>SST</a></td>')

 1060 format('<td>',a3,'</td>','<td align=right>',f5.1,'</td>',
     $     11('<td align=right>',i6,'</td>'))

 9050 FORMAT('<p>',
     $     'Climatic water balance for: Longitude&nbsp&nbsp <b>',f5.1,A1,
     $     ';',
     $     '</b>&nbsp&nbsp Latitude&nbsp&nbsp <b>',
     *     f5.1,A1,';','</b>&nbsp&nbsp Elevation<b>',i5,'<b>m</b>.<br>',
     $     '</b>Soil water-holding capacity:<b>',F7.1,
     $   'mm.</b>&nbsp Declining availability function: curve &nbsp<b>',
     $    A1,'</b>.<br>',
     $     'Willmott and Feddema''s Moisture Index:<b>',F9.2,'</b><br>')

c     $'Moisture Index:<b>',F9.2,'</b>&nbsp Temp:<b>',
c     $ F5.1,'<sup>o</sup></b>.&nbsp Precip:<b>',
c     * F5.1,'%</b>','.<br>')

      END
C***********************************************************************
C     
C     SPLINE SUBROUTINES
C     Subroutines Tested and Implemented -    March 1986   DRL
C     
C***********************************************************************
      SUBROUTINE SPLINE(Y,YY)
      implicit none
      
      integer i,nit,j1,k,j2,j
      real y,yy,z,x,a,b,c,d,e,aa,u2,v2,s2,t2,zz,r,s1,t1,v1,xdiff,u1

C     
C     CALCULATE THE SPLINES TO INTERPOLATE APE OR PREC
C     Y  -  ARRAY OF VALUES FOR THE 12 MONTHS (13=1)
C     YY -  ARRAY OF VALUES FOR THE 361 DAYS  (361=1)
C     
      DIMENSION Y(13),YY(390),Z(16)
      DIMENSION X(17),A(17),B(17),C(17),D(17),E(17),AA(17)
      LOGICAL NEGATV,THISAJ
C     
      Z(1)= Y(11)
      Z(2)= Y(12)
      DO 1 I=1,12
         Z(I+2)= Y(I)
    1 CONTINUE
      Z(15)= Y(1)
      Z(16)= Y(2)
C     
      DO 2 I=1,17
         AA(I)= 0.5
         X(I)= FLOAT(I*30-90)
         IF (I.GT.1.AND.I.LT.17) THEN
            IF (Z(I-1).EQ.0.0) AA(I)= 1.0
            IF (Z(I).EQ.0.0) AA(I)= 0.0
         END IF
    2 CONTINUE
      IF (Z(1).EQ.0.) AA(1)= 0.
      IF (Z(16).EQ.0.) AA(17)= 1.
C     
      NIT= 0
C     
 22   J1= 1
      E(1)= AA(1)*Z(1)
      DO 3 K=2,16
         ZZ= AA(K)
         E(K)= ZZ*Z(J1)+(1.-ZZ)*Z(K)
         J1= K
 3    CONTINUE
      E(17)= AA(17)*Z(16)
      B(1)= 0.
      C(1)= 0.
      DO 5 K=1,16
         J2= K+1
         U2= 1./(X(J2)-X(K))
         A(K)= U2
         V2= U2*U2
         S2= 20.*V2*Z(K)
         T2= 8.*V2*(E(K)+E(J2))
         IF (K.EQ.1) GO TO 4
         ZZ= 1./(3.*(U1+U2)+U1*B(J1))
         B(K)= -U2*ZZ
         R= S2-S1-T2+T1+4.*(V1-V2)*E(K)
         C(K)= ZZ*(R+U1*C(J1))
 4       J1= K
         U1= U2
         V1= V2
         S1= S2
         T1= T2
 5    CONTINUE
      D(16)= C(16)
      IF (Z(16).EQ.0.0.OR.Z(15).EQ.0.0) D(16)= 0.
      DO 6 J1= 2,15
         K= 17-J1
         D(K)= C(K)-B(K)*D(K+1)
         IF (Z(K).EQ.0.0.OR.Z(K-1).EQ.0.0) D(K)= 0.
 6    CONTINUE
 7    DO 8 K=1,16
         J2= K+1
         ZZ= A(K)
         C(K)= 1.5*ZZ*(-3.*D(K)+D(J2)+ZZ*(-8.*E(J2)-12.*E(K)+20.*Z(K)))
        B(K)=-4.*ZZ*ZZ*(-1.5*D(K)+D(J2)+ZZ*(-7.*E(J2)-8.*E(K)+15.*Z(K)))
         A(K)= 5.*ZZ*ZZ*ZZ*(.5*(D(J2)-D(K))-3.*ZZ*(E(J2)+E(K)-2.*Z(K)))
 8    CONTINUE
C     
      NEGATV= .FALSE.
      K= 1
      DO 10 J=3,14
         THISAJ= .FALSE.
         DO 9 I=0,29
            XDIFF= FLOAT(I)
            YY(K)= E(J)+D(J)*XDIFF+C(J)*XDIFF**2+B(J)*XDIFF**3
     #           +A(J)*XDIFF**4
            IF (Z(J).EQ.0.) YY(K)= 0.
            IF (YY(K).LT.0.0.AND.(.NOT.THISAJ)) THEN
               THISAJ= .TRUE.
               NEGATV= .TRUE.
               IF (AA(J).NE.1.0) AA(J)= AA(J)*0.5
               IF (AA(J+1).NE.1.0) AA(J+1)= 1.-(1.-AA(J+1))*0.5
               IF (NIT.GE.3) THEN
                  IF (AA(J).NE.1.0) AA(J)= 0.
                  AA(J+1)= 1.0
               END IF
            END IF
            K= K+1
    9    CONTINUE
 10   CONTINUE
C     
      NIT= NIT+1
      IF (NEGATV.AND.NIT.GE.4) GO TO 11
      IF (NEGATV) GO TO 22
C     
      GO TO 15
C     
 11   K= 1
      DO 14 J=3,14
         DO 13 I=0,29
            XDIFF= FLOAT(I)
            YY(K)= E(J)+D(J)*XDIFF+C(J)*XDIFF**2+B(J)*XDIFF**3
     #           +A(J)*XDIFF**4
            IF (Z(J).EQ.0.) YY(K)= 0.
            IF (YY(K).LT.0.) THEN
               K= (J-3)*30+1
               DO 12 J2=0,29
                  YY(K)= Z(J)
                  K= K+1
 12            CONTINUE
               GO TO 14
            END IF
            K= K+1
 13      CONTINUE
 14   CONTINUE
C     
 15   DO 16 I=361,390
         YY(I)= YY(I-360)
 16   CONTINUE
C     
      RETURN
      END
C***********************************************************************
      SUBROUTINE CUBICP(Y,YY)
      implicit none
      
      integer i,j,j1,k,j2
      real y,yy,y2,f,g,h,h1,w,h2,u,r1,r2,v,z,xdiff
C     
C     CALCULATE THE SPLINE TO INTERPOLATE TEMPERATURE
C     Y  -  ARRAY OF VALUES FOR THE 12 MONTHS (13=1)
C     YY -  ARRAY OF VALUES FOR THE 361 DAYS  (361=1,ETC.)
C     
      DIMENSION Y(13),YY(390)
      DIMENSION Y2(13),F(12),G(12),H(12)
      Y(13)= Y(1)
      J1= 1
      G(1)= 0.
      F(1)= 0.
      H(1)= -1.
      H1= 30.
      W= H1
      H2= 30.
      U= 120.
      R1= (Y(13)-Y(12))/H1
      R2= (Y(12)-Y(11))/H2
      V= 6.*(R1-R2)
      DO 2 K=1,11
         J2= K+1
         H2= 30.
         R2= (Y(J2)-Y(K))/H2
         IF (K.EQ.1) GOTO 1
         U= U-W*H(J1)
         V= V-W*F(J1)
         W= -G(J1)*W
    1    Z= 1./(2.*(H1+H2)-H1*G(J1))
         G(K)= Z*H2
         H(K)= -Z*H(J1)*H1
         F(K)= Z*(6.*(R2-R1)-H1*F(J1))
         J1= K
         H1= H2
         R1= R2
    2 CONTINUE
      H2= W+H1
      H1= (V-H2*F(11))/(U-H2*(G(11)+H(11)))
      Y2(12)= H1
      DO 3 J1= 2,12
         K= 13-J1
         Y2(K)= F(K)-G(K)*Y2(K+1)-H(K)*H1
    3 CONTINUE
      Y2(13)=Y2(1)
C     
      DO 4 I=1,12
         K= I+1
         H1= 30.
         H(I)= Y(I)
         G(I)= (Y(K)-Y(I))/H1 - H1*(Y2(K)+2.*Y2(I))/6.
         F(I)= 0.5*Y2(I)
         Y2(I)= (Y2(K)-Y2(I)) / (6.*H1)
    4 CONTINUE
      K= 15
      DO 6 J=1,12
         DO 5 I=0,29
            XDIFF= FLOAT(I)
            YY(K)= H(J)+G(J)*XDIFF+F(J)*XDIFF**2+Y2(J)*XDIFF**3
            K= K+1
            IF (K.GT.360) K= 1
    5    CONTINUE
    6 CONTINUE
C     
      DO 7 I=361,390
         YY(I)= YY(I-360)
    7 CONTINUE
C     
      RETURN
      END
C***********************************************************************
      function curve_id(csm)
      implicit none
      real curve_id
      integer i
      character*1 csm,c_id(8)

      data c_id/'A','B','C','D','E','F','G','H'/
      
      do i=1,8
         if (c_id(i) == csm) then
            curve_id=real(i)-1.0
         end if
      end do
      return
      end
