$STORAGE:2
      PROGRAM WIMP
C
C **********************************************************************
C *                                                                    *
C *         W       W     IIIIIIIII     M       M     PPPPPPPP         *
C *         W       W         I         MM     MM     P       P        *
C *         W   W   W         I         M M   M M     P       P        *
C *         W  W W  W         I         M  M M  M     PPPPPPPP         *
C *         W W   W W         I         M   M   M     P                *
C *         WW     WW         I         M       M     P                *
C *         W       W     IIIIIIIII     M       M     P                *
C **********************************************************************
C *                                                                    *
C *            THE WATER BUDGET INTERACTIVE MODELING PROGRAM           *
C *                      PERSONAL COMPUTER VERSION                     *
C *                                                                    *
C **********************************************************************
      REAL LAT,LON,T(390),P(390),APE(390),D(390),AE(390),ST(390),
     1     DST(390),DEF(390),SUR(390),SF(390),RN(390),SMT(390),SST(390),
     2     TM(13),PM(13),PEM(13),APEM(13),
     3     OUT(12,12),SUMM(5),SUMY(5),H(13),DY(13),MON(13)
      INTEGER LAND(360,133),IOFF(8),JOFF(8)
      CHARACTER*1 ICONT,CSM
      CHARACTER*3 IMON(12)
      COMMON /REAL/ LAT,LON,ILAT,ILON,FC
      COMMON /CHARS/ CSM,ICONT
      COMMON /CCHANGE/ TCC,PCCX
      DATA DY/13*15.0/,KD/15/,KM/1/
      DATA MON/1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,1./
      DATA IOFF/-1,0,1,1,1,0,-1,-1/, JOFF/1,1,1,0,-1,-1,-1,0/
      DATA IMON/'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep',
     1          'Oct','Nov','Dec'/
C
      IRND(X)= INT(SIGN(1.,X)*(ABS(X)+0.5))
C
C  INITIAL PARAMETERS:
C
C     FC - FIELD CAPACITY OF THE TOP (ONLY) SOIL LAYER IN MM.
C          THE DEFAULT VALUE IS 150 MM.
C
C     SM - DETERMINES A DIMENSIONLESS AVAILABILITY FUNCTION WHICH
C          CONTROLS THE REMOVAL OF SOIL MOISTURE BY EVAPOTRANSPIRATION.
C          THE VALUE OF THIS PARAMETER SPECIFIES WHICH ONE OF
C          SEVERAL FUNCTIONS WILL BE USED. THE DEFAULT FUNCTION IS
C          SUCH THAT THE RATE OF EVAPOTRANSPIRATION WILL DECLINE
C          LINEARLY WITH THE RATIO OF ACTUAL TO POTENTIAL MAXIMUM SOIL
C          MOISTURE (FIELD CAPACITY). A GRAPH SHOWING THE AVAILABILITY
C          FUNCTIONS AVAILABLE IN WIMP IS GIVEN BY MATHER (1974: 106).
C
C     LAT - THE LATITUDE IN DEGREES.
C
C     LON - THE LONGITUDE IN DEGREES.
C
C
C  EXTERNAL ROUTINES USED:
C
C     CLEAR - A UTILITY ROUTINE WHICH CLEARS THE TERMINAL SCREEN.
C
C     GRAPHIC SUBROUTINES ARE FROM MICROCOMPATIBLES' GRAFMATIC LIBRARY
C
C     OPEN NECESSARY FILES:
C     FILE 20 MUST BE OPENED WITH DIRECT ACCESS; UNIT 7 IS A PRINTER
C     THESE STATEMENTS ARE SYSTEM-SPECIFIC AND MAY NEED TO BE MODIFIED
C
      OPEN (5,FILE='CON:')
      OPEN (6,FILE='CON:')
C
C     SET DEFAULT VALUES OF FIELD CAPACITY AND THE AVAILABILITY FUNCTION
C
      FC= 150.0
      SM= 2.0
      CSM= 'C'
      LAT= 400.0
      LON= 400.0
C
      CALL CLEAR
      WRITE(6,9060)
      WRITE(6,9100)
      WRITE(6,9070)
      READ(5,9040) ICONT
C
C     WRITE INITIAL INSTRUCTIONS
C
      CALL HELP(1)
C
C     READ THE FILE CONTAINING LAND/WATER DESIGNATIONS WHILE THE USER
C     IS READING THE INSTRUCTIONS
C
      OPEN (10,FILE='NODETYPE.BIN',FORM='BINARY',STATUS='OLD')
      DO 2 J=1,133
         DO 1 K=1,30
            IST= (K-1)*12+1
            IEND= IST+11
            READ(10) (LAND(I,J),I=IST,IEND)
    1    CONTINUE
    2 CONTINUE
      CLOSE(10)
C
      WRITE(6,9170)
      READ(5,9040) ICONT
C
C     PROMPT USER FOR LATITUDE AND LONGITUDE
C
      CALL CLEAR
   10 CONTINUE
      WRITE(6,9000)
      CALL RDR(LAT,IER)
      IF (IER.EQ.1) GO TO 130
      WRITE(6,9050)
      CALL RDR(LON,IER)
      IF (IER.EQ.1) GO TO 130
C
      IF (ABS(LAT).GT.90.0 .OR. ABS(LON).GT.180.0) GO TO 130
      IF (ABS(LAT).GT.66.0) GO TO 140
C
      ILAT= IRND(LAT)
      ILON= IRND(LON)
      IF (ILON.EQ.180) ILON= -180
      LAT= FLOAT(ILAT)
      LON= FLOAT(ILON)
      IX= ILON+181
      JY= 67-ILAT
      IF (LAND(IX,JY).EQ.0) GO TO 100
      IFILE= LAND(IX,JY)/5000+1
      IREC= LAND(IX,JY)-(IFILE-1)*5000
   15 CLOSE(20)
      IF (IFILE.EQ.1) THEN
         OPEN(20,FILE='NAMERICA',ACCESS='DIRECT',RECL=104,
     1           FORM='FORMATTED')
      ELSE IF (IFILE.EQ.2) THEN
         OPEN(20,FILE='SAMRAUST',ACCESS='DIRECT',RECL=104,
     1           FORM='FORMATTED')
      ELSE IF (IFILE.EQ.3) THEN
         OPEN(20,FILE='EUROPE',ACCESS='DIRECT',RECL=104,
     1           FORM='FORMATTED')
      ELSE IF (IFILE.EQ.4) THEN
         OPEN(20,FILE='EASTASIA',ACCESS='DIRECT',RECL=104,
     1           FORM='FORMATTED')
      ELSE IF (IFILE.EQ.5) THEN
         OPEN(20,FILE='AFRICA',ACCESS='DIRECT',RECL=104,
     1           FORM='FORMATTED')
      END IF
   20 READ(20,1010,REC=IREC) JLAT,JLON,(TM(I),I=1,12),(PM(I),I=1,12)
      IF (ILAT.NE.JLAT.OR.ILON.NE.JLON) GO TO 160
C
C     CHOOSE FIELD CAPACITY AND RESISTANCE FUNCTION
C
      WRITE(6,9120) FC
      TCC= 0.0
      PCCX= 0.0
   21 CONTINUE
      CALL RDR(FC,IER)
      IF (FC.GT.0.0.AND.IER.NE.1) GO TO 22
      WRITE(6,9150)
      GO TO 21
C
   22 CONTINUE
      WRITE(6,9260)
  221 CALL RDR(TCC,IER)
      IF (IER.NE.1) GO TO 222
      WRITE(6,9280)
      GO TO 22
C
  222 CONTINUE
      WRITE(6,9270)
  223 CALL RDR(PCCX,IER)
      PCC= PCCX*0.01+1.0
      IF (PCC.LT.0.0) THEN
         PCC= 0.0
         PCCX= -100.
      END IF
      IF (IER.NE.1) GO TO 224
      WRITE(6,9280)
      GO TO 222
C
  224 CALL PLTCRV(SM)
C
   24 CALL CLEAR
      DO 25 I=1,12
         TM(I)= TM(I)+TCC
         PM(I)= PM(I)*PCC
   25 CONTINUE
      WRITE(6,9190)
      CSM= CHAR(INT(SM)+65)
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
   30    CONTINUE
   40 CONTINUE
C
      N= 360
      M= 390
C
      CALL DIFF(N,M,T,P,RN,SF,APE,D)
      CALL BAL(N,M,T,P,RN,SF,SMT,SST,ST,D,FC,SM,SUR,DST,DT,KM,IER)
      IF (IER.EQ.1) GO TO 150
      IF (IER.EQ.2) GO TO 170
      CALL EVAPO(N,M,D,AE,APE,P,RN,SMT,SUR,DST,DEF)
C
C     ROUND OFF TO NEAREST WHOLE NUMBER AND GET TOTALS BEFORE WRITING.
C
      CALL OUTPUT(TM,PEM,APEM,PM,D,SST,SMT,ST,DST,AE,DEF,SUR,M,OUT,SUMY)
C
      CALL TABLE(IMON,OUT,SUMY,6)
C
C     DOES USER WANT PRINTED RESULTS?
C
      WRITE(6,9160)
      READ(5,9040) ICONT
      IF (ICONT.EQ.'H'.OR.ICONT.EQ.'h') CALL HELP(4)
      IF (ICONT.EQ.'Y'.OR.ICONT.EQ.'y') THEN
         OPEN (7,FILE='LPT1:',STATUS='NEW')
         CALL TABLE(IMON,OUT,SUMY,7)
         CLOSE(7)
      END IF
C
C     GRAPHICAL REPRESENTATION OF THE WATER BALANCE
C
   26 WRITE(6,9200)
      READ(5,9040) ICONT
      IF(ICONT.EQ.'H'.OR.ICONT.EQ.'h') CALL HELP(5)
      IF(ICONT.EQ.'Y'.OR.ICONT.EQ.'y')
     *          CALL GRAPH(LAT,LON,P,APE,AE,SUR,DEF,DST,SMT,SST)
C
   55 CALL CLEAR
C
C     TEST TO SEE IF SUBSEQUENT STATIONS ARE TO BE EVALUATED.
C
   56 WRITE(6,9010)
      READ(5,9040,END=180) ICONT
      IF (ICONT.EQ.'H'.OR.ICONT.EQ.'h') CALL HELP(2)
      CALL CLEAR
      IF (ICONT.EQ.'Y'.OR.ICONT.EQ.'y') GO TO 10
      IF (ICONT.EQ.'N'.OR.ICONT.EQ.'n') GO TO 57
      WRITE(6,9180)
      GO TO 56
C
   57 STOP
C
C     DETERMINE IF THERE IS A LAND GRID-POINT WITHIN ONE DEGREE OF
C     LATITUDE AND/OR LONGITUDE OF THE WATER GRID-POINT SPECIFIED
C
  100 CONTINUE
      DO 110 K= 1,8
        IXX= IX+IOFF(K)
        JYY= JY+JOFF(K)
        IF (JYY.LT.1.OR.JYY.GT.133) GO TO 110
        IF (IXX.LT.1) IXX= IXX+360
        IF (IXX.GT.360) IXX= IXX-360
        IF (LAND(IXX,JYY).NE.0) GO TO 120
  110 CONTINUE
      WRITE(6,9020)
      GO TO 10
  120 IFILE= LAND(IXX,JYY)/5000+1
      IREC= LAND(IXX,JYY)-(IFILE-1)*5000
      ILON= IXX-181
      ILAT= 67-JYY
      WRITE(6,9030) ILAT,ILON
      READ(5,9040,END=10) ICONT
      IF (ICONT.EQ.'H'.OR.ICONT.EQ.'h') CALL HELP(3)
      CALL CLEAR
      LAT= FLOAT(ILAT)
      LON= FLOAT(ILON)
      IF (ICONT.EQ.'Y'.OR.ICONT.EQ.'y') GO TO 15
      GO TO 56
C
C     LATITUDE AND/OR LONGITUDE SPECIFIED IS OUT-OF-BOUNDS
C
  130 CALL CLEAR
      WRITE(6,9110)
      GO TO 10
  140 CALL CLEAR
      WRITE(6,9210)
      WRITE(6,9110)
      GO TO 10
C
C     NET ANNUAL SNOW ACCUMULATION
C
  150 CALL CLEAR
      WRITE(6,9220) ILAT,ILON
      GO TO 56
C
C     ERROR -- DATA DOES NOT MATCH REQUESTED GRID POINT
C
  160 WRITE(6,9240)
      STOP
C
C     BUDGET DOES NOT BALANCE AFTER 150 ITERATIONS
C
  170 CALL CLEAR
      WRITE(6,9250) ILAT,ILON
      GO TO 10
C
  180 CALL CLEAR
      STOP
C
 1010 FORMAT (2I4,12F4.1,12F4.0)
 1020 FORMAT (I2)
 9000 FORMAT(' Enter the latitude: ')
 9010 FORMAT(' Would you like to evaluate the water budget for',
     *' another location? (y or n): ')
 9020 FORMAT(' The grid point that you have selected falls in a large',
     *' body of water.',/,' Please select another location.',//)
 9030 FORMAT(' The grid point that you have selected is classified as',
     *' water.  Would you like',/,' to compute a water budget for ',
     *'latitude 'I3,' and longitude ',I4,'?',//,
     *' Answer yes, no or help: ')
 9040 FORMAT(A1)
 9050 FORMAT(' Enter the longitude: ')
 9060 FORMAT(//,35X,'Welcome to',//)
 9070 FORMAT(//,19X,'The Water-Budget Interactive Modeling Program',/,
     *          17X,'Developed by C. Willmott, C. Rowe, and D. Legates',
     */,        27X,'at the University of Delaware',////,
     *          27X,'Press RETURN for instructions')
 9100 FORMAT(16X,'W       W     IIIIIIIII     M       M     PPPPPPPP ',/
     *       16X,'W       W         I         MM     MM     P       P',/
     *       16X,'W   W   W         I         M M   M M     P       P',/
     *       16X,'W  W W  W         I         M  M M  M     PPPPPPPP ',/
     *       16X,'W W   W W         I         M   M   M     P        ',/
     *       16X,'WW     WW         I         M       M     P        ',/
     *       16X,'W       W     IIIIIIIII     M       M     P        ')
 9110 FORMAT(' Latitude must be between -66 degrees and 66 degrees;',
     *' Longitude must be between',/,' -180 degrees and 180 degrees.',
     * '  Please choose values within these ranges.',//)
 9120 FORMAT(/,' Field capacity is currently set at ',F6.1,' mm.  Enter'
     *,' a new value or press',/,' RETURN to use the current field',
     * ' capacity: ')
 9130 FORMAT(/,' The soil moisture availability function is currently',
     * ' curve ',A1,'.  Enter a new',/,' letter or press RETURN to use',
     * ' the current availability function: ')
 9140 FORMAT(/,' Availability function curves are labelled A through H;'
     *,' please choose one of',/,' these letters: ')
 9150 FORMAT(/,' Field capacity must be positive and non-zero; please ',
     * 'choose another value: ')
 9160 FORMAT(/,' Do you want to print this water balance? (y or n): ')
 9170 FORMAT(/,' Press RETURN to continue',/)
 9180 FORMAT(/,' Please answer with "y" or "n"')
 9190 FORMAT(//' Calculating the water balance...please wait...')
 9200 FORMAT(' Do you wish to see a graphical representation?  (y or',
     *' n): ')
 9210 FORMAT(' A permanent snow cover usually exists above the Arctic ',
     *'Circle (66.5 degrees',/,' North) and below the Antarctic Circle',
     *' (66.5 degrees South).  For this program,')
 9220 FORMAT(' A permanent snow cover exists at latitude ',
     *I3,' and longitude ',I4,'.',///)
 9240 FORMAT(' ERROR -- Data on disk does not match requested location',
     *//,' Please report this information to',///,' Dr. Cort Willmott',/
     *,' Department of Geography',/,' University of Delaware',/,' Newark
     *, Delaware  19716',/,'U.S.A.',///)
 9250 FORMAT(' The Water Budget did not converge after 150 iterations fo
     *r',/,' latitude ',I3,' and longitude ',I4,///)
 9260 FORMAT(' Enter the Climate Change in degrees celsius for AIR TEMPE
     *RATURE',/,'  (Press <RETURN> for no change in air temperature): ')
 9270 FORMAT(' Enter the Climate Change percentage for PRECIPITATION',/
     *,'   (Press <RETURN> for no change in precipitation): ')
 9280 FORMAT(' Please enter a valid number (positive or negative)')
      END
C***********************************************************************
      SUBROUTINE MATHER(N,M,H,T,HEAT,A,PE,APE,LAT,DL,KD,KM,DT,MON,DY)
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
   10   H(I)= 0.0
   20   CONTINUE
        HEAT= HEAT + H(I)
   30 CONTINUE
C
C     NOTE: "A" IS AN EMPIRICALLY DERIVED EXPONENT BASED UPON "HEAT".
C
   40 CONTINUE
      A= 6.75 / 10.0 ** 7.0 * HEAT ** 3.0-7.71 / 10.0 ** 5.0 * HEAT **
     1   2.0+1.79 / 10.0 ** 2.0 * HEAT+0.49
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
     1                           T(I) ** 2.0)
        GO TO 60
   50   PE(I)= 0.0
   60   CONTINUE
        KD= DY(I)
        KM= MON(I)
C
C     ADJUST PE FOR DAYLENGTH AND THE NUMBER OF DAYS IN A MONTH.
C
        CALL DAY(ALAT,KD,KM,DT,DECD,DL)
        APE(I)= PE(I) * (DAYS(KM + 1) / 30.0) * (DL / 12.0)
   70 CONTINUE
      RETURN
      END
C***********************************************************************
      SUBROUTINE DAY(LAT,KD,KM,DT,DECD,DL)
      REAL DAYS(13)
      REAL LAT
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
   10 CONTINUE
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
   20 WRITE(6,1000)
C
   30 CONTINUE
      RETURN
 1000 FORMAT ('0',' ERROR - DIVIDE BY ZERO OR LESS - LAT. ',//,
     *'  OR THE DECLINATION IS PROBABLY INCORRECT ')
      END
C***********************************************************************
      SUBROUTINE DIFF(N,M,T,P,RN,SF,APE,D)
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
      REAL ST(M),D(M),SUR(M),DST(M),T(M),P(M),RN(M),SF(M),SMT(M),SST(M)
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
   10 CONTINUE
      SFSUM= 0.0
      SMTSUM= 0.0
      INIT2= 2
      IFLAG=0
      NP1= 390
   20 CONTINUE
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
C       ESTIMATE WATER IN THE SNOWPACK AND SNOWMELT
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
C       DEMAND OR SURPLUS
C
        DS= SNMLT + D(I)
C
C       FIND SOIL MOISTURE STORAGE
C
   40   RA= RATIO(SX,FC,SM)
        IF(DS.GE.0.0) RA= 1.0
        ST(I)= ST(L) + DS * RA
        IF (ST(I).GE.FC) GO TO 50
        IF(ST(I).LE.0.1) ST(I)= 0.1
        GO TO 60
   50   SUR(I)= SUR(I) + ST(I) - FC
        ST(I)= FC
   60   CONTINUE
        ST(L)= X1
        SST(L)= X2
        DST(I)= ST(I) - ST(L)
        IF(IFLAG.EQ.0)GO TO 80
        Z= ST(I)
        Z2= SST(I)
        GO TO 10
   80 CONTINUE
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
     1   (XX2.LT.1.0.OR.XX3.LT.1.0).AND.(ZZ2.LE.1.0.OR.ZZ3.LE.1.0)
     2   .AND. ITC.GT.0) GO TO 100
      IF (XX.LT.1.0.AND.XX2.LT.1.0) GO TO 90
      GO TO 10
C
   90 CONTINUE
C
C     INITIALIZATION FOR SECONDARY BALANCING
C
      IF(K.GT.50.AND.ITC.EQ.2) GO TO 100
      ITC= ITC + 1
      IFLAG= 1
      INIT2= 1
      NP1= 1
      GO TO 20
  100 CONTINUE
      IF(K.GT.50.AND.ITC.EQ.2) IER= 2
      RETURN
C
  110 IER= 1
C
      RETURN
      END
C***********************************************************************
      SUBROUTINE EVAPO(N,M,D,AE,APE,P,RN,SMT,SUR,DST,DEF)
      REAL D(M),AE(M),APE(M),P(M),RN(M),SMT(M),SUR(M),DST(M),DEF(M)
C
C     CALCULATE ACTUAL EVAPOTRANSPIRATION AND DEFICIT.
C
      DO 10 I= 1,N
        AE(I)= RN(I) + SMT(I) - DST(I) - SUR(I)
        DEF(I)= APE(I) - AE(I)
        IF(DEF(I).LE.0.0) DEF(I)= 0.0
   10 CONTINUE
      RETURN
      END
C***********************************************************************
      SUBROUTINE OUTPUT(T,PE,APE,P,D,SST,SMT,ST,DST,AE,DEF,SUR,M,OUT,SUM
     #)
      DIMENSION T(13),PE(13),APE(13),P(13),D(M),ST(M),DST(M),AE(M),
     1          DEF(M),SUR(M),SMT(M),SST(M),OUT(12,12),SUM(5)
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
   10   CONTINUE
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
   20 CONTINUE
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
   10 RATIO = 1.0 - EXP(-6.8 * (SX / FC) )
      RETURN
C
C     CURVE C
C
   20 RATIO= SX / FC
      RETURN
C
C     CURVE D
C
   30 RATIO= 0.98E-2*EXP(6.79*(SX/FC))-0.98E-2
      IF (RATIO.GT.1.0) RATIO= 1.0
      RETURN
C
C     CURVE E
C
   40 RATIO= 0.98E-2*EXP(9.50*(SX/FC))-0.98E-2
      IF (RATIO.GT.1.0) RATIO= 1.0
      RETURN
C
C     CURVE F
C
   50 RATIO= 1.0E-1*EXP(7.8*(SX/FC))-0.1
      IF (RATIO.GT.1.0) RATIO= 1.0
      RETURN
C
C     CURVE G
C
   60 RATIO= SX / (0.7 * FC)
      IF (RATIO.GE.1.0) RATIO= 1.0
      RETURN
C
C     CURVE H
C
   70 RATIO= SX / (0.5 * FC)
      IF (RATIO.GE.1.0) RATIO= 1.0
      RETURN
      END
C***********************************************************************
      FUNCTION SMELT(TX,PX,SSTX)
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
      SUBROUTINE HELP(ICODE)
      REAL LAT,LON
      CHARACTER*1 CSM,ICONT,NO
      COMMON /REAL/ LAT,LON,ILAT,ILON,FC
      COMMON /CHARS/ CSM,ICONT
      DATA NO/'N'/
C
C     INTERACTIVE HELP ROUTINE
C
      CALL CLEAR
C
      GO TO (10,20,30,40,50),ICODE
C
   10 WRITE(6,1000)
      RETURN
C
   20 WRITE(6,1010)
      READ(5,1030,END=60) ICONT
      CALL CLEAR
      RETURN
C
   30 WRITE(6,1020) ILAT,ILON
      READ(5,1030,END=60) ICONT
      CALL CLEAR
      RETURN
C
   40 WRITE(6,1040)
      READ(5,1030,END=60) ICONT
      RETURN
C
   50 CALL CLEAR
      WRITE(6,1050)
      READ(5,1030,END=60) ICONT
      RETURN
C
   60 ICONT= NO
      RETURN
C
 1000 FORMAT(
     *' WIMP is designed to evaluate a spatially-averaged, monthly',
     *' water balance for'/' virtually any terrestrial location of',
     *' interest; that is, on any continent at'/' any node of a one',
     *' degree of latitude by one degree of longitude grid.  Monthly',/,
     *' average air temperature and precipitation data have been',
     *' interpolated to the'/' one degree by one degree grid and the',
     *' grid-point water balance is calculated by'/' a modified',
     *' Thornthwaite procedure.',//,
     *' You will be asked for the latitude and longitude of the',
     *' location for which you'/' desire a water budget calculation',
     *' (North latitude is positive and South is'/' negative; East',
     *' longitude is positive and West is negative).  These coordinates'
     */' will be rounded to the nearest whole degree and WIMP will',
     *' determine whether'/' the selected location is on land. If the',
     *' point is on land, a water budget will'/' be evaluated; if not,'
     *' you will be asked to select another location.',//,
     *' Please wait...loading data from disk...',//)
 1010 FORMAT(' Do you wish to evaluate the water budget for another loca
     *tion?'/' Just answer yes; or no, if you wish to exit from WIMP:  '
     *)
 1020 FORMAT(' The location you have selected is classified as water by
     *WIMP and a water',/,' budget evaluation cannot be made.  However,
     *the location that you specified is',/,' within one degree of',
     *' latitude and/or longitude of a terrestrial location.',//,
     *' Would you like to evaluate the water budget at',/,' latitude ',
     *I3,' and longitude ',I4,'?',/,' Answer yes or no:  ')
 1030 FORMAT(A1)
 1040 FORMAT(' You may print the table of water balance results by',
     *' answering yes to this',/,' question; any other response will',
     *' result in the table not being printed.',//,' Do you want the',
     *' results to be printed?',/,' Answer yes or no:  ')
 1050 FORMAT(' You may produce a graphical representation of the water',
     *' balance by answering',/,' yes to this question; any other resp',
     *'onse will result in no graphical plot.',//,' Do you want a grap',
     *'hical representation of the water balance?',/,' Answer yes or no:
     * ')
      END
C***********************************************************************
      SUBROUTINE TABLE(IMON,OUT,SUMY,IFILE)
      REAL OUT(12,12),SUMY(5),LAT,LON,MSTNDX
      INTEGER IOUT(12),ISUMY(5)
      CHARACTER*1 CSM,ICONT,ALAT,ALON
      CHARACTER*3 IMON(12)
      COMMON /REAL/ LAT,LON,ILAT,ILON,FC
      COMMON /CHARS/ CSM,ICONT
      COMMON /CCHANGE/ TCC,PCCX
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
      MSTNDX= ( (SUMY(2)/SUMY(1)) -1.0 ) *100.0
C
C     WRITE THE OUTPUT HEADERS
C
      IF (IFILE.EQ.6) CALL CLEAR
      WRITE(IFILE,9050) JLAT,ALAT,JLON,ALON,FC,CSM,MSTNDX,TCC,PCCX
C
C     OUTPUT WATER BUDGET RESULTS
C
      WRITE(IFILE,1050)
      DO 20 I=1,12
        DO 10 J=2,12
          IOUT(J)= OUT(J,I)
   10   CONTINUE
        WRITE(IFILE,1060) IMON(I),OUT(1,I),(IOUT(J),J=2,12)
   20 CONTINUE
C
      DO 30 I=1,5
        ISUMY(I)= RND(SUMY(I))
   30 CONTINUE
      WRITE(IFILE,1030) ISUMY
C
      IF (IFILE.EQ.7) WRITE(IFILE,1010)
C
      RETURN
 1010 FORMAT(//)
 1030 FORMAT(/,'  Yearly Totals:',2I6,18X,3I6)
 1050 FORMAT (/,' MON  TEMP   UPE   APE  PREC  DIFF    ST   DST    AE
     *DEF  SURP   SMT   SST',/)
 1060 FORMAT(1X,A3,1X,F5.1,11I6)
 9050 FORMAT(1X,'Water budget for:   Latitude',4X,I2,A1,5X,' Longitude',
     *3X,I3,A1,/,10X,'Field capacity ',F7.1,'mm      Resistance curve ',
     *A1,/,10X,'Moisture Index ',F9.2,6X,'Temp: ',F5.1,' deg.    Precip:
     * ',F5.1,'%')
      END
C***********************************************************************
      SUBROUTINE CLEAR
      CALL QSMODE(2)
      RETURN
      END
C***********************************************************************
      SUBROUTINE RDR(VALUE,IER)
C
C   READS IN NUMERIC DATA AND CHECKS FOR ERRORS
C
      CHARACTER*1 A(10)
      CHARACTER*10 AA
      INTEGER B(10)
      EQUIVALENCE (AA,A(1))
      DATA A/10*' '/
      READ(5,1000) AA
      NDEC= 0
      SGN= 1.0
      B(1)= ICHAR(A(1))-48
      IF (B(1).EQ.-2) THEN
         NDEC= 1
         GO TO 1
      ENDIF
      IF (B(1).EQ.-3) THEN
         B(1)= 0
         SGN= -1.0
         GO TO 1
      ENDIF
      IF (B(1).EQ.-16) GO TO 20
      IF (B(1).GT.9.OR.B(1).LT.0) GO TO 10
    1 DO 2 I=2,10
         B(I)= ICHAR(A(I))-48
         IF (B(I).EQ.-16) THEN
            NCHAR= I
            GO TO 3
         ENDIF
         IF (B(I).EQ.-2) THEN
            IF (NDEC.NE.0) GO TO 10
            NDEC= I
            GO TO 2
         ENDIF
         IF (B(I).GT.9.OR.B(I).LT.0) GO TO 10
    2 CONTINUE
      NCHAR= 11
    3 NCHAR= NCHAR-1
      IF (NDEC.EQ.0) NDEC= NCHAR+1
      VALUE= 0.0
      DO 4 I=1,NCHAR
         IF (I.EQ.NDEC) GO TO 4
         J= I+1
         IF (J.GT.NDEC+1) J= J-1
         VALUE= VALUE + FLOAT(B(I))*(10.**(NDEC-J))
    4 CONTINUE
      VALUE= VALUE*SGN
C
C  NORMAL END -- IER= 0  AND  VALUE RETURNS NUMBER
C
      IER= 0
      RETURN
C
C  ERROR:  NON-NUMERIC CHARACTER FOUND -- IER= 1
C
   10 IER= 1
      RETURN
C
C  CARRAGE RETURN -- USE SAME VALUE AS PREVIOUSLY ENTERED
C
   20 IER= -1
      RETURN
 1000 FORMAT(A)
      END
C***********************************************************************
C
C   SPLINE SUBROUTINES
C       Subroutines Tested and Implemented -    March 1986   DRL
C
C***********************************************************************
      SUBROUTINE SPLINE(Y,YY)
C
C  CALCULATE THE SPLINES TO INTERPOLATE APE OR PREC
C         Y  -  ARRAY OF VALUES FOR THE 12 MONTHS (13=1)
C         YY -  ARRAY OF VALUES FOR THE 361 DAYS  (361=1)
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
   22 J1= 1
      E(1)= AA(1)*Z(1)
      DO 3 K=2,16
         ZZ= AA(K)
         E(K)= ZZ*Z(J1)+(1.-ZZ)*Z(K)
         J1= K
3     CONTINUE
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
4        J1= K
         U1= U2
         V1= V2
         S1= S2
         T1= T2
5     CONTINUE
      D(16)= C(16)
      IF (Z(16).EQ.0.0.OR.Z(15).EQ.0.0) D(16)= 0.
      DO 6 J1= 2,15
         K= 17-J1
         D(K)= C(K)-B(K)*D(K+1)
         IF (Z(K).EQ.0.0.OR.Z(K-1).EQ.0.0) D(K)= 0.
6     CONTINUE
7     DO 8 K=1,16
        J2= K+1
        ZZ= A(K)
        C(K)= 1.5*ZZ*(-3.*D(K)+D(J2)+ZZ*(-8.*E(J2)-12.*E(K)+20.*Z(K)))
        B(K)=-4.*ZZ*ZZ*(-1.5*D(K)+D(J2)+ZZ*(-7.*E(J2)-8.*E(K)+15.*Z(K)))
        A(K)= 5.*ZZ*ZZ*ZZ*(.5*(D(J2)-D(K))-3.*ZZ*(E(J2)+E(K)-2.*Z(K)))
8     CONTINUE
C
      NEGATV= .FALSE.
      K= 1
      DO 10 J=3,14
         THISAJ= .FALSE.
         DO 9 I=0,29
            XDIFF= FLOAT(I)
            YY(K)= E(J)+D(J)*XDIFF+C(J)*XDIFF**2+B(J)*XDIFF**3
     #             +A(J)*XDIFF**4
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
   10 CONTINUE
C
      NIT= NIT+1
      IF (NEGATV.AND.NIT.GE.4) GO TO 11
      IF (NEGATV) GO TO 22
C
      GO TO 15
C
   11 K= 1
      DO 14 J=3,14
         DO 13 I=0,29
            XDIFF= FLOAT(I)
            YY(K)= E(J)+D(J)*XDIFF+C(J)*XDIFF**2+B(J)*XDIFF**3
     #             +A(J)*XDIFF**4
            IF (Z(J).EQ.0.) YY(K)= 0.
            IF (YY(K).LT.0.) THEN
               K= (J-3)*30+1
               DO 12 J2=0,29
                  YY(K)= Z(J)
                  K= K+1
   12          CONTINUE
               GO TO 14
            END IF
            K= K+1
   13    CONTINUE
   14 CONTINUE
C
   15 DO 16 I=361,390
         YY(I)= YY(I-360)
   16 CONTINUE
C
      RETURN
      END
C***********************************************************************
      SUBROUTINE CUBICP(Y,YY)
C
C  CALCULATE THE SPLINE TO INTERPOLATE TEMPERATURE
C         Y  -  ARRAY OF VALUES FOR THE 12 MONTHS (13=1)
C         YY -  ARRAY OF VALUES FOR THE 361 DAYS  (361=1,ETC.)
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
C
C     GRAPHIC SUBROUTINES -- USING MICROCOMPATIBLES' GRAFMATIC LIBRARY
C
C***********************************************************************
      SUBROUTINE GRAPH(LAT,LON,P,APE,AE,SUR,DEF,DST,SMT,SST)
      REAL LAT,LON,X(390),P(390)
      REAL APE(390),AE(390),SUR(390),DEF(390),DST(390),SMT(390),SST(390)
      INTEGER IX(392),IY(392)
      CHARACTER*5 CHRCTR
      IRND(XXX)= INT(SIGN(1.,XXX)*(ABS(XXX)+0.5))
      RND(XXX)= FLOAT(INT(SIGN(1.,XXX)*(ABS(XXX)+0.5)))
      YMAX= 0.
      CALL QSMODE(13)
      DO 1 I=1,30
         P(I+360)= P(I)
         APE(I+360)= APE(I)
         AE(I+360)= AE(I)
         SUR(I+360)= SUR(I)
         DEF(I+360)= DEF(I)
         DST(I+360)= DST(I)
         SMT(I+360)= SMT(I)
         SST(I+360)= SST(I)
         IF (I.GT.21) GO TO 1
         IX(I)= 250
         IX(I+21)= 249+I
         IX(I+42)= 270
         IX(I+63)= 271-I
         IY(I)= 31+I
         IY(I+21)= 52
         IY(I+42)= 53-I
         IY(I+63)= 32
    1 CONTINUE
      DO 2 I=2,390,2
         X(I-1)= FLOAT(I-1)/30.
         X(I)= FLOAT(I)/30.
         P(I-1)= P(I-1)*30.
         APE(I-1)= APE(I-1)*30.
         AE(I-1)= AE(I-1)*30.
         P(I)= P(I)*30.
         APE(I)= APE(I)*30.
         AE(I)= AE(I)*30.
         SMT(I)= SMT(I)*30.
         SST(I)= SST(I)*30.
         IF (SMT(I).GT.0.) THEN
            P(I)= P(I)+SMT(I)
         ELSE
            IF (SST(I).GT.0.) P(I)= 0.
         END IF
         YMAX= AMAX1(P(I),YMAX)
         YMAX= AMAX1(APE(I),YMAX)
    2 CONTINUE
      IF (AMOD(YMAX,25.).GT.20.0) YMAX= YMAX+25.
      YMAX= YMAX-AMOD(YMAX,25.)+25.
      ILAT= IABS(INT(LAT))
      ILON= IABS(INT(LON))
      CALL QPLOT(40,235,30,190,0.,13.,0.,YMAX,13.,YMAX,0,1,1.33)
      XX= -1.
      CALL QXAXIS(0.,13.,XX,0,0,0)
      XX= -YMAX/5.
      CALL QYAXIS(0.,YMAX,XX,0,0,0)
      CALL QPLOT(40,235,30,190,0.,13.,0.,YMAX,0.,0.,0,1,1.33)
      CALL QXAXIS(0.,13.,1.,0,0,0)
      CALL QYAXIS(0.,YMAX,YMAX/5.,0,0,0)
      CALL QGTXT(2,'MM',3,11,106,0)
      CALL QGTXT(1,'J',3,45,20,0)
      CALL QGTXT(1,'F',3,60,20,0)
      CALL QGTXT(1,'M',3,75,20,0)
      CALL QGTXT(1,'A',3,90,20,0)
      CALL QGTXT(1,'M',3,105,20,0)
      CALL QGTXT(1,'J',3,120,20,0)
      CALL QGTXT(1,'J',3,135,20,0)
      CALL QGTXT(1,'A',3,150,20,0)
      CALL QGTXT(1,'S',3,165,20,0)
      CALL QGTXT(1,'O',3,180,20,0)
      CALL QGTXT(1,'N',3,195,20,0)
      CALL QGTXT(1,'D',3,210,20,0)
      CALL QGTXT(1,'J',3,225,20,0)
      CALL QGTXT(1,'0',3,30,26,0)
      DO 3 I=1,5
         WRITE(CHRCTR,1001) INT(YMAX*FLOAT(I))/5
         CALL QGTXT(5,CHRCTR,3,0,27+159*I/5,0)
    3 CONTINUE
      WRITE(CHRCTR,1001) ILAT
      CALL QGTXT(5,CHRCTR,3,68,192,0)
      IF (LAT.GT.0.0) CALL QGTXT(1,'N',3,108,192,0)
      IF (LAT.LT.0.0) CALL QGTXT(1,'S',3,108,192,0)
      WRITE(CHRCTR,1001) ILON
      CALL QGTXT(5,CHRCTR,3,144,192,0)
      IF (LON.GT.0.0) CALL QGTXT(1,'E',3,184,192,0)
      IF (LON.LT.0.0) CALL QGTXT(1,'W',3,184,192,0)
      DO 4 I=1,21
         CALL QLINE(250,IY(I),270,IY(I),2)
    4 CONTINUE
      CALL QLINE(250,IY(1),250,IY(22),3)
      CALL QLINE(250,IY(22),270,IY(22),3)
      CALL QLINE(270,IY(22),270,IY(1),3)
      CALL QLINE(270,IY(1),250,IY(1),3)
      CALL QGTXT(4,'SURP',3,280,IY(1)+6,0)
      DO 5 I=1,84
         IY(I)= IY(I)+45
         IF (I.LE.21) CALL QLINE(250,IY(I),270,IY(I),1)
    5 CONTINUE
      CALL QLINE(250,IY(1),250,IY(22),3)
      CALL QLINE(250,IY(22),270,IY(22),3)
      CALL QLINE(270,IY(22),270,IY(1),3)
      CALL QLINE(270,IY(1),250,IY(1),3)
      CALL QGTXT(4,'+DST',3,280,IY(1)+6,0)
      DO 6 I=1,84
         IY(I)= IY(I)+45
         IF (I.LE.21) CALL QLINE(250,IY(I),270,IY(I),1)
    6 CONTINUE
      CALL QLINE(250,IY(1),250,IY(22),3)
      CALL QLINE(250,IY(22),270,IY(22),3)
      CALL QLINE(270,IY(22),270,IY(1),3)
      CALL QLINE(270,IY(1),250,IY(1),3)
      CALL DAPPLE(IX,IY,84,1,0)
      CALL QGTXT(4,'-DST',3,280,IY(1)+6,0)
      DO 7 I=1,84
         IY(I)= IY(I)+45
         IF (I.LE.21) CALL QLINE(250,IY(I),270,IY(I),2)
    7 CONTINUE
      CALL QLINE(250,IY(1),250,IY(22),3)
      CALL QLINE(250,IY(22),270,IY(22),3)
      CALL QLINE(270,IY(22),270,IY(1),3)
      CALL QLINE(270,IY(1),250,IY(1),3)
      CALL DAPPLE(IX,IY,84,2,0)
      CALL QGTXT(4,' DEF',3,280,IY(1)+6,0)
      CALL SHADE(AE,P,1,YMAX,IX,IY,X,0)
      CALL SHADE(APE,AE,2,YMAX,IX,IY,X,0)
      CALL SHADE(P,APE,1,YMAX,IX,IY,DST,1)
      CALL SHADE(P,APE,2,YMAX,IX,IY,SUR,1)
      CALL QPLOT(40,235,30,190,0.,13.,0.,YMAX,13.,YMAX,0,1,1.33)
      XX= -1.
      CALL QXAXIS(0.,13.,XX,0,0,0)
      XX= -YMAX/5.
      CALL QYAXIS(0.,YMAX,XX,0,0,0)
      CALL QPLOT(40,235,30,190,0.,13.,0.,YMAX,0.,0.,0,1,1.33)
      CALL QXAXIS(0.,13.,1.,0,0,0)
      CALL QYAXIS(0.,YMAX,YMAX/5.,0,0,0)
      CALL QGTXT(24,'PRESS RETURN TO CONTINUE',3,42,1,0)
   10 CALL QINKEY(I,J)
      IF (I.NE.1.OR.J.NE.13) GO TO 10
      CALL QSMODE(2)
      RETURN
 1001 FORMAT(I5)
      END
C***********************************************************************
      SUBROUTINE SHADE(X1,X2,KLR,YMAX,IX,IY,X,ITRIP)
      DIMENSION X1(390),X2(390),IX(392),IY(392),X(390)
      LOGICAL SIGNIF
      IRND(XXX)= INT(SIGN(1.,XXX)*(ABS(XXX)+0.5))
      J= 0
      DO 3 I=0,390,2
         IF (I.EQ.0) THEN
            XX1= X1(390)
            XX2= X2(390)
            IF (ITRIP.NE.0.AND.X(390).LE.0.) XX1= 0.
         ELSE
            XX1= X1(I)
            XX2= X2(I)
            IF (ITRIP.NE.0.AND.X(I).LE.0.) XX1= 0.
         END IF
         IF (XX1.LE.XX2) THEN
            IF (J.EQ.0) GO TO 3
            J= J+1
            IEND= I-2
            SIGNIF= .FALSE.
            DO 1 K=IBEG,IEND,2
               L = (K-IBEG)/2+1
               L2= (IEND-K)/2+J
               IX(L) = 40+K/2
               IX(L2)= IX(L)
               IF (K.EQ.0) THEN
                  IY(L) = 30+IRND(160.* X1(390)/YMAX)
                  IY(L2)= 30+IRND(160.* X2(390)/YMAX)
               ELSE
                  IY(L) = 30+IRND(160.* X1(K)/YMAX)
                  IY(L2)= 30+IRND(160.* X2(K)/YMAX)
               END IF
               IF (IY(L).NE.IY(L2)) THEN
                  SIGNIF= .TRUE.
                  CALL QLINE(IX(L),IY(L),IX(L2),IY(L2),KLR)
               END IF
    1       CONTINUE
            K= J*2-2
            IF (SIGNIF) THEN
               DO 2 J=1,K-1
                  CALL QLINE(IX(J),IY(J),IX(J+1),IY(J+1),3)
    2          CONTINUE
               CALL QLINE(IX(K),IY(K),IX(1),IY(1),3)
               IF (ITRIP.EQ.0) CALL DAPPLE(IX,IY,K,KLR,0)
            END IF
            J= 0
         ELSE
            IF (J.EQ.0.AND.I.NE.0) THEN
               J= 2
               IBEG= I-2
               IF (IBEG.LT.0) IBEG= 0
            ELSE
               IF (J.EQ.0) IBEG= 0
               J= J+1
            END IF
         END IF
    3 CONTINUE
      IF (J.EQ.0) GO TO 6
      J= J+1
      SIGNIF= .FALSE.
      DO 4 K=IBEG,390,2
         L = (K-IBEG)/2+1
         L2= (390-K)/2+J
         IX(L) = 40+K/2
         IX(L2)= IX(L)
         IF (K.EQ.0) THEN
            IY(L) = 30+IRND(160.* X1(390)/YMAX)
            IY(L2)= 30+IRND(160.* X2(390)/YMAX)
         ELSE
            IY(L) = 30+IRND(160.* X1(K)/YMAX)
            IY(L2)= 30+IRND(160.* X2(K)/YMAX)
         END IF
         IF (IY(L).NE.IY(L2)) THEN
            SIGNIF= .TRUE.
            CALL QLINE(IX(L),IY(L),IX(L2),IY(L2),KLR)
         END IF
    4 CONTINUE
      K= J*2-2
      IF (.NOT.SIGNIF) GO TO 6
      DO 5 J=1,K-1
         CALL QLINE(IX(J),IY(J),IX(J+1),IY(J+1),3)
    5 CONTINUE
      CALL QLINE(IX(K),IY(K),IX(1),IY(1),3)
      IF (ITRIP.EQ.0) CALL DAPPLE(IX,IY,K,KLR,0)
    6 RETURN
      END
C***********************************************************************
      SUBROUTINE DAPPLE(IX,IY,K,KLR1,KLR2)
      DIMENSION IX(K),IY(K)
      IXMAX= IX(1)
      IXMIN= IX(1)
      IYMAX= IY(1)
      IYMIN= IY(1)
      DO 1 I=2,K
         IF (IX(I).GT.IXMAX) IXMAX= IX(I)
         IF (IX(I).LT.IXMIN) IXMIN= IX(I)
         IF (IY(I).GT.IYMAX) IYMAX= IY(I)
         IF (IY(I).LT.IYMIN) IYMIN= IY(I)
    1 CONTINUE
      DO 3 I=IYMIN,IYMAX
         II= IYMAX+IYMIN-I
         L= 0
         IF (MOD(I,2).EQ.0) L= 1
         DO 2 J=IXMIN+L,IXMAX,2
            CALL QRPNT(J,II,KLRI)
            IF (KLRI.EQ.KLR1) CALL QSPNT(J,II,KLR2)
    2    CONTINUE
    3 CONTINUE
      RETURN
      END
C***********************************************************************
      SUBROUTINE PLTCRV(SM)
      EXTERNAL AFUN,BFUN,CFUN,DFUN,EFUN,FFUN,GFUN,HFUN,XFUN
      INTEGER*2 JCOL(10),JROW(10)
      LOGICAL PLTD
      PLTD= .FALSE.
      CALL QSMODE(13)
      X= -10.
      CALL QPLOT(70,250,50,190,0.,100.,0.,100.,100.,0.,0,1.,1.33)
      CALL QYAXIS(0.,100.,X,0,0,0)
      X= 10.
      CALL QPLOT(70,250,50,190,0.,100.,0.,100.,0.,0.,0,1.,1.33)
      CALL QXAXIS(100.,0.,X,0,0,0)
      CALL QYAXIS(0.,100.,X,0,0,0)
      CALL QSETUP(0,3,-2,0)
      CALL QCURV(AFUN,XFUN,0.,100.)
      CALL QGTXT(17,'SOIL MOISTURE (%)',3,94,30,0)
      CALL QGTXT(3,'___',3,24,120,0)
      CALL QGTXT(2,'PE',3,28,110,0)
      CALL QGTXT(2,'AE',3,28,122,0)
      CALL QGTXT(1,'%',3,52,116,0)
      CALL QGTXT(1,'0',3,246,40,0)
      CALL QGTXT(2,'20',3,205,40,0)
      CALL QGTXT(2,'40',3,169,40,0)
      CALL QGTXT(2,'60',3,133,40,0)
      CALL QGTXT(2,'80',3,97,40,0)
      CALL QGTXT(3,'100',3,59,40,0)
      CALL QGTXT(1,'0',3,60,48,0)
      CALL QGTXT(2,'20',3,52,74,0)
      CALL QGTXT(2,'40',3,52,102,0)
      CALL QGTXT(2,'60',3,52,130,0)
      CALL QGTXT(2,'80',3,52,158,0)
      CALL QGTXT(3,'100',3,44,185,0)
      CALL QSETUP(0,2,-2,0)
      CALL QCURV(XFUN,BFUN,0.,100.)
      CALL QSETUP(0,1,-2,0)
      CALL QCURV(XFUN,CFUN,0.,100.)
      CALL QSETUP(0,2,-2,0)
      CALL QCURV(XFUN,DFUN,0.,100.)
      CALL QCURV(XFUN,EFUN,0.,100.)
      CALL QSETUP(0,1,-2,0)
      CALL QCURV(XFUN,GFUN,0.,69.)
      CALL QCURV(XFUN,HFUN,0.,50.)
      CALL QSETUP(0,2,-2,0)
      CALL QCURV(XFUN,FFUN,0.,100.)
      CALL QSETUP(0,1,-2,0)
      CALL QCURV(XFUN,AFUN,0.,100.)
      CALL QGTXT(1,'A',3,220,186,0)
      CALL QGTXT(1,'B',3,228,116,0)
      CALL QGTXT(1,'C',3,90,166,0)
      CALL QGTXT(1,'D',3,154,88,0)
      CALL QGTXT(1,'E',3,188,74,0)
      CALL QGTXT(1,'F',3,199,147,0)
      CALL QGTXT(1,'G',3,154,147,0)
      CALL QGTXT(1,'H',3,181,147,0)
      CALL QGTXT(32,'Select a resistance curve <A-H>:  ',3,1,1,0)
      ISM= -1
    1 CALL QINKEY(I,KEY)
      IF (I.NE.1) GO TO 2
      IF (KEY.EQ.8) THEN
         CALL QGTXT(1,' ',3,273,1,0)
         ISM= -1
         GO TO 1
      END IF
      IF (KEY.EQ.13) THEN
         IF (ISM.GE.0.AND.ISM.LE.7) GO TO 3
         GO TO 2
      END IF
      KEY= KEY-65
      IF (KEY.GT.7) KEY= KEY-32
      IF (KEY.GE.0.AND.KEY.LE.7) THEN
         ISM= KEY
         CALL QGTXT(1,CHAR(ISM+65),3,273,1,0)
         GO TO 1
      END IF
    2 CALL QBEEP
      IF (PLTD) GO TO 1
      PLTD= .TRUE.
      CALL QGTXT(21,'<Select a curve: A-H>',3,74,14,0)
      GO TO 1
    3 CALL QSMODE(2)
      SM= FLOAT(ISM)
      RETURN
      END
C***********************************************************************
C
C   FUNCTIONS FOR SUBROUTINE PLTCRV
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
