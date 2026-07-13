      SUBROUTINE INIT
C
C Set up graphics device to display menu.
C-----------------------------------------------------------------------
      CALL PGPAP(2.5, 2.0)
      CALL PGPAGE
      CALL PGSVP(0.0,1.0,0.0,1.0)
      CALL PGSWIN(0.0,0.5,0.0,1.0)
      CALL PGSCR(0, 0.4, 0.4, 0.4)
      RETURN
C-----------------------------------------------------------------------
      END

      SUBROUTINE MENU(NP, ID)
      INTEGER NP, ID
C
C Display menu of plots.
C-----------------------------------------------------------------------
      INTEGER NBOX
      PARAMETER (NBOX=16)
      CHARACTER*12 VALUE(NBOX)
      INTEGER I, JUNK, K
      REAL X1, X2, Y(NBOX), XX, YY, R
      CHARACTER CH
      INTEGER PGCURS
C
      DATA VALUE / '1', '2', '3', '4', '5', '6', '7', '8', '9',
     :             '10', '11', '12', '13',
     :             'One panel', 'Four panels', 'EXIT' /

      DATA XX/0.5/, YY/0.5/
C
      X1 = 0.1
      X2 = 0.2
      DO 5 I=1,NBOX
         Y(I) = 1.0 - REAL(I+1)/REAL(NBOX+2)
 5    CONTINUE
C
C Display buttons.
C
      CALL PGBBUF
      CALL PGSAVE
      CALL PGERAS
      CALL PGSCI(1)
      CALL PGSCH(2.5)
      CALL PGPTXT(X1, 1.0-1.0/REAL(NBOX+2), 0.0, 0.0, '\fiMENU')
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
         CALL PGPTXT(X2, Y(I), 0.0, 0.0, VALUE(I))
 10   CONTINUE
      K = 14
      IF (NP.EQ.4) K = 15
      CALL PGSCI(2)
      CALL PGSFS(1)
      CALL PGCIRC(X1, Y(K), 0.02)
      CALL PGUNSA
      CALL PGEBUF
C
C Cursor input.
C
 20   JUNK = PGCURS(XX, YY, CH)
      IF (ICHAR(CH).EQ.0) GOTO 50
C
C Find which box and highlight it
C
      DO 30 I=1,NBOX
         R = (XX-X1)**2 +(YY-Y(I))**2
         IF (R.LT.(0.03**2)) THEN
            ID = I
            CALL PGSAVE
            CALL PGSCI(2)
            CALL PGSFS(1)
            CALL PGCIRC(X1, Y(I), 0.02)
            CALL PGUNSA
            RETURN
         END IF
 30   CONTINUE
      GOTO 20
 50   ID = 0
      RETURN
      END
