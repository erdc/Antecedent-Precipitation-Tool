module menu_var

  INTEGER NP
  INTEGER NBOX
  PARAMETER (NBOX=9)
  CHARACTER*12 VALUE(NBOX)
  INTEGER I, JUNK, K
  REAL X1, X2, Y(NBOX), XX, YY, R
  CHARACTER CH
  DATA VALUE / 'A','B','C','D','E','F','G','H','Exit'/

  DATA XX/0.5/, YY/0.5/

end module menu_var
