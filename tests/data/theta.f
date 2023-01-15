      program THETA
      real*8 thita,en,e,pi,h,e1,thita1


      E=0.764
      pi=4*atan(1.)
      EN=0.
      h=2.*pi/(48.*4.)
      !e1=2.*atan(sqrt((1-e)/(1+e)))

      !thita1=2.*ATAN(SQRT((1.+E)/(1.-E))*TAN(E1/2.))
      
      !write(*,*)'e1,thita1',e1*360/(2.*pi),thita1*360/(2.*pi)
      do i=1,(48*4)
      THiTA=2.*ATAN(SQRT((1.+E)/(1.-E))*TAN(EN/2.))
      if (thita.lt.0) then
      thita=thita+2*pi
      endif
      write(*,*)en*360/(2.*pi),thita*360/(2.*pi)
      en=en+h
      enddo  
      end 