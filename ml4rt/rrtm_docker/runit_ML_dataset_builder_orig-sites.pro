; This script creates the netCDF file that contains both the input and output patterns
; for the machine learning (ML) project that will try to approximate the shortwave
; RRTM model.
;
; Inputs:
;    The "extracted column" datasets that I have been creating from the RAP
;       From these files, I will use the PTU and cloud profiles as input to the RRTM_SW
; Outputs:
;    A netCDF file that contains the input profiles, along with the computed SW fluxes
;       and heating rate profiles
;
;---------------------------------------------------------------------------------
function create_output_file, name, z

  fid = ncdf_create(name,/clobber)
  did0 = ncdf_dimdef(fid,'time',/unlimited)
  did1 = ncdf_dimdef(fid,'height',n_elements(z))
  vid  = ncdf_vardef(fid,'time',/long,did0)
    ncdf_attput,fid,vid,'long_name','Time since 1 Jan 1970 at 00:00:00 UTC'
    ncdf_attput,fid,vid,'units','seconds'
  vid = ncdf_vardef(fid,'jday',/float,did0)
    ncdf_attput,fid,vid,'long_name','Julian day (i.e., day of the year)'
    ncdf_attput,fid,vid,'units','days'
    ncdf_attput,fid,vid,'comment','day = 1 is 1 January'
  vid = ncdf_vardef(fid,'height',/float,did1)
    ncdf_attput,fid,vid,'long_name','Height above ground'
    ncdf_attput,fid,vid,'units','km'
  vid = ncdf_vardef(fid,'sza',/float,did0)
    ncdf_attput,fid,vid,'long_name','Solar zenith angle'
    ncdf_attput,fid,vid,'units','degrees'
  vid = ncdf_vardef(fid,'stdatmos',/float,did0)
    ncdf_attput,fid,vid,'long_name','Standard atmosphere used'
    ncdf_attput,fid,vid,'units','1 -- tropics, 2 -- Midlat Summer, 3 -- Midlat Winter, ' + $
    			    '4 -- Subarctic Summer, 5 -- Subarctic Winter, 6 -- US Standard Atmos'
  vid = ncdf_vardef(fid,'lat',/float,did0)
    ncdf_attput,fid,vid,'long_name','Site latitude'
    ncdf_attput,fid,vid,'units','degrees N'
  vid = ncdf_vardef(fid,'lon',/float,did0)
    ncdf_attput,fid,vid,'long_name','Site longitude'
    ncdf_attput,fid,vid,'units','degrees E'
  vid = ncdf_vardef(fid,'albedo',/float,did0)
    ncdf_attput,fid,vid,'long_name','Surface albedo'
    ncdf_attput,fid,vid,'units','unitless'
  vid = ncdf_vardef(fid,'lwp',/float,did0)
    ncdf_attput,fid,vid,'long_name','Liquid water path'
    ncdf_attput,fid,vid,'units','g/m2'
  vid = ncdf_vardef(fid,'iwp',/float,did0)
    ncdf_attput,fid,vid,'long_name','Ice water path'
    ncdf_attput,fid,vid,'units','g/m2'
  vid = ncdf_vardef(fid,'p',/float,[did1,did0])
    ncdf_attput,fid,vid,'long_name','Pressure'
    ncdf_attput,fid,vid,'units','mb'
  vid = ncdf_vardef(fid,'t',/float,[did1,did0])
    ncdf_attput,fid,vid,'long_name','Temperature'
    ncdf_attput,fid,vid,'units','K'
  vid = ncdf_vardef(fid,'q',/float,[did1,did0])
    ncdf_attput,fid,vid,'long_name','Water vapor mixing ratio'
    ncdf_attput,fid,vid,'units','g/kg'
  vid = ncdf_vardef(fid,'lwc',/float,[did1,did0])
    ncdf_attput,fid,vid,'long_name','Layer liquid water content'
    ncdf_attput,fid,vid,'units','g/m2'
  vid = ncdf_vardef(fid,'iwc',/float,[did1,did0])
    ncdf_attput,fid,vid,'long_name','Layer ice water content'
    ncdf_attput,fid,vid,'units','g/m2'
  vid = ncdf_vardef(fid,'hr',/float,[did1,did0])
    ncdf_attput,fid,vid,'long_name','SW radiative heating rate'
    ncdf_attput,fid,vid,'units','K/day'
  vid = ncdf_vardef(fid,'fluxd',/float,[did1,did0])
    ncdf_attput,fid,vid,'long_name','SW downwelling flux'
    ncdf_attput,fid,vid,'units','W/m2'
  vid = ncdf_vardef(fid,'fluxu',/float,[did1,did0])
    ncdf_attput,fid,vid,'long_name','SW upwelling flux'
    ncdf_attput,fid,vid,'units','W/m2'
  vid = ncdf_vardef(fid,'sfcflux',/float,did0)
    ncdf_attput,fid,vid,'long_name','SW downwelling flux at the surface'
    ncdf_attput,fid,vid,'units','W/m2'
  vid = ncdf_vardef(fid,'toaflux',/float,did0)
    ncdf_attput,fid,vid,'long_name','SW upwelling flux at the top of the atmosphere'
    ncdf_attput,fid,vid,'units','W/m2'
  ncdf_attput,fid,/global,'Author','dave.turner@noaa.gov'
  ncdf_attput,fid,/global,'Comment1','Shortwave radiative flux calculations and atmospheric state inputs'
  ncdf_attput,fid,/global,'Comment2','SW calculations made with RRTM_SW'
  ncdf_attput,fid,/global,'Comment3','Inputs from NCEP RAP model at various locations'
  ncdf_control,fid,/endef
  ncdf_varput,fid,'height',z
  ncdf_close,fid
return, 0
end
;---------------------------------------------------------------------------------
function append_to_output_file, name, index, isecs, ijday, isza, istdatmos, ilat, ilon, ialbedo, $
	ilwp, iiwp, ilwc, iiwc, itt, ipp, iqq, ohr, ofluxu, ofluxd, oosr, ossr

  fid = ncdf_open(name, /write)
  ncdf_varput,fid,'time',isecs,offset=index
  ncdf_varput,fid,'jday',ijday,offset=index
  ncdf_varput,fid,'sza',isza,offset=index
  ncdf_varput,fid,'stdatmos',istdatmos,offset=index
  ncdf_varput,fid,'lat',ilat,offset=index
  ncdf_varput,fid,'lon',ilon,offset=index
  ncdf_varput,fid,'albedo',ialbedo,offset=index
  ncdf_varput,fid,'lwp',ilwp,offset=index
  ncdf_varput,fid,'iwp',iiwp,offset=index
  ncdf_varput,fid,'p',ipp,offset=[0,index]
  ncdf_varput,fid,'t',itt,offset=[0,index]
  ncdf_varput,fid,'q',iqq,offset=[0,index]
  ncdf_varput,fid,'lwc',ilwc,offset=[0,index]
  ncdf_varput,fid,'iwc',iiwc,offset=[0,index]
  ncdf_varput,fid,'hr',ohr,offset=[0,index]
  ncdf_varput,fid,'fluxu',ofluxu,offset=[0,index]
  ncdf_varput,fid,'fluxd',ofluxd,offset=[0,index]
  ncdf_varput,fid,'sfcflux',ossr,offset=index
  ncdf_varput,fid,'toaflux',oosr,offset=index
  ncdf_close,fid
  return, index + n_elements(isecs)
end
;---------------------------------------------------------------------------------
; Main routine
pro runit, year

	; This is the flag telling the code to create a new output file (=1)
  do_create_output = 1
  outname = string(format='(A,I0,A)','output_file.',year,'.cdf')

	; These are the sites that I care to process
  sites = [" ARM sgpC1 site (Lamont OK)", $
  	   " ARM nsaC1 site (Barrow AK) - ocean", $
	   " Eureka (Canada) - ocean", $
	   " NyAlesund (Norway) - land", $
	   " Vortex-SE-2016 site (Belle Mina AL)", $
	   " NY Mesonet site - Tupper Lake (NY)", $
	   " NY Mesonet site - Owego (NY)", $
	   " NOAA/ESRL/PSD 449 MHz profiler RASS site (Forks WA)", $
	   " NOAA/ESRL/PSD 449 MHz profiler RASS site (Bodega Bay CA)", $
	   " NOAA/ESRL/PSD 449 MHz profiler RASS site (Santa Barbara CA)", $
  	   " JOYCE Site (Juelich Germany)"]
  	; These are the forecast hours that I care to process
  hours = indgen(12)*2+1

	; Get the standard atmospheres loaded into memory
  restore,'/home/user/vip/src/idl_code/std_atmosphere.idl'
  satmos = {name:name, p:p, t:t, q:w, z:z}

	; Loop over the input files 
  files = file_search('profiler_sites?.*rap*nc',count=count)
  for i=0L,count-1 do begin ; { 
    parts = str_sep(files(i),'.')
    yydoy = parts(n_elements(parts)-3)
    hhmm  = parts(n_elements(parts)-2)
    yy0   = fix(strmid(yydoy,0,4))
    doy   = fix(strmid(yydoy,4,3))
    hh    = fix(strmid(hhmm,0,2))
    nn    = fix(strmid(hhmm,2,2))
    julian2ymdhms,yy0,doy,yy,mm,dd
    ymdhms2systime,yy,mm,dd,0,0,0,secs
    print,'Processing '+files(i)
    fid = ncdf_open(files(i))
    ncdf_varget,fid,'dsite',dsite
    dsite = string(dsite)
    ncdf_varget,fid,'mlat',mlat
    ncdf_varget,fid,'mlon',mlon
    ncdf_varget,fid,'forecast',fhour
    ncdf_varget,fid,'height',z
    ncdf_varget,fid,'p0',p	; mb
    ncdf_varget,fid,'t0',t	; C
    ncdf_varget,fid,'r0',q	;  g / kg
    ncdf_varget,fid,'lwc0',lwc 	; kg / kg
    ncdf_varget,fid,'iwp0',iwc 	; kg / m2
    ncdf_varget,fid,'dswsfc0',swdn	; W/m2, downwelling sfc flux
    ncdf_varget,fid,'uswsfc0',swup	; W/m2, upwelling   sfc flux
    albedo = (swup / swdn) > 0 < 1
    ncdf_close,fid
    	
    	; Now compute the valid time for these data
    dsecs = secs(0) + fhour*60*60 + hh*60*60 + nn*60
    systime2ymdhms,dsecs,dyy,dmm,ddd,dhh,dnn,dss
    systime2julian,dsecs,dyy(0),djulian

	; Convert the liquid and ice water content profiles from [kg/kg] to [kg/m2]
    airdens = t2dens(t,p,0.)
    lwp = lwc * airdens		; Now kg/m3
    for n=0,n_elements(z)-2 do begin
      lwp(n,*,*) = lwp(n,*,*) * (z(n+1)-z(n))	; Now kg/m2
    endfor
    lwc = lwp

	; Now loop over the desired sites, processing each one in turn
    npts = 0
    rrtm_sw_command = '/home/user/vip/src/rrtm_sw_v2.7.1/rrtm_sw_gfortran_v2.7.2'
    for j=0,n_elements(sites)-1 do begin ; { 
      foo = where(sites(j) eq dsite, nfoo)
      if(nfoo gt 1) then stop,'Found multiple sites -- this should never happen'
      if(nfoo eq 0) then stop,'Did not find a site -- this should not happen'
      print,'  Working on '+dsite(foo)
      for k=0,n_elements(hours)-1 do begin ; {
        bar = where(abs(hours(k) - fhour) lt 0.2, nbar)
	if(nbar ge 2) then stop,'Found multiple fhours for this time -- this should not happen'
	if(nbar eq 1) then begin
          solarpos,dyy(bar(0)),dmm(bar(0)),ddd(bar(0)),dhh(bar(0)),dnn(bar(0)),dss(bar(0)), $
	  	mlat(foo(0)),mlon(foo(0)),xhour,ra,sd,salt
          sza = 90 - salt
          if(sza(0) gt 85) then continue

          if(abs(mlat(foo(0))) lt 65) then begin
	    if(3 lt dmm(bar(0)) and dmm(bar(0)) le 9) then stdatmos = 2 $ 	; Mid-lat summer
	    else stdatmos = 3							; Mid-lat winter
	  endif else begin
	    if(3 lt dmm(bar(0)) and dmm(bar(0)) le 9) then stdatmos = 4 $ 	; Subarctic summer
	    else stdatmos = 5							; Subarctic winter
	  endelse
          feh = where(satmos.z(*,stdatmos) gt max(z/1000.+2) and satmos.z(*,stdatmos) le 50,nfeh)
	  if(nfeh le 0) then stop,'I did not expect this to happen -- fix code'
	  zz = [z/1000,                            satmos.z(feh,stdatmos)]
	  pp = [reform(p(*,foo(0),bar(0))),        satmos.p(feh,stdatmos)]
	  tt = [reform(t(*,foo(0),bar(0)))+273.16, satmos.t(feh,stdatmos)]
	  qq = [reform(q(*,foo(0),bar(0))),        satmos.q(feh,stdatmos)]
	  lw = [1000*reform(lwc(*,foo(0),bar(0))>0), replicate(0.,nfeh)]
                    ; The IWC profile is not in the "columns" netCDF files, so I am going to 
                    ; stub out this logic for a bit by using the lwc*0 as a placeholder
	  iw = [1000*reform(0*lwc(*,foo(0),bar(0))>0), replicate(0.,nfeh)]
	  lwp = total(lw)
	  iwp = total(iw)

	  l_reff = fltarr(n_elements(zz))*0+ 8.0
	  i_reff = fltarr(n_elements(zz))*0+25.0
;	  feh = where(lwc(*,foo(0),bar(0)) gt 1e-7, nfeh)
;	  if(nfeh gt 0) then l_reff(feh) =  8.0			; Pick a default effective radius for liquid clouds
;	  feh = where(iwc(*,foo(0),bar(0)) gt 1e-7, nfeh)
;	  if(nfeh gt 0) then i_reff(feh) = 25.0			; Pick a default effective radius for ice clouds

		; Quick QC check
 	  del = pp(1:n_elements(pp)-1) - pp(0:n_elements(pp)-2)
	  feh = where(del ge 0, nfeh)
	  if(nfeh gt 0) then begin
	    print,'Warning: Pressure profile increased with height -- skipping sample'
	    continue
	  endif

                ; Assume the aerosol optical depth (first index) is zero, and put nominal 
                ; values into the other aerosol fields (but it doesn't matter, because AOD = 0)
          aerosol = [0., 0.99, 0.85, -2., 1.5]

		; Run the RT model
	  out = make_rrtm_sw_calc(zz, pp, tt, qq, lw, iw, l_reff, i_reff, $
		  stdatmos, 10000./[0.1,10], [1,1]-(albedo(foo(0),bar(0))>0.03<0.99), $
		  djulian(bar(0)), sza, aerosol, rrtm_sw_command, silent=1)
	  if(out.success eq 0) then begin
	    print,'Warning: RRTM not run properly.  Time to investigate...'
	    continue
	  endif

          if(npts eq 0) then begin
	    ilwp = lwp
	    iiwp = iwp
	    ilwc = transpose(lw)
	    iiwc = transpose(iw)
	    itt  = transpose(tt)
	    ipp  = transpose(pp)
	    iqq  = transpose(qq)
	    izz  = zz
	    ialbedo = reform(albedo(foo(0),bar(0)))
	    isza    = sza(0)
	    istdatmos = stdatmos
	    ilat   = mlat(foo)
	    ilon   = mlon(foo)
	    isecs  = dsecs(bar(0))
	    ijday  = djulian(bar(0))
	    ohr    = transpose(out.hr)
	    ofluxu = transpose(out.fluxu)
	    ofluxd = transpose(out.fluxdtot)
	    oosr   = out.osr
	    ossr   = out.ssr
	  endif else begin
	    ilwp = [ilwp,lwp]
	    iiwp = [iiwp,iwp]
	    ilwc = [ilwc,transpose(lw)]
	    iiwc = [iiwc,transpose(iw)]
	    itt  = [itt,transpose(tt)]
	    ipp  = [ipp,transpose(pp)]
	    iqq  = [iqq,transpose(qq)]
	    ialbedo = [ialbedo,reform(albedo(foo(0),bar(0)))]
	    isza    = [isza,sza(0)]
	    istdatmos = [istdatmos,stdatmos]
	    ilat   = [ilat,mlat(foo)]
	    ilon   = [ilon,mlon(foo)]
	    isecs  = [isecs,dsecs(bar(0))]
	    ijday  = [ijday,djulian(bar(0))]
	    ohr    = [ohr,transpose(out.hr)]
	    ofluxu = [ofluxu,transpose(out.fluxu)]
	    ofluxd = [ofluxd,transpose(out.fluxdtot)]
	    oosr   = [oosr,out.osr]
	    ossr   = [ossr,out.ssr]
	  endelse
	  npts = n_elements(isza)
	endif
      endfor ; } loop over k
    endfor  ; } loop over j
      
		; Transpose the 2d arrays to get them in the right shape
    if(n_elements(isecs) gt 0) then begin
      ipp    = transpose(ipp)
      itt    = transpose(itt)
      iqq    = transpose(iqq)
      ilwc   = transpose(ilwc)
      iiwc   = transpose(iiwc)
      ohr    = transpose(ohr)
      ofluxu = transpose(ofluxu)
      ofluxd = transpose(ofluxd)

      		; If the netCDF file has not yet been created, then create it
      if(do_create_output eq 1) then begin
	index = 0
        do_create_output = create_output_file(outname, zz)
      endif

		; Append output to the file
      print,'Adding ',n_elements(isecs),' samples to the output file ',outname, ' at ',index, format='(A,I0,A,A,A,I0)'
      index = append_to_output_file(outname, index, isecs, ijday, isza, istdatmos, ilat, ilon, ialbedo, $
	ilwp, iiwp, ilwc, iiwc, itt, ipp, iqq, ohr, ofluxu, ofluxd, oosr, ossr)
    endif

  endfor  ; } loop over i

  return
end
