;+
; $Id: make_rrtm_lw_calc.pro,v 1.3 2021/09/27 16:12:54 dave.turner Exp $
;
; Abstract:
;	This routine makes a RRTM_LW calculation for the input atmospheric state
;   and cloud profile.  The output from the model is returned to the user.  Note
;   that the model levels used in the calculation will be the same as the input levels.
;
; Author:
;	Dave Turner
;	NOAA NSSL
;
; Date:
;	June 2014
;
; Return structure:
;	success: 	1 is successful, 0 if run failed
;
;
; Call:
  function make_rrtm_lw_calc, $		; Returns the above structure
  	z, $				; Input height array [km AGL]
	p, $				; Input pressure array [mb]
	t, $				; Input temperature array [K]
	q, $				; Input water vapor mixing ratio array [g/kg]
	lwc, $ 				; Input liquid water content array [g/m2]
	iwc, $ 				; Input ice water content array [g/m2]
	l_reff, $			; Input liquid water effective radius array [um]
	i_reff, $			; Input ice water effective radius array [um]
	stdatmos, $			; The standard atmosphere to use [1-6]
	sfctemp, $			; The input surface temperature [K]
	sfcwnum, $			; The input wnum values for emissivity spectrum
	sfcemis, $			; The input surface emissivity spectrum
	co2=co2, $			; Input CO2 concentration profile [ppm]
	n2o=n2o, $			; Input N2O concentration [ppm]
	ch4=ch4, $			; Input CH4 concentration [ppm]
	o3p=o3p,   $		; Input O3  concentration (units given by the keyword below)
    o3_units=o3_units,$ ; Units of the O3 profile (must be one of "ppmv", "cm-3", "g/kg", "g/m3")
	rrtm_lw_command, $		; The path/command needed to run the model
	silent=silent, $		; If set, then create the RRTM_INPUT files quietly
	version=version, $		; Returns the version of this routine
	dostop=dostop			; Set this to stop inside routine

  ; Note: the LWC and IWC are the amount of liquid and ice in LAYER i between the 
  ;          ith and (i+1)th levels.  The uppermost value of these arrays should be zero.
  ;          Be careful with the units here!

    ; Somewhat reasonable default profiles
  if(n_elements(co2) eq 0) then co2 = replicate(400.0,n_elements(z)) ; [ppm]
  if(n_elements(n2o) eq 0) then n2o = replicate(0.310,n_elements(z)) ; [ppm]
  if(n_elements(ch4) eq 0) then ch4 = replicate(1.793,n_elements(z)) ; [ppm]
  if(n_elements(o3p) eq 0) then begin
    o3p = replicate(0.000,n_elements(z))    ; Default will be the std atmosphere
    o3_units = fix(stdatmos+0.5)            ;   which is why I put zeros into above profile
  endif
;-

  fail = {success:0}
  version = '$Id: make_rrtm_lw_calc.pro,v 1.3 2021/09/27 16:12:54 dave.turner Exp $'

  	; Some limits on the input effective radii [um]
  min_l_reff =  2.6
  max_l_reff = 50.0
  min_i_reff =  5.1
  max_i_reff = 95.0

	; Some QC of the input
  if(n_elements(z) ne n_elements(p) or $
     	n_elements(z) ne n_elements(t) or $
     	n_elements(z) ne n_elements(q) or $
     	n_elements(z) ne n_elements(lwc) or $
     	n_elements(z) ne n_elements(iwc) or $
     	n_elements(z) ne n_elements(co2) or $
     	n_elements(z) ne n_elements(n2o) or $
     	n_elements(z) ne n_elements(ch4) or $
     	n_elements(z) ne n_elements(o3p) or $
     	n_elements(z) ne n_elements(l_reff) or $
     	n_elements(z) ne n_elements(i_reff)) then begin
    print,'Error: The input arrays do not have the same dimension size'
    if(keyword_set(dostop)) then stop,'Stopping inside routine for debugging'
    return,fail
  endif

  	; More QC
  stda = fix(stdatmos+0.5)
  if(stda lt 1 or stda gt 6) then begin
    print,'Error: The standard atmosphere must be an integer between 1 and 6'
    if(keyword_set(dostop)) then stop,'Stopping inside routine for debugging'
    return,fail
  endif

  	; And more QC
  foo = where(lwc lt 0, nfoo)
  if(nfoo gt 0) then $
    print,'   Warning: the LWC profile has negative values in it (replacing with zeros)'
  foo = where(iwc lt 0, nfoo)
  if(nfoo gt 0) then $
    print,'   Warning: the IWC profile has negative values in it (replacing with zeros)'
  foo = where(q   lt 0, nfoo)
  if(nfoo gt 0) then $
    print,'   Warning: the q profile has negative values in it (replacing with zeros)'
  foo = where(p   le 0, nfoo)
  if(nfoo gt 0) then begin
    print,'   Error: the p profile has negative values in it'
    if(keyword_set(dostop)) then stop,'Stopping inside routine for debugging'
    return,fail
  endif
  bar = where(lwc gt 0, nbar)
  if(nbar gt 0) then begin
    foo = where(l_reff(bar) le min_l_reff or l_reff(bar) gt max_l_reff, nfoo)
    if(nfoo gt 0) then $
      print,'   Warning: the liquid Reff profile has values outside the reasonable limits'
  endif
  bar = where(iwc gt 0, nbar)
  if(nbar gt 0) then begin
    foo = where(i_reff(bar) le min_i_reff or i_reff(bar) gt max_i_reff, nfoo)
    if(nfoo gt 0) then $
      print,'   Warning: the  ice   Reff profile has values outside the reasonable limits'
  endif

	; Make sure there are no residual RRTM input files hanging around
  spawn,'rm -f TAPE6 TAPE7 INPUT_RRTM IN_CLD_RRTM OUTPUT_RRTM'

	; Are there clouds in this run?  If so, then create the needed input
	; cloud file for the calculation
  lw = lwc(0:n_elements(z)-2) > 0	; Temporary array, emphasizing the layers
  iw = iwc(0:n_elements(z)-2) > 0	; Temporary array, emphasizing the layers
  lwp = total(lw)  ; Units g/m2
  iwp = total(iw)  ; Units g/m2
  if(lwp + iwp gt 0) then icld = 2 else icld = 0    ; Set Icld = 2 for max/random overlap
  if(icld ge 1) then begin
    openw,lun,'IN_CLD_RRTM',/get_lun
    printf,lun,format='(3x,I2,4x,I1,4x,I1)',2,3,1
    for i=0,n_elements(z)-2 do begin
      if(lw(i) gt 0 or iw(i) gt 0) then begin
        cwp = lw(i) + iw(i) 
	    fracice = iw(i) / cwp
	    cldfrac = 1.0
        printf,lun,format='(A1,1x,I3,E10.4,E10.4,E10.4,E10.4,E10.4)', $
		    ' ',i+1,cldfrac,cwp,fracice, $
		    (i_reff(i)>min_i_reff)<max_i_reff, $
		    (l_reff(i)>min_l_reff)<max_l_reff
      endif
    endfor 
    free_lun,lun
  endif

	; Create the input thermodynamic file needed for the calculation
  iout = 99
  numangs = 4
  		; These are the centers of the RRTM_LW bands
  lw_wnum_bands = [180, 425, 565, 665, 760, 900, 1030, 1130, 1285, 1435, $
  					1640, 1940, 2165, 2315, 2490, 2925]
  		; Interpolate the input surface emissivity spectrum to the RRTM
		; bnads, but do not extrapolate beyond the bounds of the input spectrum
  lw_band_emis = interpol(sfcemis,sfcwnum,lw_wnum_bands)
  foo = where(lw_wnum_bands lt min(sfcwnum), nfoo)
  bar = where(sfcwnum eq min(sfcwnum))
  if(nfoo gt 0) then lw_band_emis(foo) = sfcemis(bar(0))
  foo = where(lw_wnum_bands gt min(sfcwnum), nfoo)
  bar = where(sfcwnum eq max(sfcwnum))
  if(nfoo gt 0) then lw_band_emis(foo) = sfcemis(bar(0))
  lw_band_emis = (lw_band_emis > 0) < 1

	; Now write the INPUT_RRTM file
  rundecker,0,stda,z,p,t,q,iout=iout,icld=icld,numangs=numangs,mlayers=z, $
        co2_profile=co2, n2o_profile=n2o, ch4_prof=ch4, o3_prof=o3p, o3_units=o3_units, $
  	    sfc_temp=sfctemp, sfc_emis=lw_band_emis,co2_mix=0,silent=silent

	; Run the model
  spawn,rrtm_lw_command

	; Read in the output, if it exists
  files = findfile('OUTPUT_RRTM',count=count)
  if(count ne 1) then begin
    print,'Error: The RRTM_LW model did not run properly -- input files may be messed up'
    if(keyword_set(dostop)) then stop,'Stopping inside routine for debugging'
    return,fail
  endif else begin
    		; Read in the output file
    read_rrtm_lw, nregions=n_elements(lw_wnum_bands) + 1, path='.', filename='OUTPUT_RRTM', olr=olr, slr=slr, $
    	wnum=wnum, wbounds=wbounds, fluxu=fluxu, fluxd=fluxd, fluxn=fluxn, pres=pres, hr=hr

    		; Simple QC to make sure model ran correctly
    if(n_elements(pres) ne n_elements(p)) then begin
      print,'Error: Pressure levels in OUTPUT_RRTM do not match input pressure levels'
      if(keyword_set(dostop)) then stop,'Stopping inside routine for debugging'
      return,fail
    endif

		; Otherwise, I need to reverse all of the arrays, as the RRTM
		; outputs from TOA to SFC, but I desire it the other way around.
		; Note that I am only capturing the entire LW band, at the moment
    output = {success:1, hr:reform(reverse(hr)), fluxn:reform(reverse(fluxn)), $
    		fluxu:reform(reverse(fluxu)), fluxd:reform(reverse(fluxd)), $
		    olr:olr, slr:slr}
  endelse

  if(keyword_set(dostop)) then stop,'Stopping at end of routine before returning'

	; Make sure there are no residual RRTM input files hanging around
  spawn,'rm -f TAPE6 TAPE7 INPUT_RRTM IN_CLD_RRTM OUTPUT_RRTM OUT_CLD_RRTM'

  return,output
end
