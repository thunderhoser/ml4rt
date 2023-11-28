;+
; $Id: make_rrtm_sw_calc.pro,v 1.6 2021/09/27 16:12:32 dave.turner Exp $
;
; Abstract:
;	This routine makes a RRTM_SW calculation for the input atmospheric state
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
  function make_rrtm_sw_calc, $		; Returns the above structure
  	z, $				; Input height array [km AGL]
	p, $				; Input pressure array [mb]
	t, $				; Input temperature array [K]
	q, $				; Input water vapor mixing ratio array [g/kg]
	lwc, $ 				; Input liquid water content array [g/m2]
	iwc, $ 				; Input ice water content array [g/m2]
	l_reff, $			; Input liquid water effective radius array [um]
	i_reff, $			; Input ice water effective radius array [um]
	stdatmos, $			; The standard atmosphere to use [1-6]
	sfcwnum, $			; The input wnum values for emissivity spectrum
	sfcemis, $			; The input surface emissivity spectrum
	juldate, $			; The julian date (1 = Jan 1)
	sza, $				; The solar zenith angle [deg]; 0 is sun overhead
    aerosol, $          ; Structure for aerosol: {aod, SSA, asym, angstrom exp, scale height}, where:
                              ;   (0) AOD is the aerosol optical depth at 0.5 um, 
                              ;   (1) SSA is single scatter albedo, 
                              ;   (2) asym is the asymmetry parameter, 
                              ;   (3) angtrom exponent, and
                              ;   (4) the 1/e scale height [km AGL] (a value of 1.5 km is good)
    aer_ext=aer_ext, $  ; If exists and has same size of z, then the AOD in the aerosol variale is ignored
                              ; Note that the aer_ext is at 0.5 um, and must have units of 1/km.
	co2=co2, $			; Input CO2 concentration profile [ppm]
	ch4=ch4, $			; Input CH4 concentration profile [ppm]
	n2o=n2o, $			; Input N2O concentration profile [ppm]
	o3p=o3p, $			; Input O3 concentration profile
    o3_units=o3_units,$ ; Units of the O3 profile (must be one of "ppmv", "cm-3", "g/kg", "g/m3")
	rrtm_sw_command, $	; The path/command needed to run the model
	silent=silent, $	; If set, then create the RRTM_INPUT files quietly
	version=version, $	; Returns the version of this routine
	dostop=dostop		; Set this to stop inside routine


  ; Note: the LWC and IWC are the amount of liquid and ice in LAYER i between the 
  ;          ith and (i+1)th levels.  The uppermost value of these arrays should be zero.
  ;          Be careful with the units here!
;-

  fail = {success:0}
  version = '$Id: make_rrtm_sw_calc.pro,v 1.6 2021/09/27 16:12:32 dave.turner Exp $'

        ; Check the aerosol structure first
  if(n_elements(aerosol) eq 0) then aerosol = [0., 0.99, 0.85, -2., 1.5]
  if(n_elements(aerosol) ne 5) then begin
    print,'Error: The aerosol input must be a 5-element vector -- see instructions'
    return, fail
  endif
  if(aerosol(0) lt 0) then begin
    print,'Error: the aerosol optical depth must be non-negative'
    return, fail
  end
  if(aerosol(1) le 0 or aerosol(1) gt 1) then begin
    print,'Error: the aerosol single scatter albedo must be 0 < SSA <= 1'
    return, fail
  end
  if(aerosol(2) le 0 or aerosol(2) ge 1) then begin
    print,'Error: the aerosol asymmetry parameter must be 0 < g < 1'
    return, fail
  end
  if(aerosol(3) le -4 or aerosol(3) ge 0) then begin
    print,'Error: the aerosol angstrom exponent must be -4 < angExp < 0'
    return, fail
  end
  if(aerosol(4) le 0 or aerosol(4) ge 3) then begin
    print,'Error: the aerosol scale height coefficient must be 0 < ScaleHeight < 3'
    return, fail
  end
  if(n_elements(aer_ext) gt 0 and n_elements(aer_ext) ne n_elements(z)) then begin
    print,'Error: an aerosol extinction profile was input, but its dimension does not match z'
    return, fail
  endif

  	; Some limits on the input effective radii [um]
  min_l_reff =  2.6
  max_l_reff = 50.0
  min_i_reff =  5.1
  max_i_reff = 95.0
    
    ; Somewhat reasonable default profiles
  if(n_elements(co2) eq 0) then co2 = replicate(400.0,n_elements(z)) ; [ppm]
  if(n_elements(n2o) eq 0) then n2o = replicate(0.310,n_elements(z)) ; [ppm]
  if(n_elements(ch4) eq 0) then ch4 = replicate(1.793,n_elements(z)) ; [ppm]
  if(n_elements(o3p) eq 0) then begin
    o3p = replicate(0.000,n_elements(z))    ; Default will be the std atmosphere
    o3_units = fix(stdatmos+0.5)            ;   which is why I put zeros into above profile
  endif

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
    print,n_elements(z),' ',n_elements(p),' ',n_elements(t),' ',n_elements(q),' ',n_elements(lwc),' ',n_elements(iwc),' ',n_elements(co2),' ',n_elements(n2o),' ',n_elements(ch4),' ',n_elements(o3p),' ',n_elements(l_reff),' ',n_elements(i_reff)
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
  spawn,'rm -f TAPE6 TAPE7 INPUT_RRTM IN_CLD_RRTM OUTPUT_RRTM OUT_CLD_RRTM'

	; Are there clouds in this run?  If so, then create the needed input
	; cloud file for the calculation
  lw = lwc(0:n_elements(z)-2) > 0	; Temporary array, emphasizing the layers
  iw = iwc(0:n_elements(z)-2) > 0	; Temporary array, emphasizing the layers
  lwp = total(lw)  ; Units g/m2
  iwp = total(iw)  ; Units g/m2
  if(lwp + iwp gt 0) then icld = 1 else icld = 0
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

    ; Are there aerosols in this run?  If so, then create the needed input
    ; aerosol file for the calculation
  iaer = 0
  if(aerosol(0) gt 0) then begin
    iaer = 1
        ; Build an extinction profile using the scale height (units km^(-1))
    ext  = exp(-z / aerosol(4))
    ext  = ext / total(ext) * aerosol(0)
        ; If an extinction profile was passed in, then override the one computed above
    if(n_elements(aer_ext) eq n_elements(z)) then ext = aer_ext > 0
        ; Convert extinction from 500 nm to 1000 nm, as that is what RRTM_SW needs
    ext  = ext * (1.0/0.5)^aerosol(3)   ; Convert EXT_500nm to EXT_1000nm
        ; Write the aerosol file used by the RRTM_SW
    openw,lun,'IN_AER_RRTM',/get_lun
    printf,lun,format='(3x,I2)',1
    printf,lun,format='(2x,I3,4x,I1,4x,I1,4x,I1,3(F8.2))',n_elements(ext),0,0,0,aerosol(3),1,0
    for i=0,n_elements(z)-1 do begin
      printf,lun,format='(2x,I3,F7.4)',i,ext(i)
    endfor
    printf,lun,format='(F5.2)',aerosol(1)
    printf,lun,format='(F5.2)',aerosol(2)
    free_lun,lun
  endif

	; Create the input thermodynamic file needed for the calculation
  iout = 98
  numangs = 0		; in RRTM_SW, this is numangs=1 implies 8 stream, while 0 implies 4 stream
  		; These are the centers of the RRTM_SW bands
  sw_wnum_bands = [2925, 3625, 4325, 4900, 5650, 6925, 7875, 10450, 14425, 19325, $
  				25825, 33500, 44000, 1710]

  		; Interpolate the input surface emissivity spectrum to the RRTM
		; bnads, but do not extrapolate beyond the bounds of the input spectrum
  sw_band_emis = interpol(sfcemis,sfcwnum,sw_wnum_bands)
  foo = where(sw_wnum_bands lt min(sfcwnum), nfoo)
  bar = where(sfcwnum eq min(sfcwnum))
  if(nfoo gt 0) then sw_band_emis(foo) = sfcemis(bar(0))
  foo = where(sw_wnum_bands gt min(sfcwnum), nfoo)
  bar = where(sfcwnum eq max(sfcwnum))
  if(nfoo gt 0) then sw_band_emis(foo) = sfcemis(bar(0))
  sw_band_emis = (sw_band_emis > 0) < 1

	; Now write the INPUT_RRTM file
  rundecker,0,stda,z,p,t,q,iout=iout,icld=icld,iaer=iaer,numangs=numangs,mlayers=z, $
        co2_profile=co2, n2o_profile=n2o, ch4_prof=ch4, o3_prof=o3p, o3_units=o3_units, $
  	    sfc_emis=sw_band_emis,co2_mix=0,juldate=juldate,sza=sza,silent=silent

	; Run the model
  if(keyword_set(silent)) then spawn,'('+rrtm_sw_command+') >& /dev/null' else spawn,rrtm_sw_command

	; Read in the output, if it exists
  files = findfile('OUTPUT_RRTM',count=count)
  if(count ne 1) then begin
    print,'Error: The RRTM_SW model did not run properly -- input files may be messed up'
    if(keyword_set(dostop)) then stop,'Stopping inside routine for debugging'
    return,fail
  endif else begin
    		; Read in the output file
    read_rrtm_sw, nregions=n_elements(sw_wnum_bands) + 1, path='.', filename='OUTPUT_RRTM', osr=osr, ssr=ssr, $     
      wnum=wnum, wbounds=wbounds, fluxu=fluxu, fluxdtot=fluxdtot, fluxddir=fluxddir,$
      fluxddif=fluxddif, fluxn=fluxn, pres=pres, hr=hr

    		; Simple QC to make sure model ran correctly
    print,size(pres, /dimensions)[1]
    if(n_elements(pres) ne n_elements(p)) then begin
      print,'Error: Pressure levels in OUTPUT_RRTM do not match input pressure levels'
      if(keyword_set(dostop)) then stop,'Stopping inside routine for debugging'
      return,fail
    endif

		; Otherwise, I need to reverse all of the arrays, as the RRTM
		; outputs from TOA to SFC, but I desire it the other way around.
		; Note that I am only capturing the entire SW band, at the moment
    output = {success:1, hr:reform(reverse(hr)), fluxn:reform(reverse(fluxn)), $
    		fluxu:reform(reverse(fluxu)), fluxdtot:reform(reverse(fluxdtot)), $
		    osr:osr, ssr:ssr}
  endelse

  if(keyword_set(dostop)) then stop,'Stopping at end of routine before returning'

	; Make sure there are no residual RRTM input files hanging around
  spawn,'rm -f TAPE6 TAPE7 INPUT_RRTM IN_CLD_RRTM OUTPUT_RRTM OUT_CLD_RRTM IN_AER_RRTM'

  return,output
end
