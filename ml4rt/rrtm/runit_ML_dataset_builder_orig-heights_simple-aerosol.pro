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
function create_output_file, name, num_heights, num_bands

  fid = ncdf_create(name,/clobber)
  did0 = ncdf_dimdef(fid,'time',/unlimited)
  did1 = ncdf_dimdef(fid,'height',num_heights)
  did2 = ncdf_dimdef(fid,'wavenumber',num_bands)
  vid  = ncdf_vardef(fid,'valid_time_unix_sec',/long,did0)
    ncdf_attput,fid,vid,'long_name','Time since 1 Jan 1970 at 00:00:00 UTC'
    ncdf_attput,fid,vid,'units','seconds'
  vid = ncdf_vardef(fid,'julian_day',/float,did0)
    ncdf_attput,fid,vid,'long_name','Julian day (i.e., day of the year)'
    ncdf_attput,fid,vid,'units','days'
    ncdf_attput,fid,vid,'comment','day = 1 is 1 January'
  vid = ncdf_vardef(fid,'height_m_agl',/float,[did1,did0])
    ncdf_attput,fid,vid,'long_name','Height above ground'
    ncdf_attput,fid,vid,'units','m'
  vid = ncdf_vardef(fid,'height_thickness_metres',/float,[did1,did0])
    ncdf_attput,fid,vid,'long_name','Height thickness'
    ncdf_attput,fid,vid,'units','m'
  vid = ncdf_vardef(fid,'solar_zenith_angle_deg',/float,did0)
    ncdf_attput,fid,vid,'long_name','Solar zenith angle'
    ncdf_attput,fid,vid,'units','degrees'
  vid = ncdf_vardef(fid,'standard_atmosphere_enum',/float,did0)
    ncdf_attput,fid,vid,'long_name','Standard atmosphere used'
    ncdf_attput,fid,vid,'units','1 -- tropics, 2 -- Midlat Summer, 3 -- Midlat Winter, ' + $
    			    '4 -- Subarctic Summer, 5 -- Subarctic Winter, 6 -- US Standard Atmos'
  vid = ncdf_vardef(fid,'site_latitude_deg_n',/float,did0)
    ncdf_attput,fid,vid,'long_name','Site latitude'
    ncdf_attput,fid,vid,'units','degrees N'
  vid = ncdf_vardef(fid,'site_longitude_deg_e',/float,did0)
    ncdf_attput,fid,vid,'long_name','Site longitude'
    ncdf_attput,fid,vid,'units','degrees E'
  vid = ncdf_vardef(fid,'surface_albedo',/float,did0)
    ncdf_attput,fid,vid,'long_name','Surface albedo'
    ncdf_attput,fid,vid,'units','unitless'
  vid = ncdf_vardef(fid,'total_liquid_water_path_kg_m02',/float,did0)
    ncdf_attput,fid,vid,'long_name','Liquid water path'
    ncdf_attput,fid,vid,'units','kg/m2'
  vid = ncdf_vardef(fid,'total_ice_water_path_kg_m02',/float,did0)
    ncdf_attput,fid,vid,'long_name','Ice water path'
    ncdf_attput,fid,vid,'units','kg/m2'
  vid = ncdf_vardef(fid,'pressure_pascals',/float,[did1,did0])
    ncdf_attput,fid,vid,'long_name','Pressure'
    ncdf_attput,fid,vid,'units','Pa'
  vid = ncdf_vardef(fid,'pressure_thickness_pascals',/float,[did1,did0])
    ncdf_attput,fid,vid,'long_name','Pressure thickness'
    ncdf_attput,fid,vid,'units','Pa'
  vid = ncdf_vardef(fid,'temperature_kelvins',/float,[did1,did0])
    ncdf_attput,fid,vid,'long_name','Temperature'
    ncdf_attput,fid,vid,'units','K'
  vid = ncdf_vardef(fid,'vapour_mixing_ratio_kg_kg01',/float,[did1,did0])
    ncdf_attput,fid,vid,'long_name','Water vapor mixing ratio'
    ncdf_attput,fid,vid,'units','kg/kg'
  vid = ncdf_vardef(fid,'layerwise_liquid_water_path_kg_m02',/float,[did1,did0])
    ncdf_attput,fid,vid,'long_name','Layer liquid water content'
    ncdf_attput,fid,vid,'units','kg/m2'
  vid = ncdf_vardef(fid,'layerwise_ice_water_path_kg_m02',/float,[did1,did0])
    ncdf_attput,fid,vid,'long_name','Layer ice water content'
    ncdf_attput,fid,vid,'units','kg/m2'
  vid = ncdf_vardef(fid,'ozone_mixing_ratio_kg_kg01',/float,[did1,did0])
      ncdf_attput,fid,vid,'long_name','Ozone mixing ratio'
      ncdf_attput,fid,vid,'units','kg/kg'
  vid = ncdf_vardef(fid,'co2_concentration_ppmv',/float,[did1,did0])
      ncdf_attput,fid,vid,'long_name','CO2 concentration'
      ncdf_attput,fid,vid,'units','ppmv'
  vid = ncdf_vardef(fid,'ch4_concentration_ppmv',/float,[did1,did0])
      ncdf_attput,fid,vid,'long_name','CH4 concentration'
      ncdf_attput,fid,vid,'units','ppmv'
  vid = ncdf_vardef(fid,'n2o_concentration_ppmv',/float,[did1,did0])
      ncdf_attput,fid,vid,'long_name','N2O concentration'
      ncdf_attput,fid,vid,'units','ppmv'
  vid = ncdf_vardef(fid,'aerosol_extinction_metres01',/float,[did1,did0])
      ncdf_attput,fid,vid,'long_name','Aerosol extinction coefficient'
      ncdf_attput,fid,vid,'units','1/m'
  vid = ncdf_vardef(fid,'liquid_eff_radius_metres',/float,[did1,did0])
      ncdf_attput,fid,vid,'long_name','Effective radius of liquid particles'
      ncdf_attput,fid,vid,'units','m'
  vid = ncdf_vardef(fid,'ice_eff_radius_metres',/float,[did1,did0])
      ncdf_attput,fid,vid,'long_name','Effective radius of ice particles'
      ncdf_attput,fid,vid,'units','m'
  vid = ncdf_vardef(fid,'aerosol_albedo',/float,did0)
      ncdf_attput,fid,vid,'long_name','Aerosol single_scattering albedo'
      ncdf_attput,fid,vid,'units','unitless'
  vid = ncdf_vardef(fid,'aerosol_asymmetry_param',/float,did0)
      ncdf_attput,fid,vid,'long_name','Aerosol asymmetry parameter'
      ncdf_attput,fid,vid,'units','unitless'
  vid = ncdf_vardef(fid,'heating_rate_k_day01',/float,[did1,did2,did0])
    ncdf_attput,fid,vid,'long_name','SW radiative heating rate'
    ncdf_attput,fid,vid,'units','K/day'
  vid = ncdf_vardef(fid,'downwelling_flux_w_m02',/float,[did1,did2,did0])
    ncdf_attput,fid,vid,'long_name','SW downwelling flux'
    ncdf_attput,fid,vid,'units','W/m2'
  vid = ncdf_vardef(fid,'upwelling_flux_w_m02',/float,[did1,did2,did0])
    ncdf_attput,fid,vid,'long_name','SW upwelling flux'
    ncdf_attput,fid,vid,'units','W/m2'
  vid = ncdf_vardef(fid,'surface_downwelling_flux_w_m02',/float,[did2,did0])
    ncdf_attput,fid,vid,'long_name','SW downwelling flux at the surface'
    ncdf_attput,fid,vid,'units','W/m2'
  vid = ncdf_vardef(fid,'toa_upwelling_flux_w_m02',/float,[did2,did0])
    ncdf_attput,fid,vid,'long_name','SW upwelling flux at the top of the atmosphere'
    ncdf_attput,fid,vid,'units','W/m2'
  ncdf_attput,fid,/global,'Author','dave.turner@noaa.gov'
  ncdf_attput,fid,/global,'Comment1','Shortwave radiative flux calculations and atmospheric state inputs'
  ncdf_attput,fid,/global,'Comment2','SW calculations made with RRTM_SW'
  ncdf_control,fid,/endef
  ncdf_close,fid
return, 0
end
;---------------------------------------------------------------------------------
function append_to_output_file, name, index, valid_times_unix_sec, julian_days, $
    zenith_angles_deg, standard_atmo_enums, latitudes_deg_n, longitudes_deg_e, surface_albedos, $
    total_liquid_paths_g_m02, total_ice_paths_g_m02, liquid_paths_g_m02, ice_paths_g_m02, $
    temps_kelvins, pressures_mb, pressure_thicknesses_mb, heights_km_agl, height_thicknesses_km, vapour_mixing_ratios_g_kg01, $
    ozone_mixing_ratios_g_kg01, co2_concentrations_ppmv, ch4_concentrations_ppmv, n2o_concentrations_ppmv, $
    aerosol_extinctions_km01, liquid_eff_radii_microns, ice_eff_radii_microns, aerosol_albedos, aerosol_asymmetry_params, $
    heating_rate_matrix_k_day01, upwelling_flux_matrix_w_m02, downwelling_flux_matrix_w_m02, toa_upwelling_fluxes_w_m02, sfc_downwelling_fluxes_w_m02
  
  fid = ncdf_open(name, /write)
  ncdf_varput,fid,'valid_time_unix_sec',valid_times_unix_sec,offset=index
  ncdf_varput,fid,'julian_day',julian_days,offset=index
  ncdf_varput,fid,'solar_zenith_angle_deg',zenith_angles_deg,offset=index
  ncdf_varput,fid,'standard_atmosphere_enum',standard_atmo_enums,offset=index
  ncdf_varput,fid,'site_latitude_deg_n',latitudes_deg_n,offset=index
  ncdf_varput,fid,'site_longitude_deg_e',longitudes_deg_e,offset=index
  ncdf_varput,fid,'surface_albedo',surface_albedos,offset=index
  ncdf_varput,fid,'total_liquid_water_path_kg_m02',0.001*total_liquid_paths_g_m02,offset=index
  ncdf_varput,fid,'total_ice_water_path_kg_m02',0.001*total_ice_paths_g_m02,offset=index
  ncdf_varput,fid,'layerwise_liquid_water_path_kg_m02',0.001*liquid_paths_g_m02,offset=[0,index]
  ncdf_varput,fid,'layerwise_ice_water_path_kg_m02',0.001*ice_paths_g_m02,offset=[0,index]
  ncdf_varput,fid,'temperature_kelvins',temps_kelvins,offset=[0,index]
  ncdf_varput,fid,'pressure_pascals',100*pressures_mb,offset=[0,index]
  ncdf_varput,fid,'pressure_thickness_pascals',100*pressure_thicknesses_mb,offset=[0,index]
  ncdf_varput,fid,'height_m_agl',1000*heights_km_agl,offset=[0,index]
  ncdf_varput,fid,'height_thickness_metres',1000*height_thicknesses_km,offset=[0,index]
  ncdf_varput,fid,'vapour_mixing_ratio_kg_kg01',0.001*vapour_mixing_ratios_g_kg01,offset=[0,index]
  ncdf_varput,fid,'ozone_mixing_ratio_kg_kg01',0.001*ozone_mixing_ratios_g_kg01,offset=[0,index]
  ncdf_varput,fid,'co2_concentration_ppmv',co2_concentrations_ppmv,offset=[0,index]
  ncdf_varput,fid,'ch4_concentration_ppmv',ch4_concentrations_ppmv,offset=[0,index]
  ncdf_varput,fid,'n2o_concentration_ppmv',n2o_concentrations_ppmv,offset=[0,index]
  ncdf_varput,fid,'aerosol_extinction_metres01',0.001*aerosol_extinctions_km01,offset=[0,index]
  ncdf_varput,fid,'liquid_eff_radius_metres',0.000001*liquid_eff_radii_microns,offset=[0,index]
  ncdf_varput,fid,'ice_eff_radius_metres',0.000001*ice_eff_radii_microns,offset=[0,index]
  ncdf_varput,fid,'aerosol_albedo',aerosol_albedos,offset=index
  ncdf_varput,fid,'aerosol_asymmetry_param',aerosol_asymmetry_params,offset=index
  ncdf_varput,fid,'heating_rate_k_day01',heating_rate_matrix_k_day01,offset=[0,0,index]
  ncdf_varput,fid,'upwelling_flux_w_m02',upwelling_flux_matrix_w_m02,offset=[0,0,index]
  ncdf_varput,fid,'downwelling_flux_w_m02',downwelling_flux_matrix_w_m02,offset=[0,0,index]
  ncdf_varput,fid,'surface_downwelling_flux_w_m02',sfc_downwelling_fluxes_w_m02,offset=[0,index]
  ncdf_varput,fid,'toa_upwelling_flux_w_m02',toa_upwelling_fluxes_w_m02,offset=[0,index]
  ncdf_close,fid
  return, index + n_elements(valid_times_unix_sec)
end
;---------------------------------------------------------------------------------
; Main routine
pro runit, year

	; This is the flag telling the code to create a new output file (=1)
  do_create_output = 1
  outname = string(format='(A,I0,A)','output_file.',year,'.cdf')

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
    ncdf_varget,fid,'site_name',site_names
    site_names = string(site_names)
    ncdf_varget,fid,'site_latitude_deg_n',site_latitudes_deg_n
    ncdf_varget,fid,'site_longitude_deg_e',site_longitudes_deg_e
    ncdf_varget,fid,'forecast_hour',forecast_hours
    ncdf_varget,fid,'height_m_agl',height_matrix_m_agl
    ncdf_varget,fid,'height_thickness_metres',height_thickness_matrix_metres
    ncdf_varget,fid,'pressure_pascals',pressure_matrix_pascals
    ncdf_varget,fid,'pressure_thickness_pascals',pressure_thickness_matrix_pascals
    ncdf_varget,fid,'temperature_kelvins',temperature_matrix_kelvins
    ncdf_varget,fid,'vapour_mixing_ratio_kg_kg01',vapour_mixr_matrix_kg_kg01
    ncdf_varget,fid,'layerwise_liquid_water_path_kg_m02',layerwise_liquid_path_matrix_kg_m02
    ncdf_varget,fid,'layerwise_ice_water_path_kg_m02',layerwise_ice_path_matrix_kg_m02
    ncdf_varget,fid,'layerwise_snow_path_kg_m02',layerwise_snow_path_kg_m02
    ncdf_varget,fid,'ozone_mixing_ratio_kg_kg01',ozone_mixr_matrix_kg_kg01
    ncdf_varget,fid,'co2_concentration_ppmv',co2_concentration_matrix_ppmv
    ncdf_varget,fid,'ch4_concentration_ppmv',ch4_concentration_matrix_ppmv
    ncdf_varget,fid,'n2o_concentration_ppmv',n2o_concentration_matrix_ppmv
    ncdf_varget,fid,'aerosol_albedo',aerosol_albedo_matrix
    ncdf_varget,fid,'aerosol_asymmetry_param',aerosol_asymmetry_param_matrix
    ncdf_varget,fid,'aerosol_extinction_metres01',aerosol_extinction_matrix_metres01
    ncdf_varget,fid,'liquid_eff_radius_metres',liquid_eff_radius_matrix_metres
    ncdf_varget,fid,'ice_eff_radius_metres',ice_eff_radius_matrix_metres
    ncdf_varget,fid,'surface_albedo',surface_albedo_matrix
    ncdf_close,fid

    layerwise_ice_path_matrix_kg_m02 = layerwise_ice_path_matrix_kg_m02 + layerwise_snow_path_kg_m02
    sites = site_names
    hours = forecast_hours

    	; Now compute the valid time for these data
    dsecs = secs(0) + forecast_hours*60*60 + hh*60*60 + nn*60
    systime2ymdhms,dsecs,dyy,dmm,ddd,dhh,dnn,dss
    systime2julian,dsecs,dyy(0),djulian

	; Now loop over the desired sites, processing each one in turn
    rrtm_sw_command = '/home/user/vip/src/rrtm_sw_v2.7.1/rrtm_sw_gfortran_v2.7.2'
    
    for j=0,n_elements(sites)-1 do begin ; {
      npts = 0
    
      foo = where(sites(j) eq site_names, nfoo)
      if(nfoo gt 1) then stop,'Found multiple sites -- this should never happen'
      if(nfoo eq 0) then stop,'Did not find a site -- this should not happen'
      print,'  Working on '+site_names(foo)
      for k=0,n_elements(hours)-1 do begin ; {
        bar = where(abs(hours(k) - forecast_hours) lt 0.2, nbar)
        print,'  Working on forecast hour '+string(forecast_hours(bar))
  	    if(nbar ge 2) then stop,'Found multiple fhours for this time -- this should not happen'
	    if(nbar eq 1) then begin
          solarpos,dyy(bar(0)),dmm(bar(0)),ddd(bar(0)),dhh(bar(0)),dnn(bar(0)),dss(bar(0)), $
	  	    site_latitudes_deg_n(foo(0)),site_longitudes_deg_e(foo(0)),xhour,ra,sd,salt
          sza = 90 - salt
          if(sza(0) gt 85) then begin
            print,'Warning: solar zenith angle (' + string(sza(0)) + ') > 85 deg'
	        continue
          endif

          if(abs(site_latitudes_deg_n(foo(0))) lt 65) then begin
	        if(3 lt dmm(bar(0)) and dmm(bar(0)) le 9) then stdatmos = 2 $ 	; Mid-lat summer
	        else stdatmos = 3							; Mid-lat winter
	      endif else begin
	        if(3 lt dmm(bar(0)) and dmm(bar(0)) le 9) then stdatmos = 4 $ 	; Subarctic summer
	        else stdatmos = 5							; Subarctic winter
	      endelse
	      
	      these_heights_km_agl = reform(0.001*height_matrix_m_agl(*,foo(0),bar(0)))
	      these_height_thicknesses_km = reform(0.001*height_thickness_matrix_metres(*,foo(0),bar(0)))
          feh = where(satmos.z(*,stdatmos) gt max(these_heights_km_agl+2) and satmos.z(*,stdatmos) le 50,nfeh)
	      if(nfeh gt 0) then begin
                ; Append on the bit of standard atmosphere needed to get high into the stratosphere
	        these_heights_km_agl = [these_heights_km_agl,                           satmos.z(feh,stdatmos)]
	        these_height_thicknesses_km = [these_height_thicknesses_km,                           replicate(0.,nfeh)]
	        these_pressures_mb = [0.01*reform(pressure_matrix_pascals(*,foo(0),bar(0))),        satmos.p(feh,stdatmos)]
	        these_pressure_thicknesses_mb = [0.01*reform(pressure_thickness_matrix_pascals(*,foo(0),bar(0))),        replicate(0.,nfeh)]
	        these_temps_kelvins = [reform(temperature_matrix_kelvins(*,foo(0),bar(0))), satmos.t(feh,stdatmos)]
	        these_vapour_mixing_ratios_g_kg01 = [1000*reform(vapour_mixr_matrix_kg_kg01(*,foo(0),bar(0))),        satmos.q(feh,stdatmos)]
	        these_liquid_paths_g_m02 = [1000*reform(layerwise_liquid_path_matrix_kg_m02(*,foo(0),bar(0))>0), replicate(0.,nfeh)]
	        these_ice_paths_g_m02 = [1000*reform(layerwise_ice_path_matrix_kg_m02(*,foo(0),bar(0))>0), replicate(0.,nfeh)]
	        these_ozone_mixing_ratios_g_kg01 = [1000*reform(ozone_mixr_matrix_kg_kg01(*,foo(0),bar(0))>0), replicate(0.,nfeh)]
	        these_co2_concentrations_ppmv = [reform(co2_concentration_matrix_ppmv(*,foo(0),bar(0))>0), replicate(0.,nfeh)]
	        these_ch4_concentrations_ppmv = [reform(ch4_concentration_matrix_ppmv(*,foo(0),bar(0))>0), replicate(0.,nfeh)]
	        these_n2o_concentrations_ppmv = [reform(n2o_concentration_matrix_ppmv(*,foo(0),bar(0))>0), replicate(0.,nfeh)]
	        these_aerosol_extinctions_km01 = [1000*reform(aerosol_extinction_matrix_metres01(*,foo(0),bar(0))>0), replicate(0.,nfeh)]
	        these_liquid_eff_radii_microns = [1000000*reform(liquid_eff_radius_matrix_metres(*,foo(0),bar(0))>0), replicate(0.,nfeh)]
	        these_ice_eff_radii_microns = [1000000*reform(ice_eff_radius_matrix_metres(*,foo(0),bar(0))>0), replicate(0.,nfeh)]
          endif else begin
                ; The model profile is fine, but reform and rename the variables to be consistent
            these_pressures_mb = reform(0.01*pressure_matrix_pascals(*,foo(0),bar(0)))
            these_pressure_thicknesses_mb = reform(0.01*pressure_thickness_matrix_pascals(*,foo(0),bar(0)))
            these_temps_kelvins = reform(temperature_matrix_kelvins(*,foo(0),bar(0)))
            these_vapour_mixing_ratios_g_kg01 = 1000*reform(vapour_mixr_matrix_kg_kg01(*,foo(0),bar(0)))
            these_liquid_paths_g_m02 = 1000*reform(layerwise_liquid_path_matrix_kg_m02(*,foo(0),bar(0))>0)
            these_ice_paths_g_m02 = 1000*reform(layerwise_ice_path_matrix_kg_m02(*,foo(0),bar(0))>0)
            these_ozone_mixing_ratios_g_kg01 = 1000*reform(ozone_mixr_matrix_kg_kg01(*,foo(0),bar(0))>0)
            these_co2_concentrations_ppmv = reform(co2_concentration_matrix_ppmv(*,foo(0),bar(0))>0)
            these_ch4_concentrations_ppmv = reform(ch4_concentration_matrix_ppmv(*,foo(0),bar(0))>0)
            these_n2o_concentrations_ppmv = reform(n2o_concentration_matrix_ppmv(*,foo(0),bar(0))>0)
            these_aerosol_extinctions_km01 = 1000*reform(aerosol_extinction_matrix_metres01(*,foo(0),bar(0))>0)
            these_liquid_eff_radii_microns = 1000000*reform(liquid_eff_radius_matrix_metres(*,foo(0),bar(0))>0)
            these_ice_eff_radii_microns = 1000000*reform(ice_eff_radius_matrix_metres(*,foo(0),bar(0))>0)
          endelse

	      this_total_liquid_path_g_m02 = total(these_liquid_paths_g_m02)
	      this_total_ice_path_g_m02 = total(these_ice_paths_g_m02)

		; Quick QC check
 	      del = these_pressures_mb(1:n_elements(these_pressures_mb)-1) - these_pressures_mb(0:n_elements(these_pressures_mb)-2)
	      feh = where(del ge 0, nfeh)
	      if(nfeh gt 0) then begin
	        print,'Warning: Pressure profile increased with height -- skipping sample'
	        continue
	      endif

                ; Assume the aerosol optical depth (first index) is zero, and put nominal 
                ; values into the other aerosol fields (but it doesn't matter, because AOD = 0)
          aerosol_vector = [1., aerosol_albedo_matrix(foo(0),bar(0)), aerosol_asymmetry_param_matrix(foo(0),bar(0)), -1.7, 1.5]

		        ; Run the RT model RRTM_SW
	      outsw = make_rrtm_sw_calc(these_heights_km_agl, these_pressures_mb, these_temps_kelvins, these_vapour_mixing_ratios_g_kg01, $
	          these_liquid_paths_g_m02, these_ice_paths_g_m02, these_liquid_eff_radii_microns, these_ice_eff_radii_microns, $
	          stdatmos, 10000./[0.1,10], [1,1]-(surface_albedo_matrix(foo(0),bar(0))>0.03<0.99), djulian(bar(0)), sza, $
	          aerosol_vector, $
              co2=these_co2_concentrations_ppmv, ch4=these_ch4_concentrations_ppmv, n2o=these_n2o_concentrations_ppmv, o3p=these_ozone_mixing_ratios_g_kg01, o3_units='g/kg', $
              rrtm_sw_command, silent=1)
	      if(outsw.success eq 0) then begin
	        print,'Warning: RRTM_SW not run properly.  Time to investigate...'
	        continue
	      endif
	      
	      print,size(outsw.hr)
	      these_dim = size(outsw.hr, /dimensions)
	      this_heating_rate_matrix_k_day01 = replicate(1, these_dim[0], these_dim[1], 2)
	      this_heating_rate_matrix_k_day01[*,*,0] = outsw.hr
	      print,size(this_heating_rate_matrix_k_day01)
	      new_heating_rate_matrix_k_day01 = concatenate(3, this_heating_rate_matrix_k_day01, this_heating_rate_matrix_k_day01)
	      print,size(new_heating_rate_matrix_k_day01)

          if(npts eq 0) then begin
	        output_total_liquid_paths_g_m02 = this_total_liquid_path_g_m02
	        output_total_ice_paths_g_m02 = this_total_ice_path_g_m02
	        output_liquid_paths_g_m02 = transpose(these_liquid_paths_g_m02)
	        output_ice_paths_g_m02 = transpose(these_ice_paths_g_m02)
	        output_ozone_mixing_ratios_g_kg01 = transpose(these_ozone_mixing_ratios_g_kg01)
	        output_co2_concentrations_ppmv = transpose(these_co2_concentrations_ppmv)
	        output_ch4_concentrations_ppmv = transpose(these_ch4_concentrations_ppmv)
	        output_n2o_concentrations_ppmv = transpose(these_n2o_concentrations_ppmv)
	        output_aerosol_extinctions_km01 = transpose(these_aerosol_extinctions_km01)
	        output_liquid_eff_radii_microns = transpose(these_liquid_eff_radii_microns)
	        output_ice_eff_radii_microns = transpose(these_ice_eff_radii_microns)
	        output_temps_kelvins = transpose(these_temps_kelvins)
	        output_pressures_mb = transpose(these_pressures_mb)
	        output_pressure_thicknesses_mb = transpose(these_pressure_thicknesses_mb)
	        output_heights_km_agl = transpose(these_heights_km_agl)
	        output_height_thicknesses_km = transpose(these_height_thicknesses_km)
	        output_vapour_mixing_ratios_g_kg01 = transpose(these_vapour_mixing_ratios_g_kg01)
	        output_surface_albedos = reform(surface_albedo_matrix(foo(0),bar(0)))
	        output_aerosol_albedos = reform(aerosol_albedo_matrix(foo(0),bar(0)))
	        output_aerosol_asymmetry_params = reform(aerosol_asymmetry_param_matrix(foo(0),bar(0)))
	        output_zenith_angles_deg = sza(0)
	        output_standard_atmo_enums = stdatmos
	        output_latitudes_deg_n = site_latitudes_deg_n(foo)
	        output_longitudes_deg_e = site_longitudes_deg_e(foo)
	        output_times_unix_sec = dsecs(bar(0))
	        output_julian_days = djulian(bar(0))
	        output_heating_rate_matrix_k_day01 = transpose(outsw.hr)
	        output_upwelling_flux_matrix_w_m02 = transpose(outsw.fluxu)
	        output_downwelling_flux_matrix_w_m02 = transpose(outsw.fluxdtot)
	        output_toa_upwelling_fluxes_w_m02 = outsw.osr
	        output_sfc_downwelling_fluxes_w_m02 = outsw.ssr
	      endif else begin
	        output_total_liquid_paths_g_m02 = [output_total_liquid_paths_g_m02,this_total_liquid_path_g_m02]
	        output_total_ice_paths_g_m02 = [output_total_ice_paths_g_m02,this_total_ice_path_g_m02]
	        output_liquid_paths_g_m02 = [output_liquid_paths_g_m02,transpose(these_liquid_paths_g_m02)]
	        output_ice_paths_g_m02 = [output_ice_paths_g_m02,transpose(these_ice_paths_g_m02)]
	        
	        print,size(output_ozone_mixing_ratios_g_kg01)
	        print,size(transpose(these_ozone_mixing_ratios_g_kg01))
	        output_ozone_mixing_ratios_g_kg01 = [output_ozone_mixing_ratios_g_kg01,transpose(these_ozone_mixing_ratios_g_kg01)]
	        print,size(output_ozone_mixing_ratios_g_kg01)
	        
            output_co2_concentrations_ppmv = [output_co2_concentrations_ppmv,transpose(these_co2_concentrations_ppmv)]
            output_ch4_concentrations_ppmv = [output_ch4_concentrations_ppmv,transpose(these_ch4_concentrations_ppmv)]
            output_n2o_concentrations_ppmv = [output_n2o_concentrations_ppmv,transpose(these_n2o_concentrations_ppmv)]
            output_aerosol_extinctions_km01 = [output_aerosol_extinctions_km01,transpose(these_aerosol_extinctions_km01)]
            output_liquid_eff_radii_microns = [output_liquid_eff_radii_microns,transpose(these_liquid_eff_radii_microns)]
            output_ice_eff_radii_microns = [output_ice_eff_radii_microns,transpose(these_ice_eff_radii_microns)]
	        output_temps_kelvins = [output_temps_kelvins,transpose(these_temps_kelvins)]
	        output_pressures_mb = [output_pressures_mb,transpose(these_pressures_mb)]
	        output_pressure_thicknesses_mb = [output_pressure_thicknesses_mb,transpose(these_pressure_thicknesses_mb)]
	        output_heights_km_agl = [output_heights_km_agl,transpose(these_heights_km_agl)]
	        output_height_thicknesses_km = [output_height_thicknesses_km,transpose(these_height_thicknesses_km)]
	        output_vapour_mixing_ratios_g_kg01 = [output_vapour_mixing_ratios_g_kg01,transpose(these_vapour_mixing_ratios_g_kg01)]
	        output_surface_albedos = [output_surface_albedos,reform(surface_albedo_matrix(foo(0),bar(0)))]
	        output_aerosol_albedos = [output_aerosol_albedos,reform(aerosol_albedo_matrix(foo(0),bar(0)))]
	        output_aerosol_asymmetry_params = [output_aerosol_asymmetry_params,reform(aerosol_asymmetry_param_matrix(foo(0),bar(0)))]
	        output_zenith_angles_deg = [output_zenith_angles_deg,sza(0)]
	        output_standard_atmo_enums = [output_standard_atmo_enums,stdatmos]
	        output_latitudes_deg_n = [output_latitudes_deg_n,site_latitudes_deg_n(foo)]
	        output_longitudes_deg_e = [output_longitudes_deg_e,site_longitudes_deg_e(foo)]
	        output_times_unix_sec= [output_times_unix_sec,dsecs(bar(0))]
	        output_julian_days = [output_julian_days,djulian(bar(0))]
	        output_heating_rate_matrix_k_day01 = [output_heating_rate_matrix_k_day01,transpose(outsw.hr)]
	        output_upwelling_flux_matrix_w_m02 = [output_upwelling_flux_matrix_w_m02,transpose(outsw.fluxu)]
	        output_downwelling_flux_matrix_w_m02 = [output_downwelling_flux_matrix_w_m02,transpose(outsw.fluxdtot)]
	        output_toa_upwelling_fluxes_w_m02 = [output_toa_upwelling_fluxes_w_m02,outsw.osr]
	        output_sfc_downwelling_fluxes_w_m02 = [output_sfc_downwelling_fluxes_w_m02,outsw.ssr]
	        
	        print,size(output_heating_rate_matrix_k_day01)
	        
	        
	      endelse
	      npts = n_elements(output_zenith_angles_deg)
	    endif
      endfor ; } loop over k
      
    	  ; Transpose the 2d arrays to get them in the right shape
      if(npts gt 0) then begin
        output_liquid_paths_g_m02 = transpose(output_liquid_paths_g_m02)
        output_ice_paths_g_m02 = transpose(output_ice_paths_g_m02)
        output_temps_kelvins = transpose(output_temps_kelvins)
        output_pressures_mb = transpose(output_pressures_mb)
        output_pressure_thicknesses_mb = transpose(output_pressure_thicknesses_mb)
        output_heights_km_agl = transpose(output_heights_km_agl)
        output_height_thicknesses_km = transpose(output_height_thicknesses_km)
        output_vapour_mixing_ratios_g_kg01 = transpose(output_vapour_mixing_ratios_g_kg01)
        output_ozone_mixing_ratios_g_kg01 = transpose(output_ozone_mixing_ratios_g_kg01)
        output_co2_concentrations_ppmv = transpose(output_co2_concentrations_ppmv)
        output_ch4_concentrations_ppmv = transpose(output_ch4_concentrations_ppmv)
        output_n2o_concentrations_ppmv = transpose(output_n2o_concentrations_ppmv)
        output_aerosol_extinctions_km01 = transpose(output_aerosol_extinctions_km01)
        output_liquid_eff_radii_microns = transpose(output_liquid_eff_radii_microns)
        output_ice_eff_radii_microns = transpose(output_ice_eff_radii_microns)
        output_heating_rate_matrix_k_day01 = transpose(output_heating_rate_matrix_k_day01)
        output_upwelling_flux_matrix_w_m02 = transpose(output_upwelling_flux_matrix_w_m02)
        output_downwelling_flux_matrix_w_m02 = transpose(output_downwelling_flux_matrix_w_m02)

            ; If the netCDF file has not yet been created, then create it
        if(do_create_output eq 1) then begin
	      index = 0
          do_create_output = create_output_file(outname, n_elements(these_heights_km_agl), n_elements(output_toa_upwelling_fluxes_w_m02))
        endif

		    ; Append output to the file
        print,'Adding ',n_elements(output_times_unix_sec),' samples to the output file ',outname, ' at ',index, format='(A,I0,A,A,A,I0)'
        index = append_to_output_file(outname, index, output_times_unix_sec, output_julian_days, $
          output_zenith_angles_deg, output_standard_atmo_enums, output_latitudes_deg_n, output_longitudes_deg_e, output_surface_albedos, $
	      output_total_liquid_paths_g_m02, output_total_ice_paths_g_m02, output_liquid_paths_g_m02, output_ice_paths_g_m02, $
	      output_temps_kelvins, output_pressures_mb, output_pressure_thicknesses_mb, output_heights_km_agl, output_height_thicknesses_km, output_vapour_mixing_ratios_g_kg01, $
	      output_ozone_mixing_ratios_g_kg01, output_co2_concentrations_ppmv, output_ch4_concentrations_ppmv, output_n2o_concentrations_ppmv, $
          output_aerosol_extinctions_km01, output_liquid_eff_radii_microns, output_ice_eff_radii_microns, output_aerosol_albedos, output_aerosol_asymmetry_params, $
	      output_heating_rate_matrix_k_day01, output_upwelling_flux_matrix_w_m02, output_downwelling_flux_matrix_w_m02, output_toa_upwelling_fluxes_w_m02, output_sfc_downwelling_fluxes_w_m02)
      endif
    endfor  ; } loop over j
  endfor  ; } loop over i

  return
end
